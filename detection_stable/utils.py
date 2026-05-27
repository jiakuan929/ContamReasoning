import os
import random
import math
from typing import List, Literal

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _logdet_active_and_rank_from_gram(
    X_c: torch.Tensor,
    ridge: float = 1e-4,
    eps_rel: float = 1e-12,
) -> tuple[float, int]:
    X_c = X_c.to(torch.float64)
    N = X_c.shape[0]
    denom = max(N - 1, 1)

    # Gram: K = X X^T / (N - 1)
    K = (X_c @ X_c.T) / denom  # [N, N]

    evals = torch.linalg.eigvalsh(K)  # [N]
    evals = torch.clamp(evals, min=0.0)

    if evals.numel() == 0:
        return 0.0, 0

    max_e = evals.max()
    if max_e <= 0:
        return 0.0, 0

    thresh = max_e * eps_rel
    mask = evals > thresh
    evals_eff = evals[mask]
    r = int(mask.sum().item())

    if r == 0:
        return 0.0, 0

    L = torch.sum(torch.log(evals_eff + ridge)).item()
    return L, r


@torch.no_grad()
def gaussian_mi_gram_proper(
    A_samples: torch.Tensor,
    G_samples: torch.Tensor,
    ridge: float = 1e-4,
    eps_rel: float = 1e-12,
) -> float:
    A = A_samples.detach().to(torch.float64)
    G = G_samples.detach().to(torch.float64)

    N, D_a = A.shape
    _, D_g = G.shape
    D_joint = D_a + D_g

    Ac = A - A.mean(dim=0, keepdim=True)
    Gc = G - G.mean(dim=0, keepdim=True)

    L_a, r_a = _logdet_active_and_rank_from_gram(Ac, ridge=ridge, eps_rel=eps_rel)
    L_g, r_g = _logdet_active_and_rank_from_gram(Gc, ridge=ridge, eps_rel=eps_rel)

    # Joint: Gram([A,G]) = Ac Ac^T + Gc Gc^T
    denom = max(N - 1, 1)
    K_a = Ac @ Ac.T
    K_g = Gc @ Gc.T
    K_joint = (K_a + K_g) / denom


    evals_j = torch.linalg.eigvalsh(K_joint)
    evals_j = torch.clamp(evals_j, min=0.0)

    if evals_j.numel() == 0:
        return 0.0

    max_ej = evals_j.max()
    if max_ej <= 0:
        return 0.0

    thresh_j = max_ej * eps_rel
    mask_j = evals_j > thresh_j
    evals_j_eff = evals_j[mask_j]
    r_j = int(mask_j.sum().item())

    if r_j == 0:
        return 0.0

    L_j = torch.sum(torch.log(evals_j_eff + ridge)).item()

    log_eps = math.log(ridge)
    const_term = (r_j - r_a - r_g) * log_eps

    mi = 0.5 * ((L_a + L_g - L_j) + const_term)
    return float(mi)


@torch.no_grad()
def free_generate(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    **generation_kwargs,
) -> dict:
    # print("Free generation...")
    model.eval()
    messages = [{"role": "user", "content": question}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt_text, return_tensors="pt").to(model.device)
    input_ids = enc.input_ids
    out = model.generate(
        **enc,
        **generation_kwargs
    )
    gen_ids = out[0, input_ids.size(1):]  # response tokens only
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    return {
        "prompt_text": prompt_text,
        "prompt_ids": input_ids,
        "gen_ids": gen_ids.unsqueeze(0),  # [1, L_resp]
        "gen_text": gen_text
    }


class AGCollector:
    """Collect per-position inputs (a) and grad_outputs (g) for selected Linear modules."""

    def __init__(self, name: str):
        self.name = name
        self.as_list = []  # list of [N_pos, d_in]
        self.gs_list = []  # list of [N_pos, d_out]
        self._fh = None
        self._bh = None

    def _forward_hook(self, module, inp, out):
        x: torch.Tensor = inp[0] if isinstance(inp, tuple) else inp
        a = x[0].detach()
        self.as_list.append(a)

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple; for Linear, grad_out[0] has shape like the forward output
        g = grad_out[0] if isinstance(grad_out, tuple) else grad_out
        gg = g[0].detach()
        self.gs_list.append(gg)

    def attach(self, module: torch.nn.Module):
        self._fh = module.register_forward_hook(self._forward_hook)
        # full backward hook catches grad wrt output activations
        self._bh = module.register_full_backward_hook(self._backward_hook)

    def detach(self):
        if self._fh is not None:
            self._fh.remove()
        if self._bh is not None:
            self._bh.remove()

    def stacked(self):
        # Concatenate across potentially multiple calls (should be one call here)
        A = torch.cat(self.as_list, dim=0) if self.as_list else None  # [N_pos, d_in]
        G = torch.cat(self.gs_list, dim=0) if self.gs_list else None  # [N_pos, d_out]
        return A, G
    

def collect_AG(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    target_response: str | None,
    module_names: List[str],
    generation_kwargs: dict | None,
):
    if target_response is None:
        assert generation_kwargs is not None
        gen = free_generate(model, tok, question, **generation_kwargs)
        prompt_ids = gen["prompt_ids"].to(model.device)
        gen_ids = gen["gen_ids"].to(model.device)
        prompt_len = prompt_ids.shape[1]
        input_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    else:
        messages = [{"role": "user", "content": question}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = text + target_response
        full_enc = tok(full_text, return_tensors="pt").to(model.device)
        prompt_enc = tok(text, return_tensors="pt").to(model.device)
        prompt_len = prompt_enc.input_ids.shape[1]
        input_ids = full_enc.input_ids
    B, S = input_ids.shape

    target_mods = [
        (name, mod) for name, mod in model.named_modules() if name in module_names
    ]
    assert len(target_mods) > 0, "No target modules found for given module_names"

    # Only set requires_grad for our target modules.
    # To reduce occupied memory caused by too many unneccessary grads
    # print("Selecting target modules requires_grad...")
    for name, param in model.named_parameters():
        is_target = False
        for mod_name, _ in target_mods:
            if mod_name in name:
                is_target = True
                break
        param.requires_grad_(is_target)

    collectors = {}
    for name, mod in target_mods:
        col = AGCollector(name)
        col.attach(mod)
        collectors[name] = col

    logits = model(input_ids=input_ids).logits
    V = logits.shape[-1]

    logits_shifted = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    label_mask = torch.ones_like(labels, dtype=torch.bool)
    label_mask[:, :prompt_len - 1] = False
    labels_masked = labels.masked_fill(~label_mask, -100)

    loss = torch.nn.functional.cross_entropy(
        logits_shifted.view(-1, V), labels_masked.view(-1),
        reduction="mean", ignore_index=-100
    )

    model.zero_grad(set_to_none=True)
    loss.backward()

    for col in collectors.values():
        col.detach()

    A = {}
    G = {}
    for name, col in collectors.items():
        A_samples, G_samples = col.stacked()
        A[name] = A_samples[prompt_len - 1: -1, :]
        G[name] = G_samples[prompt_len - 1: -1, :]
    return A, G
