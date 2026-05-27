import os
import random
from typing import List, Literal

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class NameMap:
    model_name = {
        "qwen2.5-1.5b": "/netcache/huggingface/Qwen2.5-1.5B",
        "qwen2.5-1.5b-it": "/netcache/huggingface/Qwen2.5-1.5B-Instruct",
        "qwen2.5-7b": "/netcache/huggingface/Qwen2.5-7B",
        "qwen2.5-7b-it": '/netcache/huggingface/Qwen2.5-7B-Instruct',
        "qwen2.5-math-1.5b": "/netcache/huggingface/Qwen2.5-Math-1.5B",
        "qwen2.5-math-1.5b-it": "/netcache/huggingface/Qwen2.5-Math-1.5B-Instruct",
        "qwen2.5-math-7b": "/netcache/huggingface/Qwen2.5-Math-7B",
        "qwen2.5-math-7b-it": "/netcache/huggingface/Qwen2.5-Math-7B-Instruct",

        "qwen2.5-32b-it": "/netcache/huggingface/Qwen2.5-32B-Instruct",
        "qwen3-1.7b": "/netcache/huggingface/Qwen3-1.7B",
        "qwen3-4b": "/netcache/huggingface/Qwen3-4B",

        "llama3.1-8b-it": "/netcache/huggingface/Llama-3.1-8B-Instruct"
    }


def load_hf_model_tokenizer(model_name_or_path: str):
    """
    Load standard huggingface model and tokenizer.
    """
    if model_name_or_path in NameMap.model_name:
        model_name_or_path = NameMap.model_name[model_name_or_path]
    if not os.path.exists(model_name_or_path):
        model_name_or_path = model_name_or_path.replace("netcache", "mnt/usercache")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()
    return model, tok


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def single_forward(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    ret: Literal["logits", "probs", "token"] = "token",
):
    model_inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**model_inputs).logits
    if ret == "logits":
        return logits
    elif ret == "probs":
        return torch.softmax(logits, dim=-1)

    next_token_id = torch.argmax(logits[0, -1, :], dim=-1).item()
    next_token_str = tok.decode(next_token_id)
    next_token_prob = torch.softmax(logits, dim=-1)[0, -1, next_token_id].item()
    return {"id": next_token_id, "str": next_token_str, "prob": next_token_prob}


def model_regular_completion(
    model: AutoModelForCausalLM, tok: AutoTokenizer, prefix: str, max_new_tokens: int
):
    """
    Execute regular model completion given a context prefix. The generation uses greedy decoding.
    """
    enc = tok(prefix, return_tensors="pt").to(model.device)
    len_prefix = enc.input_ids.shape[1]
    with torch.no_grad():
        gen_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    predicted_ids = gen_ids[0, len_prefix:]
    resp = tok.decode(predicted_ids, skip_special_tokens=True)
    return predicted_ids, resp


def empirical_cov(X: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Compute centered covariance Cov(X) with ridge; X=[N, D]."""
    Xc = X - X.mean(dim=0, keepdim=True)
    N = X.shape[0]
    C = (Xc.t() @ Xc) / max(N - 1, 1)
    if eps > 0:
        C = C + eps * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
    return C


def logdet_psd(M: torch.Tensor) -> torch.Tensor:
    """Stable log|M| for (near) PSD matrices."""
    # Use slogdet; if negative/zero due to numerical issues, add a small ridge
    sign, ld = torch.slogdet(M)
    if (sign <= 0) or torch.isnan(ld):
        M = M + 1e-6 * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        sign, ld = torch.slogdet(M)
    return ld


def gaussian_mi(
    A_samples: torch.Tensor, G_samples: torch.Tensor, ridge: float = 1e-4
) -> float:
    """
    Gaussian MI between random vectors a ~ A_samples and g ~ G_samples.
    - A_samples: [N, d_in]
    - G_samples: [N, d_out]
    I(a;g) = 0.5 * log ( |Σ_a| |Σ_g| / |Σ| ), Σ = [[Σ_a, Σ_ag], [Σ_ga, Σ_g]]
    """
    A_samples = A_samples.to(torch.float64)
    G_samples = G_samples.to(torch.float64)

    Sa = empirical_cov(A_samples, eps=ridge)  # [d_in, d_in]
    Sg = empirical_cov(G_samples, eps=ridge)  # [d_out, d_out]

    # Cross-covariance Σ_ag
    Ac = A_samples - A_samples.mean(dim=0, keepdim=True)
    Gc = G_samples - G_samples.mean(dim=0, keepdim=True)
    N = A_samples.shape[0]
    Sag = (Ac.t() @ Gc) / max(N - 1, 1)  # [d_in, d_out]

    # Joint covariance
    top = torch.cat([Sa, Sag], dim=1)
    bottom = torch.cat([Sag.t(), Sg], dim=1)
    S_joint = torch.cat([top, bottom], dim=0)

    # log-determinant MI
    ld_a = logdet_psd(Sa)
    ld_g = logdet_psd(Sg)
    ld_joint = logdet_psd(S_joint)
    mi = 0.5 * float((ld_a + ld_g - ld_joint).clamp_min(0.0))
    return mi


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
        # print(f"FREE Generated response:\n{gen['gen_text']}")
        # print(f"prompt_ids.shape: {prompt_ids.shape}  gen_ids.shape: {gen_ids.shape}")
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
