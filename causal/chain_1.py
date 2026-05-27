import os
import json
import sys
import argparse
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bas import load_hf_model_tokenizer, free_generate


class ActivationCollector:
    def __init__(self, mod: torch.nn.Module):
        self.activations: List[torch.Tensor] = []
        self.h = mod.register_forward_hook(self.hook)

    def hook(self, module, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        a = x[:, -1, :].detach()
        self.activations.append(a)

    def detach(self):
        self.h.remove()


def compute_spectrum_from_A(A: torch.Tensor, max_samples: int = 20000):
    """
    A: [N, d] activation matrix (rows = samples/tokens)
    Returns: eigenvalues λ ∈ [d] and eigenvectors, in descending order
    """
    # optional subsampling when N is huge
    N, d = A.shape
    if N > max_samples:
        idx = torch.randperm(N, device=A.device)[:max_samples]
        A = A[idx]

    # center along sample dimension
    A = A.to(torch.float64)  # better numerical stability
    A = A - A.mean(dim=0, keepdim=True)  # [N, d]

    # covariance C = (1/N) A^T A, shape [d, d]
    C = (A.T @ A) / A.shape[0]
    ridge = 1e-8
    C = C + ridge * torch.eye(d, device=C.device, dtype=C.dtype)

    # eigen decomposition (symmetric PSD)
    # eigh returns eigenvalues in ascending order
    evals, evecs = torch.linalg.eigh(C)  # evals: [d]
    evals = torch.clamp(evals, min=0.0)  # avoid tiny negatives
    # sort descending
    sorted_indices = torch.argsort(evals, descending=True)
    evals_sorted = evals[sorted_indices]
    evecs_sorted = evecs[:, sorted_indices]
    return evals_sorted, evecs_sorted


def reshape_spectrum(A: torch.Tensor, tau=0.5, eps=1e-8):
    """
    调整激活值 A 的谱分布 (Spectrum)
    A: [N, d] 的张量
    tau: 0 (原状) 到 1 (完全平滑/白化) 之间的缩放因子
    """
    mu = A.mean(dim=0, keepdim=True)
    A_centered = A - mu

    U, S, Vh = torch.linalg.svd(A_centered, full_matrices=False)

    S_new = torch.pow(S + eps, 1 - tau)

    A_reshaped = U @ torch.diag(S_new) @ Vh

    return A_reshaped + mu

def reshape_spectrum_keep_evecs(A: torch.Tensor, tau=0.5, eps=1e-8, max_samples: int = 20000):

    N, d = A.shape
    if N > max_samples:
        idx = torch.randperm(N, device=A.device)[:max_samples]
        A = A[idx]

    mu = A.mean(dim=0, keepdim=True)
    Ac = (A - mu).to(torch.float64)

    # covariance eigendecomposition
    C = (Ac.T @ Ac) / Ac.shape[0]
    C = C + 1e-8 * torch.eye(d, device=A.device, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(C)         # ascending
    evals = torch.clamp(evals, min=0.0)
    idx = torch.argsort(evals, descending=True)
    lam = evals[idx]                            # [d]
    U = evecs[:, idx]                           # [d, d]

    lam_new = torch.pow(lam + eps, 1 - tau)

    B = Ac @ U                                  # [N, d]
    scale = torch.sqrt(lam_new / (lam + eps))   # [d]
    B_new = B * scale.unsqueeze(0)              # [N, d]

    Ac_new = B_new @ U.T                        # [N, d]
    return (Ac_new + mu.to(torch.float64)).to(A.dtype)

@torch.no_grad()
def reshape_spectrum_standard(
    A: torch.Tensor,
    tau: float = 0.5,
    eps: float = 1e-8,
    ridge: float = 1e-8,
    max_samples: int = 20000,
    return_centered: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reshape the spectrum (eigenvalues) of Cov(A) while keeping eigenvectors (directions) fixed,
    and preserving total variance (trace of covariance).

    A: [N, d] activation matrix (rows = samples/tokens)
    tau: in [0, 1]. tau=0 -> no change; tau->1 -> flatten spectrum more strongly.
         We use: lambda' = (lambda + eps)^(1 - tau), then rescale to keep sum(lambda') == sum(lambda).
    eps: numerical stability for power / division
    ridge: small ridge added to covariance for stable eigendecomposition
    max_samples: optional subsampling if N is huge (for speed)
    return_centered: if True, returns centered A' (without adding mean back)

    Returns:
      A_new: [N, d] spectrum-reshaped activation (same dtype/device as A)
      U:     [d, d] eigenvectors of Cov(A_centered) sorted by descending eigenvalues
      lam_new: [d] reshaped eigenvalues (descending), with sum(lam_new) == sum(lam)
    Notes:
      - Directions are kept fixed by operating in the eigenbasis of Cov(A_centered).
      - Total variance is preserved by rescaling lam_new to match sum(lam).
      - A_new is constructed so that (approximately) Cov(A_new_centered) = U diag(lam_new) U^T.
    """
    if A.dim() != 2:
        raise ValueError(f"A must be 2D [N, d], got shape {tuple(A.shape)}")
    # if not (0.0 <= tau <= 1.0):
    #     raise ValueError(f"tau must be in [0,1], got {tau}")

    device = A.device
    dtype_in = A.dtype

    # Optional subsampling for decomposition (affects U/lam estimation).
    N, d = A.shape
    if N > max_samples:
        idx = torch.randperm(N, device=device)[:max_samples]
        A_est = A[idx]
    else:
        A_est = A

    # Center (use float64 for stable covariance/eigh)
    mu = A_est.mean(dim=0, keepdim=True)
    Ac = (A_est - mu).to(torch.float64)  # [N_est, d]

    # Covariance in feature space: [d, d]
    C = (Ac.T @ Ac) / Ac.shape[0]
    C = C + ridge * torch.eye(d, device=device, dtype=torch.float64)

    # Eigen-decomposition (ascending) -> sort descending
    evals, evecs = torch.linalg.eigh(C)
    evals = torch.clamp(evals, min=0.0)
    idx_sort = torch.argsort(evals, descending=True)
    lam = evals[idx_sort]          # [d]
    U = evecs[:, idx_sort]         # [d, d]

    # Reshape eigenvalues: flatten while keeping directions fixed
    lam_new = torch.pow(lam + eps, 1.0 - tau)

    # Preserve total variance: sum(lam_new) == sum(lam)
    sum_lam = lam.sum()
    sum_lam_new = lam_new.sum()
    if sum_lam_new.item() <= 0:
        # pathological case: all zeros; just return centered original estimate direction
        lam_new = lam.clone()
    else:
        lam_new = lam_new * (sum_lam / (sum_lam_new + eps))

    # Construct A_new for the FULL A (not only A_est):
    # Work in eigenbasis of Cov(A_est_centered), scale coordinates to match lam_new.
    mu_full = A.mean(dim=0, keepdim=True)
    A_full_c = (A - mu_full).to(torch.float64)      # [N, d]
    B = A_full_c @ U                                 # [N, d]  coordinates in eigenbasis

    # If Cov(B) ≈ diag(lam), then scaling by sqrt(lam_new/lam) gives Cov(B_new) ≈ diag(lam_new)
    scale = torch.sqrt(lam_new / (lam + eps))        # [d]
    B_new = B * scale.unsqueeze(0)                   # [N, d]
    A_new_c = B_new @ U.T                            # [N, d]

    if return_centered:
        A_new = A_new_c
    else:
        A_new = A_new_c + mu_full.to(torch.float64)

    # Cast back to original dtype
    A_new = A_new.to(dtype_in)

    # Return U and lam_new in float64 by default (more stable for downstream computations)
    return A_new, U, lam_new


def per_sample_processing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    k: int,
    tau: float,
    module_names: List[str],
    **generation_kwargs,
):

    def _get_acts(model: AutoModelForCausalLM, question: str):
        modules = {n: m for n, m in model.named_modules() if n in module_names}
        collectors = {name: ActivationCollector(mod) for name, mod in modules.items()}
        with torch.no_grad():
            out = free_generate(model, tok, question, **generation_kwargs)
        for col in collectors.values():
            col.detach()

        all_activations = {
            name: torch.cat(col.activations, dim=0) for name, col in collectors.items()
        }
        results = []
        for i, name in enumerate(module_names):
            w: torch.Tensor = modules[name].weight.detach()  # [d_out, d_in]
            acts = all_activations[name]
            # Whiten transformation to flatten spectrum
            # acts = reshape_spectrum_keep_evecs(acts)
            acts, evecs, evals = reshape_spectrum_standard(acts, tau=tau)

            assert evals[0] > evals[1]

            bound = min(k, len(evals))
            evals_small = evals[bound:]  # [n,]
            evecs_small = evecs[:, bound:]  # [d_in, n]
            wu = w @ evecs_small.to(dtype=w.dtype)  # [d_out, n]
            delta_y_k = (
                (evals_small.to(dtype=wu.dtype) * (wu.norm(dim=0) ** 2)).sum().item()
            )
            wu = w @ evecs.to(dtype=w.dtype)
            denom = (evals.to(dtype=wu.dtype) * (wu.norm(dim=0) ** 2)).sum().item()
            delta_y_k = delta_y_k / denom
            # p = p.norm(p=2, dim=0) ** 2    # [n]
            # delta_y_k = (evals_small * p).sum().item()
            results.append(delta_y_k)
        return results

    return _get_acts(model, question)


def get_sample_question(sample: dict) -> str:
    if "question" in sample:
        return sample["question"]
    elif "problem" in sample:
        return sample["problem"]
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--model_identifier", type=str)
    parser.add_argument("--datatype", type=str)
    parser.add_argument("--component", choices=["mlp", "self_attn"], default="mlp")
    parser.add_argument("--proj", type=str)
    parser.add_argument("--seen_data", type=str)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--maxn", type=int, default=None)
    args = parser.parse_args()

    with open(args.seen_data, "r") as f:
        seen_data = json.load(f)

    if args.maxn is not None:
        print(f"Truncate dataset to {args.maxn} samples.")
        seen_data = seen_data[: args.maxn]

    model, tok = load_hf_model_tokenizer(args.model)
    base_model, _ = load_hf_model_tokenizer(args.base_model)
    num_layers = model.config.num_hidden_layers

    module_names = [
        f"model.layers.{x}.{args.component}.{args.proj}_proj" for x in range(num_layers)
    ]  # up seen < unseen

    sft_results = []
    for sample in tqdm(seen_data, desc="SFT"):
        q = get_sample_question(sample)
        delta_yk = per_sample_processing(
            model,
            tok,
            q,
            args.k,
            args.tau,
            module_names,
            max_new_tokens=1024,
            do_sample=False,
        )
        sft_results.append(delta_yk)
    sft_results = np.array(sft_results)

    this_time = datetime.now().strftime("%y-%m-%d-%H-%M")
    os.makedirs(args.output_dir, exist_ok=True)

    fname = "flat" if args.tau > 0 else "dominance"
    np.save(f"{args.output_dir}/{fname}.npy", sft_results)
