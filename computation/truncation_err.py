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


def per_sample_processing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    k: int,
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
            evals, evecs = compute_spectrum_from_A(acts)
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
    parser.add_argument("--sft_model", type=str)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--model_identifier", type=str)
    parser.add_argument("--datatype", type=str)
    parser.add_argument("--component", choices=["mlp", "self_attn"], default="mlp")
    parser.add_argument("--proj", type=str)
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--maxn", type=int, default=None)
    args = parser.parse_args()

    with open(args.datapath, "r") as f:
        seen_data = json.load(f)

    if args.maxn is not None:
        print(f"Truncate dataset to {args.maxn} samples.")
        seen_data = seen_data[: args.maxn]

    model, tok = load_hf_model_tokenizer(args.sft_model)
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
            module_names,
            max_new_tokens=1024,
            do_sample=False,
        )
        sft_results.append(delta_yk)
    sft_results = np.array(sft_results)

    orig_results = []
    for sample in tqdm(seen_data, desc="Base"):
        q = get_sample_question(sample)
        delta_yk = per_sample_processing(
            base_model,
            tok,
            q,
            args.k,
            module_names,
            max_new_tokens=1024,
            do_sample=False,
        )
        orig_results.append(delta_yk)
    orig_results = np.array(orig_results)

    this_time = datetime.now().strftime("%y-%m-%d-%H-%M")
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(f"{args.output_dir}/sft.npy", sft_results)
    np.save(f"{args.output_dir}/base.npy", orig_results)

    plt.figure()
    plt.plot(orig_results.mean(axis=0), label="Base", marker="s", markersize=4)
    plt.plot(sft_results.mean(axis=0), label="SFT", marker="s", markersize=4)
    plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("Relative Truncation Error")
    plt.title(f"k = {args.k}")
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/err.png")
