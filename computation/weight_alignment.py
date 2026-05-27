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
        results_max_min = []
        results_median = []
        for i, name in enumerate(module_names):
            w: torch.Tensor = modules[name].weight.detach()  # [d_out, d_in]
            acts = all_activations[name]
            evals, evecs = compute_spectrum_from_A(acts)
            assert evals[0] > evals[1]

            bound = min(k, len(evals))
            U_head = evecs[:, :bound]
            U_tail = evecs[:, bound:]
            g_head: torch.Tensor = w @ U_head.to(dtype=w.dtype)
            # g_head = g_head.norm(p=2, dim=0).mean().item()
            g_tail: torch.Tensor = w @ U_tail.to(dtype=w.dtype)
            # g_tail = g_tail.norm(p=2, dim=0).mean().item()

            head_min = g_head.norm(p=2, dim=0).min().item()
            head_median = g_head.norm(p=2, dim=0).median().item()
            tail_max = g_tail.norm(p=2, dim=0).max().item()
            tail_median = g_tail.norm(p=2, dim=0).median().item()

            results_max_min.append(tail_max / (head_min + 1e-9))
            results_median.append(tail_median / (head_median + 1e-9))

        return results_max_min, results_median

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

    sft_max_min = []
    sft_medians = []
    for sample in tqdm(seen_data, desc="SFT"):
        q = get_sample_question(sample)
        max_min, median_median = per_sample_processing(
            model,
            tok,
            q,
            args.k,
            module_names,
            max_new_tokens=1024,
            do_sample=False,
        )
        sft_max_min.append(max_min)
        sft_medians.append(median_median)
    sft_max_min = np.array(sft_max_min)
    sft_medians = np.array(sft_medians)

    orig_max_min = []
    orig_medians = []
    for sample in tqdm(seen_data, desc="Base"):
        q = get_sample_question(sample)
        max_min, median_median = per_sample_processing(
            base_model,
            tok,
            q,
            args.k,
            module_names,
            max_new_tokens=1024,
            do_sample=False,
        )
        orig_max_min.append(max_min)
        orig_medians.append(median_median)
    orig_max_min = np.array(orig_max_min)
    orig_medians = np.array(orig_medians)

    os.makedirs(args.output_dir, exist_ok=True)

    np.save(f"{args.output_dir}/sft_max_min.npy", sft_max_min)
    np.save(f"{args.output_dir}/sft_median.npy", sft_medians)
    np.save(f"{args.output_dir}/base_max_min.npy", orig_max_min)
    np.save(f"{args.output_dir}/base_median.npy", orig_medians)

    plt.figure()
    plt.plot(orig_max_min.mean(axis=0), marker="o", markersize=4, label="Base")
    plt.plot(sft_max_min.mean(axis=0), marker="o", markersize=4, label="SFT")
    plt.xlabel("Layer")
    plt.ylabel("tail/head")
    plt.title(f"k = {args.k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/max_min.png")

    plt.figure()
    plt.plot(orig_medians.mean(axis=0), marker="o", markersize=4, label="Base")
    plt.plot(sft_medians.mean(axis=0), marker="o", markersize=4, label="SFT")
    plt.xlabel("Layer")
    plt.ylabel("tail/head")
    plt.title(f"k = {args.k}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/median_median.png")
