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
from activation_rank.utils import compute_spectrum_from_A, effective_rank, spectral_entropy_effective_dim


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


def per_sample_processing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    module_names: List[str],
    **generation_kwargs,
):

    def _get_acts(model: AutoModelForCausalLM, question: str):
        modules = {n: m for n, m in model.named_modules() if n in module_names}
        collectors = {name: ActivationCollector(mod) for name, mod in modules.items()}

        out = free_generate(model, tok, question, **generation_kwargs)
        for col in collectors.values():
            col.detach()

        all_activations = {
            name: torch.cat(col.activations, dim=0) for name, col in collectors.items()
        }
        results_eff = []
        results_H = []
        for i, name in enumerate(module_names):
            acts = all_activations[name]
            evals = compute_spectrum_from_A(acts)
            r_eff = effective_rank(evals)
            r_H = spectral_entropy_effective_dim(evals)
            results_eff.append(r_eff)
            results_H.append(r_H)
        return results_eff, results_H

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
    parser.add_argument("--model_identifier", type=str)
    parser.add_argument("--datatype", type=str)
    parser.add_argument("--component", choices=["mlp", "self_attn"], default="mlp")
    parser.add_argument("--proj", type=str)
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--maxn", type=int, default=None)
    args = parser.parse_args()

    with open(args.datapath, "r") as f:
        dataset = json.load(f)

    if args.maxn is not None:
        print(f"Truncate dataset to {args.maxn} samples.")
        dataset = dataset[: args.maxn]

    model, tok = load_hf_model_tokenizer(args.model)
    num_layers = model.config.num_hidden_layers

    module_names = [
        f"model.layers.{x}.{args.component}.{args.proj}_proj" for x in range(num_layers)
    ]  # up seen < unseen

    results_eff_seen = []
    results_H_seen = []

    tag = "Seen" if "train" in args.datapath else "Unseen" 
    for sample in tqdm(dataset, desc=tag):
        q = get_sample_question(sample)

        r_eff, r_H = per_sample_processing(
            model,
            tok,
            q,
            module_names,
            max_new_tokens=1024,
            do_sample=False,
        )
        results_eff_seen.append(r_eff)
        results_H_seen.append(r_H)

    results_eff_seen = np.array(results_eff_seen)
    results_H_seen = np.array(results_H_seen)

    this_time = datetime.now().strftime("%y-%m-%d-%H-%M")
    # output_dir = f"outputs2/effective_rank/{args.datatype}/{args.model_identifier}/{args.component}_{args.proj}"
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(f"{args.output_dir}/r_eff.npy", results_eff_seen)
    np.save(f"{args.output_dir}/r_H.npy", results_H_seen)
    