import os
import json
import sys
import math
import argparse
from datetime import datetime
from typing import List, Dict, Literal

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bas import load_hf_model_tokenizer
from detection2.det_utils import collect_AG


def compute_ag_mi_teacher_forcing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    target_response: str | None,
    module_names: List[str],
):
    modules = {n: m for n, m in model.named_modules() if n in module_names}
    A, G = collect_AG(
        model,
        tok,
        question,
        target_response,
        module_names,
        generation_kwargs={"max_new_tokens": 1024, "do_sample": False},
    )
    names = list(A.keys())
    results = []
    results_mean = []
    for name in names:
        A_samples, G_samples = A[name], G[name]
        N = A_samples.shape[0]

        G_c = G_samples - G_samples.mean(dim=0)  # [N, d_out]

        A_c = A_samples - A_samples.mean(dim=0)
        grad_w_center = (G_c.T @ A_c) / N
        r = grad_w_center.norm().item()

        results.append(r)

    return results


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
    parser.add_argument("--component", type=str)
    parser.add_argument("--proj", type=str, default="up")
    parser.add_argument("--unseen_data", type=str)
    parser.add_argument("--seen_data", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--maxn", type=int)
    args = parser.parse_args()

    with open(args.unseen_data, "r") as f:
        unseen_data = json.load(f)
    with open(args.seen_data, "r") as f:
        seen_data = json.load(f)

    if args.maxn is not None:
        seen_data = seen_data[: args.maxn]
        unseen_data = unseen_data[: args.maxn]

    model, tok = load_hf_model_tokenizer(args.model)
    num_layers = model.config.num_hidden_layers

    temp = "model.layers.{x}.{component}.{proj}_proj"
    module_names = [
        temp.format(x=x, component=args.component, proj=args.proj)
        for x in range(num_layers)
    ]

    unseen_results = []
    for sample in tqdm(unseen_data, desc="Unseen"):
        target_response = None
        question = get_sample_question(sample)

        with torch.enable_grad():
            r = compute_ag_mi_teacher_forcing(
                model, tok, question, target_response, module_names
            )
        unseen_results.append(r)

    unseen_results = np.array(unseen_results)

    seen_results = []
    for sample in tqdm(seen_data, desc="Seen"):
        target_response = None
        question = get_sample_question(sample)

        with torch.enable_grad():
            r = compute_ag_mi_teacher_forcing(
                model, tok, question, target_response, module_names
            )
        seen_results.append(r)

    seen_results = np.array(seen_results)

    this_time = datetime.now().strftime("%y-%m-%d-%H-%M")
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(f"{args.output_dir}/seen.npy", seen_results)
    np.save(f"{args.output_dir}/unseen.npy", unseen_results)

    seen_results = seen_results.mean(axis=0)
    unseen_results = unseen_results.mean(axis=0)

    plt.figure()
    plt.plot(unseen_results, marker="o", markersize=4, label="Unseen")
    plt.plot(seen_results, marker="^", markersize=4, label="Seen")
    plt.xlabel("Layer")
    plt.ylabel("G_center Norm")
    plt.title(f"{args.model_identifier}  {args.component}.{args.proj}_proj")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/g_center.png")
    plt.close()
