import os
import json
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Literal

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bas import load_hf_model_tokenizer
from detection_stable.utils import collect_AG, gaussian_mi_gram_proper


def compute_ag_mi_teacher_forcing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    target_response: str | None,
    module_names: List[str],
):
    A, G = collect_AG(
        model,
        tok,
        question,
        target_response,
        module_names,
        generation_kwargs={"max_new_tokens": 1024, "do_sample": False},
    )
    names = list(A.keys())
    results = {}
    for name in names:
        A_samples, G_samples = A[name], G[name]
        gnorm = torch.norm(G_samples, p=2, dim=1).mean().item()
        mi = gaussian_mi_gram_proper(A_samples, G_samples)
        results[name] = {"mi": mi, "grad_norm": gnorm}
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
    parser.add_argument(
        "--component", type=str, choices=["mlp", "self_attn"], default="mlp"
    )
    parser.add_argument("--proj", type=str, default="up")
    parser.add_argument(
        "--unseen_data", type=str, default="dataset/training/arith_test.json"
    )
    parser.add_argument(
        "--seen_data", type=str, default="dataset/training/arith_train.json"
    )
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    with open(args.unseen_data, "r") as f:
        unseen_data = json.load(f)
    with open(args.seen_data, "r") as f:
        seen_data = json.load(f)

    model, tok = load_hf_model_tokenizer(args.model)
    num_layers = model.config.num_hidden_layers

    temp = "model.layers.{x}.{component}.{proj}_proj"
    module_names = [
        temp.format(x=x, component=args.component, proj=args.proj)
        for x in range(num_layers)
    ]

    unseen_results = []
    unseen_gnorms = []
    for sample in tqdm(unseen_data, desc="Unseen"):
        target_response = None
        question = get_sample_question(sample)

        with torch.enable_grad():
            r = compute_ag_mi_teacher_forcing(
                model, tok, question, target_response, module_names
            )
        mi = [r[module_names[i]]["mi"] for i in range(num_layers)]
        gn = [r[module_names[i]]["grad_norm"] for i in range(num_layers)]
        unseen_results.append(mi)
        unseen_gnorms.append(gn)

    seen_results = []
    seen_gnorms = []
    for sample in tqdm(seen_data, desc="Seen"):
        target_response = None
        question = get_sample_question(sample)

        with torch.enable_grad():
            r = compute_ag_mi_teacher_forcing(
                model, tok, question, target_response, module_names
            )
        mi = [r[module_names[i]]["mi"] for i in range(num_layers)]
        gn = [r[module_names[i]]["grad_norm"] for i in range(num_layers)]
        seen_results.append(mi)
        seen_gnorms.append(gn)

    this_time = datetime.now().strftime("%y-%m-%d-%H-%M")
    os.makedirs(args.output_dir, exist_ok=True)

    unseen_results = np.array(unseen_results)
    seen_results = np.array(seen_results)
    unseen_gnorms = np.array(unseen_gnorms)
    seen_gnorms = np.array(seen_gnorms)
    np.save(f"{args.output_dir}/unseen_mi.npy", unseen_results)
    np.save(f"{args.output_dir}/seen_mi.npy", seen_results)
    np.save(f"{args.output_dir}/unseen_gnorms.npy", unseen_gnorms)
    np.save(f"{args.output_dir}/seen_gnorms.npy", seen_gnorms)

    unseen_results = unseen_results.mean(axis=0)
    seen_results = seen_results.mean(axis=0)
    unseen_gnorms = unseen_gnorms.mean(axis=0)
    seen_gnorms = seen_gnorms.mean(axis=0)

    plt.figure()
    plt.plot(unseen_results, marker="o", markersize=4, label="Unseen")
    plt.plot(seen_results, marker="^", markersize=4, label="Seen")
    plt.xlabel("Layer")
    plt.ylabel("MI")
    plt.title(f"{args.model_identifier}  {args.component}.{args.proj}_proj")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/mutual_info.png")
    plt.close()

    plt.figure()
    plt.plot(unseen_gnorms, marker="o", markersize=4, label="Unseen")
    plt.plot(seen_gnorms, marker="^", markersize=4, label="Seen")
    plt.xlabel("Layer")
    plt.ylabel("MI")
    plt.title(f"{args.model_identifier}  {args.component}.{args.proj}_proj")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/grad_norm.png")
    plt.close()
