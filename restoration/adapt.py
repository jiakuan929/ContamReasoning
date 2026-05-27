import os
import json
import argparse
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bas import load_hf_model_tokenizer, free_generate
from mathruler.grader import extract_boxed_content, grade_answer


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


@torch.no_grad()
def compute_shortcut_subspace(
    A: torch.Tensor,
    k: int,
    max_samples: int = 20000,
    ridge: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if A.dim() != 2:
        raise ValueError(f"A must be 2D [N, d], got shape {tuple(A.shape)}")

    device = A.device
    dtype_in = A.dtype
    N, d = A.shape

    if N > max_samples:
        idx = torch.randperm(N, device=device)[:max_samples]
        A_est = A[idx]
    else:
        A_est = A

    mu = A_est.mean(dim=0, keepdim=True)           # [1, d]
    Ac = (A_est - mu).to(torch.float64)            # [N_est, d]

    C = (Ac.T @ Ac) / Ac.shape[0]                  # [d, d]
    C = C + ridge * torch.eye(d, device=device, dtype=torch.float64)

    evals, evecs = torch.linalg.eigh(C)            
    idx_sort = torch.argsort(evals, descending=True)
    U = evecs[:, idx_sort]                         # [d, d]
    S = U[:, :k]                                   # [d, k]

    S = S.to(dtype_in)
    mu_full = A.mean(dim=0, keepdim=True).to(dtype_in)

    return S, mu_full


class ShortcutOrthogonalIntervention:

    def __init__(
        self,
        module: torch.nn.Module,
        S: torch.Tensor,          # [d, k] shortcut 子空间正交基
        mu: torch.Tensor,         # [1, d] 中心化用的均值
        power: float,
        apply_last_token: bool = False,
        max_forwards: int = 8,
    ):

        self.module = module
        self.S = S
        self.mu = mu
        self.power = power
        self.apply_last_token = apply_last_token
        self.max_forwards = max_forwards

        device = next(module.parameters()).device
        self.S = self.S.to(device)
        self.mu = self.mu.to(device)

        self.fwd_cnt = 0

        self.handle = module.register_forward_pre_hook(self.hook)

    def _orthogonalize(self, x: torch.Tensor) -> torch.Tensor:

        d = x.shape[-1]
        x_flat = x.reshape(-1, d)               # [N_flat, d]

        x_c = x_flat - self.mu                  # [N_flat, d]

        # S: [d, k], S^T: [k, d]
        x_proj = (x_c @ self.S) @ self.S.t()    # [N_flat, d]

        alpha = (x_proj.norm()) / (x_c.norm() + 1e-9)
        self.strength = alpha ** self.power

        x_c_new = x_c - self.strength * x_proj  # [N_flat, d]

        x_new = x_c_new + self.mu               # [N_flat, d]

        return x_new.reshape_as(x)

    def hook(self, module, inp):

        self.fwd_cnt += 1
        if (self.max_forwards is not None) and (self.fwd_cnt > self.max_forwards):
            return None

        x = inp[0] if isinstance(inp, tuple) else inp  # [B, T, d]
        B, T_len, d = x.shape

        if self.apply_last_token:
            x_last = x[:, -1, :]                        # [B, d]
            x_last_new = self._orthogonalize(x_last)    # [B, d]

            x_new = x.clone()
            x_new[:, -1, :] = x_last_new
        else:
            x_new = self._orthogonalize(x)              # [B, T, d]

        if isinstance(inp, tuple):
            return (x_new,) + inp[1:]
        else:
            return x_new

    def remove(self):
        self.handle.remove()


def estimate_params(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    module_names: List[str],
    **generation_kwargs
):
    modules = {n: m for n, m in model.named_modules() if n in module_names}
    cols: Dict[str, ActivationCollector] = {}
    for n, m in modules.items():
        col = ActivationCollector(m)
        cols[n] = col

    free_generate(model, tok, question, **generation_kwargs)

    for col in cols.values():
        col.detach()

    params = {}
    for n in module_names:
        A = torch.cat(cols[n].activations, dim=0)
        params[n] = compute_shortcut_subspace(A, k=64)
    return params


def per_sample_processing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    power: float,
    module_names: List[str],
    **generation_kwargs,
):
    params = estimate_params(model, tok, question, module_names, **generation_kwargs)

    reshapers: Dict[str, ShortcutOrthogonalIntervention] = {}
    modules = {n: m for n, m in model.named_modules() if n in module_names}
    for n, m in modules.items():
        T, mu = params[n]
        intervention = ShortcutOrthogonalIntervention(m, T, mu, power=power, apply_last_token=True)
        reshapers[n] = intervention

    out = free_generate(model, tok, question, **generation_kwargs)
    response: str = out["gen_text"]

    for v in reshapers.values():
        v.remove()
    return response


def judge_flag2(gt: str, sep, resp):
    pos = resp.find(sep)
    if pos == -1:
        return False, ""
    answer: str = resp[pos + len(sep) :]
    answer = answer.replace(",", "")
    gt = gt.replace(",", "")
    is_cor = answer == gt
    return is_cor, answer


def get_question_and_gt(sample):
    if "question" in sample:
        q = sample["question"]
        answer_text: str = sample["answer"]
        sep = "#### "
        gt = answer_text[answer_text.find(sep) + len(sep) :]
    elif "problem" in sample:
        q = sample["problem"]
        gt = sample["answer"][0]
    else:
        raise NotImplementedError
    return q, gt


def iter_dataset(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    dataset: List,
    module_names: List[str],
    power: float,
    desc: str,
):

    accs = []
    r_samples = []
    for sample in tqdm(dataset, desc=desc):
        question, gt = get_question_and_gt(sample)
        sep = "#### "
        base_result = sample["is_correct"]

        response = per_sample_processing(
            model,
            tok,
            question,
            power,
            module_names,
            max_new_tokens=1024,
            do_sample=False,
        )
        print(f"Intervened Response:\n{response}\n\n")

        boxed_answer = extract_boxed_content(response)
        if boxed_answer == "None":
            is_correct, prediction = judge_flag2(gt, sep, response)
        else:
            is_correct = grade_answer(boxed_answer, gt)
            prediction = boxed_answer
        sample["steer_answer"] = prediction
        sample["steer_response"] = response
        is_perfect = prediction.strip() == sample["pred"].strip()
        is_match = is_correct == base_result
        accs.append(is_match)
        sample["is_perfect"] = is_perfect
        sample["is_match"] = is_match
        r_samples.append(sample)

    return np.mean(accs), r_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--recovery_data", type=str)
    parser.add_argument("--retain_data", type=str)
    parser.add_argument("--modules", nargs="+")
    parser.add_argument("--power", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--maxn", type=int, default=None)
    args = parser.parse_args()

    with open(args.recovery_data, "r") as f:
        recovery_data = json.load(f)
    with open(args.retain_data, "r") as f:
        retain_data = json.load(f)

    if args.maxn is not None:
        recovery_data = recovery_data[: args.maxn]
        retain_data = retain_data[: args.maxn]

    model, tok = load_hf_model_tokenizer(args.model)

    recovery_acc, recovery_match = iter_dataset(
        model,
        tok,
        recovery_data,
        module_names=args.modules,
        power=args.power,
        desc="Recovery",
    )

    retain_acc, retain_match = iter_dataset(
        model,
        tok,
        retain_data,
        module_names=args.modules,
        power=args.power,
        desc="Retain",
    )

    print(f"Recovery: {recovery_acc}")
    print(f"Retain: {retain_acc}")

    this_time = datetime.now().strftime("%y-%m-%d-%H-%M")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/recovery_details.json", "w") as f:
        json.dump(recovery_match, f, indent=4)
    with open(f"{args.output_dir}/retain_details.json", "w") as f:
        json.dump(retain_match, f, indent=4)
    with open(f"{args.output_dir}/acc.json", "w") as f:
        json.dump({"recovery": recovery_acc, "retain": retain_acc}, f, indent=4)
