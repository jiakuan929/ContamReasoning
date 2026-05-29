import os
import sys
import json
import argparse
from typing import List, Dict

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bas import load_hf_model_tokenizer, free_generate
from detection_stable.utils import collect_AG


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


def get_sample_question(sample: dict) -> str:
    if "question" in sample:
        return sample["question"]
    elif "problem" in sample:
        return sample["problem"]
    else:
        raise NotImplementedError


def compute_Uk(A: torch.Tensor, k: int, max_samples: int = 20000):
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
    assert A.shape[0] > 1
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
    U_k = evecs_sorted[:, :k]
    return U_k


class OnlineRMatrixCalculator:
    def __init__(self, k, d, eps=1e-6, device='cpu'):
        self.k = k
        self.d = d
        self.eps = eps
        self.device = device

        self.sum_z = torch.zeros(k, dtype=torch.float64, device=device)
        self.sum_g = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_zzT = torch.zeros(k, k, dtype=torch.float64, device=device)
        self.sum_ggT = torch.zeros(d, d, dtype=torch.float64, device=device)
        self.sum_zgT = torch.zeros(k, d, dtype=torch.float64, device=device)
        self.total_n = 0

    @torch.no_grad()
    def update(self, Z, G):
        """
        Z: [n, k], G: [n, d]
        """
        Z = Z.to(self.device, dtype=torch.float64)
        G = G.to(self.device, dtype=torch.float64)
        n = Z.shape[0]
        
        self.sum_z += Z.sum(dim=0)
        self.sum_g += G.sum(dim=0)
        self.sum_zzT += Z.T @ Z
        self.sum_ggT += G.T @ G
        self.sum_zgT += Z.T @ G
        self.total_n += n

    def _inv_sqrt(self, mat):
        U, S, Vh = torch.linalg.svd(mat)
        
        S_inv_sqrt = 1.0 / torch.sqrt(S + self.eps)
        
        return (U * S_inv_sqrt) @ U.T

    def compute_R(self):
        n = self.total_n
        if n < 2: return None
        
        mu_z = self.sum_z / n
        mu_g = self.sum_g / n
        
        sigma_z = (self.sum_zzT / n) - torch.outer(mu_z, mu_z)
        sigma_g = (self.sum_ggT / n) - torch.outer(mu_g, mu_g)
        sigma_zg = (self.sum_zgT / n) - torch.outer(mu_z, mu_g)
        
        sigma_z_inv_sqrt = self._inv_sqrt(sigma_z)
        sigma_g_inv_sqrt = self._inv_sqrt(sigma_g)
        
        R = sigma_z_inv_sqrt @ sigma_zg @ sigma_g_inv_sqrt
        
        return R.to(torch.float32) 


def per_sample_processing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    k: int,
    all_r_calcs: Dict[str, OnlineRMatrixCalculator],
    module_names: List[str],
    **generation_kwargs,
):
    modules = {n: m for n, m in model.named_modules() if n in module_names}

    allA, allG = collect_AG(
        model, tok, question, None, module_names, generation_kwargs
    )

    for i, name in enumerate(module_names):
        # col = cols[name]
        # A = torch.cat(col.activations, dim=0)
        A = allA[name]
        G = allG[name]

        U_k = compute_Uk(A, k)
        A_c = A - A.mean(dim=0)
        Z_k = A_c.to(dtype=U_k.dtype) @ U_k

        all_r_calcs[name].update(Z_k, G)


def compute_batch_R_matrices(all_r_calcs):
    results = {}

    stats_list = []
    names = []
    for name, calc in all_r_calcs.items():
        if calc.total_n < 2:
            continue
        stats_list.append({
            'sum_z': calc.sum_z,
            'sum_g': calc.sum_g,
            'sum_zzT': calc.sum_zzT,
            'sum_ggT': calc.sum_ggT,
            'sum_zgT': calc.sum_zgT,
            'n': calc.total_n,
            'k': calc.k,
            'd': calc.d
        })
        names.append(name)
    
    if not stats_list:
        return results
    
    device = stats_list[0]['sum_z'].device
    
    n_tensors = torch.tensor([stat['n'] for stat in stats_list], 
                            device=device, dtype=torch.float64)
    sum_z_tensors = torch.stack([stat['sum_z'] for stat in stats_list])
    sum_g_tensors = torch.stack([stat['sum_g'] for stat in stats_list])
    
    mu_z_batch = sum_z_tensors / n_tensors.unsqueeze(1)
    mu_g_batch = sum_g_tensors / n_tensors.unsqueeze(1)
    
    sum_zzT_batch = torch.stack([stat['sum_zzT'] for stat in stats_list])
    sum_ggT_batch = torch.stack([stat['sum_ggT'] for stat in stats_list])
    sum_zgT_batch = torch.stack([stat['sum_zgT'] for stat in stats_list])
    
    mu_z_outer = torch.bmm(mu_z_batch.unsqueeze(2), mu_z_batch.unsqueeze(1))
    mu_g_outer = torch.bmm(mu_g_batch.unsqueeze(2), mu_g_batch.unsqueeze(1))
    mu_zg_outer = torch.bmm(mu_z_batch.unsqueeze(2), mu_g_batch.unsqueeze(1))
    
    sigma_z_batch = (sum_zzT_batch / n_tensors.view(-1, 1, 1)) - mu_z_outer
    sigma_g_batch = (sum_ggT_batch / n_tensors.view(-1, 1, 1)) - mu_g_outer
    sigma_zg_batch = (sum_zgT_batch / n_tensors.view(-1, 1, 1)) - mu_zg_outer
    
    def batch_inv_sqrt(matrices):
        L_batch, Q_batch = torch.linalg.eigh(matrices)
        L_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(L_batch, min=all_r_calcs[names[0]].eps))
        return torch.bmm(torch.bmm(Q_batch, torch.diag_embed(L_inv_sqrt)), Q_batch.transpose(1, 2))
    
    sigma_z_inv_sqrt_batch = batch_inv_sqrt(sigma_z_batch)
    sigma_g_inv_sqrt_batch = batch_inv_sqrt(sigma_g_batch)
    
    # R = sigma_z_inv_sqrt @ sigma_zg @ sigma_g_inv_sqrt
    R_batch = torch.bmm(torch.bmm(sigma_z_inv_sqrt_batch, sigma_zg_batch), sigma_g_inv_sqrt_batch)
    
    svd_vals = torch.linalg.svdvals(R_batch)
    rho_max_batch = svd_vals.max(dim=1).values

    for i, name in enumerate(names):
        results[name] =  svd_vals[i]

    return results


def get_single_module(model, mod_name):
    target_mod = None
    for n, m in model.named_modules():
        if n == mod_name:
            target_mod = m
            break
    return target_mod


def iter_dataset(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    dataset: List,
    desc: str,
    module_names: List[str],
    k: int,
):
    n0 = module_names[0]
    if "mlp.up" in n0 or "mlp.gate" in n0:
        g_dim = model.config.intermediate_size
    else:
        g_dim = get_single_module(model, n0).out_features
    all_r_calcs = {name: OnlineRMatrixCalculator(k, g_dim, device="cuda") for name in module_names}

    for sample in tqdm(dataset, desc=desc):
        q = get_sample_question(sample)

        r = per_sample_processing(
            model,
            tok,
            q,
            k,
            all_r_calcs,
            module_names,
            max_new_tokens=1024,
            do_sample=False,
        )

    return compute_batch_R_matrices(all_r_calcs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--layers", nargs="+")
    parser.add_argument("--component", type=str)
    parser.add_argument("--proj", type=str)
    parser.add_argument("--seen_data", type=str)
    parser.add_argument("--unseen_data", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--maxn", type=int, default=None)
    args = parser.parse_args()

    model, tok = load_hf_model_tokenizer(args.model)
    num_layers = model.config.num_hidden_layers

    with open(args.seen_data, "r") as f:
        seen_data = json.load(f)
    with open(args.unseen_data, "r") as f:
        unseen_data = json.load(f)

    if args.maxn is not None:
        seen_data = seen_data[: args.maxn]
        unseen_data = unseen_data[: args.maxn]

    module_names = [f"model.layers.{x}.{args.component}.{args.proj}_proj" for x in args.layers]
    print(module_names)

    seen_rhos = iter_dataset(
        model, tok, seen_data, desc="Seen", module_names=module_names, k=args.k
    )
    unseen_rhos = iter_dataset(
        model, tok, unseen_data, desc="Unseen", module_names=module_names, k=args.k
    )

    for name in module_names:
        seen_rho_i = seen_rhos[name]
        unseen_rho_i = unseen_rhos[name]
        print(name, seen_rho_i.shape, unseen_rho_i.shape)
        print("  leq:", bool(torch.all(seen_rho_i <= unseen_rho_i)))
        print("  strict:", bool(torch.any(seen_rho_i < unseen_rho_i)))

        os.makedirs(f"{args.output_dir}/{name}", exist_ok=True)
        np.save(f"{args.output_dir}/{name}/seen.npy", seen_rho_i.detach().cpu().numpy())
        np.save(f"{args.output_dir}/{name}/unseen.npy", unseen_rho_i.detach().cpu().numpy())
