import os
import json
import sys
import math
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
from detection2.det_utils import collect_AG


def _logdet_active_and_rank_from_gram(
    X_c: torch.Tensor,
    ridge: float = 1e-4,
    eps_rel: float = 1e-12,
) -> tuple[float, int]:
    """
    给定中心化数据 X_c [N, D]，用 Gram 矩阵求协方差的非零特征值：
        Σ = Cov(X) = X_c^T X_c / (N-1)
    并返回：
        L = sum_i log(λ_i + ridge)   （只对非零 λ_i）
        r = rank(Σ) = 非零特征值个数
    """
    X_c = X_c.to(torch.float64)
    N = X_c.shape[0]
    denom = max(N - 1, 1)

    # Gram: K = X X^T / (N - 1)
    K = (X_c @ X_c.T) / denom  # [N, N]

    # 特征值（协方差的非零特征值）
    evals = torch.linalg.eigvalsh(K)  # [N]
    # 数值噪声可能有微小负值，截断到 0
    evals = torch.clamp(evals, min=0.0)

    if evals.numel() == 0:
        return 0.0, 0

    # 以最大特征值为参照，确定“非零”阈值
    max_e = evals.max()
    if max_e <= 0:
        # 完全零方差（几乎不可能），直接返回 0, 0
        return 0.0, 0

    thresh = max_e * eps_rel
    mask = evals > thresh
    evals_eff = evals[mask]
    r = int(mask.sum().item())

    if r == 0:
        return 0.0, 0

    # 对非零特征值算 sum log(λ_i + ridge)
    L = torch.sum(torch.log(evals_eff + ridge)).item()
    return L, r


@torch.no_grad()
def gaussian_mi_gram_proper(
    A_samples: torch.Tensor,
    G_samples: torch.Tensor,
    ridge: float = 1e-4,
    eps_rel: float = 1e-12,
) -> float:
    """
    基于 Gram 矩阵的高斯互信息估计：
        I(A;G) = 0.5 [ log|Σ_A+epsI| + log|Σ_G+epsI| - log|Σ_AG+epsI| ]
    其中每个 logdet 拆成：
        log|Σ + eps I| = L_active + (D - r)*log(eps)
    并在代码里显式合并 (D - r) log(eps) 项，避免数值抵消。
    """
    # 转 double，避免谱分解时的数值不稳定
    A = A_samples.detach().to(torch.float64)
    G = G_samples.detach().to(torch.float64)

    N, D_a = A.shape
    _, D_g = G.shape
    D_joint = D_a + D_g

    # 中心化
    Ac = A - A.mean(dim=0, keepdim=True)
    Gc = G - G.mean(dim=0, keepdim=True)

    # A 的活跃 logdet 和秩
    L_a, r_a = _logdet_active_and_rank_from_gram(Ac, ridge=ridge, eps_rel=eps_rel)
    # G 的活跃 logdet 和秩
    L_g, r_g = _logdet_active_and_rank_from_gram(Gc, ridge=ridge, eps_rel=eps_rel)

    # Joint: Gram([A,G]) = Ac Ac^T + Gc Gc^T
    denom = max(N - 1, 1)
    K_a = Ac @ Ac.T
    K_g = Gc @ Gc.T
    K_joint = (K_a + K_g) / denom

    # Joint 特征值
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

    # 把 (D - r)*log(ridge) 那些常数项解析地合并
    log_eps = math.log(ridge)
    const_term = (r_j - r_a - r_g) * log_eps

    mi = 0.5 * ((L_a + L_g - L_j) + const_term)
    return float(mi)


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


def compute_mi_alpha(
    A: torch.Tensor,
    G: torch.Tensor,
    U_k: torch.Tensor,
    W: torch.Tensor,
    center_A=True,
    center_G=True,
):
    assert A.dim() == 2 and G.dim() == 2
    N, d_in = A.shape
    Ng, d_out = G.shape
    assert N == Ng, "A and G must have same batch size"
    assert U_k.shape[0] == d_in
    k = U_k.shape[1]
    assert W.shape == (d_out, d_in)

    # 1) 中心化 A
    if center_A:
        mu = A.mean(dim=0, keepdim=True)  # [1, d_in]
        A_c = A - mu
    else:
        A_c = A

    # 2) alpha = U_k^T (A - mu)  -> [N, k]
    alpha = A_c @ U_k  # [N, k]

    V = W @ U_k  # [d_out, k]

    # 4) g_alpha = V^T G^T, 对每个样本 i:
    #    g_alpha[i, j] = v_j^T g_y[i]
    #    形式上就是 G @ V -> [N, k]
    if center_G:
        G_c = G - G.mean(dim=0, keepdim=True)
    else:
        G_c = G
    g_alpha = G_c @ V  # [N, k]
    return gaussian_mi_gram_proper(alpha, g_alpha)


def compute_ag_mi_teacher_forcing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    target_response: str | None,
    module_names: List[str],
    k: int,
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
    new_results = []
    for name in names:
        A_samples, G_samples = A[name], G[name]
        U_k = compute_Uk(A_samples, k).to(G_samples.dtype)

        alpha = (A_samples - A_samples.mean(dim=0)) @ U_k  # [N, k]
        N = alpha.shape[0]
        # mi_alpha = gaussian_mi_gram_proper(alpha, G_samples)   #  完全等于！！！
        G_c = G_samples - G_samples.mean(dim=0, keepdim=True)
        Sigma_alpha_G = (alpha.T @ G_c) / (N - 1)
        fro_sigma_alpha_g = torch.norm(Sigma_alpha_G).item()

        A_c = A_samples - A_samples.mean(dim=0, keepdim=True)
        Sigma_AG = (A_c.T @ G_c) / (N - 1)
        fro_sigma_A_G = torch.norm(Sigma_AG).item()

        results.append(fro_sigma_alpha_g)
        new_results.append(fro_sigma_A_G)

    return results, new_results


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
    parser.add_argument("--k", type=int)
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
    unseen_new = []
    for sample in tqdm(unseen_data, desc="Unseen"):
        target_response = None
        question = get_sample_question(sample)

        with torch.enable_grad():
            r_alpha, r_ag = compute_ag_mi_teacher_forcing(
                model, tok, question, target_response, module_names, args.k
            )
        unseen_results.append(r_alpha)
        unseen_new.append(r_ag)
    unseen_results = np.array(unseen_results)
    unseen_new = np.array(unseen_new)

    seen_results = []
    seen_new = []
    for sample in tqdm(seen_data, desc="Seen"):
        target_response = None
        question = get_sample_question(sample)

        with torch.enable_grad():
            r_alpha, r_ag = compute_ag_mi_teacher_forcing(
                model, tok, question, target_response, module_names, args.k
            )
        seen_results.append(r_alpha)
        seen_new.append(r_ag)
    seen_results = np.array(seen_results)
    seen_new = np.array(seen_new)

    this_time = datetime.now().strftime("%y-%m-%d-%H-%M")
    os.makedirs(args.output_dir, exist_ok=True)

    np.save(f"{args.output_dir}/unseen_alpha_g.npy", unseen_results)
    np.save(f"{args.output_dir}/seen_alpha_g.npy", seen_results)
    np.save(f"{args.output_dir}/unseen_A_g.npy", unseen_new)
    np.save(f"{args.output_dir}/seen_A_g.npy", seen_new)

    seen_results = seen_results.mean(axis=0)
    unseen_results = unseen_results.mean(axis=0)
    seen_new = seen_new.mean(axis=0)
    unseen_new = unseen_new.mean(axis=0)

    plt.figure()
    plt.plot(unseen_results, marker="o", markersize=4, label="Unseen")
    plt.plot(seen_results, marker="^", markersize=4, label="Seen")
    plt.xlabel("Layer")
    plt.ylabel(r"$||\Sigma_{Z_k G}||$")
    plt.title(f"{args.model_identifier}  {args.component}.{args.proj}_proj")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/alpha_g_norm.png")
    plt.close()

    plt.figure()
    plt.plot(unseen_new, marker="o", markersize=4, label="Unseen")
    plt.plot(seen_new, marker="^", markersize=4, label="Seen")
    plt.xlabel("Layer")
    plt.ylabel(r"$||\Sigma_{A G}||$")
    plt.title(f"{args.model_identifier}  {args.component}.{args.proj}_proj")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/A_g_norm.png")
    plt.close()
