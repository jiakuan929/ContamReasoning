import torch


def compute_spectrum_from_A(A: torch.Tensor, max_samples: int = 20000):
    """
    A: [N, d] activation matrix (rows = samples/tokens)
    返回: eigenvalues λ ∈ [d], in descending order
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
    evals, _ = torch.linalg.eigh(C)  # evals: [d]
    evals = torch.clamp(evals, min=0.0)  # avoid tiny negatives
    # sort descending
    evals = torch.flip(evals, dims=[0])
    return evals


def effective_rank(evals: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Participation ratio: r_eff = (sum λ)^2 / sum λ^2
    evals: [d], non-negative eigenvalues
    """
    # remove extremely small eigenvalues if you want
    # evals = evals[evals > eps]
    s1 = evals.sum()
    s2 = (evals**2).sum()
    if s2 < eps:
        return 0.0
    r_eff = (s1**2) / s2
    return r_eff.item()


def spectral_entropy_effective_dim(evals: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Spectral entropy-based effective dimension:
      p_i = λ_i / sum λ_i
      H   = - Σ p_i log p_i
      r_H = exp(H)
    """
    s = evals.sum()
    if s < eps:
        return 0.0
    p = evals / s  # [d]
    p = torch.clamp(p, min=eps)  # avoid log(0)
    H = -(p * torch.log(p)).sum()
    r_H = torch.exp(H)
    return r_H.item()