from __future__ import annotations
import math
import torch


def qubo_energy(Q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return x @ Q @ x


def ising_energy(J: torch.Tensor, h: torch.Tensor, s: torch.Tensor, c0=0.0) -> torch.Tensor:
    h = h.reshape(-1)
    return -0.5 * s @ J @ s - h @ s + torch.as_tensor(c0, dtype=J.dtype, device=J.device)


def qubo_to_ising(Q: torch.Tensor):
    """Exact mapping for symmetric Q and x=(s+1)/2.

    Returns J, h, c0 so that x^T Q x == -1/2 s^T J s - h^T s + c0.
    """
    Q = 0.5 * (Q + Q.T)
    d = torch.diag(Q)
    Qo = Q - torch.diag(d)
    J = -0.5 * Qo
    h = -0.5 * Q.sum(dim=1)
    c0 = 0.5 * torch.trace(Q) + 0.25 * Qo.sum()
    return J, h, c0


def equivalence_check(Q: torch.Tensor, trials=1000, atol=1e-8, seed=7):
    g = torch.Generator(device=Q.device).manual_seed(seed)
    J, h, c0 = qubo_to_ising(Q)
    n = Q.shape[0]
    max_err = 0.0
    for _ in range(trials):
        x = torch.randint(0, 2, (n,), generator=g, device=Q.device, dtype=Q.dtype)
        s = 2.0 * x - 1.0
        eq = qubo_energy(Q, x)
        ei = ising_energy(J, h, s, c0)
        max_err = max(max_err, float((eq - ei).abs()))
    return max_err, max_err <= atol


def tts_seconds(t_run: float, p_success: float, target=0.99) -> float:
    if p_success <= 0:
        return math.inf
    if p_success >= target:
        return t_run
    return t_run * math.log(1.0 - target) / math.log(1.0 - p_success)
