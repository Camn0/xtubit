from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from .common.math import qubo_to_ising, equivalence_check

@dataclass
class QuboBundle:
    Q: torch.Tensor
    onehot_constant: float
    variable_map: list[dict]


def build_yanagisawa_qubo(
    dG: torch.Tensor,
    clash: torch.Tensor,
    conn: torch.Tensor,
    fragment_id: torch.Tensor,
    A=1.0, B=5.0, C=5.0, D=25.0,
    pair_convention="upper",
):
    """Build the upper-triangle version of the published four-term Hamiltonian.

    H = A sum dG_i x_i + B sum_{i<j} clash_ij x_i x_j
        + C sum_{i<j} conn_ij x_i x_j
        + D/2 sum_k (sum_{i in F_k} x_i - 1)^2.

    For x^T Q x, off-diagonal coefficients are divided by two because x^TQx
    counts symmetric off-diagonal entries twice.
    """
    if pair_convention not in {"upper", "ordered"}:
        raise ValueError("pair_convention must be 'upper' or 'ordered'")
    n = int(dG.numel())
    Q = torch.zeros((n, n), dtype=dG.dtype, device=dG.device)
    scale = 0.5 if pair_convention == "upper" else 1.0
    pair = scale * (B * clash + C * conn)
    same = fragment_id[:, None].eq(fragment_id[None, :])
    Q += pair
    # H4 = D/2 * (sum x - 1)^2 => diag -D/2, physical pair coefficient D.
    Q[same] += 0.5 * D
    Q.fill_diagonal_(0.0)
    Q.diagonal().copy_(A * dG - 0.5 * D)
    Q = 0.5 * (Q + Q.T)
    nfrag = int(torch.unique(fragment_id).numel())
    return QuboBundle(Q=Q, onehot_constant=0.5 * D * nfrag, variable_map=[])


def build_variable_map(fragment_id: torch.Tensor, placement_ids: Sequence[str] | None = None):
    vm = []
    for i, f in enumerate(fragment_id.tolist()):
        vm.append({"index": i, "fragment_id": int(f), "placement_id": str(placement_ids[i]) if placement_ids else str(i)})
    return vm


def validate_qubo(bundle: QuboBundle, equivalence_trials=1000, atol=1e-8):
    Q = bundle.Q
    if not torch.allclose(Q, Q.T, atol=atol, rtol=0):
        raise AssertionError("Q is not symmetric")
    err, ok = equivalence_check(Q, trials=equivalence_trials, atol=atol)
    if not ok:
        raise AssertionError(f"Q->Ising equivalence failed; max error={err}")
    return {"max_equivalence_error": err, "passed": ok}
