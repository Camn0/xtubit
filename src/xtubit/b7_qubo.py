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
    placement_ids: Sequence[str] | None = None,
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
    vmap = build_variable_map(fragment_id, placement_ids=placement_ids)
    return QuboBundle(Q=Q, onehot_constant=0.5 * D * nfrag, variable_map=vmap)



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


def build_yanagisawa_pyqubo(
    dG: torch.Tensor,
    clash: torch.Tensor,
    conn: torch.Tensor,
    fragment_id: torch.Tensor,
    A: float = 1.0,
    B: float = 5.0,
    C: float = 5.0,
    D: float = 25.0,
    half_penalty: bool = True,
) -> tuple[QuboBundle, Any]:
    """Symbolic construction and compilation of the Yanagisawa docking Hamiltonian using PyQUBO.

    Constructs the four-term Hamiltonian symbolically:
    H = A * H_1 (protein-fragment interaction)
      + B * H_2 (steric clash penalty)
      + C * H_3 (covalent connectivity)
      + D_eff * H_4 (one-hot subpocket placement constraint with tagged labels)

    Returns:
    - bundle: QuboBundle containing symmetric Q matrix, constant offset, and variable map.
    - pyqubo_model: Compiled PyQUBO Model object for constraint verification and BQM export.
    """
    try:
        from pyqubo import Binary, Constraint
    except ImportError as exc:
        raise RuntimeError("Install pyqubo to use symbolic QUBO compilation") from exc

    n = int(dG.numel())
    x = [Binary(f"x_{i}") for i in range(n)]

    # H1: Protein-fragment binding affinity
    H1 = sum(dG[i].item() * x[i] for i in range(n))

    # H2: Clash penalty for overlapping poses across different fragments
    H2 = sum(
        clash[i, j].item() * x[i] * x[j]
        for i in range(n)
        for j in range(i + 1, n)
        if clash[i, j] != 0
    )

    # H3: Covalent connectivity constraint rewarding linked poses
    H3 = sum(
        conn[i, j].item() * x[i] * x[j]
        for i in range(n)
        for j in range(i + 1, n)
        if conn[i, j] != 0
    )

    # H4: One-hot constraint per subpocket with tagged label
    unique_frags = sorted(list(set(fragment_id.tolist())))
    D_term = (0.5 * D) if half_penalty else float(D)
    H4 = sum(
        Constraint(
            (sum(x[i] for i in range(n) if fragment_id[i] == f) - 1) ** 2,
            label=f"onehot_frag_{f}",
        )
        for f in unique_frags
    )

    H = A * H1 + B * H2 + C * H3 + D_term * H4
    model = H.compile()
    qubo_dict, offset = model.to_qubo(index_label=False)

    Q = torch.zeros((n, n), dtype=dG.dtype, device=dG.device)
    for (k1, k2), val in qubo_dict.items():
        i = int(k1.split("_")[1])
        j = int(k2.split("_")[1])
        if i == j:
            Q[i, i] += val
        else:
            Q[i, j] += 0.5 * val
            Q[j, i] += 0.5 * val

    vmap = build_variable_map(fragment_id)
    bundle = QuboBundle(Q=Q, onehot_constant=float(offset), variable_map=vmap)
    return bundle, model
