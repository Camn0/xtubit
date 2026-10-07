import torch
import numpy as np
import pytest
from xtubit.b7_qubo import build_yanagisawa_qubo, validate_qubo
from xtubit.common.math import qubo_to_ising, qubo_energy, ising_energy, tts_seconds
from xtubit.solvers.exact import brute_force_qubo
from xtubit.solvers.psa_pd import spsa, tapsa
from xtubit.solvers.tesb_port import two_stage_tesb
from xtubit.b9_metrics import repair_onehot, summarize_solver_runs
from xtubit.qpmhi import pareto_front, hypervolume, qpmhi_scores


def test_qubo_ising_conversion_and_validation():
    torch.manual_seed(42)
    n = 8
    dG = torch.randn(n, dtype=torch.float64)
    clash = torch.zeros((n, n), dtype=torch.float64)
    clash[0, 2] = clash[2, 0] = 1.0
    conn = torch.zeros((n, n), dtype=torch.float64)
    conn[1, 3] = conn[3, 1] = -1.0
    fragment_id = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)

    bundle = build_yanagisawa_qubo(dG, clash, conn, fragment_id)
    val = validate_qubo(bundle, equivalence_trials=200, atol=1e-8)
    assert val["passed"]
    assert val["max_equivalence_error"] < 1e-8


def test_exact_and_annealing_solvers():
    torch.manual_seed(7)
    n = 6
    fragment_id = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    dG = torch.tensor([-3.0, -1.0, -4.0, -2.0, -5.0, -1.0], dtype=torch.float64)
    clash = torch.zeros((n, n), dtype=torch.float64)
    conn = torch.zeros((n, n), dtype=torch.float64)
    bundle = build_yanagisawa_qubo(dG, clash, conn, fragment_id)

    x_exact, e_exact = brute_force_qubo(bundle.Q)
    assert x_exact.shape[0] == n

    J, h, c0 = qubo_to_ising(bundle.Q)
    s_tapsa = tapsa(J, h, cycles=100, batch=16, seed=0)
    s_spsa = spsa(J, h, cycles=100, batch=16, seed=0)
    s_tsb = two_stage_tesb(J, h, warm_iter=50, run_iter=100, seed=0)

    assert s_tapsa.shape[0] == n
    assert s_spsa.shape[0] == n
    assert s_tsb.shape[0] == n


def test_onehot_repair():
    bits = [1, 1, 0, 0]
    groups = [[0, 1], [2, 3]]
    dE = [-2.5, -1.0, -4.0, -2.0]
    repaired, violations = repair_onehot(bits, groups, dE)
    assert violations == 2
    # Group 0 should pick index 0 (lowest dE: -2.5)
    # Group 1 should pick index 2 (lowest dE: -4.0)
    assert repaired[0] == 1 and repaired[1] == 0
    assert repaired[2] == 1 and repaired[3] == 0


def test_pareto_front_and_hypervolume():
    Y = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 3.0],
        [0.5, 0.5, 0.5],
        [3.0, 3.0, 3.0],
    ])
    front = pareto_front(Y)
    # [3.0, 3.0, 3.0] dominates everything
    assert len(front) == 1
    np.testing.assert_array_equal(front[0], [3.0, 3.0, 3.0])

    ref = np.array([0.0, 0.0, 0.0])
    hv = hypervolume(front, ref)
    assert np.isclose(hv, 27.0)
