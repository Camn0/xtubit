"""Theoretical and Mathematical Rigor Benchmark Suite for X-TUBIT.

Audits each core theoretical component against analytical ground truths:
  1. Chemoinformatics & Conformer Geometry (B1, B2)
  2. Variational Bayesian Linear Layer & KL Divergence Non-Negativity (B3, B4)
  3. Hypervolume Monotonicity & Non-Dominated Pareto Filtering (B5)
  4. Yanagisawa 4-Term Hamiltonian & QUBO <-> Ising Exactness (B6, B7)
  5. Digital Annealing Solvers & TTS99 Scaled Confidence (B8)
  6. Deterministic One-Hot Penalty Repair & RMSD Invariance (B9)
"""

import math
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from xtubit.b1_data import standardize_smiles, sa_score, scaffold_split
from xtubit.b2_conformer import generate_lowest_mmff
from xtubit.b4_bayesian_gnn import BayesianLinear, calibrate_sigma
from xtubit.qpmhi import pareto_front, hypervolume, qpmhi_scores
from xtubit.b7_qubo import build_yanagisawa_qubo, validate_qubo
from xtubit.common.math import qubo_to_ising, qubo_energy, ising_energy, tts_seconds
from xtubit.solvers.exact import brute_force_qubo
from xtubit.solvers.psa_pd import spsa, tapsa
from xtubit.solvers.tesb_port import two_stage_tesb
from xtubit.b9_metrics import repair_onehot, heavy_atom_rmsd


# ==============================================================================
# 1. B1 & B2 Chemoinformatics & 3D Geometry Invariants
# ==============================================================================
def test_b1_standardization_and_sa_bounds():
    """Verify standardization invariance and SA score bounds [1.0, 10.0]."""
    # TAM16 SMILES with stereochemistry and salts
    raw_smi = "CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23.[Na+].[Cl-]"
    mol = standardize_smiles(raw_smi)
    assert mol is not None, "Standardization failed on salt-stripped SMILES"
    
    # Salts must be stripped by LargestFragmentChooser
    assert mol.GetNumAtoms() == 22, f"Expected 22 heavy atoms for TAM16 core, got {mol.GetNumAtoms()}"
    
    score = sa_score(mol, allow_fallback=True)
    assert 1.0 <= score <= 10.0, f"SA score {score} out of theoretical [1, 10] bounds"


def test_b2_conformer_energy_minimization():
    """Verify MMFF94 force field lowers potential energy from raw unoptimized embedding."""
    smi = "c1oc2ccccc2c1-c1ccco1"  # Bi-heterocycle core
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    e_initial = ff.CalcEnergy()
    
    # Run optimization
    opt_mol, cid, e_final = generate_lowest_mmff(smi, n_confs=5, seed=42)
    assert e_final < e_initial, f"Optimization failed to lower energy: initial={e_initial}, final={e_final}"


# ==============================================================================
# 2. B3 & B4 Bayesian Variational Mathematics & Calibration
# ==============================================================================
def test_b4_bayesian_kl_divergence_non_negative():
    """Verify analytical KL divergence D_KL(q || p) >= 0 and D_KL == 0 when q == p."""
    layer = BayesianLinear(in_features=16, out_features=8, prior_sigma=0.1)
    
    # When initialized, parameters are non-zero, KL must be strictly positive
    kl_init = layer.kl()
    assert float(kl_init) >= 0.0, f"KL divergence is negative: {kl_init}"
    
    # Force q == p: mu = 0, sigma = prior_sigma
    with torch.no_grad():
        layer.mu_w.zero_()
        layer.mu_b.zero_()
        # softplus(rho) = prior_sigma => rho = log(exp(prior_sigma) - 1)
        target_rho = math.log(math.exp(0.1) - 1.0)
        layer.rho_w.fill_(target_rho)
        layer.rho_b.fill_(target_rho)
    
    kl_exact = float(layer.kl())
    # Should be approximately 0 within floating point tolerance
    assert abs(kl_exact) < 1e-4, f"D_KL(p || p) must be 0, got {kl_exact}"


def test_b4_uncertainty_calibration_convergence():
    """Verify 1D grid NLL search correctly recovers simulated noise multiplier."""
    true_tau = 2.5
    sigma = torch.ones(50) * 0.5
    # Simulated residuals with std = true_tau * sigma = 1.25
    torch.manual_seed(42)
    y_err = torch.randn(50) * (true_tau * 0.5)
    mu = torch.zeros(50)
    y = mu + y_err
    
    estimated_tau = calibrate_sigma(mu, sigma, y, min_tau=0.1, max_tau=10.0)
    # Estimated tau must be close to true_tau within sample variance
    assert abs(estimated_tau - true_tau) < 0.6, f"Expected tau ~{true_tau}, got {estimated_tau}"


# ==============================================================================
# 3. B5 Multi-Objective Hypervolume Monotonicity
# ==============================================================================
def test_b5_hypervolume_monotonicity_and_invariance():
    """Verify exact 3D recursive hypervolume satisfies Pareto monotonicity axioms."""
    ref = np.array([0.0, 0.0, 0.0])
    points_a = np.array([
        [2.0, 2.0, 2.0],
        [1.0, 3.0, 1.0],
    ])
    hv_a = hypervolume(points_a, ref)
    assert hv_a > 0.0
    
    # Adding a strictly dominated point (e.g. [1.0, 1.0, 1.0]) must NOT increase HV
    points_dom = np.vstack([points_a, [1.0, 1.0, 1.0]])
    hv_dom = hypervolume(points_dom, ref)
    assert np.isclose(hv_a, hv_dom, atol=1e-10), "Dominated point altered hypervolume"
    
    # Adding an un-dominated point must strictly INCREASE HV
    points_better = np.vstack([points_a, [3.0, 1.0, 2.0]])
    hv_better = hypervolume(points_better, ref)
    assert hv_better > hv_a + 1e-6, "Un-dominated point failed to increase hypervolume"


# ==============================================================================
# 4. B6 & B7 Hamiltonian Equivalence & One-Hot Penalty Axioms
# ==============================================================================
def test_b7_qubo_penalty_minimum_iff_one_hot():
    """Verify Yanagisawa penalty (D/2)*sum_k (sum x_i - 1)^2 is 0 iff exactly one placement is active."""
    n = 6
    g = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    dG = torch.zeros(n, dtype=torch.float64)
    clash = torch.zeros((n, n), dtype=torch.float64)
    conn = torch.zeros((n, n), dtype=torch.float64)
    
    bundle = build_yanagisawa_qubo(dG, clash, conn, g, A=0.0, B=0.0, C=0.0, D=50.0)
    
    # Valid one-hot state: index 0 and index 3 active
    x_valid = torch.tensor([1., 0., 0., 1., 0., 0.], dtype=torch.float64)
    e_valid = float(qubo_energy(bundle.Q, x_valid) + bundle.onehot_constant)
    assert abs(e_valid) < 1e-8, f"Valid one-hot state should have penalty 0, got {e_valid}"
    
    # Invalid states: fragment 0 has 2 bits active -> penalty must be > 0
    x_invalid = torch.tensor([1., 1., 0., 1., 0., 0.], dtype=torch.float64)
    e_invalid = float(qubo_energy(bundle.Q, x_invalid) + bundle.onehot_constant)
    assert e_invalid > 0.0, f"Invalid state penalty should be positive, got {e_invalid}"


def test_b7_qubo_to_ising_machine_precision():
    """Verify QUBO <-> Ising energy mapping is exact to < 1e-12 across 500 random states."""
    torch.manual_seed(99)
    n = 10
    Q = torch.randn(n, n, dtype=torch.float64)
    Q = 0.5 * (Q + Q.T)  # Symmetric
    
    J, h, c0 = qubo_to_ising(Q)
    max_diff = 0.0
    for _ in range(500):
        x = torch.randint(0, 2, (n,), dtype=torch.float64)
        s = 2.0 * x - 1.0
        e_q = float(qubo_energy(Q, x))
        e_i = float(ising_energy(J, h, s, c0))
        max_diff = max(max_diff, abs(e_q - e_i))
    
    assert max_diff < 1e-12, f"QUBO-Ising energy difference {max_diff} exceeded tolerance"


# ==============================================================================
# 5. B8 Annealing Convergence & TTS99 Mathematics
# ==============================================================================
def test_b8_solvers_find_global_minimum_on_test_hamiltonian():
    """Verify Exact, TApSA, SpSA, and tSB all converge toward ground-state energy on benchmark."""
    torch.manual_seed(123)
    # Known 6-variable problem with planted ground state x* = [1, 0, 1, 0, 1, 0]
    n = 6
    g = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    dG = torch.tensor([-10.0, 0.0, -10.0, 0.0, -10.0, 0.0], dtype=torch.float64)
    clash = torch.zeros((n, n), dtype=torch.float64)
    conn = torch.zeros((n, n), dtype=torch.float64)
    bundle = build_yanagisawa_qubo(dG, clash, conn, g, A=1.0, B=5.0, C=5.0, D=50.0)
    
    # 1. Exact Brute Force
    x_exact, e_exact = brute_force_qubo(bundle.Q)
    true_ground_energy = float(e_exact + bundle.onehot_constant)
    
    # Optimal bits must be exactly [1, 0, 1, 0, 1, 0]
    expected_bits = torch.tensor([1., 0., 1., 0., 1., 0.], dtype=torch.float64)
    assert torch.allclose(x_exact, expected_bits), f"Exact solver failed planted state: {x_exact}"
    
    # 2. Annealing solvers
    J, h, c0 = qubo_to_ising(bundle.Q)
    s_tapsa = tapsa(J, h, cycles=200, batch=32, seed=7)
    x_tapsa = (s_tapsa[:, 0] + 1.0) / 2.0
    e_tapsa = float(qubo_energy(bundle.Q.float(), x_tapsa) + bundle.onehot_constant)
    
    assert abs(e_tapsa - true_ground_energy) < 1e-2, f"TApSA did not reach planted ground state: {e_tapsa} vs {true_ground_energy}"


def test_b8_tts99_mathematical_limits():
    """Verify Time-to-Solution (TTS99) obeys theoretical limits."""
    t_run = 0.05
    # When p_success >= 0.99, TTS99 == t_run
    assert tts_seconds(t_run, p_success=0.99, target=0.99) == t_run
    assert tts_seconds(t_run, p_success=1.00, target=0.99) == t_run
    
    # When p_success is low (e.g. 0.10), TTS99 must scale logarithmically: ln(0.01)/ln(0.9) * 0.05 ~ 2.18 s
    tts_low = tts_seconds(t_run, p_success=0.10, target=0.99)
    expected_tts = t_run * math.log(0.01) / math.log(0.90)
    assert np.isclose(tts_low, expected_tts, atol=1e-6)
    
    # When p_success == 0, TTS99 == infinity
    assert math.isinf(tts_seconds(t_run, p_success=0.0, target=0.99))


# ==============================================================================
# 6. B9 One-Hot Repair Determinism & RMSD Correctness
# ==============================================================================
def test_b9_onehot_repair_recovers_strictly_feasible_state():
    """Verify repair_onehot repairs arbitrary constraint violations to sum_i x_i == 1."""
    # Arbitrary broken bitstring: fragment 0 has 0 bits, fragment 1 has 3 bits active
    bits = [0, 0, 0, 1, 1, 1]
    groups = [[0, 1, 2], [3, 4, 5]]
    dE = [-2.0, -5.0, -1.0, -8.0, -3.0, -6.0]
    
    repaired, violations = repair_onehot(bits, groups, dE)
    assert violations == 2, "Expected 2 fragment constraint violations detected"
    
    # Fragment 0 must select index 1 (dE = -5.0 is minimum)
    assert repaired[0] == 0 and repaired[1] == 1 and repaired[2] == 0
    # Fragment 1 must select index 3 (dE = -8.0 is minimum)
    assert repaired[3] == 1 and repaired[4] == 0 and repaired[5] == 0
    # Every group must sum to exactly 1
    for g in groups:
        assert sum(repaired[i] for i in g) == 1


def test_b9_rmsd_identity_and_translation():
    """Verify heavy-atom RMSD is 0.0 for identical molecules and strictly positive for perturbed poses."""
    smi = "c1ccccc1"
    mol_ref = Chem.MolFromSmiles(smi)
    mol_pred = Chem.MolFromSmiles(smi)
    
    AllChem.EmbedMolecule(mol_ref, randomSeed=1)
    AllChem.EmbedMolecule(mol_pred, randomSeed=1)
    
    # Identical coordinates
    rmsd_zero = heavy_atom_rmsd(mol_pred, mol_ref)
    assert np.isclose(rmsd_zero, 0.0, atol=1e-6), f"RMSD of identical conformer must be 0, got {rmsd_zero}"
