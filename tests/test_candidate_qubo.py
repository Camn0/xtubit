import pytest
import torch
import numpy as np
from pathlib import Path

from xtubit.b6_pairs import (
    load_receptor_pocket_atoms,
    compute_protein_fragment_contact_potential,
    partition_molecule_to_subpockets,
    generate_candidate_pocket_placements,
    build_candidate_qubo,
)
from xtubit.post_anneal import TAM16_SMILES
from xtubit.pipeline import run_cross_docking_benchmark


X20403_SMILES = "CN(CC1(CC1)COC)C(=O)c2ccc(cc2)CCn3cc(nn3)c4ccc(nc4)c5cc(ccc5OC)OC"


def test_receptor_pocket_atoms_loading():
    """Verify receptor heavy atoms are extracted from 5V3Y and 8TQV crystal structures."""
    atoms_5v3y = load_receptor_pocket_atoms("5V3Y", radius=10.0)
    assert len(atoms_5v3y) > 50, "5V3Y pocket must contain >50 heavy atoms"

    atoms_8tqv = load_receptor_pocket_atoms("8TQV", radius=10.0)
    assert len(atoms_8tqv) > 50, "8TQV pocket must contain >50 heavy atoms"

    # Distinct crystallographic center frames
    mean_5v3y = np.mean(atoms_5v3y, axis=0)
    mean_8tqv = np.mean(atoms_8tqv, axis=0)
    assert np.linalg.norm(mean_5v3y - mean_8tqv) > 20.0, "5V3Y and 8TQV must be in distinct crystallographic frames"


def test_protein_fragment_contact_potential():
    """Verify contact potential calculates attractive binding wells and steric overlap."""
    pocket = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [0.0, 3.8, 0.0]])

    # Optimum contact at ~3.8 A
    frag_opt = np.array([[0.0, 0.0, 3.8]])
    e_opt = compute_protein_fragment_contact_potential(frag_opt, pocket)
    assert e_opt < 0.0, f"Optimal contact should be attractive (negative dG): {e_opt}"

    # Clashing contact at 1.0 A
    frag_clash = np.array([[0.0, 0.0, 1.0]])
    e_clash = compute_protein_fragment_contact_potential(frag_clash, pocket)
    assert e_clash > 0.0, f"Steric clash should be penalized (positive dG): {e_clash}"


def test_candidate_dependent_qubo_divergence():
    """Verify candidate A and candidate B yield structurally different QUBO matrices."""
    res_tam = build_candidate_qubo(TAM16_SMILES, receptor="5V3Y", poses_per_subpocket=3, n_subpockets=4)
    res_x20 = build_candidate_qubo(X20403_SMILES, receptor="5V3Y", poses_per_subpocket=3, n_subpockets=4)

    Q_tam = res_tam["Q"]
    Q_x20 = res_x20["Q"]

    # Invariant: QUBO must not be identical across different molecules
    diff = float(torch.norm(Q_tam - Q_x20))
    assert diff > 5.0, f"QUBO matrix must differ between TAM16 and X20403, got diff={diff}"
    assert not torch.allclose(Q_tam, Q_x20, atol=1e-3)


def test_receptor_dependent_qubo_divergence():
    """Verify the exact same candidate in receptor 5V3Y vs 8TQV yields different QUBO matrices."""
    res_5v3y = build_candidate_qubo(TAM16_SMILES, receptor="5V3Y", poses_per_subpocket=3, n_subpockets=4)
    res_8tqv = build_candidate_qubo(TAM16_SMILES, receptor="8TQV", poses_per_subpocket=3, n_subpockets=4)

    Q_5v3y = res_5v3y["Q"]
    Q_8tqv = res_8tqv["Q"]

    # Invariant: QUBO must not be identical across different receptor conformations
    diff = float(torch.norm(Q_5v3y - Q_8tqv))
    assert diff > 5.0, f"QUBO matrix must differ between 5V3Y and 8TQV, got diff={diff}"
    assert not torch.allclose(Q_5v3y, Q_8tqv, atol=1e-3)


def test_cross_docking_benchmark_execution(tmp_path):
    """Verify execution of full cross-docking protocol and metric persistence."""
    cross_res = run_cross_docking_benchmark(tmp_path)
    assert cross_res["status"] == "COMPLETED"
    assert len(cross_res["experiments"]) == 4

    metrics_file = tmp_path / "metrics" / "cross_docking.json"
    assert metrics_file.exists(), "cross_docking.json must be saved"

    exp_names = [e["experiment"] for e in cross_res["experiments"]]
    assert "Cognate (TAM16 in 5V3Y)" in exp_names
    assert "Cognate (X20403 in 8TQV)" in exp_names
    assert "Cross-Docking (X20403 in 5V3Y)" in exp_names
    assert "Cross-Docking (TAM16 in 8TQV)" in exp_names

    for e in cross_res["experiments"]:
        assert "receptor_contact_dG" in e
        assert "receptor_clashes" in e
        assert "ligand_strain_energy_kcal_mol" in e
        assert "conformer_aligned_rmsd_A" in e
        assert "in_pocket_cartesian_rmsd_A" in e


def test_3d_rigid_fragment_grid_placements():
    """Verify 3D rigid fragment docking explores SO(3) rotations and translational cavity space."""
    res = generate_candidate_pocket_placements(TAM16_SMILES, receptor="5V3Y", poses_per_subpocket=3, n_subpockets=4)
    assert len(res["coords"]) == 12
    assert len(res["variable_meta"]) == 12

    # Check that poses within the same sub-pocket are spatially diverse
    c0 = np.array(res["variable_meta"][0]["coord"])
    c1 = np.array(res["variable_meta"][1]["coord"])
    assert np.linalg.norm(c0 - c1) > 0.5, "Poses within subpocket must be spatially diverse"


def test_streamlit_bayesian_checkpoint_dynamic_loading():
    """Verify evaluate_single_smiles in Streamlit dynamically loads 7-D Bayesian checkpoint without fallback."""
    from app.streamlit_app import evaluate_single_smiles
    res = evaluate_single_smiles(TAM16_SMILES, mol_id="TAM16_TEST")
    assert res is not None
    assert "surrogate_source" in res
    assert "Trained Bayesian Model" in res["surrogate_source"], f"Expected Trained Bayesian Model, got: {res['surrogate_source']}"
    assert 4.0 <= res["mu"] <= 9.0
    assert res["sigma"] > 0.1
