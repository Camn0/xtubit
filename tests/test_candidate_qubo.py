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


def test_no_silent_receptor_aliasing():
    """Verify that all 4 bundled PDB structures (5V3Y, 8TQV, 5V40, 8TQG) load distinct coordinates."""
    coords_5v3y = load_receptor_pocket_atoms("5V3Y")
    coords_8tqv = load_receptor_pocket_atoms("8TQV")
    coords_5v40 = load_receptor_pocket_atoms("5V40")
    coords_8tqg = load_receptor_pocket_atoms("8TQG")

    assert len(coords_5v3y) > 0, "5V3Y pocket coordinates must not be empty"
    assert len(coords_8tqv) > 0, "8TQV pocket coordinates must not be empty"
    assert len(coords_5v40) > 0, "5V40 pocket coordinates must not be empty"
    assert len(coords_8tqg) > 0, "8TQG pocket coordinates must not be empty"

    # Coordinates must not be identical across different crystallographic receptors
    assert not np.allclose(coords_5v40.mean(axis=0), coords_8tqg.mean(axis=0))
    assert not np.allclose(coords_5v3y.mean(axis=0), coords_8tqv.mean(axis=0))
    assert not np.allclose(coords_5v3y.mean(axis=0), coords_8tqg.mean(axis=0))

    # Missing receptor must raise FileNotFoundError, not silently fall back to 5V3Y
    import pytest
    with pytest.raises(FileNotFoundError):
        load_receptor_pocket_atoms("NON_EXISTENT_PDB")


def test_receptor_specific_qubo_four_pdbs():
    """Verify that QUBO matrices differ across all 4 authentic PDB crystal structures."""
    q_5v3y = build_candidate_qubo(TAM16_SMILES, receptor="5V3Y")["Q"]
    q_8tqv = build_candidate_qubo(TAM16_SMILES, receptor="8TQV")["Q"]
    q_5v40 = build_candidate_qubo(TAM16_SMILES, receptor="5V40")["Q"]
    q_8tqg = build_candidate_qubo(TAM16_SMILES, receptor="8TQG")["Q"]

    assert not torch.allclose(q_5v3y, q_8tqv)
    assert not torch.allclose(q_5v3y, q_8tqg)
    assert not torch.allclose(q_8tqv, q_8tqg)
    assert not torch.allclose(q_5v40, q_8tqg)


def test_brics_chemical_decomposition():
    """Verify candidate ligands decompose via BRICS / rotatable bonds with full atom conservation."""
    from rdkit import Chem
    from xtubit.b6_pairs import decompose_candidate_to_fragments
    mol_tam16 = Chem.MolFromSmiles(TAM16_SMILES)
    grps_tam16, atts_tam16 = decompose_candidate_to_fragments(mol_tam16, n_subpockets=4)
    total_atoms_tam16 = sum(len(g) for g in grps_tam16)
    assert total_atoms_tam16 == mol_tam16.GetNumHeavyAtoms(), "All heavy atoms must be conserved in TAM16"
    assert len(grps_tam16) >= 3, "TAM16 must decompose into at least 3 chemical fragments"
    assert len(atts_tam16) >= 2, "TAM16 fragments must track inter-fragment attachments"

    mol_x = Chem.MolFromSmiles(X20403_SMILES)
    grps_x, atts_x = decompose_candidate_to_fragments(mol_x, n_subpockets=4)
    total_atoms_x = sum(len(g) for g in grps_x)
    assert total_atoms_x == mol_x.GetNumHeavyAtoms(), "All heavy atoms must be conserved in X20403"
    assert len(grps_x) >= 3, "X20403 must decompose into multiple chemical fragments"


def test_solver_bits_causally_alter_3d_geometry():
    """Verify that distinct QUBO spin bitstrings causally reconstruct distinct 3D poses (RMSD > 1.0 A)."""
    from rdkit import Chem
    from xtubit.post_anneal import stitch_fragments_to_molecule
    qubo_res = build_candidate_qubo(TAM16_SMILES, receptor="5V3Y", poses_per_subpocket=3, n_subpockets=4)
    frag_poses = qubo_res["fragment_poses_coords"]
    atom_grps = qubo_res["atom_groups"]

    # Reconstruct pose for state 1 (pose 0 for all fragments)
    mol1 = stitch_fragments_to_molecule(
        {0: 0, 1: 0, 2: 0, 3: 0},
        variable_coords=qubo_res["coords"],
        poses_per_subpocket=3,
        candidate_smiles=TAM16_SMILES,
        fragment_poses_coords=frag_poses,
        atom_groups=atom_grps
    )

    # Reconstruct pose for state 2 (alternative poses)
    mol2 = stitch_fragments_to_molecule(
        {0: 2, 1: 1, 2: 2, 3: 1},
        variable_coords=qubo_res["coords"],
        poses_per_subpocket=3,
        candidate_smiles=TAM16_SMILES,
        fragment_poses_coords=frag_poses,
        atom_groups=atom_grps
    )

    n_h = Chem.RemoveHs(mol1).GetNumHeavyAtoms()
    c1 = np.array([list(mol1.GetConformer().GetAtomPosition(i)) for i in range(n_h)])
    c2 = np.array([list(mol2.GetConformer().GetAtomPosition(i)) for i in range(n_h)])

    rmsd = np.sqrt(np.mean(np.sum((c1 - c2) ** 2, axis=-1)))
    assert rmsd > 1.0, f"Expected causal geometric divergence (RMSD > 1.0 A), got {rmsd:.2f} A"


def test_receptor_conformation_causally_changes_pose():
    """Verify that docking into distinct receptors (5V3Y vs 8TQV) generates distinct 3D poses."""
    from rdkit import Chem
    from xtubit.post_anneal import stitch_fragments_to_molecule
    q_5v3y = build_candidate_qubo(TAM16_SMILES, receptor="5V3Y")
    q_8tqv = build_candidate_qubo(TAM16_SMILES, receptor="8TQV")

    mol_5v3y = stitch_fragments_to_molecule(
        {0: 0, 1: 0, 2: 0, 3: 0},
        variable_coords=q_5v3y["coords"],
        candidate_smiles=TAM16_SMILES,
        fragment_poses_coords=q_5v3y["fragment_poses_coords"],
        atom_groups=q_5v3y["atom_groups"]
    )

    mol_8tqv = stitch_fragments_to_molecule(
        {0: 0, 1: 0, 2: 0, 3: 0},
        variable_coords=q_8tqv["coords"],
        candidate_smiles=TAM16_SMILES,
        fragment_poses_coords=q_8tqv["fragment_poses_coords"],
        atom_groups=q_8tqv["atom_groups"]
    )

    n_h = Chem.RemoveHs(mol_5v3y).GetNumHeavyAtoms()
    c_5v3y = np.array([list(mol_5v3y.GetConformer().GetAtomPosition(i)) for i in range(n_h)])
    c_8tqv = np.array([list(mol_8tqv.GetConformer().GetAtomPosition(i)) for i in range(n_h)])

    cartesian_rmsd = np.sqrt(np.mean(np.sum((c_5v3y - c_8tqv) ** 2, axis=-1)))
    assert cartesian_rmsd > 5.0, f"Expected distinct in-pocket frames across 5V3Y and 8TQV, got {cartesian_rmsd:.2f} A"


def test_atom_typed_docking_score_penalizes_steric_clashes():
    """Verify atom-typed scoring function heavily penalizes steric overlap while rewarding H-bonds."""
    from xtubit.b6_pairs import compute_atom_typed_docking_score
    pock_coords = np.array([[0.0, 0.0, 0.0]])
    pock_elems = ["O"]

    # Clash: donor N placed 1.2 A away (far below sum of vdW radii ~ 3.07 A)
    frag_clash = np.array([[0.0, 0.0, 1.2]])
    score_clash = compute_atom_typed_docking_score(frag_clash, pock_coords, ["N"], pock_elems)

    # Ideal H-bond contact: donor N placed 2.85 A away
    frag_hbond = np.array([[0.0, 0.0, 2.85]])
    score_hbond = compute_atom_typed_docking_score(frag_hbond, pock_coords, ["N"], pock_elems)

    assert score_clash > 0.0, "Steric overlap must produce positive penalty"
    assert score_hbond < 0.0, "Optimal H-bond contact must produce favorable negative score"
    assert score_clash > score_hbond + 3.0, "Clash penalty must strongly exceed favorable contact"


def test_attachment_pair_connectivity_evaluation():
    """Verify connectivity matrix rewards poses where exact cleaved attachment atoms are close in 3D."""
    from xtubit.b6_pairs import generate_candidate_pocket_placements
    res = generate_candidate_pocket_placements(TAM16_SMILES, receptor="5V3Y", poses_per_subpocket=3, n_subpockets=4)
    conn = res["conn"]
    attachments = res["attachments"]
    atom_groups = res["atom_groups"]
    fragment_poses_coords = res["fragment_poses_coords"]
    fragment_id = res["fragment_id"]

    assert torch.allclose(conn, conn.T), "Connectivity matrix must be symmetric"
    assert len(attachments) > 0, "TAM16 must have cut attachment bonds across BRICS fragments"

    # For any rewarded pair, verify the specific attachment atoms are within bond distance
    for i in range(len(fragment_id)):
        f_i = int(fragment_id[i])
        for j in range(i + 1, len(fragment_id)):
            f_j = int(fragment_id[j])
            if conn[i, j] < 0:
                # Must be bonded fragments
                relevant_att = [att for att in attachments if (att[1] == f_i and att[3] == f_j) or (att[1] == f_j and att[3] == f_i)]
                assert len(relevant_att) > 0, f"Fragments {f_i} and {f_j} rewarded without an attachment record"
                # Check that at least one attachment pair is within the bond window
                min_d = min(
                    np.linalg.norm(fragment_poses_coords[i][atom_groups[f_i].index(u)] - fragment_poses_coords[j][atom_groups[f_j].index(v)])
                    if f_u == f_i else
                    np.linalg.norm(fragment_poses_coords[i][atom_groups[f_i].index(v)] - fragment_poses_coords[j][atom_groups[f_j].index(u)])
                    for (u, f_u, v, f_v) in relevant_att
                )
                assert 1.2 <= min_d <= 2.2, f"Rewarded pair has non-physical attachment distance: {min_d:.2f} A"


