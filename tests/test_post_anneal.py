import numpy as np
import torch
from rdkit import Chem
from xtubit.post_anneal import (
    decode_bitstring_to_subpockets,
    stitch_fragments_to_molecule,
    minimize_ligand_in_pocket,
    compute_crystal_rmsd,
    export_multi_model_sdf,
    TAM16_SMILES,
)


def test_bitstring_subpocket_decoding():
    """Verify bitstring correctly decodes one-hot active poses per sub-pocket."""
    # 60 bits: 6 pockets x 10 poses
    bits = np.zeros(60, dtype=int)
    bits[3] = 1   # Pocket 0, pose 3
    bits[15] = 1  # Pocket 1, pose 5
    bits[22] = 1  # Pocket 2, pose 2
    bits[37] = 1  # Pocket 3, pose 7
    bits[41] = 1  # Pocket 4, pose 1
    bits[59] = 1  # Pocket 5, pose 9

    decoded = decode_bitstring_to_subpockets(bits, poses_per_subpocket=10, n_subpockets=6)
    assert decoded[0] == 3
    assert decoded[1] == 5
    assert decoded[2] == 2
    assert decoded[3] == 7
    assert decoded[4] == 1
    assert decoded[5] == 9


def test_stitch_fragments_valency_and_aromaticity():
    """Verify assembled molecular topology is chemically valid and matches TAM16."""
    decoded = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    coords = torch.zeros((60, 3), dtype=torch.float64)

    mol = stitch_fragments_to_molecule(decoded, variable_coords=coords, poses_per_subpocket=10)

    # 1. Structure sanity
    assert mol is not None
    assert mol.GetNumConformers() >= 1

    # 2. Heavy atom count (True TAM16 has exactly 22 heavy atoms: C20H20O4 core)
    mol_no_h = Chem.RemoveHs(mol)
    assert mol_no_h.GetNumHeavyAtoms() == 22, f"Expected 22 heavy atoms for TAM16, got {mol_no_h.GetNumHeavyAtoms()}"

    # 3. Valency & aromaticity check
    sanitized = Chem.SanitizeMol(mol_no_h, catchErrors=True)
    assert sanitized == Chem.SanitizeFlags.SANITIZE_NONE, f"Sanitization error: {sanitized}"


def test_mmff94_force_field_energy_minimization():
    """Verify MMFF94 relaxation reduces potential energy under pocket restraints."""
    decoded = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    mol = stitch_fragments_to_molecule(decoded)

    # Freeze core atoms 0-5 (representing anchor pocket)
    frozen_atoms = [0, 1, 2, 3, 4, 5]
    res = minimize_ligand_in_pocket(mol, frozen_atom_indices=frozen_atoms, max_steps=100)

    assert res["minimized_energy_kcal_mol"] < res["initial_energy_kcal_mol"], "Energy was not reduced by minimization"
    assert res["delta_energy_kcal_mol"] < 0.0, "Delta energy should be negative"
    assert res["steps_executed"] == 100
    assert res["force_field"] in ["MMFF94", "UFF"]


def test_crystal_rmsd_and_multi_model_sdf_export():
    """Verify heavy-atom RMSD calculation and multi-model SDF block export."""
    decoded = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    mol_unrelaxed = stitch_fragments_to_molecule(decoded)
    res = minimize_ligand_in_pocket(mol_unrelaxed, max_steps=50)
    mol_relaxed = res["minimized_mol"]

    rmsd = compute_crystal_rmsd(mol_relaxed)
    assert 0.0 <= rmsd <= 5.0, f"RMSD {rmsd} A is unreasonable"

    sdf_str = export_multi_model_sdf(
        mol_unrelaxed,
        mol_relaxed,
        metadata={
            "mol_id": "TAM16",
            "initial_energy_kcal_mol": res["initial_energy_kcal_mol"],
            "minimized_energy_kcal_mol": res["minimized_energy_kcal_mol"],
            "delta_energy_kcal_mol": res["delta_energy_kcal_mol"],
            "rmsd_A": rmsd,
        }
    )

    assert "TAM16_Unrelaxed" in sdf_str
    assert "TAM16_MMFF94_Relaxed" in sdf_str
    assert "$$$$" in sdf_str


def test_stitch_candidate_independence():
    """Anti-cheating test: Verify reconstructed molecular graphs are strictly candidate-dependent."""
    decoded = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    cand_a_smi = TAM16_SMILES
    cand_b_smi = "Cc1ccc(C(=O)NCc2cccs2)c2ccccc12"  # distinct ligand structure

    mol_a = stitch_fragments_to_molecule(decoded, candidate_smiles=cand_a_smi)
    mol_b = stitch_fragments_to_molecule(decoded, candidate_smiles=cand_b_smi)

    smi_a = Chem.MolToSmiles(Chem.RemoveHs(mol_a), canonical=True)
    smi_b = Chem.MolToSmiles(Chem.RemoveHs(mol_b), canonical=True)

    assert smi_a != smi_b, "Candidate A and B must not produce identical molecular topologies"
    assert mol_a.GetNumHeavyAtoms() != mol_b.GetNumHeavyAtoms()
