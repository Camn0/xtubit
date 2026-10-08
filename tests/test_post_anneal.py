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

    # 2. Heavy atom count (Authentic TAM16 / 5V8 in PDB 5V3Y has exactly 28 heavy atoms: C22H24N2O4)
    mol_no_h = Chem.RemoveHs(mol)
    assert mol_no_h.GetNumHeavyAtoms() == 28, f"Expected 28 heavy atoms for TAM16, got {mol_no_h.GetNumHeavyAtoms()}"


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


def test_pdb_reference_identity():
    """Verify authentic PDB crystal ligands match deposited identities: 5V3Y (5V8) and 8TQV (JS9)."""
    from pathlib import Path
    from rdkit.Chem import rdMolDescriptors

    # 1. TAM16 Reference (5V3Y / 5V8)
    sdf_5v3y = Path("data/raw/5v3y_ligand.sdf")
    assert sdf_5v3y.exists(), "PDB 5V3Y crystal ligand SDF must exist"
    suppl_5v3y = Chem.SDMolSupplier(str(sdf_5v3y))
    ref_5v8 = suppl_5v3y[0]
    assert ref_5v8.GetNumHeavyAtoms() == 28
    assert rdMolDescriptors.CalcMolFormula(ref_5v8) == "C22H24N2O4"
    assert Chem.MolToInchiKey(ref_5v8) == "PQGCMFVNJWTUFH-UHFFFAOYSA-N"

    # 2. X20403 Reference (8TQV / JS9)
    sdf_8tqv = Path("data/raw/8tqv_ligand.sdf")
    assert sdf_8tqv.exists(), "PDB 8TQV crystal ligand SDF must exist"
    suppl_8tqv = Chem.SDMolSupplier(str(sdf_8tqv))
    ref_js9 = suppl_8tqv[0]
    assert ref_js9.GetNumHeavyAtoms() == 40
    assert rdMolDescriptors.CalcMolFormula(ref_js9) == "C31H35N5O4"
    assert Chem.MolToInchiKey(ref_js9) == "MGTHESHOEDYGFD-UHFFFAOYSA-N"


def test_reference_compound_registry_matches_pdb():
    """Verify reference_compounds.yaml matches authentic crystallographic coordinates for TAM16 and X20403."""
    from pathlib import Path
    import yaml

    yaml_path = Path("data/raw/reference_compounds.yaml")
    assert yaml_path.exists(), "reference_compounds.yaml must exist"

    with open(yaml_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    # TAM16
    assert "TAM16" in registry
    tam16 = registry["TAM16"]
    assert tam16["pdb_ligand_id"] == "5V8"
    assert tam16["pdb_target"] == "5V3Y"
    assert tam16["inchikey"] == "PQGCMFVNJWTUFH-UHFFFAOYSA-N"
    assert tam16["num_heavy_atoms"] == 28
    assert tam16["formula"] == "C22H24N2O4"

    # X20403
    assert "X20403" in registry
    x20403 = registry["X20403"]
    assert x20403["pdb_ligand_id"] == "JS9"
    assert x20403["pdb_target"] == "8TQV"
    assert x20403["inchikey"] == "MGTHESHOEDYGFD-UHFFFAOYSA-N"
    assert x20403["num_heavy_atoms"] == 40
    assert x20403["formula"] == "C31H35N5O4"


def test_candidate_reaches_physics():
    """Verify candidate SMILES and placement coordinates propagate into reconstructed 3D pose."""
    decoded = {0: 1, 1: 0, 2: 2, 3: 0, 4: 1, 5: 0}
    test_coords = torch.ones((60, 3), dtype=torch.float64) * 15.0

    cand_smi = "CN(CC1(CC1)COC)C(=O)c2ccc(cc2)CCn3cc(nn3)c4ccc(nc4)c5cc(ccc5OC)OC"  # Authentic X20403 / JS9
    mol_custom = stitch_fragments_to_molecule(
        decoded,
        variable_coords=test_coords,
        candidate_smiles=cand_smi,
    )
    assert mol_custom.GetNumHeavyAtoms() == 40
    # Centroid shifted towards test_coords
    conf = mol_custom.GetConformer()
    centroid = np.mean([list(conf.GetAtomPosition(i)) for i in range(40)], axis=0)
    assert np.all(np.abs(centroid) > 5.0), "Conformer should be placed near test_coords"


