from rdkit import Chem
from xtubit.wetlab_dossier import (
    generate_preclinical_specification_sheet,
    map_commercial_building_blocks,
    generate_forward_synthetic_scheme,
    build_turnkey_dossier_package,
    BIOASSAY_PROTOCOLS,
)


def test_preclinical_specification_generation():
    """Verify chemical specification sheet contains IUPAC, formula, MW, and InChI."""
    tam16_smi = "CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23"
    mol = Chem.MolFromSmiles(tam16_smi)
    specs = generate_preclinical_specification_sheet(mol, mol_id="TAM16")

    assert specs["mol_id"] == "TAM16"
    assert specs["canonical_smiles"] is not None
    assert specs["formula_weight_Da"] > 250.0
    assert specs["clogp"] > 3.0
    assert specs["inchikey"] is not None
    assert len(specs["inchikey"]) == 27  # standard InChIKey length


def test_commercial_building_blocks_mapping():
    """Verify decomposition into commercial starting material building blocks."""
    tam16_smi = "CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23"
    blocks = map_commercial_building_blocks(tam16_smi)

    assert len(blocks) >= 2
    for b in blocks:
        assert "chemical_name" in b
        assert "enamine_id" in b
        assert "mcule_id" in b
        assert "cas_number" in b
        assert "est_cost_per_gram" in b
        assert "est_lead_time" in b


def test_standardized_bioassay_protocols_attached():
    """Verify both Pks13 enzymatic and Mtb H37Rv cellular bioassay protocols exist."""
    assert "Pks13_Fluorogenic_Esterase_IC50" in BIOASSAY_PROTOCOLS
    assert "Mtb_H37Rv_Cellular_MIC90" in BIOASSAY_PROTOCOLS

    p1 = BIOASSAY_PROTOCOLS["Pks13_Fluorogenic_Esterase_IC50"]
    assert "buffer_conditions" in p1
    assert "4-methylumbelliferyl heptanoate" in p1["substrate"]
    assert "positive_control" in p1

    p2 = BIOASSAY_PROTOCOLS["Mtb_H37Rv_Cellular_MIC90"]
    assert "H37Rv" in p2["strain"]
    assert "positive_control" in p2


def test_turnkey_dossier_package_assembly():
    """Verify end-to-end turnkey dossier assembly for wet-lab handoff."""
    row = {
        "mol_id": "TAM16",
        "smiles_can": "CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23",
    }
    dossier = build_turnkey_dossier_package(row, reviewer="Lead Chemist")

    assert dossier["candidate_id"] == "TAM16"
    assert dossier["reviewer"] == "Lead Chemist"
    assert len(dossier["building_blocks"]) >= 2
    assert len(dossier["synthetic_scheme"]) == 2
    assert dossier["estimated_starting_materials_cost_USD"] > 0

