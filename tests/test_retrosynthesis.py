from rdkit import Chem
from xtubit.retrosynthesis import (
    estimate_synthetic_complexity,
    estimate_synthetic_route,
    update_qpmhi_with_scscore,
)


def test_scscore_calibration_hierarchy():
    """Verify SCScore properly ranks compounds from simple starting materials to complex polycycles."""
    bz = Chem.MolFromSmiles("c1ccccc1")
    tam16 = Chem.MolFromSmiles("CNC(=O)c1c(-c2ccc(O)cc2)oc2ccc(O)c(CN3CCCCC3)c12")
    taxol = Chem.MolFromSmiles(
        "CC(=O)OC1C(=O)C2(C)C(O)CC3OCC3(OC(=O)C)C2C(OC(=O)c4ccccc4)C(O)(C1(C)C)CC(NC(=O)c5ccccc5)C(O)c6ccccc6"
    )

    sc_bz = estimate_synthetic_complexity(bz)
    sc_tam16 = estimate_synthetic_complexity(tam16)
    sc_taxol = estimate_synthetic_complexity(taxol)

    # 1. Hierarchy: Simple < Clinical Lead < Complex Natural Product
    assert 1.0 <= sc_bz <= 1.8, f"Benzene SCScore {sc_bz} out of expected [1.0, 1.8] range"
    assert 2.5 <= sc_tam16 <= 3.8, f"TAM16 SCScore {sc_tam16} out of expected [2.5, 3.8] lead range"
    assert sc_taxol >= 4.5, f"Taxol SCScore {sc_taxol} should be >= 4.5"
    assert sc_bz < sc_tam16 < sc_taxol, "Synthetic complexity ordering violated"


def test_tam16_three_step_synthesis_benchmark():
    """Verify TAM16 retrosynthetic route recapitulates published 3-step synthesis (Aggarwal et al. 2017)."""
    tam16 = Chem.MolFromSmiles("CNC(=O)c1c(-c2ccc(O)cc2)oc2ccc(O)c(CN3CCCCC3)c12")
    route = estimate_synthetic_route(tam16)

    assert route["num_steps"] == 3, f"Expected 3 synthetic steps for TAM16, got {route['num_steps']}"
    assert route["is_synthetically_tractable"] is True
    assert route["building_blocks_available"] is True
    assert any("Suzuki-Miyaura" in rxn for rxn in route["reactions"]), "Suzuki-Miyaura coupling not identified in TAM16 route"




def test_synthetic_step_hard_constraint():
    """Verify that molecules requiring > 4 steps are flagged as untractable."""
    # Highly substituted poly-branched molecule requiring > 4 strategic coupling steps
    complex_mol = Chem.MolFromSmiles(
        "O=C(Nc1ccc(C(=O)Nc2ccc(C(=O)Nc3ccc(C(=O)Nc4ccccc4)cc3)cc2)cc1)c5ccccc5-c6ccccc6"
    )
    route = estimate_synthetic_route(complex_mol)

    assert route["num_steps"] > 4, f"Expected > 4 steps, got {route['num_steps']}"
    assert route["is_synthetically_tractable"] is False
    assert "Exceeds max synthetic steps constraint" in route["rejection_reason"]


def test_qpmhi_retrosynthesis_penalty():
    """Verify qPMHI penalizes high SCScore and routes exceeding step constraints."""
    mu = 8.0
    qed = 0.8
    sa = 2.5
    scscore_easy = 2.0
    scscore_hard = 4.5

    # Case 1: Tractable 2-step synthesis
    qpmhi_easy = update_qpmhi_with_scscore(mu, qed, sa, scscore_easy, steps=2)

    # Case 2: Difficult chemistry (high SCScore, 3 steps)
    qpmhi_hard = update_qpmhi_with_scscore(mu, qed, sa, scscore_hard, steps=3)

    # Case 3: Rejected route (> 4 steps)
    qpmhi_rejected = update_qpmhi_with_scscore(mu, qed, sa, scscore_easy, steps=5)

    assert qpmhi_easy > qpmhi_hard > qpmhi_rejected
    # Verify severe penalty applied for steps > 4
    assert qpmhi_rejected <= 0.3 * qpmhi_easy
