import pytest
from rdkit import Chem
from xtubit.medchem_filters import (
    evaluate_pains,
    evaluate_brenk,
    evaluate_rule_of_two,
    evaluate_medchem_cleanliness,
)

def test_pains_detection():
    # Rhodanine core (classic PAINS false positive)
    rhodanine = Chem.MolFromSmiles("O=C1NC(=S)SC1=Cc2ccccc2")
    res = evaluate_pains(rhodanine)
    assert res["has_pains"] is True
    assert len(res["pains_matches"]) > 0

    # TAM16 lead (clean lead, no PAINS)
    tam16 = Chem.MolFromSmiles("CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23")
    res_tam = evaluate_pains(tam16)
    assert res_tam["has_pains"] is False
    assert len(res_tam["pains_matches"]) == 0

def test_brenk_structural_alerts():
    # Alkyl halide / reactive electrophile
    alkyl_halide = Chem.MolFromSmiles("CCCCBr")
    res = evaluate_brenk(alkyl_halide)
    assert res["has_brenk"] is True

    # Hydrazine derivative
    hydrazine = Chem.MolFromSmiles("c1ccccc1NN")
    res_h = evaluate_brenk(hydrazine)
    assert res_h["has_brenk"] is True

    # TAM16 lead (clean, no reactive alerts)
    tam16 = Chem.MolFromSmiles("CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23")
    res_tam = evaluate_brenk(tam16)
    assert res_tam["has_brenk"] is False

def test_rule_of_two_fragment_boundaries():
    # Small fragment (Benzofuran Core)
    benzofuran = Chem.MolFromSmiles("Cc1oc2ccccc2c1")
    res_frag = evaluate_rule_of_two(benzofuran)
    assert res_frag["ro2_compliant"] is True
    assert len(res_frag["ro2_violations"]) == 0

    # Large full-sized molecule (violates MW > 300, cLogP > 3.0)
    large_mol = Chem.MolFromSmiles("CC1=C(C(=O)NCC2=CC=CS2)C3=C(O1)C=CC(=C3)C4=CC=CC=C4")
    res_large = evaluate_rule_of_two(large_mol)
    assert res_large["ro2_compliant"] is False
    assert any("MW" in v for v in res_large["ro2_violations"])

def test_medchem_cleanliness_suite():
    tam16 = Chem.MolFromSmiles("CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23")
    audit = evaluate_medchem_cleanliness(tam16)
    assert audit["is_clean"] is True
    assert audit["has_pains"] is False
    assert audit["has_brenk"] is False

