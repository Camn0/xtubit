import pytest
from rdkit import Chem
from xtubit.admet_predictors import (
    predict_delaney_esol,
    predict_herg_liability,
    predict_microsomal_stability,
    predict_admet_profile,
)

def test_delaney_esol_solubility():
    # Hydrophilic small molecule (ethanol / glycine)
    glycine = Chem.MolFromSmiles("NCC(=O)O")
    sol_g = predict_delaney_esol(glycine)
    assert sol_g["logs"] > -2.0
    assert sol_g["is_soluble_50uM"] is True

    # Lipophilic TAM16 lead
    tam16 = Chem.MolFromSmiles("CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23")
    sol_tam = predict_delaney_esol(tam16)
    assert sol_tam["logs"] < -4.5
    assert sol_tam["solubility_uM"] < 50.0

def test_herg_liability():
    # Known potent hERG blocker (Terfenadine-like pharmacophore: basic tertiary amine + high lipophilicity)
    terfenadine = Chem.MolFromSmiles("CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(c3ccccc3)c4ccccc4")
    res_terf = predict_herg_liability(terfenadine)
    assert res_terf["herg_risk"] == "High Risk"
    assert res_terf["is_herg_safe"] is False

    # Neutral clean molecule (Aspirin)
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    res_asp = predict_herg_liability(aspirin)
    assert res_asp["is_herg_safe"] is True

def test_microsomal_stability():
    # Compound with labile ester linkage
    ester_mol = Chem.MolFromSmiles("CCCC(=O)OCC")
    res_est = predict_microsomal_stability(ester_mol)
    assert res_est["microsomal_t12_min"] < 45.0

    # Stable heterocyclic core without labile esters
    stable_core = Chem.MolFromSmiles("c1ccccc1-c2ncccc2")
    res_stab = predict_microsomal_stability(stable_core)
    assert res_stab["is_stable_30min"] is True

def test_admet_profile_tam16():
    tam16 = Chem.MolFromSmiles("CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23")
    prof = predict_admet_profile(tam16)
    assert "logs" in prof
    assert "herg_risk" in prof
    assert "microsomal_t12_min" in prof
