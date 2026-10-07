from __future__ import annotations
from typing import Dict, Any, List
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import FilterCatalog

# Initialize singletons for fast reusable filtering
_pains_params = FilterCatalog.FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_pains_catalog = FilterCatalog.FilterCatalog(_pains_params)

_brenk_params = FilterCatalog.FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
_brenk_catalog = FilterCatalog.FilterCatalog(_brenk_params)


def evaluate_pains(mol: Chem.Mol) -> Dict[str, Any]:
    """Evaluate mol against Baell & Holloway PAINS filter (A, B, C).
    
    Returns:
        dict: {'has_pains': bool, 'pains_matches': list[str]}
    """
    if mol is None:
        return {"has_pains": False, "pains_matches": []}
    matches = _pains_catalog.GetMatches(mol)
    desc_list = [entry.GetDescription() for entry in matches]
    return {
        "has_pains": len(desc_list) > 0,
        "pains_matches": desc_list,
    }


def evaluate_brenk(mol: Chem.Mol) -> Dict[str, Any]:
    """Evaluate mol against Brenk structural alert toxicophores.
    
    Returns:
        dict: {'has_brenk': bool, 'brenk_matches': list[str]}
    """
    if mol is None:
        return {"has_brenk": False, "brenk_matches": []}
    matches = _brenk_catalog.GetMatches(mol)
    desc_list = [entry.GetDescription() for entry in matches]
    return {
        "has_brenk": len(desc_list) > 0,
        "brenk_matches": desc_list,
    }


def evaluate_rule_of_two(mol: Chem.Mol) -> Dict[str, Any]:
    """Evaluate fragment suitability against BMS / Abbott Rule-of-Two (Ro2).
    
    Criteria for high-quality fragment libraries:
    - MW <= 300 Da
    - cLogP <= 3.0
    - Rotatable Bonds <= 3
    - H-Bond Donors <= 3
    - H-Bond Acceptors <= 3
    
    Returns:
        dict: {'ro2_compliant': bool, 'ro2_violations': list[str]}
    """
    if mol is None:
        return {"ro2_compliant": False, "ro2_violations": ["Invalid Molecule"]}
    
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    rotb = int(Lipinski.NumRotatableBonds(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))

    violations = []
    if mw > 300.0:
        violations.append(f"MW {mw:.1f} > 300 Da")
    if logp > 3.0:
        violations.append(f"cLogP {logp:.2f} > 3.0")
    if rotb > 3:
        violations.append(f"RotB {rotb} > 3")
    if hbd > 3:
        violations.append(f"HBD {hbd} > 3")
    if hba > 3:
        violations.append(f"HBA {hba} > 3")

    return {
        "ro2_compliant": len(violations) == 0,
        "ro2_violations": violations,
        "mw": mw,
        "logp": logp,
        "rotb": rotb,
        "hbd": hbd,
        "hba": hba,
    }


def evaluate_medchem_cleanliness(mol: Chem.Mol) -> Dict[str, Any]:
    """Run comprehensive medicinal chemistry cleanliness audit.
    
    Returns:
        dict: Combined results including overall 'is_clean' flag.
    """
    pains = evaluate_pains(mol)
    brenk = evaluate_brenk(mol)
    ro2 = evaluate_rule_of_two(mol)
    is_clean = (not pains["has_pains"]) and (not brenk["has_brenk"])

    return {
        "is_clean": is_clean,
        "has_pains": pains["has_pains"],
        "pains_matches": pains["pains_matches"],
        "has_brenk": brenk["has_brenk"],
        "brenk_matches": brenk["brenk_matches"],
        "ro2_compliant": ro2["ro2_compliant"],
        "ro2_violations": ro2["ro2_violations"],
    }
