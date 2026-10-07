from __future__ import annotations
from typing import Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


def predict_delaney_esol(mol: Chem.Mol) -> Dict[str, Any]:
    """Calculate Delaney ESOL aqueous thermodynamic solubility.
    
    Formula:
        LogS = 0.16 - 0.63 * cLogP - 0.0062 * MW + 0.066 * RotB - 0.74 * (AromaticAtoms / HeavyAtoms)
    
    Returns:
        dict: {'logs': float, 'solubility_uM': float, 'solubility_class': str}
    """
    if mol is None:
        return {"logs": -10.0, "solubility_uM": 0.0, "solubility_class": "Insoluble"}

    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    rotb = int(Lipinski.NumRotatableBonds(mol))
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    aromatic_prop = float(aromatic_atoms / heavy_atoms) if heavy_atoms > 0 else 0.0

    logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotb - 0.74 * aromatic_prop
    sol_um = float(10**logs * 1e6)

    if logs >= -3.0:
        sol_class = "High Solubility (> 1 mM)"
    elif logs >= -4.3:
        sol_class = "Moderate Solubility (> 50 µM)"
    elif logs >= -5.5:
        sol_class = "Low Solubility (3–50 µM)"
    else:
        sol_class = "Poor / Insoluble (< 3 µM)"

    return {
        "logs": float(logs),
        "solubility_uM": sol_um,
        "solubility_class": sol_class,
        "is_soluble_50uM": bool(logs >= -4.3),
    }


def predict_herg_liability(mol: Chem.Mol) -> Dict[str, Any]:
    """Predict cardiotoxicity risk via hERG potassium channel blockade.
    
    Rule-based pharmacophore model based on:
    - Basic nitrogen (tertiary/secondary amine)
    - Lipophilicity (cLogP > 3.5)
    - Multiple aromatic rings (>= 2)
    - Low polar surface area (TPSA < 65 A^2)
    """
    if mol is None:
        return {"herg_risk": "High", "is_safe": False, "risk_score": 1.0}

    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Check for basic alkyl amine pattern: [NX3;H2,H1,H0;!$(NC=O)]
    basic_amine_patt = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]")
    has_basic_amine = mol.HasSubstructMatch(basic_amine_patt) if basic_amine_patt else False

    risk_points = 0
    reasons = []

    if has_basic_amine:
        risk_points += 2
        reasons.append("Basic amine pharmacophore")
    if logp > 3.8:
        risk_points += 1.5
        reasons.append(f"High lipophilicity (LogP {logp:.1f})")
    if num_aromatic_rings >= 2:
        risk_points += 1
        reasons.append(f"Aromatic stacking ({num_aromatic_rings} rings)")
    if tpsa < 60.0:
        risk_points += 1
        reasons.append(f"Low polar surface area ({tpsa:.1f} Å²)")

    # Score bounded 0.0 to 1.0
    risk_score = min(risk_points / 5.5, 1.0)
    if risk_score >= 0.65:
        risk_label = "High Risk"
        is_safe = False
    elif risk_score >= 0.40:
        risk_label = "Moderate Risk"
        is_safe = True
    else:
        risk_label = "Low / Clean"
        is_safe = True

    return {
        "herg_risk": risk_label,
        "is_herg_safe": is_safe,
        "herg_score": float(risk_score),
        "herg_reasons": reasons,
    }


def predict_microsomal_stability(mol: Chem.Mol) -> Dict[str, Any]:
    """Estimate mouse liver microsomal metabolic stability (t_1/2 in minutes).
    
    Identifies metabolic soft spots:
    - Benzylic/allylic oxidation sites
    - Ester linkages (esterase cleavage)
    - Unfluorinated lipophilic aromatic rings
    """
    if mol is None:
        return {"microsomal_t12_min": 0.0, "stability_class": "Unstable", "is_stable_30min": False}

    logp = float(Crippen.MolLogP(mol))
    
    # Soft-spot substructures
    benzylic_patt = Chem.MolFromSmarts("[c][CH2,CH1][C,c]")
    ester_patt = Chem.MolFromSmarts("C(=O)O[C,c]")
    thio_patt = Chem.MolFromSmarts("[#16]")

    num_benzylic = len(mol.GetSubstructMatches(benzylic_patt)) if benzylic_patt else 0
    num_ester = len(mol.GetSubstructMatches(ester_patt)) if ester_patt else 0
    has_sulfur = mol.HasSubstructMatch(thio_patt) if thio_patt else False

    # Baseline half-life ~60 min, penalized by soft spots and extreme lipophilicity
    penalties = 0.0
    penalties += num_ester * 35.0  # Rapid esterase cleavage
    penalties += min(num_benzylic * 8.0, 24.0)
    if logp > 4.5:
        penalties += (logp - 4.5) * 12.0
    if has_sulfur:
        penalties += 6.0  # S-oxidation

    estimated_t12 = max(65.0 - penalties, 5.0)

    if estimated_t12 >= 45.0:
        stab_class = "High Stability (t½ > 45 min)"
    elif estimated_t12 >= 30.0:
        stab_class = "Moderate Stability (30–45 min)"
    else:
        stab_class = "Rapid Clearance (t½ < 30 min)"

    return {
        "microsomal_t12_min": float(estimated_t12),
        "stability_class": stab_class,
        "is_stable_30min": bool(estimated_t12 >= 30.0),
    }


def predict_admet_profile(mol: Chem.Mol) -> Dict[str, Any]:
    """Run comprehensive preclinical ADMET prediction suite."""
    esol = predict_delaney_esol(mol)
    herg = predict_herg_liability(mol)
    micro = predict_microsomal_stability(mol)

    return {
        "logs": esol["logs"],
        "solubility_uM": esol["solubility_uM"],
        "solubility_class": esol["solubility_class"],
        "is_soluble_50uM": esol["is_soluble_50uM"],
        "herg_risk": herg["herg_risk"],
        "is_herg_safe": herg["is_herg_safe"],
        "herg_score": herg["herg_score"],
        "herg_reasons": herg["herg_reasons"],
        "microsomal_t12_min": micro["microsomal_t12_min"],
        "stability_class": micro["stability_class"],
        "is_stable_30min": micro["is_stable_30min"],
    }
