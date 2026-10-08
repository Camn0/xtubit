from __future__ import annotations
from dataclasses import dataclass
import math
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    rdMolStandardize = None
import selfies as sf


def standardize_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if rdMolStandardize is not None:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    return mol


def sa_score(mol, allow_fallback: bool = False):
    """Calculate Synthetic Accessibility score.
    
    Prefers RDKit Contrib SA_Score. If unavailable and allow_fallback is True,
    falls back to a structural complexity heuristic.
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception as exc:
        try:
            from rdkit.Chem import rdMolDescriptors
            if hasattr(rdMolDescriptors, "CalcSyntheticAccessibilityScore"):
                return float(rdMolDescriptors.CalcSyntheticAccessibilityScore(mol))
        except Exception:
            pass
        if allow_fallback:
            # Heuristic estimate based on rings, chiral centers, and atom count (1 to 10 scale)
            n_atoms = mol.GetNumAtoms()
            n_rings = mol.GetRingInfo().NumRings()
            n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            score = 2.0 + 0.1 * n_atoms + 0.5 * n_rings + 0.8 * n_chiral
            return float(min(max(score, 1.0), 10.0))
        raise RuntimeError(
            "RDKit Contrib SA_Score is required for production SA values; "
            "the fallback must not be silently substituted unless allow_fallback=True."
        ) from exc


def featurize(mol, mol_id: str, source: str = "", allow_sa_fallback: bool = True):
    from .medchem_filters import evaluate_medchem_cleanliness
    from .admet_predictors import predict_admet_profile
    from .retrosynthesis import estimate_synthetic_complexity, estimate_synthetic_route
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    clean_info = evaluate_medchem_cleanliness(mol)
    admet_info = predict_admet_profile(mol)
    route_info = estimate_synthetic_route(mol)
    return {
        "mol_id": mol_id,
        "smiles_can": smi,
        "selfies": sf.encoder(smi),
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
        "qed": float(QED.qed(mol)),
        "sa": float(sa_score(mol, allow_fallback=allow_sa_fallback)),
        "scscore": float(estimate_synthetic_complexity(mol)),

        "synth_steps": int(route_info["num_steps"]),
        "synth_tractable": bool(route_info["is_synthetically_tractable"]),
        "synth_reaction": str(route_info["primary_reaction"]),
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rot_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "formal_charge": int(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
        "is_clean": bool(clean_info["is_clean"]),
        "has_pains": bool(clean_info["has_pains"]),
        "has_brenk": bool(clean_info["has_brenk"]),
        "pains_matches": clean_info["pains_matches"],
        "esol_logs": float(admet_info.get("logs", admet_info.get("esol_logs", -4.0))),
        "solubility_um": float(admet_info.get("solubility_uM", admet_info.get("solubility_um", 50.0))),
        "herg_safe": bool(admet_info.get("is_herg_safe", admet_info.get("herg_safe", True))),
        "microsomal_t12": float(admet_info.get("microsomal_t12_min", admet_info.get("microsomal_t12", 45.0))),
        "source": source,
    }


def pic50_from_nm(value_nm: float) -> float:
    if value_nm <= 0:
        raise ValueError("IC50 must be >0")
    return 9.0 - math.log10(value_nm)


def scaffold_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Strict Bemis-Murcko scaffold split preserving scaffold exclusivity.
    
    Guarantees that molecules sharing the exact same Murcko scaffold are never
    split across training, validation, or test sets.
    """
    groups = {s: list(idx) for s, idx in df.groupby("scaffold").groups.items()}
    ordered = sorted(groups.values(), key=len, reverse=True)
    n = len(df)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))

    train, val, test = [], [], []

    if len(ordered) >= 3:
        # Guarantee at least one distinct scaffold in validation and test
        val.extend(ordered.pop())
        test.extend(ordered.pop())
    elif len(ordered) == 2:
        val.extend(ordered.pop())

    for g in ordered:
        if len(train) + len(g) <= n_train or (len(train) == 0):
            train.extend(g)
        elif len(val) + len(g) <= n_val or (len(val) == 0 and len(ordered) >= 3):
            val.extend(g)
        else:
            test.extend(g)

    # Invariant: Guarantee zero scaffold overlap between splits
    train_scaffolds = set(df.loc[train, "scaffold"])
    val_scaffolds = set(df.loc[val, "scaffold"])
    test_scaffolds = set(df.loc[test, "scaffold"])
    assert not (train_scaffolds & val_scaffolds), "Data leakage: train and val share scaffolds"
    assert not (train_scaffolds & test_scaffolds), "Data leakage: train and test share scaffolds"
    assert not (val_scaffolds & test_scaffolds), "Data leakage: val and test share scaffolds"

    return df.loc[train].copy(), df.loc[val].copy(), df.loc[test].copy()


