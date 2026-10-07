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
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return {
        "mol_id": mol_id,
        "smiles_can": smi,
        "selfies": sf.encoder(smi),
        "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
        "qed": float(QED.qed(mol)),
        "sa": float(sa_score(mol, allow_fallback=allow_sa_fallback)),
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rot_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "formal_charge": int(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
        "source": source,
    }


def pic50_from_nm(value_nm: float) -> float:
    if value_nm <= 0:
        raise ValueError("IC50 must be >0")
    return 9.0 - math.log10(value_nm)


def scaffold_split(df: pd.DataFrame, train_frac=0.75, val_frac=0.125):
    groups = {s: list(idx) for s, idx in df.groupby("scaffold").groups.items()}
    ordered = sorted(groups.values(), key=len, reverse=True)
    n = len(df)
    n_train = max(1, int(train_frac * n))
    n_val = max(1, int((train_frac + val_frac) * n))
    train, val, test = [], [], []
    for g in ordered:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(train) + len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            for idx in g:
                if len(train) < n_train:
                    train.append(idx)
                elif len(train) + len(val) < n_val:
                    val.append(idx)
                else:
                    test.append(idx)
    if n >= 3 and not val and len(train) > 1:
        val.append(train.pop())
    if n >= 3 and not test and len(train) > 1:
        test.append(train.pop())
    return df.loc[train].copy(), df.loc[val].copy(), df.loc[test].copy()
