from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _set_xyz(mol, xyz):
    m = Chem.Mol(mol)
    if m.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    c = m.GetConformer()
    for i, (x, y, z) in enumerate(np.asarray(xyz, dtype=float)):
        c.SetAtomPosition(i, (float(x), float(y), float(z)))
    return m


def uff_energy(mol, xyz):
    m = _set_xyz(mol, xyz)
    if not AllChem.UFFHasAllMoleculeParams(m):
        return np.inf
    return float(AllChem.UFFGetMoleculeForceField(m).CalcEnergy())


def interaction_without_bond(frag_i, xyz_i, frag_j, xyz_j):
    combo = Chem.CombineMols(_set_xyz(frag_i, xyz_i), _set_xyz(frag_j, xyz_j))
    xyz = np.vstack([xyz_i, xyz_j])
    eij = uff_energy(combo, xyz)
    return eij - uff_energy(frag_i, xyz_i) - uff_energy(frag_j, xyz_j)


def interaction_with_bond(frag_i, xyz_i, atom_i, frag_j, xyz_j, atom_j):
    # Caller is responsible for using attachment atoms with valence available.
    combo = Chem.CombineMols(_set_xyz(frag_i, xyz_i), _set_xyz(frag_j, xyz_j))
    rw = Chem.RWMol(combo)
    offset = frag_i.GetNumAtoms()
    rw.AddBond(int(atom_i), offset + int(atom_j), Chem.BondType.SINGLE)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    xyz = np.vstack([xyz_i, xyz_j])
    return uff_energy(m, xyz) - uff_energy(frag_i, xyz_i) - uff_energy(frag_j, xyz_j)


def classify_pair(e_nb, e_b, chemically_bonded, threshold=500.0):
    conn = -1.0 if chemically_bonded and e_b is not None and e_b <= threshold else 0.0
    clash = 1.0 if conn == 0.0 and e_nb > threshold else 0.0
    return clash, conn
