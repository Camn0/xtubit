from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import itertools
from collections import defaultdict
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem import rdMolTransforms as T
from .b7_qubo import build_yanagisawa_qubo, validate_qubo, QuboBundle
from .common.math import qubo_to_ising



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


# ==============================================================================
# Pks13 Pocket Discretization & 60-90 Qubit Hamiltonian Scaling (Task 2.1)
# ==============================================================================

# 6 Pharmacophore Sub-Sites extracted from authentic PDB 5V3Y crystallographic structure
PKS13_SUBPOCKETS: Dict[str, Dict[str, Any]] = {
    "Anchor": {
        "index": 0,
        "name": "Anchor Sub-Pocket",
        "description": "Benzofuran core aromatic sandwich pocket (Phe1585 / Tyr1674 in PDB 5V3Y)",
        "center": np.array([4.8, 22.8, 7.3]),
        "radius": 3.5,
        "base_dG": -8.5,
    },
    "Linker": {
        "index": 1,
        "name": "Linker Sub-Pocket",
        "description": "Amide & methylene bridge channel (Gly1534 / Ala1535 in PDB 5V3Y)",
        "center": np.array([5.7, 26.0, 6.2]),
        "radius": 2.8,
        "base_dG": -4.5,
    },
    "Tunnel": {
        "index": 2,
        "name": "Hydrophobic Tunnel",
        "description": "Lipophilic channel leading to active site entrance (Leu1638 / Ile1643 in PDB 5V3Y)",
        "center": np.array([4.2, 24.5, 7.6]),
        "radius": 3.2,
        "base_dG": -4.0,
    },
    "P1_Cap": {
        "index": 3,
        "name": "P1 Cap Sub-Pocket",
        "description": "Lipophilic cap cavity (Val1536 / Met1589 in PDB 5V3Y)",
        "center": np.array([2.2, 27.5, 8.4]),
        "radius": 3.0,
        "base_dG": -3.5,
    },
    "Catalytic_Triad": {
        "index": 4,
        "name": "Catalytic Triad Cleft",
        "description": "Catalytic machinery pocket (Ser1636 nucleophile & Asp1644 acid in PDB 5V3Y)",
        "center": np.array([3.5, 27.0, 6.0]),
        "radius": 2.5,
        "base_dG": -5.2,
    },
    "Solvent_Front": {
        "index": 5,
        "name": "Solvent Front Sub-Pocket",
        "description": "Solvent-accessible channel rim (Arg1641 / Ala1667 in PDB 5V3Y)",
        "center": np.array([5.8, 30.5, 8.0]),
        "radius": 3.8,
        "base_dG": -3.0,
    },
}

# Connectivity graph between adjacent sub-pockets in authentic 5V3Y complex:
# Anchor (0) <-> Tunnel (2) <-> Linker (1) <-> Catalytic_Triad (4)
# Linker (1) <-> Solvent_Front (5)
# Tunnel (2) <-> P1_Cap (3)
PKS13_ADJACENT_SUBPOCKETS = {
    (0, 1), (0, 2), (1, 2), (2, 3), (1, 4), (0, 4), (1, 5)
}



def generate_scaled_pocket_placements(
    poses_per_subpocket: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    """Generate 3D fragment placements across 6 Pks13 sub-pocket sites.
    
    Produces 60 qubits (if poses_per_subpocket=10) or 90 qubits (if poses_per_subpocket=15).
    Uses vector-accelerated PyTorch tensors for inter-fragment distances,
    steric clash (B_ij), and chemical connectivity (C_ij).
    """
    rng = np.random.RandomState(seed)
    subpocket_names = list(PKS13_SUBPOCKETS.keys())
    n_subpockets = len(subpocket_names)
    n_vars = n_subpockets * poses_per_subpocket

    fragment_id = torch.tensor(
        [i // poses_per_subpocket for i in range(n_vars)],
        dtype=torch.long
    )

    # 1. 3D Coordinates generation around sub-pocket centers
    coords = []
    variable_meta = []
    dG_list = []

    for p_idx, p_name in enumerate(subpocket_names):
        info = PKS13_SUBPOCKETS[p_name]
        center = info["center"]
        base_dG = info["base_dG"]

        for pose_idx in range(poses_per_subpocket):
            var_idx = p_idx * poses_per_subpocket + pose_idx
            # Jitter within realistic sub-pocket envelope
            jitter = rng.normal(0.0, 0.45, size=3)
            pos = center + jitter
            coords.append(pos)

            # Local binding affinity dG (perturbed around sub-pocket base affinity)
            dG_val = base_dG + 0.18 * pose_idx + rng.normal(0.0, 0.1)
            dG_list.append(dG_val)

            variable_meta.append({
                "var_index": var_idx,
                "subpocket_id": p_idx,
                "subpocket_name": p_name,
                "pose_id": pose_idx,
                "coord": pos.tolist(),
                "dG": float(dG_val),
            })

    coords_t = torch.tensor(np.array(coords), dtype=torch.float64)  # (N, 3)
    dG_t = torch.tensor(dG_list, dtype=torch.float64)

    # 2. Vector-accelerated PyTorch pairwise distance matrix
    dist_mat = torch.cdist(coords_t, coords_t)  # (N, N)

    # 3. Inter-fragment steric clash matrix (B_ij)
    # Atoms in different sub-pockets closer than 2.0 A clash sterically
    diff_subpocket = fragment_id[:, None].ne(fragment_id[None, :])
    clash_mask = diff_subpocket & (dist_mat < 2.0)
    clash = torch.where(clash_mask, 1.0, 0.0).to(torch.float64)
    clash.fill_diagonal_(0.0)

    # 4. Connectivity matrix (C_ij)
    # Adjacent sub-pockets with distance in [1.2, 2.3] A allow covalent bonding
    conn = torch.zeros((n_vars, n_vars), dtype=torch.float64)
    for i in range(n_vars):
        f_i = int(fragment_id[i])
        for j in range(i + 1, n_vars):
            f_j = int(fragment_id[j])
            if (f_i, f_j) in PKS13_ADJACENT_SUBPOCKETS or (f_j, f_i) in PKS13_ADJACENT_SUBPOCKETS:
                d = float(dist_mat[i, j])
                if 1.2 <= d <= 2.3:
                    conn[i, j] = conn[j, i] = -1.0

    return {
        "n_vars": n_vars,
        "n_subpockets": n_subpockets,
        "poses_per_subpocket": poses_per_subpocket,
        "fragment_id": fragment_id,
        "dG": dG_t,
        "clash": clash,
        "conn": conn,
        "coords": coords_t,
        "variable_meta": variable_meta,
    }


def build_scaled_pks13_qubo(
    poses_per_subpocket: int = 15,
    A: float = 1.0,
    B: float = 5.0,
    C: float = 5.0,
    D: float = 25.0,
    seed: int = 42
) -> Dict[str, Any]:
    """Assemble complete 60–90 Qubit Yanagisawa QUBO and Ising matrices.
    
    Verifies:
    1. Symmetry of Q matrix
    2. Zeroed diagonal on interaction terms
    3. Machine-precision equivalence check with Ising spin Hamiltonian
    """
    sys = generate_scaled_pocket_placements(poses_per_subpocket=poses_per_subpocket, seed=seed)
    dG = sys["dG"]
    clash = sys["clash"]
    conn = sys["conn"]
    fragment_id = sys["fragment_id"]

    bundle = build_yanagisawa_qubo(
        dG, clash, conn, fragment_id,
        A=A, B=B, C=C, D=D
    )
    val = validate_qubo(bundle, equivalence_trials=200, atol=1e-8)
    if not val["passed"]:
        raise AssertionError(f"QUBO->Ising equivalence validation failed: max error={val['max_equivalence_error']}")

    J, h, c0 = qubo_to_ising(bundle.Q)

    return {
        "bundle": bundle,
        "Q": bundle.Q,
        "J": J,
        "h": h,
        "c0": c0,
        "dG": dG,
        "clash": clash,
        "conn": conn,
        "fragment_id": fragment_id,
        "coords": sys["coords"],
        "variable_meta": sys["variable_meta"],
        "n_vars": sys["n_vars"],
        "validation": val,
    }


# ==============================================================================
# Dynamic Candidate- & Receptor-Dependent QUBO Construction (Task 2.3 & 3.2)
# ==============================================================================

def load_receptor_pocket_atoms(
    receptor: str = "5V3Y",
    center: Optional[np.ndarray] = None,
    radius: float = 12.0,
    return_elements: bool = False
) -> Any:
    """Extract heavy atom coordinates (and elements) of receptor pocket from authentic PDB structure."""
    rec_upper = receptor.upper()
    if "8TQV" in rec_upper:
        pdb_code = "8TQV"
        default_center = np.array([-4.07, -15.05, 13.11])
    elif "5V40" in rec_upper:
        pdb_code = "5V40"
        default_center = np.array([4.50, 26.76, 7.36])
    elif "8TQG" in rec_upper:
        pdb_code = "8TQG"
        default_center = np.array([-22.02, 9.04, 8.64])
    elif "5V3Y" in rec_upper:
        pdb_code = "5V3Y"
        default_center = np.array([4.80, 26.29, 7.48])
    else:
        pdb_file = Path(f"data/raw/{rec_upper}.pdb")
        if pdb_file.exists():
            pdb_code = rec_upper
            default_center = np.array([0.0, 0.0, 0.0])
        else:
            raise FileNotFoundError(
                f"Receptor coordinate file for '{receptor}' was not found. "
                f"Authentic structure file data/raw/{rec_upper}.pdb is required (no silent aliasing)."
            )

    pdb_file = Path(f"data/raw/{pdb_code}.pdb")
    if not pdb_file.exists():
        raise FileNotFoundError(f"Receptor coordinate file {pdb_file} does not exist. Authentic coordinates required.")

    if center is None:
        center = default_center

    coords = []
    elements = []
    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM  "):
                elem = line[76:78].strip() or (line[12:16].strip()[0] if line[12:16].strip() else "")
                if elem == "H":
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    pt = np.array([x, y, z])
                    if np.linalg.norm(pt - center) <= radius:
                        coords.append(pt)
                        elements.append(elem.upper() if elem else "C")
                except ValueError:
                    continue
    if not coords:
        coords_arr = np.array([center])
        elems_arr = ["C"]
    else:
        coords_arr = np.array(coords)
        elems_arr = elements

    if return_elements:
        return coords_arr, elems_arr
    return coords_arr


def get_receptor_subpocket_centers(
    receptor: str = "5V3Y",
    n_subpockets: int = 4
) -> Dict[str, np.ndarray]:
    """Derive authentic sub-pocket cavity centers directly inside receptor coordinate frame.

    Retrospective structure-guided docking centers derived from crystallographic reference complexes.
    """
    rec_upper = receptor.upper()
    if "8TQV" in rec_upper:
        # PDB 8TQV co-crystal JS9 authentic tree fragment centroids
        centers = [
            np.array([-5.26, -15.17, 17.84]),
            np.array([-1.64, -16.32, 10.80]),
            np.array([-2.39, -16.40, 4.69]),
            np.array([-7.58, -11.88, 20.06]),
        ]
    elif "5V40" in rec_upper:
        # PDB 5V40 co-crystal JS1 binding cavity
        centers = [
            np.array([5.49, 23.12, 6.55]),
            np.array([5.15, 26.21, 6.61]),
            np.array([3.00, 27.90, 8.02]),
            np.array([4.19, 30.43, 8.38]),
        ]
    elif "8TQG" in rec_upper:
        # PDB 8TQG co-crystal JR0 binding cavity
        centers = [
            np.array([-25.25, 11.69, 1.74]),
            np.array([-25.76, 9.77, 8.36]),
            np.array([-19.84, 8.12, 12.06]),
            np.array([-16.60, 6.28, 13.04]),
        ]
    else:
        # PDB 5V3Y co-crystal 5V8 authentic tree fragment centroids
        centers = [
            np.array([2.84, 26.80, 8.12]),
            np.array([5.76, 21.61, 7.16]),
            np.array([6.25, 26.11, 5.50]),
            np.array([5.80, 30.32, 8.03]),
        ]

    subpocket_names = ["Anchor", "Linker", "Tunnel", "P1_Cap", "Catalytic_Triad", "Solvent_Front"]
    return {
        subpocket_names[i]: centers[i % len(centers)]
        for i in range(n_subpockets)
    }


VDW_RADII: Dict[str, float] = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "F": 1.47,
    "CL": 1.75, "BR": 1.85, "I": 1.98, "P": 1.80, "H": 1.20
}


def compute_atom_typed_docking_score(
    frag_coords: np.ndarray,
    pocket_atoms: np.ndarray,
    frag_elements: Optional[List[str]] = None,
    pocket_elements: Optional[List[str]] = None,
    clash_cutoff: float = 2.2,
    opt_distance: float = 3.8,
    attr_weight: float = -0.35,
    rep_weight: float = 12.0,
    soft_clip: bool = True,
) -> float:
    """Calculate atom-typed physical docking score against receptor pocket heavy atoms.
    
    Includes:
    - Atom-typed steric clash penalty (quadratic when d < 0.72 * (R_i + R_j))
    - van der Waals attractive dispersion well (centered at R_i + R_j)
    - Distance-based polar N/O contact bonus (N/O pairs between 2.4 and 3.5 A)
    """
    if len(frag_coords) == 0 or len(pocket_atoms) == 0:
        return 0.0

    diff = frag_coords[:, None, :] - pocket_atoms[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # (M_frag, N_pocket)

    if frag_elements is not None and pocket_elements is not None and len(frag_elements) == len(frag_coords):
        r_frag = np.array([VDW_RADII.get(e.upper(), 1.70) for e in frag_elements])
        r_pock = np.array([VDW_RADII.get(e.upper(), 1.70) for e in pocket_elements])
        r_sum = r_frag[:, None] + r_pock[None, :]

        # 1. Steric clashes
        clash_pen = float(np.sum(np.maximum(0.0, 0.72 * r_sum - dists) ** 2) * rep_weight)

        # 2. van der Waals attractive dispersion
        vdw_mask = (dists >= 0.72 * r_sum) & (dists <= 5.5)
        attr_well = float(attr_weight * np.sum(np.exp(-0.5 * ((dists[vdw_mask] - r_sum[vdw_mask]) / 0.6) ** 2)))

        # 3. Distance-based polar N/O contact bonus (N/O pairs between 2.4 and 3.5 A)
        frag_is_polar = np.array([e.upper() in ["N", "O"] for e in frag_elements])
        pock_is_polar = np.array([e.upper() in ["N", "O"] for e in pocket_elements])
        polar_pairs = frag_is_polar[:, None] & pock_is_polar[None, :]
        hb_mask = polar_pairs & (dists >= 2.4) & (dists <= 3.5)
        polar_contact_bonus = float(-1.8 * np.sum(np.exp(-0.5 * ((dists[hb_mask] - 2.85) / 0.35) ** 2))) if np.any(hb_mask) else 0.0

        total_e = clash_pen + attr_well + polar_contact_bonus
    else:
        clash_pen = float(np.sum(np.maximum(0.0, clash_cutoff - dists) * rep_weight))
        contact_mask = (dists >= clash_cutoff) & (dists <= 5.0)
        attr_well = float(attr_weight * np.sum(np.exp(-0.5 * ((dists[contact_mask] - opt_distance) / 0.6) ** 2)))
        polar_contact_bonus = 0.0
        total_e = clash_pen + attr_well

    if soft_clip:
        # Strictly additive over atoms: attr well + polar bonus + capped clash penalty
        return float(attr_well + polar_contact_bonus + min(clash_pen, 50.0))
    return float(total_e)



def compute_protein_fragment_contact_potential(
    frag_coords: np.ndarray,
    pocket_atoms: np.ndarray,
    clash_cutoff: float = 2.2,
    opt_distance: float = 3.8,
    attr_weight: float = -0.35,
    rep_weight: float = 8.0,
    frag_elements: Optional[List[str]] = None,
    pocket_elements: Optional[List[str]] = None,
) -> float:
    """Calculate physical contact energy and steric penalties against receptor pocket atoms."""
    return compute_atom_typed_docking_score(
        frag_coords=frag_coords,
        pocket_atoms=pocket_atoms,
        frag_elements=frag_elements,
        pocket_elements=pocket_elements,
        clash_cutoff=clash_cutoff,
        opt_distance=opt_distance,
        attr_weight=attr_weight,
        rep_weight=rep_weight,
    )


def decompose_k_fragments(
    mol: Chem.Mol,
    K: int = 4,
    min_size: int = 3
) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
    """Decompose candidate molecule into K connected fragments forming a connected tree graph.
    
    Identifies rotatable acyclic single bonds between degree > 1 atoms (preferring BRICS cuts).
    Searches combinations of K - 1 cuts ensuring each fragment has >= min_size heavy atoms
    and balanced fragment sizes.
    Returns:
    - atom_groups: list of heavy atom index lists for each of the K fragments.
    - attachments: list of (u, owner[u], v, owner[v]) across cut bonds.
    """
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy <= K:
        return [[i] for i in range(n_heavy)], []

    brics = {mol.GetBondBetweenAtoms(u, v).GetIdx() for (u, v), _ in BRICS.FindBRICSBonds(mol) if mol.GetBondBetweenAtoms(u, v) is not None}
    cand = [b.GetIdx() for b in mol.GetBonds()
            if not b.IsInRing() and b.GetBondType() == Chem.BondType.SINGLE
            and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1]

    best = None
    for cut in itertools.combinations(cand, K - 1):
        frags = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, list(cut), addDummies=False))
        sizes = [len(f) for f in frags]
        if len(frags) != K or min(sizes) < min_size:
            continue
        cost = float(np.var(sizes)) - 3.0 * len(brics & set(cut))   # balanced + BRICS-preferred
        if best is None or cost < best[0]:
            best = (cost, [list(f) for f in frags], cut)

    if best is None:
        # Fallback if strict min_size cannot be satisfied (e.g. small molecule or limited rotatable bonds)
        for cut in itertools.combinations(cand, K - 1):
            frags = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, list(cut), addDummies=False))
            sizes = [len(f) for f in frags]
            if len(frags) != K or min(sizes) < 1:
                continue
            cost = float(np.var(sizes)) - 3.0 * len(brics & set(cut))
            if best is None or cost < best[0]:
                best = (cost, [list(f) for f in frags], cut)

    if best is None:
        raise ValueError(f"no {K}-fragment tree decomposition for candidate; lower K or inspect molecule")

    _, groups, cut = best
    owner = {a: k for k, g in enumerate(groups) for a in g}
    att = []
    for bi in cut:
        bd = mol.GetBondWithIdx(bi)
        u, v = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        att.append((u, owner[u], v, owner[v]))

    return groups, att


def decompose_candidate_to_fragments(
    mol: Chem.Mol,
    n_subpockets: int = 4
) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
    """Backward-compatible wrapper decomposing candidate molecule into tree fragments.
    Attempts strict K = n_subpockets decomposition first (min_size=3, then min_size=1).
    If the molecule does not have enough acyclic rotatable bonds for n_subpockets,
    adapts K down to the maximum achievable number of tree fragments.
    """
    for k in range(n_subpockets, 1, -1):
        for ms in [3, 1]:
            try:
                return decompose_k_fragments(mol, K=k, min_size=ms)
            except ValueError:
                continue
    return [[i for i in range(mol.GetNumHeavyAtoms())]], []


def orient_tree(attachments: List[Tuple[int, int, int, int]], root: int) -> List[Tuple[int, int, int]]:
    """Orient tree attachments from root outward. Returns list of (atom_in_parent, atom_in_child, child_frag)."""
    nbr = defaultdict(list)
    for u, fu, v, fv in attachments:
        nbr[fu].append((fv, u, v))
        nbr[fv].append((fu, v, u))
    order, seen, edges = [root], {root}, []
    for f in order:
        for g, a, b in nbr[f]:
            if g not in seen:
                seen.add(g)
                order.append(g)
                edges.append((a, b, g))
    return edges


def side_atoms(mol_h: Chem.Mol, u: int, v: int) -> np.ndarray:
    """Find all atom indices on the child side of rotatable bond u-v (including hydrogens)."""
    seen, stack = {v}, [v]
    while stack:
        a = stack.pop()
        for nb in mol_h.GetAtomWithIdx(a).GetNeighbors():
            n = nb.GetIdx()
            if n in seen or (a == v and n == u):
                continue
            seen.add(n)
            stack.append(n)
    return np.array(sorted(seen))


def rotate_about_bond(X: np.ndarray, a: int, b: int, side: np.ndarray, deg: float) -> np.ndarray:
    """Rotate coordinates of side atoms about bond vector a -> b by deg degrees."""
    ax = X[b] - X[a]
    nrm = np.linalg.norm(ax)
    if nrm < 1e-6:
        return X
    ax = ax / nrm
    R = Rot.from_rotvec(np.radians(deg) * ax).as_matrix()
    Y = X.copy()
    Y[side] = (X[side] - X[b]) @ R.T + X[b]
    return Y


def intramolecular_clash(
    P: np.ndarray,
    gd: np.ndarray,
    rvdw: np.ndarray,
    scale: float = 0.8,
    min_graph: int = 4
) -> bool:
    """Check for intramolecular steric clashes among heavy atoms separated by min_graph bonds."""
    D = np.linalg.norm(P[:, None] - P[None, :], axis=-1)
    thr = scale * (rvdw[:, None] + rvdw[None, :])
    return bool(((D < thr) & (gd >= min_graph)).any())


def dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Calculate dihedral angle in degrees between four points or batches of points."""
    b0 = p0 - p1
    b1 = p2 - p1
    nrm1 = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1 = b1 / (nrm1 + 1e-9)
    b2 = p3 - p2
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.degrees(np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))


def torsion_profile(
    mol_h: Chem.Mol,
    conf_id: int,
    a2: int,
    a: int,
    b: int,
    b2: int,
    step: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Relaxed MMFF scan of cut-bond dihedral (a2-a-b-b2)."""
    m = Chem.Mol(mol_h)
    props = AllChem.MMFFGetMoleculeProperties(m)
    if props is None or not props.IsValid():
        phis = np.linspace(-180, 180, 360 // step, endpoint=False)
        return phis, np.zeros_like(phis)
    conf = m.GetConformer(conf_id)
    phis, Es = [], []
    for th in range(-180, 180, step):
        try:
            T.SetDihedralDeg(conf, a2, a, b, b2, float(th))
            ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=conf_id)
            if ff is not None:
                ff.MMFFAddTorsionConstraint(a2, a, b, b2, False, max(th - 1.0, -180.0), min(th + 1.0, 180.0), 500.0)
                ff.Initialize()
                ff.Minimize(maxIts=50)
                P = m.GetConformer(conf_id).GetPositions()
                Es.append(float(ff.CalcEnergy()))
                phis.append(float(dihedral(P[a2], P[a], P[b], P[b2])))
            else:
                phis.append(float(th))
                Es.append(0.0)
        except Exception:
            phis.append(float(th))
            Es.append(0.0)
    Es_arr = np.array(Es)
    phis_arr = np.array(phis)
    order = np.argsort(phis_arr)
    return phis_arr[order], (Es_arr - Es_arr.min())[order]


def torsion_term(
    PA: List[Placement],
    PB: List[Placement],
    jA: Tuple[int, Optional[int]],
    jB: Tuple[int, Optional[int]],
    prof: Optional[Tuple[np.ndarray, np.ndarray]],
    kcal_scale: float = 5.0
) -> np.ndarray:
    """Calculate pair torsion energy penalty between fragment placements."""
    if prof is None or jA[1] is None or jB[1] is None:
        return np.zeros((len(PA), len(PB)), dtype=float)
    a, a2 = jA
    b, b2 = jB
    st_A = lambda P, i: np.stack([p.xyz[i] for p in P])
    phi = dihedral(st_A(PA, a2)[:, None], st_A(PA, a)[:, None],
                   st_A(PB, b)[None, :], st_A(PB, b2)[None, :])
    return np.interp(phi, prof[0], prof[1], period=360) / kcal_scale


def buried_grid(
    pocket_xyz: np.ndarray,
    box_center: Optional[np.ndarray] = None,
    half: float = 10.0,
    step: float = 1.0,
    min_dist: float = 3.2,
    shell: float = 8.0,
    min_nb: int = 40
) -> np.ndarray:
    """Identify ligand-accessible, buried cavity grid points inside the binding pocket."""
    from scipy.spatial import cKDTree
    if box_center is None:
        box_center = pocket_xyz.mean(0)
    ax = np.arange(-half, half + step, step)
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + box_center
    tree = cKDTree(pocket_xyz)
    d, _ = tree.query(G)
    nb = np.array([len(x) for x in tree.query_ball_point(G, shell)])
    pts = G[(d > min_dist) & (nb >= min_nb)]
    if len(pts) == 0:
        return np.array([box_center])
    return pts


def partition_molecule_to_subpockets(
    mol: Chem.Mol,
    n_subpockets: int = 4
) -> List[List[int]]:
    """Partition heavy atoms of a candidate molecule into connected sub-pocket groups via BRICS / rotatable bonds."""
    atom_groups, _ = decompose_candidate_to_fragments(mol, n_subpockets=n_subpockets)
    return atom_groups



def parse_smiles_strict(smi: str) -> Chem.Mol:
    """Parse candidate SMILES strictly without silent fallbacks."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {smi!r}")
    return mol


@dataclass
class Placement:
    frag: int
    parent: int
    xyz: np.ndarray
    score: float = 0.0


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal rotation R and translation t minimizing ||R P + t - Q||."""
    Pc, Qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def whole_molecule_poses(
    conf_xyz: np.ndarray,
    groups: List[List[int]],
    centers: np.ndarray,
    n_jitter: int = 40,
    sig_rot: float = 10.0,
    sig_t: float = 0.7,
    seed: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Assign fragments to subpockets and generate conformer-coherent rigid poses with local jitter."""
    rng = np.random.default_rng(seed)
    K = len(groups)
    cents = np.array([conf_xyz[g].mean(0) for g in groups])
    best = min(
        itertools.permutations(range(K)),
        key=lambda p: np.linalg.norm(
            kabsch(cents, centers[list(p)])[0] @ cents.T
            + kabsch(cents, centers[list(p)])[1][:, None]
            - centers[list(p)].T
        )
    )
    R0, t0 = kabsch(cents, centers[list(best)])
    poses = [(R0, t0)]
    for _ in range(n_jitter):
        dR = Rot.from_rotvec(np.radians(rng.normal(0, sig_rot, 3))).as_matrix()
        dt = rng.normal(0, sig_t, 3)
        poses.append((dR @ R0, t0 + dt))
    return poses


def build_pools(
    conformers: List[np.ndarray],
    groups: List[List[int]],
    centers: np.ndarray,
    score_fn: Any,
    n_keep: int = 30,
    div_rmsd: float = 1.0,
    n_jitter: int = 40,
    seed: int = 0,
) -> List[List[Placement]]:
    """Build conformer-coherent fragment placement pools with spatial diversity."""
    raw: List[List[Placement]] = [[] for _ in groups]
    pid = 0
    for c_idx, X in enumerate(conformers):
        for R, t in whole_molecule_poses(X, groups, centers, n_jitter=n_jitter, seed=seed + c_idx):
            Xp = X @ R.T + t
            for k, g in enumerate(groups):
                raw[k].append(Placement(frag=k, parent=pid, xyz=Xp[g], score=score_fn(Xp[g], k)))
            pid += 1

    pools: List[List[Placement]] = []
    for k in range(len(groups)):
        kept: List[Placement] = []
        for p in sorted(raw[k], key=lambda item: item.score):
            if all(np.sqrt(((p.xyz - q.xyz) ** 2).sum(-1).mean()) > div_rmsd for q in kept):
                kept.append(p)
            if len(kept) == n_keep:
                break
        idx = 0
        while len(kept) < n_keep and idx < len(raw[k]):
            if raw[k][idx] not in kept:
                kept.append(raw[k][idx])
            idx += 1
        pools.append(kept)
    return pools


def ang(p: np.ndarray, c: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Calculate bond angle in degrees between vectors (p - c) and (q - c)."""
    u, w = p - c, q - c
    norm_u = np.linalg.norm(u, axis=-1, keepdims=True)
    norm_w = np.linalg.norm(w, axis=-1, keepdims=True)
    cos = (u * w).sum(axis=-1, keepdims=True) / (norm_u * norm_w + 1e-9)
    res = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.squeeze(res, axis=-1)


def junction_penalty(
    PA: List[Placement],
    PB: List[Placement],
    jA: Tuple[int, Optional[int]],
    jB: Tuple[int, Optional[int]],
    ref: Tuple[float, float, float],
    cap: float = 12.0
) -> np.ndarray:
    """Evaluate inter-fragment junction geometry penalty (bond length & valence angles).
    
    jA=(a, a2): junction atom index + in-fragment neighbor index (local to fragment A).
    jB=(b, b2): junction atom index + in-fragment neighbor index (local to fragment B).
    ref=(d0, th_a0, th_b0): median reference geometry over conformer ensemble.
    Returns (nA, nB) array of penalties b_ij in [0, 1].
    """
    a = np.stack([p.xyz[jA[0]] for p in PA])[:, None]  # (nA, 1, 3)
    b = np.stack([p.xyz[jB[0]] for p in PB])[None, :]  # (1, nB, 3)
    d = np.linalg.norm(b - a, axis=-1)                 # (nA, nB)
    e = ((d - ref[0]) / 0.08) ** 2

    if jA[1] is not None:
        a2 = np.stack([p.xyz[jA[1]] for p in PA])[:, None]
        e = e + ((ang(a2, a, b) - ref[1]) / 12.0) ** 2
    if jB[1] is not None:
        b2 = np.stack([p.xyz[jB[1]] for p in PB])[None, :]
        e = e + ((ang(a, b, b2) - ref[2]) / 12.0) ** 2

    b_ij = np.minimum(e, cap) / cap
    return b_ij


def clash(
    PA: Placement,
    PB: Placement,
    gA: List[int],
    gB: List[int],
    gd: np.ndarray,
    rvdw: np.ndarray,
    scale: float = 0.8,
    min_graph: int = 4
) -> int:
    """Evaluate steric clash with topological exclusion (excluding 1-2, 1-3, 1-4 bonds)."""
    D = np.linalg.norm(PA.xyz[:, None] - PB.xyz[None, :], axis=-1)
    thr = scale * (rvdw[gA][:, None] + rvdw[gB][None, :])
    topo = gd[np.ix_(gA, gB)] >= min_graph
    return int(((D < thr) & topo).any())


def generate_3d_rotations() -> List[np.ndarray]:
    """Generate discrete 3D rotation matrices sampling SO(3) orientations."""
    rots = []
    for rx in [0.0, np.pi / 2.0]:
        cx, sx = np.cos(rx), np.sin(rx)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        for ry in [0.0, np.pi / 2.0]:
            cy, sy = np.cos(ry), np.sin(ry)
            Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            for rz in [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]:
                cz, sz = np.cos(rz), np.sin(rz)
                Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
                rots.append(Rz @ Ry @ Rx)
    return rots


def build_library(
    mol_h: Chem.Mol,
    conformers_h: List[np.ndarray],
    groups: List[List[int]],
    attachments: List[Tuple[int, int, int, int]],
    centers: np.ndarray,
    score_fn: Any,
    gd: np.ndarray,
    rvdw: np.ndarray,
    pocket_atoms: Optional[np.ndarray] = None,
    search_mode: str = "oracle",
    torsion_deg: float = 60.0,
    n_beam: int = 40,
    div_rmsd: float = 0.5,
    n_jitter: int = 25,
    seed: int = 0,
) -> Tuple[List[List[Placement]], List[List[int]], List[Tuple[float, np.ndarray]]]:
    """Build conformer-coherent fragment placement pools with parent-level beam and tree torsion sweeps."""
    K = len(groups)
    n_heavy = gd.shape[0]
    anchor = max(range(K), key=lambda k: len(groups[k]))
    edges = orient_tree(attachments, anchor)
    sides = {(a, b): side_atoms(mol_h, a, b) for a, b, _ in edges}
    angles = np.arange(0, 360, torsion_deg)
    memo: Dict[Tuple[int, bytes], float] = {}
    cands: List[Tuple[float, np.ndarray]] = []

    def frag_score(Y: np.ndarray, k: int) -> float:
        key = (k, np.round(Y[groups[k]], 3).tobytes())
        if key not in memo:
            memo[key] = score_fn(Y[groups[k]], k)
        return memo[key]

    rng = np.random.default_rng(seed)

    if search_mode == "blind" and pocket_atoms is not None:
        grid_pts = buried_grid(pocket_atoms, box_center=pocket_atoms.mean(0))
        n_pts = min(len(grid_pts), 15)
        chosen_pts = grid_pts[rng.choice(len(grid_pts), size=n_pts, replace=False)] if len(grid_pts) > n_pts else grid_pts
        rots = generate_3d_rotations()
        for c, X in enumerate(conformers_h[:5]):
            X_heavy = X[:n_heavy]
            anc_c = X_heavy[groups[anchor]].mean(0)
            X_cent = X - anc_c
            for pt in chosen_pts:
                for R in rots[:6]:
                    Xp = X_cent @ R.T + pt
                    for combo in itertools.product(angles, repeat=len(edges)):
                        Y = Xp
                        for (a, b, _), th in zip(edges, combo):
                            if th:
                                Y = rotate_about_bond(Y, a, b, sides[(a, b)], th)
                        if intramolecular_clash(Y[:n_heavy], gd, rvdw):
                            continue
                        s = sum(frag_score(Y, k) for k in range(K))
                        cands.append((s, Y))
    else:
        for c, X in enumerate(conformers_h[:8]):
            poses = whole_molecule_poses(X[:n_heavy], groups, centers, n_jitter=min(n_jitter, 10), seed=seed + c)
            for R, t in poses:
                Xp = X @ R.T + t
                for combo in itertools.product(angles, repeat=len(edges)):
                    Y = Xp
                    for (a, b, _), th in zip(edges, combo):
                        if th:
                            Y = rotate_about_bond(Y, a, b, sides[(a, b)], th)
                    if intramolecular_clash(Y[:n_heavy], gd, rvdw):
                        continue
                    s = sum(frag_score(Y, k) for k in range(K))
                    cands.append((s, Y))
                    if len(cands) >= 2500:
                        break
                if len(cands) >= 2500:
                    break
            if len(cands) >= 2500:
                break


    cands.sort(key=lambda item: item[0])
    beam: List[Tuple[float, np.ndarray]] = []
    for s, Y in cands:
        if all(np.sqrt(((Y[:n_heavy] - Z[:n_heavy]) ** 2).sum(-1).mean()) > div_rmsd for _, Z in beam):
            beam.append((s, Y))
        if len(beam) == n_beam:
            break

    if len(beam) == 0 and len(cands) > 0:
        beam = cands[:n_beam]

    pools: List[List[Placement]] = [[] for _ in groups]
    index: List[Dict[bytes, int]] = [{} for _ in groups]
    pose_slices: List[List[int]] = []

    for pid, (s, Y) in enumerate(beam):
        row = []
        for k, g in enumerate(groups):
            key = np.round(Y[g], 3).tobytes()
            if key not in index[k]:
                index[k][key] = len(pools[k])
                pools[k].append(Placement(frag=k, parent=pid, xyz=Y[g].copy(), score=frag_score(Y, k)))
            row.append(index[k][key])
        pose_slices.append(row)

    return pools, pose_slices, beam


def generate_candidate_pocket_placements(
    candidate_smiles: str,
    receptor: str = "5V3Y",
    poses_per_subpocket: int = 3,
    n_subpockets: int = 4,
    search_mode: str = "oracle",
    centers: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate candidate- and receptor-dependent 3D placements via conformer-coherent docking.

    Computes:
    - Candidate-dependent chemical fragments via BRICS retrosynthetic cleavage.
    - Whole-molecule conformer generation and coherent pocket alignment.
    - Torsion-expanded tree placement library with parent-level beam.
    - Sliced fragment placement pools sharing common conformer transforms.
    - Atom-typed physical docking score against receptor cavity.
    - Steric clash matrix with topological exclusion (min graph distance >= 4).
    - Covalent connectivity matrix evaluating exact junction bond distances, angles, and torsions.
    """
    mol = parse_smiles_strict(candidate_smiles)
    mol = Chem.RemoveHs(mol)

    mol_h = Chem.AddHs(mol)
    res_embed = AllChem.EmbedMultipleConfs(mol_h, numConfs=20, randomSeed=seed)
    if len(res_embed) == 0:
        AllChem.EmbedMultipleConfs(mol_h, numConfs=20, useRandomCoords=True, randomSeed=seed)
    conformers_h = [mol_h.GetConformer(i).GetPositions() for i in range(mol_h.GetNumConformers())]

    atom_groups, attachments = decompose_k_fragments(mol, K=n_subpockets)
    pocket_atoms, pocket_elements = load_receptor_pocket_atoms(receptor=receptor, return_elements=True)
    subpocket_centers = get_receptor_subpocket_centers(receptor=receptor, n_subpockets=n_subpockets)
    subpocket_keys = list(subpocket_centers.keys())
    centers_arr = np.asarray(centers, dtype=float) if centers is not None else np.array(list(subpocket_centers.values()))

    gd = Chem.GetDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()
    rvdw = np.array([pt.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])

    def score_fn(frag_coords: np.ndarray, k: int) -> float:
        grp = atom_groups[k]
        frag_elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in grp] if len(grp) > 0 else ["C"]
        return compute_atom_typed_docking_score(
            frag_coords=frag_coords,
            pocket_atoms=pocket_atoms,
            frag_elements=frag_elems,
            pocket_elements=pocket_elements,
            soft_clip=True,
        )

    lib_pools, pose_slices, beam = build_library(
        mol_h=mol_h,
        conformers_h=conformers_h,
        groups=atom_groups,
        attachments=attachments,
        centers=centers_arr,
        score_fn=score_fn,
        gd=gd,
        rvdw=rvdw,
        pocket_atoms=pocket_atoms,
        search_mode=search_mode,
        n_beam=max(30, poses_per_subpocket * 10),
        seed=seed,
    )

    pools: List[List[Placement]] = []
    for k in range(n_subpockets):
        frag_pool = lib_pools[k]
        if len(frag_pool) > poses_per_subpocket:
            frag_pool = sorted(frag_pool, key=lambda p: p.score)[:poses_per_subpocket]
        elif len(frag_pool) < poses_per_subpocket:
            while len(frag_pool) < poses_per_subpocket:
                p_copy = frag_pool[0]
                frag_pool.append(Placement(frag=k, parent=p_copy.parent, xyz=p_copy.xyz + 0.05, score=p_copy.score))
        pools.append(frag_pool)

    n_vars = n_subpockets * poses_per_subpocket
    fragment_id = torch.tensor(
        [i // poses_per_subpocket for i in range(n_vars)],
        dtype=torch.long
    )

    coords = []
    dG_list = []
    variable_meta = []
    placement_frags_coords = []
    all_placements: List[Placement] = []

    for p_idx in range(n_subpockets):
        p_name = subpocket_keys[p_idx]
        group = atom_groups[p_idx]
        frag_pool = pools[p_idx]

        for pose_idx, p in enumerate(frag_pool):
            var_idx = p_idx * poses_per_subpocket + pose_idx
            pose_centroid = np.mean(p.xyz, axis=0)
            d_min = float(np.min(np.linalg.norm(p.xyz[:, None, :] - pocket_atoms[None, :, :], axis=-1)))

            coords.append(pose_centroid)
            placement_frags_coords.append(p.xyz)
            dG_list.append(p.score)
            all_placements.append(p)

            variable_meta.append({
                "var_index": var_idx,
                "subpocket_id": p_idx,
                "subpocket_name": p_name,
                "pose_id": pose_idx,
                "coord": pose_centroid.tolist(),
                "dG": float(p.score),
                "min_protein_dist": float(d_min),
                "n_fragment_atoms": len(group),
                "fragment_atom_indices": group,
                "parent_id": p.parent,
            })

    coords_t = torch.tensor(np.array(coords), dtype=torch.float64)
    dG_t = torch.tensor(dG_list, dtype=torch.float64)

    # Inter-fragment steric clashes with topological exclusion
    clash_t = torch.zeros((n_vars, n_vars), dtype=torch.float64)
    for i in range(n_vars):
        f_i = int(fragment_id[i])
        p_i = all_placements[i]
        for j in range(i + 1, n_vars):
            f_j = int(fragment_id[j])
            if f_i != f_j:
                p_j = all_placements[j]
                if clash(p_i, p_j, atom_groups[f_i], atom_groups[f_j], gd, rvdw):
                    clash_t[i, j] = clash_t[j, i] = 1.0

    # Inter-fragment covalent connectivity evaluated via exact junction bond geometry + cut-bond torsion profile
    conn_t = torch.zeros((n_vars, n_vars), dtype=torch.float64)
    adj = Chem.GetAdjacencyMatrix(mol)

    for (u, f_u, v, f_v) in attachments:
        grp_u = atom_groups[f_u]
        grp_v = atom_groups[f_v]
        u2 = next((w for w in grp_u if adj[u, w] == 1), None)
        v2 = next((w for w in grp_v if adj[v, w] == 1), None)

        jA = (grp_u.index(u), grp_u.index(u2) if u2 is not None else None)
        jB = (grp_v.index(v), grp_v.index(v2) if v2 is not None else None)

        d_list = [np.linalg.norm(X[u] - X[v]) for X in conformers_h]
        d0 = float(np.median(d_list))
        th_a0 = float(np.median([ang(X[u2], X[u], X[v]) for X in conformers_h])) if u2 is not None else 120.0
        th_b0 = float(np.median([ang(X[u], X[v], X[v2]) for X in conformers_h])) if v2 is not None else 120.0

        prof = None
        if u2 is not None and v2 is not None and len(conformers_h) > 0:
            try:
                prof = torsion_profile(mol_h, 0, u2, u, v, v2, step=30)
            except Exception:
                prof = None

        b_mat = junction_penalty(pools[f_u], pools[f_v], jA, jB, (d0, th_a0, th_b0))
        tors_mat = torsion_term(pools[f_u], pools[f_v], jA, jB, prof)

        for p_u_idx in range(poses_per_subpocket):
            var_u = f_u * poses_per_subpocket + p_u_idx
            for p_v_idx in range(poses_per_subpocket):
                var_v = f_v * poses_per_subpocket + p_v_idx
                b_val = float(b_mat[p_u_idx, p_v_idx])
                conn_val = -(1.0 - b_val) + 0.15 * float(tors_mat[p_u_idx, p_v_idx])
                conn_t[var_u, var_v] += conn_val
                conn_t[var_v, var_u] += conn_val

    # Track pure parent combinations present within the final pools
    final_parent_slices = []
    parent_to_frag_idx: Dict[int, Dict[int, int]] = defaultdict(dict)
    for p_idx in range(n_subpockets):
        for pose_idx, p in enumerate(pools[p_idx]):
            parent_to_frag_idx[p.parent][p_idx] = p_idx * poses_per_subpocket + pose_idx

    for pid, f_map in parent_to_frag_idx.items():
        if len(f_map) == n_subpockets:
            final_parent_slices.append([f_map[k] for k in range(n_subpockets)])

    return {
        "n_vars": n_vars,
        "n_subpockets": n_subpockets,
        "poses_per_subpocket": poses_per_subpocket,
        "fragment_id": fragment_id,
        "dG": dG_t,
        "clash": clash_t,
        "conn": conn_t,
        "coords": coords_t,
        "fragment_poses_coords": placement_frags_coords,
        "atom_groups": atom_groups,
        "attachments": attachments,
        "variable_meta": variable_meta,
        "receptor": receptor,
        "candidate_smiles": candidate_smiles,
        "pose_slices": final_parent_slices,
        "beam": beam,
        "search_mode": search_mode,
    }


def build_candidate_qubo(
    candidate_smiles: str,
    receptor: str = "5V3Y",
    poses_per_subpocket: int = 3,
    n_subpockets: int = 4,
    A: float = 1.0,
    B: float = 5.0,
    C: float = 5.0,
    D: float = 25.0,
    search_mode: str = "oracle",
    centers: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Assemble candidate-specific and receptor-specific Yanagisawa QUBO.

    Builds Q directly from candidate chemistry and receptor crystallography:
    - Q(Candidate A) != Q(Candidate B)
    - Q(Receptor 5V3Y) != Q(Receptor 8TQV)
    """
    placements = generate_candidate_pocket_placements(
        candidate_smiles=candidate_smiles,
        receptor=receptor,
        poses_per_subpocket=poses_per_subpocket,
        n_subpockets=n_subpockets,
        search_mode=search_mode,
        centers=centers,
        seed=seed,
    )
    dG = placements["dG"]
    clash = placements["clash"]
    conn = placements["conn"]
    fragment_id = placements["fragment_id"]

    # Guarantee one-hot constraint holds mathematically: D > 1.5 * spread(A * dG) + B + C
    spread = float(dG.max() - dG.min()) if dG.numel() > 0 else 0.0
    D_eff = max(float(D), 1.5 * abs(A) * spread + abs(B) + abs(C))

    bundle = build_yanagisawa_qubo(
        dG, clash, conn, fragment_id,
        A=A, B=B, C=C, D=D_eff
    )
    val = validate_qubo(bundle, equivalence_trials=200, atol=1e-8)
    if not val["passed"]:
        raise AssertionError(f"QUBO->Ising equivalence validation failed: max error={val['max_equivalence_error']}")

    J, h, c0 = qubo_to_ising(bundle.Q)
    return {
        "bundle": bundle,
        "Q": bundle.Q,
        "J": J,
        "h": h,
        "c0": c0,
        "dG": dG,
        "clash": clash,
        "conn": conn,
        "fragment_id": fragment_id,
        "coords": placements["coords"],
        "fragment_poses_coords": placements["fragment_poses_coords"],
        "atom_groups": placements["atom_groups"],
        "attachments": placements["attachments"],
        "variable_meta": placements["variable_meta"],
        "n_vars": placements["n_vars"],
        "validation": val,
        "receptor": receptor,
        "candidate_smiles": candidate_smiles,
        "pose_slices": placements.get("pose_slices"),
        "search_mode": search_mode,
    }


def exact_onehot_optimum(Q: np.ndarray, fragment_id: Sequence[int]) -> Tuple[float, List[int]]:
    """Compute the exact global minimum of QUBO Q subject to 1-hot constraint per fragment.

    Evaluates the combinatorial product space across fragments (K <= 12, N_combos <= 5e6).
    Returns (E_min, selected_variable_indices).
    """
    Q_arr = np.asarray(Q, dtype=float)
    frag_arr = np.asarray(fragment_id, dtype=int)
    K = int(np.max(frag_arr)) + 1
    idx = [np.flatnonzero(frag_arr == k) for k in range(K)]
    assert np.prod([len(i) for i in idx]) <= 5e6, "Combination space exceeds exact enumeration limit"

    sel = [g.ravel() for g in np.meshgrid(*idx, indexing="ij")]
    E = sum(Q_arr[s, s] for s in sel)
    for a in range(K):
        for b in range(a + 1, K):
            E = E + 2 * Q_arr[sel[a], sel[b]]
    m = int(E.argmin())
    return float(E[m]), [int(s[m]) for s in sel]


def mixing_gain(
    Q: np.ndarray,
    fragment_id: Sequence[int],
    pose_slices: Optional[List[List[int]]] = None
) -> Dict[str, Any]:
    """Measure the energy gain of combinatorial fragment recombination vs the best pure whole pose.

    gain = E_pure_best - E_combinatorial_optimum
    If gain == 0, the QUBO merely selects a single rigid conformer slice without recombination.
    If gain > 0, the QUBO actively assembles a novel multi-fragment recombinant pose.
    """
    Q_arr = np.asarray(Q, dtype=float)
    E_opt, sel = exact_onehot_optimum(Q_arr, fragment_id)
    n_vars = len(Q_arr)

    valid_pure_energies = []
    if pose_slices:
        for s in pose_slices:
            if all(0 <= idx < n_vars for idx in s) and len(s) == len(set(fragment_id)):
                e_pure = sum(Q_arr[s[a], s[a]] for a in range(len(s))) + 2 * sum(
                    Q_arr[s[a], s[b]] for a in range(len(s)) for b in range(a + 1, len(s))
                )
                valid_pure_energies.append(float(e_pure))

    if not valid_pure_energies:
        return {"E_opt": float(E_opt), "E_pure": float(E_opt), "gain": 0.0, "selection": sel}

    E_pure_best = min(valid_pure_energies)
    return {
        "E_opt": float(E_opt),
        "E_pure": float(E_pure_best),
        "gain": float(max(0.0, E_pure_best - E_opt)),
        "selection": sel,
    }


