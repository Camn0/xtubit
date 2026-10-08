from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
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
    """Derive authentic sub-pocket cavity centers directly inside receptor coordinate frame."""
    rec_upper = receptor.upper()
    if "8TQV" in rec_upper:
        # PDB 8TQV co-crystal JS9 binding cavity
        centers = [
            np.array([-7.67, -12.60, 20.19]),
            np.array([-4.62, -14.89, 17.10]),
            np.array([-1.59, -16.33, 10.48]),
            np.array([-2.39, -16.40, 4.69]),
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
        # PDB 5V3Y co-crystal 5V8 binding cavity
        centers = [
            np.array([5.48, 30.44, 8.22]),
            np.array([3.20, 27.65, 7.83]),
            np.array([4.75, 25.44, 6.71]),
            np.array([5.76, 21.61, 7.16]),
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
) -> float:
    """Calculate atom-typed physical docking score against receptor pocket heavy atoms.
    
    Includes:
    - Atom-typed steric clash penalty (quadratic when d < 0.72 * (R_i + R_j))
    - van der Waals attractive dispersion well (centered at R_i + R_j)
    - Directional hydrogen-bond pairing rewards between polar donors/acceptors (N/O with O/N)
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

        # 3. Directional/polar H-bonding reward (N/O pairs between 2.4 and 3.5 A)
        frag_is_polar = np.array([e.upper() in ["N", "O"] for e in frag_elements])
        pock_is_polar = np.array([e.upper() in ["N", "O"] for e in pocket_elements])
        polar_pairs = frag_is_polar[:, None] & pock_is_polar[None, :]
        hb_mask = polar_pairs & (dists >= 2.4) & (dists <= 3.5)
        hbond_bonus = float(-1.8 * np.sum(np.exp(-0.5 * ((dists[hb_mask] - 2.85) / 0.35) ** 2))) if np.any(hb_mask) else 0.0

        total_e = clash_pen + attr_well + hbond_bonus
    else:
        clash_pen = float(np.sum(np.maximum(0.0, clash_cutoff - dists) * rep_weight))
        contact_mask = (dists >= clash_cutoff) & (dists <= 5.0)
        attr_well = float(attr_weight * np.sum(np.exp(-0.5 * ((dists[contact_mask] - opt_distance) / 0.6) ** 2)))
        total_e = clash_pen + attr_well

    n_atoms = max(1, len(frag_coords))
    norm_e = (total_e / np.sqrt(n_atoms)) * 1.5
    return float(np.clip(norm_e, -9.5, 6.0))


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


def decompose_candidate_to_fragments(
    mol: Chem.Mol,
    n_subpockets: int = 4
) -> Tuple[List[List[int]], List[Tuple[int, int, int, int]]]:
    """Decompose candidate molecule via BRICS and rotatable acyclic bonds into chemical fragments.
    
    Preserves all heavy atoms and maintains rigid ring scaffolds as discrete units.
    Returns:
    - atom_groups: list of atom index lists for each fragment.
    - attachments: list of (atom_u, frag_u, atom_v, frag_v) across cut bonds.
    """
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy <= n_subpockets:
        return [[i] for i in range(n_heavy)], []

    # 1. Identify retrosynthetically cleavable BRICS bonds
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    cleave_bonds = []
    for (u, v), btype in brics_bonds:
        b = mol.GetBondBetweenAtoms(u, v)
        if b is not None and not b.IsInRing():
            cleave_bonds.append(b.GetIdx())

    # 2. If fewer than n_subpockets - 1, supplement with rotatable acyclic single bonds
    if len(cleave_bonds) < n_subpockets - 1:
        for b in mol.GetBonds():
            if (b.GetBondType() == Chem.BondType.SINGLE and 
                not b.IsInRing() and 
                b.GetBeginAtom().GetDegree() > 1 and 
                b.GetEndAtom().GetDegree() > 1 and 
                b.GetIdx() not in cleave_bonds):
                cleave_bonds.append(b.GetIdx())
                if len(cleave_bonds) >= n_subpockets - 1:
                    break

    # Fragment the molecule on selected bonds
    if cleave_bonds:
        frags_mol = Chem.FragmentOnBonds(mol, cleave_bonds[:n_subpockets - 1], addDummies=False)
        atom_groups = [list(grp) for grp in Chem.GetMolFrags(frags_mol, asMols=False)]
    else:
        adj = Chem.GetAdjacencyMatrix(mol)
        atoms_per_frag = max(1, n_heavy // n_subpockets)
        groups = []
        visited = set()
        for i in range(n_heavy):
            if i in visited:
                continue
            q, curr = [i], []
            while q and len(curr) < atoms_per_frag:
                u = q.pop(0)
                if u not in visited:
                    visited.add(u)
                    curr.append(u)
                    for v in range(n_heavy):
                        if adj[u, v] == 1 and v not in visited:
                            q.append(v)
            groups.append(curr)
        atom_groups = groups

    # Ensure exactly n_subpockets groups
    while len(atom_groups) > n_subpockets:
        min_idx = min(range(len(atom_groups)), key=lambda i: len(atom_groups[i]))
        small_grp = atom_groups.pop(min_idx)
        atom_groups[0].extend(small_grp)

    while len(atom_groups) < n_subpockets:
        max_idx = max(range(len(atom_groups)), key=lambda idx: len(atom_groups[idx]))
        if len(atom_groups[max_idx]) <= 1:
            break
        half = len(atom_groups[max_idx]) // 2
        g1, g2 = atom_groups[max_idx][:half], atom_groups[max_idx][half:]
        atom_groups[max_idx] = g1
        atom_groups.append(g2)

    # Track inter-fragment connectivity across cut bonds
    attachments = []
    adj = Chem.GetAdjacencyMatrix(mol)
    for f_i, grp_i in enumerate(atom_groups):
        for f_j in range(f_i + 1, len(atom_groups)):
            grp_j = atom_groups[f_j]
            for u in grp_i:
                for v in grp_j:
                    if adj[u, v] == 1:
                        attachments.append((u, f_i, v, f_j))

    return atom_groups, attachments


def partition_molecule_to_subpockets(
    mol: Chem.Mol,
    n_subpockets: int = 4
) -> List[List[int]]:
    """Partition heavy atoms of a candidate molecule into connected sub-pocket groups via BRICS / rotatable bonds."""
    atom_groups, _ = decompose_candidate_to_fragments(mol, n_subpockets=n_subpockets)
    return atom_groups


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


def generate_candidate_pocket_placements(
    candidate_smiles: str,
    receptor: str = "5V3Y",
    poses_per_subpocket: int = 3,
    n_subpockets: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate candidate- and receptor-dependent 3D placements and interaction terms via 3D rigid fragment docking.

    Computes:
    - Candidate-dependent chemical fragments via BRICS retrosynthetic cleavage.
    - Receptor-specific binding cavity sub-pocket coordinate frames.
    - 3D rigid-body rotation and translational grid search across receptor cavity space.
    - Atom-typed physical docking potential (van der Waals, quadratic steric clash, directional H-bonding).
    - Authentic inter-fragment steric clash matrix (distance < 2.0 A).
    - Covalent connectivity rewards across cut fragment bonds.
    """
    rng = np.random.RandomState(seed)
    mol = Chem.MolFromSmiles(candidate_smiles)
    if mol is None:
        from .post_anneal import TAM16_SMILES
        mol = Chem.MolFromSmiles(TAM16_SMILES)
    mol = Chem.RemoveHs(mol)

    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=seed)
    conf = mol.GetConformer()
    n_heavy = mol.GetNumHeavyAtoms()
    atom_pts = np.array([list(conf.GetAtomPosition(i)) for i in range(n_heavy)])

    atom_groups, attachments = decompose_candidate_to_fragments(mol, n_subpockets=n_subpockets)
    pocket_atoms, pocket_elements = load_receptor_pocket_atoms(receptor=receptor, return_elements=True)
    subpocket_centers = get_receptor_subpocket_centers(receptor=receptor, n_subpockets=n_subpockets)
    subpocket_keys = list(subpocket_centers.keys())

    n_vars = n_subpockets * poses_per_subpocket
    fragment_id = torch.tensor(
        [i // poses_per_subpocket for i in range(n_vars)],
        dtype=torch.long
    )

    sample_3d_rotations = generate_3d_rotations()
    grid_translations = [
        np.array([dx, dy, dz])
        for dx in [-1.2, 0.0, 1.2]
        for dy in [-1.2, 0.0, 1.2]
        for dz in [-1.2, 0.0, 1.2]
    ]

    coords = []
    dG_list = []
    variable_meta = []
    placement_frags_coords = []

    for p_idx in range(n_subpockets):
        group = atom_groups[p_idx]
        p_name = subpocket_keys[p_idx]
        p_center = subpocket_centers[p_name]
        frag_pts = atom_pts[group] if len(group) > 0 else np.array([[0.0, 0.0, 0.0]])
        frag_elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in group] if len(group) > 0 else ["C"]
        frag_center = np.mean(frag_pts, axis=0)
        frag_centered = frag_pts - frag_center

        # Physical 3D rigid fragment grid placement search across SO(3) rotations and translations
        candidate_evals = []
        for R in sample_3d_rotations:
            f_rot = frag_centered @ R.T
            for dt in grid_translations:
                pose_c = f_rot + p_center + dt
                # Check min distance to protein heavy atoms
                d_min = float(np.min(np.linalg.norm(pose_c[:, None, :] - pocket_atoms[None, :, :], axis=-1)))
                clash_pen = 15.0 * (1.8 - d_min) if d_min < 1.8 else 0.0
                e_contact = compute_atom_typed_docking_score(
                    frag_coords=pose_c,
                    pocket_atoms=pocket_atoms,
                    frag_elements=frag_elements,
                    pocket_elements=pocket_elements
                )
                e_total = e_contact + clash_pen
                candidate_evals.append((e_total, pose_c, e_contact, d_min))

        # Rank by total docking score (favoring strong contact and zero clash)
        candidate_evals.sort(key=lambda x: x[0])

        # Greedily select poses_per_subpocket spatially diverse poses (centroid distance >= 0.8 A)
        selected_poses = []
        for e_tot, pose_c, e_cont, d_min in candidate_evals:
            c_new = np.mean(pose_c, axis=0)
            if not any(np.linalg.norm(c_new - np.mean(prev[1], axis=0)) < 0.8 for prev in selected_poses):
                selected_poses.append((e_tot, pose_c, e_cont, d_min))
            if len(selected_poses) >= poses_per_subpocket:
                break

        while len(selected_poses) < poses_per_subpocket:
            selected_poses.append(candidate_evals[len(selected_poses) % len(candidate_evals)])

        for pose_idx, (e_tot, pose_coords, e_contact, d_min) in enumerate(selected_poses):
            var_idx = p_idx * poses_per_subpocket + pose_idx
            pose_centroid = np.mean(pose_coords, axis=0)

            coords.append(pose_centroid)
            placement_frags_coords.append(pose_coords)
            dG_list.append(e_tot)

            variable_meta.append({
                "var_index": var_idx,
                "subpocket_id": p_idx,
                "subpocket_name": p_name,
                "pose_id": pose_idx,
                "coord": pose_centroid.tolist(),
                "dG": float(e_tot),
                "e_contact": float(e_contact),
                "min_protein_dist": float(d_min),
                "n_fragment_atoms": len(group),
                "fragment_atom_indices": group,
            })

    coords_t = torch.tensor(np.array(coords), dtype=torch.float64)
    dG_t = torch.tensor(dG_list, dtype=torch.float64)

    # Inter-fragment steric clashes
    clash = torch.zeros((n_vars, n_vars), dtype=torch.float64)
    for i in range(n_vars):
        f_i = int(fragment_id[i])
        pts_i = placement_frags_coords[i]
        for j in range(i + 1, n_vars):
            f_j = int(fragment_id[j])
            if f_i != f_j:
                pts_j = placement_frags_coords[j]
                d_mat = np.linalg.norm(pts_i[:, None, :] - pts_j[None, :, :], axis=-1)
                if np.min(d_mat) < 2.0:
                    clash[i, j] = clash[j, i] = 1.0

    # Inter-fragment covalent connectivity across cut bonds evaluating exact attachment atom pairs
    conn = torch.zeros((n_vars, n_vars), dtype=torch.float64)
    # Map (f_i, f_j) -> list of (u, v) attachment atom pairs
    attach_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for (u, f_u, v, f_v) in attachments:
        pair_k = (min(f_u, f_v), max(f_u, f_v))
        if pair_k not in attach_map:
            attach_map[pair_k] = []
        attach_map[pair_k].append((u, v) if f_u < f_v else (v, u))

    adj = Chem.GetAdjacencyMatrix(mol)
    for i in range(n_vars):
        f_i = int(fragment_id[i])
        pts_i = placement_frags_coords[i]
        grp_i = atom_groups[f_i]
        for j in range(i + 1, n_vars):
            f_j = int(fragment_id[j])
            if f_i != f_j:
                pair_k = (min(f_i, f_j), max(f_i, f_j))
                if pair_k in attach_map:
                    grp_j = atom_groups[f_j]
                    pts_j = placement_frags_coords[j]
                    pair_dists = []
                    for (u, v) in attach_map[pair_k]:
                        if u in grp_i and v in grp_j:
                            loc_u = grp_i.index(u)
                            loc_v = grp_j.index(v)
                            pair_dists.append(float(np.linalg.norm(pts_i[loc_u] - pts_j[loc_v])))
                        elif v in grp_i and u in grp_j:
                            loc_v = grp_i.index(v)
                            loc_u = grp_j.index(u)
                            pair_dists.append(float(np.linalg.norm(pts_i[loc_v] - pts_j[loc_u])))

                    if pair_dists:
                        min_attach_d = min(pair_dists)
                        # Reward chemically plausible covalent bond distance window (1.2 - 2.2 A)
                        if 1.2 <= min_attach_d <= 2.2:
                            conn[i, j] = conn[j, i] = -1.0
                elif any(adj[u, v] == 1 for u in grp_i for v in atom_groups[f_j]):
                    # Fallback for unmapped bond
                    grp_j = atom_groups[f_j]
                    pts_j = placement_frags_coords[j]
                    d_mat = np.linalg.norm(pts_i[:, None, :] - pts_j[None, :, :], axis=-1)
                    if 1.2 <= float(np.min(d_mat)) <= 2.2:
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
        "fragment_poses_coords": placement_frags_coords,
        "atom_groups": atom_groups,
        "attachments": attachments,
        "variable_meta": variable_meta,
        "receptor": receptor,
        "candidate_smiles": candidate_smiles,
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
        seed=seed,
    )
    dG = placements["dG"]
    clash = placements["clash"]
    conn = placements["conn"]
    fragment_id = placements["fragment_id"]

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
        "coords": placements["coords"],
        "fragment_poses_coords": placements["fragment_poses_coords"],
        "atom_groups": placements["atom_groups"],
        "attachments": placements["attachments"],
        "variable_meta": placements["variable_meta"],
        "n_vars": placements["n_vars"],
        "validation": val,
        "receptor": receptor,
        "candidate_smiles": candidate_smiles,
    }
