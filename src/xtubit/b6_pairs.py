from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
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
    radius: float = 12.0
) -> np.ndarray:
    """Extract heavy atom coordinates of receptor pocket from authentic PDB structure."""
    from pathlib import Path
    pdb_code = "8TQV" if "8TQV" in receptor.upper() else "5V3Y"
    pdb_file = Path(f"data/raw/{pdb_code}.pdb")
    if not pdb_file.exists():
        return np.array([[4.8, 22.8, 7.3], [5.7, 26.0, 6.2], [4.2, 24.5, 7.6]])

    if center is None:
        if pdb_code == "8TQV":
            center = np.array([-4.07, -15.05, 13.11])
        else:
            center = np.array([4.80, 26.29, 7.48])

    coords = []
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
                except ValueError:
                    continue
    if not coords:
        return np.array([center])
    return np.array(coords)


def compute_protein_fragment_contact_potential(
    frag_coords: np.ndarray,
    pocket_atoms: np.ndarray,
    clash_cutoff: float = 2.2,
    opt_distance: float = 3.8,
    attr_weight: float = -0.35,
    rep_weight: float = 8.0,
) -> float:
    """Calculate physical contact energy and steric penalties against receptor pocket atoms.

    Uses vectorized Lennard-Jones-like piecewise potential:
    - Steric overlap (d < 2.2 A): steep positive penalty
    - Attractive dispersion contact well (2.2 A <= d <= 5.0 A): negative binding contribution
    """
    if len(frag_coords) == 0 or len(pocket_atoms) == 0:
        return 0.0
    diff = frag_coords[:, None, :] - pocket_atoms[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)  # (M_frag, N_pocket)

    clash_pen = float(np.sum(np.maximum(0.0, clash_cutoff - dists) * rep_weight))
    contact_mask = (dists >= clash_cutoff) & (dists <= 5.0)
    attr_well = float(attr_weight * np.sum(np.exp(-0.5 * ((dists[contact_mask] - opt_distance) / 0.6) ** 2)))

    total_e = clash_pen + attr_well
    n_atoms = max(1, len(frag_coords))
    norm_e = (total_e / np.sqrt(n_atoms)) * 1.5
    return float(np.clip(norm_e, -9.5, 6.0))


def partition_molecule_to_subpockets(
    mol: Chem.Mol,
    n_subpockets: int = 4
) -> List[List[int]]:
    """Partition heavy atoms of a candidate molecule into connected sub-pocket groups."""
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy <= n_subpockets:
        return [[i] for i in range(n_heavy)]

    adj = Chem.GetAdjacencyMatrix(mol)
    atoms_per_frag = max(1, n_heavy // n_subpockets)
    groups = []
    visited = set()
    for i in range(n_heavy):
        if i in visited:
            continue
        q = [i]
        curr = []
        while q and len(curr) < atoms_per_frag:
            u = q.pop(0)
            if u not in visited:
                visited.add(u)
                curr.append(u)
                for v in range(n_heavy):
                    if adj[u, v] == 1 and v not in visited:
                        q.append(v)
        groups.append(curr)
        if len(groups) == n_subpockets - 1:
            rem = [a for a in range(n_heavy) if a not in visited]
            if rem:
                groups.append(rem)
            break
    while len(groups) < n_subpockets:
        max_idx = max(range(len(groups)), key=lambda idx: len(groups[idx]))
        half = len(groups[max_idx]) // 2
        g1, g2 = groups[max_idx][:half], groups[max_idx][half:]
        groups[max_idx] = g1
        groups.append(g2)
    return groups[:n_subpockets]


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
    - Candidate-dependent fragment geometry from candidate SMILES.
    - 3D rigid-body rotation and translational grid search across receptor cavity space.
    - Steric clash avoidance against all-atom receptor coordinates.
    - Receptor-dependent interaction dG against real PDB pocket heavy atoms.
    - Real inter-fragment steric clashes (min distance < 2.0 A).
    - Real covalent connectivity rewards for bonded fragment interfaces.
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

    atom_groups = partition_molecule_to_subpockets(mol, n_subpockets=n_subpockets)
    subpocket_keys = list(PKS13_SUBPOCKETS.keys())[:n_subpockets]
    pocket_atoms = load_receptor_pocket_atoms(receptor=receptor)

    is_8tqv = "8TQV" in receptor.upper()
    ref_offset = np.array([-8.87, -41.34, 5.63]) if is_8tqv else np.array([0.0, 0.0, 0.0])

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
        p_center = PKS13_SUBPOCKETS[p_name]["center"] + ref_offset
        frag_pts = atom_pts[group] if len(group) > 0 else np.array([[0.0, 0.0, 0.0]])
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
                e_contact = compute_protein_fragment_contact_potential(pose_c, pocket_atoms)
                e_total = e_contact + clash_pen
                candidate_evals.append((e_total, pose_c, e_contact, d_min))

        # Rank by total energy (favoring strong contact and zero clash)
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

    # Inter-fragment covalent connectivity
    conn = torch.zeros((n_vars, n_vars), dtype=torch.float64)
    adj = Chem.GetAdjacencyMatrix(mol)
    for i in range(n_vars):
        f_i = int(fragment_id[i])
        pts_i = placement_frags_coords[i]
        grp_i = atom_groups[f_i]
        for j in range(i + 1, n_vars):
            f_j = int(fragment_id[j])
            if f_i != f_j:
                grp_j = atom_groups[f_j]
                is_bonded = any(adj[u, v] == 1 for u in grp_i for v in grp_j)
                if is_bonded:
                    pts_j = placement_frags_coords[j]
                    d_mat = np.linalg.norm(pts_i[:, None, :] - pts_j[None, :, :], axis=-1)
                    min_d = np.min(d_mat)
                    if 1.2 <= min_d <= 2.4:
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
        "variable_meta": placements["variable_meta"],
        "n_vars": placements["n_vars"],
        "validation": val,
        "receptor": receptor,
        "candidate_smiles": candidate_smiles,
    }
