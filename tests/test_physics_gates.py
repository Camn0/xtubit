import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from xtubit.b6_pairs import (
    decompose_candidate_to_fragments,
    get_receptor_subpocket_centers,
    load_receptor_pocket_atoms,
    compute_atom_typed_docking_score,
    build_pools,
    junction_penalty,
    clash,
    ang,
    Placement,
)
from xtubit.b7_qubo import build_yanagisawa_qubo
from xtubit.post_anneal import (
    stitch_fragments_to_molecule,
    close_molecule,
    reference_min_energy,
    TAM16_SMILES,
)


# Pre-registered pass/fail gate thresholds (v0.2-PHYSICS-VALIDATION)
GATES = {
    "oracle_rmsd": 1e-3,          # Gate A: oracle roundtrip assembly must be identity
    "sampling_floor_A": 1.5,      # Gate B: per-fragment best sampling floor in pool
    "native_vs_best_dE": 0.0,     # Gate C: native pose must be global optimum (dE <= 0.0)
    "max_junction_dev_A": 0.05,   # Staged closure: every cut bond within tolerance
    "closure_shift_A": 1.0,       # Staged closure: pose shift bounded < 1.0 A
    "strain_kcal": 15.0,          # Staged closure: real strain bounded < 15 kcal/mol
}


def load_crystal_ligand(pdb_id: str = "5V3Y") -> Chem.Mol:
    """Load authentic reference crystal ligand."""
    pdb_upper = pdb_id.upper()
    if "8TQV" in pdb_upper:
        sdf_path = Path("data/raw/8tqv_ligand.sdf")
    else:
        sdf_path = Path("data/raw/5v3y_ligand.sdf")

    if not sdf_path.exists():
        raise FileNotFoundError(f"Missing raw crystal ligand SDF at {sdf_path}")

    suppl = Chem.SDMolSupplier(str(sdf_path))
    mol = suppl[0]
    return Chem.RemoveHs(mol)


def heavy_rmsd(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """Compute heavy-atom root mean square deviation between two conformers."""
    m1 = Chem.RemoveHs(Chem.Mol(mol1))
    m2 = Chem.RemoveHs(Chem.Mol(mol2))
    return float(rdMolAlign.GetBestRMS(m1, m2))


def mmff_energy(mol: Chem.Mol) -> float:
    """Calculate absolute MMFF94 force field potential energy."""
    mh = Chem.AddHs(Chem.RemoveHs(Chem.Mol(mol)), addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(mh)
    if props is None or not AllChem.MMFFHasAllMoleculeParams(mh):
        ff = AllChem.UFFGetMoleculeForceField(mh)
    else:
        ff = AllChem.MMFFGetMoleculeForceField(mh, props)
    return float(ff.CalcEnergy())


def sampling_floor(native_xyz: np.ndarray, groups: List[List[int]], pools: List[List[Placement]]) -> List[float]:
    """Calculate best RMSD any placement in pool can achieve per fragment."""
    floors = []
    for k, g in enumerate(groups):
        pool_pts = np.array([p.xyz for p in pools[k]])
        native_frag = native_xyz[g]
        # (N_pool, |g|, 3) -> RMSD per pose
        rmsds = np.sqrt(np.mean(np.sum((pool_pts - native_frag[None, :, :]) ** 2, axis=-1), axis=-1))
        floors.append(float(np.min(rmsds)))
    return floors


def brute_force_onehot_qubo(Q: np.ndarray, n_subpockets: int, poses_per_subpocket: int) -> Tuple[np.ndarray, float]:
    """Brute force the one-hot candidate state space (P^K combinations)."""
    best_E = float("inf")
    best_bits = None
    n_vars = n_subpockets * poses_per_subpocket
    for combo in itertools.product(range(poses_per_subpocket), repeat=n_subpockets):
        bits = np.zeros(n_vars, dtype=float)
        for k, p in enumerate(combo):
            bits[k * poses_per_subpocket + p] = 1.0
        E = float(bits @ Q @ bits)
        if E < best_E:
            best_E = E
            best_bits = bits
    return best_bits, best_E


# ==============================================================================
# Gate A: Oracle Assembly Roundtrip
# ==============================================================================

def test_oracle_assembly_roundtrip():
    """Gate A: Verify that slicing crystal ligand into fragments and reassembling is exact identity."""
    ref = load_crystal_ligand("5V3Y")
    groups, attachments = decompose_candidate_to_fragments(ref, n_subpockets=4)
    xyz = ref.GetConformer().GetPositions()

    out = stitch_fragments_to_molecule(
        decoded_poses=ref,
        atom_groups=groups,
        fragment_poses_coords=[xyz[g] for g in groups],
        poses_per_subpocket=1,
    )

    rmsd = heavy_rmsd(out, ref)
    assert rmsd < GATES["oracle_rmsd"], f"Gate A failed: Oracle assembly RMSD={rmsd:.6f} A >= {GATES['oracle_rmsd']}"

    dE = abs(mmff_energy(out) - mmff_energy(ref))
    assert dE < 1e-2, f"Gate A failed: MMFF energy divergence dE={dE:.4f} kcal/mol"


# ==============================================================================
# Gate B: Sampling Floor
# ==============================================================================

def test_sampling_floor_gate():
    """Gate B: Verify that placement pool achieves <= 1.5 A floor per fragment."""
    ref = load_crystal_ligand("5V3Y")
    groups, _ = decompose_candidate_to_fragments(ref, n_subpockets=4)
    xyz_native = ref.GetConformer().GetPositions()
    centers_dict = get_receptor_subpocket_centers("5V3Y", n_subpockets=4)
    centers = np.array(list(centers_dict.values()))

    # Report distance between subpocket search centers and native fragment centroids
    cents_native = [xyz_native[g].mean(0) for g in groups]
    center_dists = [float(np.linalg.norm(cents_native[k] - centers[k])) for k in range(4)]
    for k, d in enumerate(center_dists):
        assert d < 1.0, f"Search center {k} is {d:.2f} A from native fragment centroid (must be < 1.0 A)"

    mol_h = Chem.AddHs(ref)
    AllChem.EmbedMultipleConfs(mol_h, numConfs=50, randomSeed=42)
    mol_embed = Chem.RemoveHs(mol_h)
    confs = [mol_embed.GetConformer(i).GetPositions() for i in range(mol_embed.GetNumConformers())]

    pocket_atoms, pocket_elements = load_receptor_pocket_atoms("5V3Y", return_elements=True)
    def dummy_score(coords, k):
        return compute_atom_typed_docking_score(coords, pocket_atoms, [ref.GetAtomWithIdx(i).GetSymbol() for i in groups[k]], pocket_elements)

    pools = build_pools(confs, groups, centers, dummy_score, n_keep=30, div_rmsd=0.8, n_jitter=30, seed=42)
    floors = sampling_floor(xyz_native, groups, pools)

    for k, floor in enumerate(floors):
        assert floor < GATES["sampling_floor_A"], (
            f"Gate B failed: Fragment {k} sampling floor {floor:.3f} A exceeds limit {GATES['sampling_floor_A']} A"
        )


# ==============================================================================
# Gate C: Scoring Gate (Native Injected Pose)
# ==============================================================================

def test_scoring_gate():
    """Gate C: Verify that injecting the native pose into the pool yields global QUBO optimum."""
    ref = load_crystal_ligand("5V3Y")
    groups, attachments = decompose_candidate_to_fragments(ref, n_subpockets=4)
    xyz_native = ref.GetConformer().GetPositions()
    pocket_atoms, pocket_elements = load_receptor_pocket_atoms("5V3Y", return_elements=True)

    gd = Chem.GetDistanceMatrix(ref)
    pt = Chem.GetPeriodicTable()
    rvdw = np.array([pt.GetRvdw(a.GetAtomicNum()) for a in ref.GetAtoms()])

    # Construct pool of 2 poses per subpocket: Pose 0 = Native, Pose 1 = Decoy (+2.5 A shift)
    poses_per_subpocket = 2
    placements: List[Placement] = []
    dG_list = []
    fragment_id = []

    for k, g in enumerate(groups):
        elem_g = [ref.GetAtomWithIdx(i).GetSymbol() for i in g]
        # Native pose
        p0_xyz = xyz_native[g]
        s0 = compute_atom_typed_docking_score(p0_xyz, pocket_atoms, elem_g, pocket_elements, soft_clip=True)
        placements.append(Placement(frag=k, parent=0, xyz=p0_xyz, score=s0))
        dG_list.append(s0)
        fragment_id.append(k)

        # Decoy pose (perturbed)
        p1_xyz = p0_xyz + np.array([2.5, 2.5, 2.5])
        s1 = compute_atom_typed_docking_score(p1_xyz, pocket_atoms, elem_g, pocket_elements, soft_clip=True)
        placements.append(Placement(frag=k, parent=1, xyz=p1_xyz, score=s1))
        dG_list.append(s1)
        fragment_id.append(k)

    n_vars = len(dG_list)
    clash_t = torch.zeros((n_vars, n_vars), dtype=torch.float64)
    conn_t = torch.zeros((n_vars, n_vars), dtype=torch.float64)
    adj = Chem.GetAdjacencyMatrix(ref)

    # Clashes with topological exclusion
    for i in range(n_vars):
        fi = fragment_id[i]
        pi = placements[i]
        for j in range(i + 1, n_vars):
            fj = fragment_id[j]
            if fi != fj:
                pj = placements[j]
                if clash(pi, pj, groups[fi], groups[fj], gd, rvdw):
                    clash_t[i, j] = clash_t[j, i] = 1.0

    # Junction penalties
    for (u, fu, v, fv) in attachments:
        gu = groups[fu]
        gv = groups[fv]
        u2 = next((w for w in gu if adj[u, w] == 1), None)
        v2 = next((w for w in gv if adj[v, w] == 1), None)
        d0 = float(np.linalg.norm(xyz_native[u] - xyz_native[v]))
        th_a0 = float(ang(xyz_native[u2], xyz_native[u], xyz_native[v])) if u2 is not None else 120.0
        th_b0 = float(ang(xyz_native[u], xyz_native[v], xyz_native[v2])) if v2 is not None else 120.0

        pool_u = [placements[fu * 2], placements[fu * 2 + 1]]
        pool_v = [placements[fv * 2], placements[fv * 2 + 1]]
        jA = (gu.index(u), gu.index(u2) if u2 is not None else None)
        jB = (gv.index(v), gv.index(v2) if v2 is not None else None)
        b_mat = junction_penalty(pool_u, pool_v, jA, jB, (d0, th_a0, th_b0))

        for pu in range(2):
            for pv in range(2):
                idx_u = fu * 2 + pu
                idx_v = fv * 2 + pv
                conn_val = -(1.0 - float(b_mat[pu, pv]))
                conn_t[idx_u, idx_v] = conn_t[idx_v, idx_u] = conn_val

    dG_t = torch.tensor(dG_list, dtype=torch.float64)
    fid_t = torch.tensor(fragment_id, dtype=torch.long)
    bundle = build_yanagisawa_qubo(dG_t, clash_t, conn_t, fid_t, A=1.0, B=5.0, C=10.0, D=25.0)
    Q = bundle.Q.numpy()

    native_bits = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    E_native = float(native_bits @ Q @ native_bits)

    best_bits, E_best = brute_force_onehot_qubo(Q, n_subpockets=4, poses_per_subpocket=2)
    dE = E_native - E_best

    assert dE <= GATES["native_vs_best_dE"], (
        f"Gate C failed: Model preferred decoy pose over native. E_native={E_native:.3f}, E_best={E_best:.3f}, dE={dE:.3f}"
    )
    assert np.all(native_bits == best_bits), "Optimal bitstring must be exact native pose"


# ==============================================================================
# P4 / P5: Staged Restrained Closure and Strain Gate
# ==============================================================================

def test_staged_closure_and_strain_gate():
    """Verify staged restrained closure achieves shift < 1.0 A and strain < 15.0 kcal/mol."""
    ref = load_crystal_ligand("5V3Y")
    xyz_heavy = ref.GetConformer().GetPositions()
    e_ref = reference_min_energy(ref, n=30, seed=7)

    res = close_molecule(ref, xyz_heavy, e_ref=e_ref)

    assert res["closure_shift"] < GATES["closure_shift_A"], (
        f"Closure shift {res['closure_shift']:.3f} A exceeded threshold {GATES['closure_shift_A']} A"
    )
    assert res["strain"] < GATES["strain_kcal"], (
        f"Calculated strain {res['strain']:.2f} kcal/mol exceeded threshold {GATES['strain_kcal']} kcal/mol"
    )
