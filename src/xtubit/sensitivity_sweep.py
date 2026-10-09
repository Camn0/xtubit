"""X-TUBIT Center-Sensitivity Sweep and Multi-Seed Benchmark.

Evaluates docking sensitivity to crystallographic search center seeding:
Jitters fragment subpocket centers by:
    c = centers + rng.normal(0, sigma, 3) + rng.normal(0, sigma/2, centers.shape)
over a grid of sigma in {0.0, 0.5, 1.0, 2.0, 3.0} Angstroms and multiple random seeds.
Quantifies the basin of convergence: the perturbation scale sigma at which pose recovery
success (<2.0 A) falls below 50%.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem

from xtubit.b6_pairs import (
    build_candidate_qubo,
    exact_onehot_optimum,
    get_receptor_subpocket_centers,
    load_receptor_pocket_atoms,
    mixing_gain,
)
from xtubit.post_anneal import (
    TAM16_SMILES,
    compute_crystal_rmsd_detailed,
    decode_bitstring_to_subpockets,
    minimize_ligand_in_pocket,
    stitch_fragments_to_molecule,
)

logger = logging.getLogger("xtubit.sensitivity")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

X20403_SMILES = "CN(CC1(CC1)COC)C(=O)c2ccc(cc2)CCn3cc(nn3)c4ccc(nc4)c5cc(ccc5OC)OC"


def dock_instance(
    candidate_id: str = "TAM16",
    receptor: str = "5V3Y",
    centers: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Execute a single docking instance with explicit fragment centers and random seed."""
    smi = X20403_SMILES if candidate_id == "X20403" else TAM16_SMILES
    qubo_res = build_candidate_qubo(
        candidate_smiles=smi,
        receptor=receptor,
        poses_per_subpocket=3,
        n_subpockets=4,
        centers=centers,
        seed=seed,
    )
    Q = qubo_res["Q"].numpy()
    frag_id = qubo_res["fragment_id"].numpy()
    pose_slices = qubo_res.get("pose_slices", [])

    e_opt, sel_vars = exact_onehot_optimum(Q, frag_id)
    mix = mixing_gain(Q, frag_id, pose_slices) if len(pose_slices) > 0 else {"gain": 0.0, "E_opt": e_opt, "E_pure": e_opt}

    bit_list = [0] * qubo_res["n_vars"]
    for v in sel_vars:
        bit_list[v] = 1

    decoded = decode_bitstring_to_subpockets(bit_list, poses_per_subpocket=3, n_subpockets=4)
    stitched = stitch_fragments_to_molecule(
        decoded,
        variable_coords=qubo_res["coords"],
        poses_per_subpocket=3,
        candidate_smiles=smi,
        fragment_poses_coords=qubo_res.get("fragment_poses_coords"),
        atom_groups=qubo_res.get("atom_groups"),
    )
    relaxed = minimize_ligand_in_pocket(stitched, max_steps=40)

    rmsd_pre = compute_crystal_rmsd_detailed(stitched, ref_pdb=receptor)
    rmsd_post = compute_crystal_rmsd_detailed(relaxed["minimized_mol"], ref_pdb=receptor)

    var_meta = qubo_res.get("variable_meta", [])
    parent_ids = [var_meta[v]["parent_id"] for v in sel_vars if v < len(var_meta)]
    from collections import Counter
    if parent_ids:
        counts = Counter(parent_ids)
        parent_purity = round(max(counts.values()) / len(parent_ids), 3)
    else:
        parent_purity = 1.0

    pocket_atoms = load_receptor_pocket_atoms(receptor)
    c_clean = Chem.RemoveHs(relaxed["minimized_mol"]).GetConformer().GetPositions()
    d_prot = np.min(np.linalg.norm(c_clean[:, None, :] - pocket_atoms[None, :, :], axis=-1), axis=-1)
    clashes = int(np.sum(d_prot < 2.0))

    return {
        "qubo_e": round(float(e_opt), 3),
        "mixing_gain": round(float(mix["gain"]), 4),
        "rmsd_pre_closure": round(float(rmsd_pre["in_pocket_cartesian_rmsd_A"]), 3),
        "rmsd_post_closure": round(float(rmsd_post["in_pocket_cartesian_rmsd_A"]), 3),
        "aligned_rmsd": round(float(rmsd_post["conformer_aligned_rmsd_A"]), 3),
        "strain": round(float(relaxed["ligand_strain_energy_kcal_mol"]), 2),
        "closure_shift": round(float(relaxed.get("closure_shift_A", relaxed.get("closure_shift", 0.0))), 3),
        "clashes": clashes,
        "parent_purity": parent_purity,
        "success_under_2A": bool(rmsd_post["in_pocket_cartesian_rmsd_A"] < 2.0),
    }


def run_center_sensitivity_sweep(
    candidate_id: str = "TAM16",
    receptor: str = "5V3Y",
    sigmas: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 3.0),
    n_seeds: int = 20,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute center sensitivity sweep across perturbation sigmas and random seeds.

    Per-seed centers:
        c = centers + rng.normal(0, sigma, 3) + rng.normal(0, sigma/2, centers.shape)
    """
    logger.info("Starting center-sensitivity sweep for %s in %s (%d seeds per sigma)...",
                candidate_id, receptor, n_seeds)
    subpocket_centers = get_receptor_subpocket_centers(receptor=receptor, n_subpockets=4)
    base_centers = np.array(list(subpocket_centers.values()))

    curve = []
    all_runs = []

    for sigma in sigmas:
        sigma_runs = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            if sigma == 0.0:
                c = base_centers.copy()
            else:
                # Global translational shift + per-subpocket jitter
                c = base_centers + rng.normal(0, sigma, 3) + rng.normal(0, sigma / 2.0, base_centers.shape)

            res = dock_instance(
                candidate_id=candidate_id,
                receptor=receptor,
                centers=c,
                seed=seed,
            )
            run_data = {
                "sigma": float(sigma),
                "seed": seed,
                **res,
            }
            sigma_runs.append(run_data)
            all_runs.append(run_data)
            logger.info("sigma=%.1f A, seed=%d -> pocket RMSD pre=%.2f A, post=%.2f A, strain=%.1f, clashes=%d, purity=%.2f",
                        sigma, seed, res["rmsd_pre_closure"], res["rmsd_post_closure"],
                        res["strain"], res["clashes"], res["parent_purity"])

        post_rmsds = [r["rmsd_post_closure"] for r in sigma_runs]
        strains = [r["strain"] for r in sigma_runs]
        clashes = [r["clashes"] for r in sigma_runs]
        successes = [r["success_under_2A"] for r in sigma_runs]

        curve.append({
            "sigma_A": float(sigma),
            "n_runs": len(sigma_runs),
            "success_rate_under_2A": float(np.mean(successes)),
            "median_pocket_rmsd": float(np.median(post_rmsds)),
            "p25_pocket_rmsd": float(np.percentile(post_rmsds, 25)),
            "p75_pocket_rmsd": float(np.percentile(post_rmsds, 75)),
            "median_strain": float(np.median(strains)),
            "median_clashes": int(np.median(clashes)),
        })

    # Find basin of convergence: sigma at which success drops below 50%
    basin_sigma = None
    for pt in curve:
        if pt["success_rate_under_2A"] < 0.50:
            basin_sigma = pt["sigma_A"]
            break

    summary = {
        "candidate": candidate_id,
        "receptor": receptor,
        "sigmas": list(sigmas),
        "n_seeds_per_sigma": n_seeds,
        "basin_of_convergence_sigma_A": basin_sigma,
        "sensitivity_curve": curve,
        "detailed_runs": all_runs,
    }

    if out_dir is not None:
        metrics_dir = out_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_file = metrics_dir / f"sensitivity_sweep_{candidate_id.lower()}_{receptor.lower()}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Sensitivity sweep results written to %s", out_file)

    return summary


if __name__ == "__main__":
    out_path = Path("data/processed")
    run_center_sensitivity_sweep(candidate_id="TAM16", receptor="5V3Y", sigmas=[0.0, 0.5, 1.0, 2.0, 3.0], n_seeds=20, out_dir=out_path)
