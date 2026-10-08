"""X-TUBIT End-to-End Execution Pipeline.

Coordinates the unified screening and digital annealing workflow:
  B1: Standardization, SA scoring, and Bemis-Murcko scaffold splitting
  B2: ETKDGv3 3D conformer generation and MMFF94 force-field minimization
  B3/B4: Bayesian GNN surrogate inference and epistemic uncertainty calibration
  B5: Pareto frontier filtering and Monte Carlo qPMHI acquisition ranking
  B6/B7: Pharmacophore pocket placement and Yanagisawa QUBO/Ising formulation
  B8: Digital Annealing Solver Execution (Exact, TApSA, SpSA, tSB)
  B9: One-hot constraint repair, 3D pose reconstitution, RMSD, and TTS99 metrics
"""

from __future__ import annotations
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from xtubit.b1_data import standardize_smiles, featurize, scaffold_split
from xtubit.b2_conformer import generate_lowest_mmff
from xtubit.b4_bayesian_gnn import BayesianGNN, BayesianLinear, loss as bnn_loss, calibrate_sigma
from xtubit.qpmhi import pareto_front, qpmhi_scores
from xtubit.b7_qubo import build_yanagisawa_qubo, validate_qubo
from xtubit.common.math import qubo_to_ising, qubo_energy, ising_energy, tts_seconds
from xtubit.solvers.exact import brute_force_qubo
from xtubit.solvers.psa_pd import spsa, tapsa
from xtubit.solvers.tesb_port import two_stage_tesb
from xtubit.b9_metrics import repair_onehot, heavy_atom_rmsd, summarize_solver_runs

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("xtubit.pipeline")


def run_stage_b1(raw_csv: Path, out_dir: Path) -> pd.DataFrame:
    """Stage B1: Data standardization, featurization, and scaffold splitting."""
    logger.info("Executing Stage B1: Data standardization and featurization...")
    df = pd.read_csv(raw_csv)
    records = []
    for _, row in df.iterrows():
        mol_id = str(row["mol_id"])
        smi = str(row["smiles"])
        mol = standardize_smiles(smi)
        if mol is None:
            logger.warning("Failed to parse SMILES for %s: %s", mol_id, smi)
            continue
        feat = featurize(mol, mol_id=mol_id, source=str(row.get("source", "")), allow_sa_fallback=True)
        feat["pIC50"] = float(row["pIC50"])
        feat["ic50_uM"] = float(row.get("ic50_uM", 0.0))
        records.append(feat)

    res_df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_parquet(out_dir / "candidates.parquet", index=False)

    train_df, val_df, test_df = scaffold_split(res_df, train_frac=0.70, val_frac=0.15)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    logger.info("Stage B1 complete: %d candidates processed (train=%d, val=%d, test=%d).",
                len(res_df), len(train_df), len(val_df), len(test_df))
    return res_df


def run_stage_b2(df: pd.DataFrame, out_dir: Path) -> Path:
    """Stage B2: 3D conformer generation with ETKDGv3 and MMFF94."""
    logger.info("Executing Stage B2: Conformer generation and MMFF94 minimization...")
    conf_dir = out_dir / "conformers"
    conf_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        mol_id = row["mol_id"]
        smi = row["smiles_can"]
        sdf_path = conf_dir / f"{mol_id}.sdf"
        try:
            mol, best_cid, best_e = generate_lowest_mmff(smi, n_confs=15, seed=7)
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol, confId=best_cid)
            writer.close()
        except Exception as e:
            logger.warning("Stage B2 conformer generation note for %s: %s", mol_id, e)

    logger.info("Stage B2 complete: conformers saved to %s", conf_dir)
    return conf_dir


def run_stage_b3_b4(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Stage B3/B4: Bayesian GNN surrogate modeling and uncertainty calibration."""
    logger.info("Executing Stage B3/B4: Bayesian surrogate and uncertainty calibration...")
    feats = ["qed", "sa", "mw", "logp", "hbd", "hba", "rot_bonds"]

    # Load strict train/val/test splits to eliminate calibration data leakage
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    if train_path.exists() and val_path.exists():
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()
    else:
        train_df, val_df, test_df = scaffold_split(df, train_frac=0.70, val_frac=0.15)

    X_train = torch.tensor(train_df[feats].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["pIC50"].values, dtype=torch.float32)

    # Variational Bayesian linear model
    in_dim = len(feats)
    torch.manual_seed(42)
    bayes_head = torch.nn.Sequential(
        BayesianLinear(in_dim, 32, prior_sigma=0.1),
        torch.nn.SiLU(),
        BayesianLinear(32, 1, prior_sigma=0.1)
    )

    optimizer = torch.optim.Adam(bayes_head.parameters(), lr=0.01)
    for epoch in range(120):
        optimizer.zero_grad()
        preds = bayes_head(X_train).squeeze(-1)
        mse = torch.nn.functional.mse_loss(preds, y_train)
        kl = sum(m.kl() for m in bayes_head.modules() if isinstance(m, BayesianLinear))
        loss = mse + 1e-3 * kl / len(y_train)
        loss.backward()
        optimizer.step()

    # Out-of-sample uncertainty calibration strictly on validation set
    if len(val_df) > 0:
        X_val = torch.tensor(val_df[feats].values, dtype=torch.float32)
        y_val = torch.tensor(val_df["pIC50"].values, dtype=torch.float32)
        with torch.no_grad():
            val_samples = [bayes_head(X_val).squeeze(-1) for _ in range(50)]
        val_stacked = torch.stack(val_samples, dim=0)
        mu_val = val_stacked.mean(dim=0)
        sigma_val = val_stacked.std(dim=0, unbiased=True).clamp_min(1e-4)
        tau = calibrate_sigma(mu_val, sigma_val, y_val)
    else:
        tau = 1.0

    # Posterior inference across all candidate compounds
    X_all = torch.tensor(df[feats].values, dtype=torch.float32)
    samples = []
    with torch.no_grad():
        for _ in range(50):
            samples.append(bayes_head(X_all).squeeze(-1))
    stacked = torch.stack(samples, dim=0)
    mu = stacked.mean(dim=0)
    sigma = stacked.std(dim=0, unbiased=True).clamp_min(1e-4)
    calibrated_sigma = sigma * tau

    df_out = df.copy()
    df_out["mu"] = mu.numpy()
    df_out["sigma"] = calibrated_sigma.numpy()
    df_out["tau"] = tau

    # Save trained Bayesian neural network checkpoint for live interactive inference
    model_save_path = out_dir / "bayes_head.pt"
    torch.save({
        "model_state": bayes_head.state_dict(),
        "feats": feats,
        "tau": tau,
    }, model_save_path)
    logger.info("Bayesian surrogate checkpoint saved to %s", model_save_path)

    # Annotate partition provenance
    train_ids = set(train_df["mol_id"])
    val_ids = set(val_df["mol_id"]) if len(val_df) > 0 else set()
    df_out["split"] = df_out["mol_id"].apply(
        lambda m: "train" if m in train_ids else ("val" if m in val_ids else "test")
    )

    logger.info("Stage B3/B4 complete: calibrated tau=%.4f (validation set only), mean sigma=%.4f",
                tau, float(calibrated_sigma.mean()))
    return df_out



def run_stage_b5(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Stage B5: Pareto frontier filtering and Monte Carlo qPMHI acquisition."""
    logger.info("Executing Stage B5: Multi-objective Pareto filtering and qPMHI acquisition...")
    mu = df["mu"].values
    sigma = df["sigma"].values
    qed = df["qed"].values
    sa_inv = 1.0 / np.maximum(df["sa"].values, 1e-4)

    Y = np.column_stack([mu, qed, sa_inv])
    front = pareto_front(Y)
    ref = np.array([mu.min() - 0.5, 0.0, 0.0])

    scores = qpmhi_scores(mu, sigma, qed, sa_inv, front, ref, samples=256, seed=7)
    df_out = df.copy()
    df_out["qpmhi_score"] = scores
    df_out["sa_inv"] = sa_inv
    df_out = df_out.sort_values(by="qpmhi_score", ascending=False).reset_index(drop=True)
    df_out["rank"] = df_out.index + 1

    selected_path = out_dir / "selected.parquet"
    df_out.to_parquet(selected_path, index=False)
    logger.info("Stage B5 complete: selected %d candidates saved to %s", len(df_out), selected_path)
    return df_out


def run_stage_b6_b7(
    out_dir: Path,
    candidate_smiles: Optional[str] = None,
    candidate_mol_id: str = "TAM16",
    receptor: Optional[str] = None
) -> Dict[str, Any]:
    """Stage B6/B7: Dynamic candidate- and receptor-dependent Yanagisawa 4-term QUBO/Ising assembly."""
    logger.info("Executing Stage B6/B7: Dynamic pocket placement and Yanagisawa Hamiltonian formulation for %s...", candidate_mol_id)
    if candidate_smiles is None:
        from .post_anneal import TAM16_SMILES
        candidate_smiles = TAM16_SMILES

    target_receptor = receptor if receptor is not None else ("8TQV" if candidate_mol_id == "X20403" else "5V3Y")

    from .b6_pairs import build_candidate_qubo
    cand_qubo_res = build_candidate_qubo(
        candidate_smiles=candidate_smiles,
        receptor=target_receptor,
        poses_per_subpocket=3,
        n_subpockets=4,
        A=1.0, B=5.0, C=5.0, D=25.0
    )

    bundle = cand_qubo_res["bundle"]
    Q = cand_qubo_res["Q"]
    J = cand_qubo_res["J"]
    h = cand_qubo_res["h"]
    c0 = cand_qubo_res["c0"]
    dG = cand_qubo_res["dG"]
    coords = cand_qubo_res["coords"]
    fragment_id = cand_qubo_res["fragment_id"]
    placement_ids = [f"F{m['subpocket_id']}_P{m['pose_id']}" for m in cand_qubo_res["variable_meta"]]
    validation = cand_qubo_res["validation"]

    logger.info("Stage B7 validation: max equivalence error = %.2e (passed=%s)",
                validation["max_equivalence_error"], validation["passed"])

    qubo_dir = out_dir / "qubo"
    qubo_dir.mkdir(parents=True, exist_ok=True)
    bundle_data = {
        "Q": Q,
        "J": J,
        "h": h,
        "c0": c0,
        "dG": dG,
        "fragment_id": fragment_id,
        "coords": coords,
        "variable_map": bundle.variable_map,
        "placement_ids": placement_ids,
        "onehot_constant": bundle.onehot_constant,
        "poses_per_subpocket": 3,
        "candidate_smiles": candidate_smiles,
        "candidate_mol_id": candidate_mol_id,
        "receptor": target_receptor,
        "fragment_poses_coords": cand_qubo_res.get("fragment_poses_coords"),
        "atom_groups": cand_qubo_res.get("atom_groups"),
    }
    torch.save(bundle_data, qubo_dir / "tam16_qubo.pt")
    logger.info("Stage B6/B7 complete: Candidate- and receptor-dependent QUBO saved to %s", qubo_dir / "tam16_qubo.pt")
    return bundle_data



def run_stage_b8(bundle_data: Dict[str, Any], out_dir: Path) -> List[Dict[str, Any]]:
    """Stage B8: Digital annealing solver benchmark (Exact, TApSA, SpSA, tSB)."""
    logger.info("Executing Stage B8: Quantum/Digital annealing solver comparison...")
    Q = bundle_data["Q"]
    J = bundle_data["J"]
    h = bundle_data["h"]
    c0 = bundle_data["c0"]
    onehot_c = bundle_data["onehot_constant"]

    results = []

    # 1. Exact Brute Force Solver
    t0 = time.perf_counter()
    x_exact, e_exact = brute_force_qubo(Q)
    t_exact = time.perf_counter() - t0
    full_e_exact = float(e_exact + onehot_c)
    results.append({
        "solver": "Exact (Brute Force)",
        "energy": full_e_exact,
        "qubo_e": float(e_exact),
        "wall_s": t_exact,
        "best_bits": x_exact.tolist(),
        "p_success": 1.0,
        "tts_99": t_exact,
    })
    logger.info("Exact solver: energy=%.4f (time=%.4fs)", full_e_exact, t_exact)

    # Helper to evaluate solver runs
    def evaluate_solver(name, runs_fn, n_runs=10):
        solver_runs = []
        best_e = math.inf
        best_x = None
        for run_idx in range(n_runs):
            t_start = time.perf_counter()
            spins = runs_fn(run_idx)
            t_wall = time.perf_counter() - t_start
            # Convert spin s in {-1, 1} to binary x in {0, 1}
            x = (spins + 1.0) / 2.0
            if x.dim() > 1:
                x = x[:, 0]
            q_e = float(qubo_energy(Q.float(), x))
            tot_e = q_e + onehot_c
            solver_runs.append({"energy": tot_e, "wall_s": t_wall, "x": x})
            if tot_e < best_e:
                best_e = tot_e
                best_x = x

        stats = summarize_solver_runs(solver_runs)
        results.append({
            "solver": name,
            "energy": stats["best_energy"],
            "wall_s": stats["median_wall_s"],
            "best_bits": best_x.tolist() if best_x is not None else [],
            "p_success": stats["p_success_at_best"],
            "tts_99": stats["tts_99"],
        })
        logger.info("%s: best_energy=%.4f, p_succ=%.2f, TTS99=%.4fs",
                    name, stats["best_energy"], stats["p_success_at_best"], stats["tts_99"])

    # 2. TApSA (Time-Averaged Perturbed Simulated Annealing)
    evaluate_solver("TApSA", lambda s: tapsa(J, h, cycles=300, batch=32, seed=s))

    # 3. SpSA (Space-Perturbed Simulated Annealing)
    evaluate_solver("SpSA", lambda s: spsa(J, h, cycles=300, batch=32, seed=s))

    # 4. Two-Stage Tabu Simulated Bifurcation (tSB)
    evaluate_solver("Two-Stage tSB", lambda s: two_stage_tesb(J, h, warm_iter=200, run_iter=400, seed=s))

    solver_dir = out_dir / "solver_out"
    solver_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(solver_dir / "solver_comparison.parquet", index=False)
    logger.info("Stage B8 complete: solver benchmarks written to %s", solver_dir)
    return results


def run_stage_b9(bundle_data: Dict[str, Any], solver_results: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    """Stage B9: Multi-solver constraint repair, 3D pose reconstitution, and RMSD evaluation."""
    logger.info("Executing Stage B9: One-hot repair, pose reconstitution, and RMSD evaluation...")
    fragment_id = bundle_data["fragment_id"].tolist()
    dG = bundle_data["dG"].tolist()
    coords = bundle_data.get("coords")
    candidate_smiles = bundle_data.get("candidate_smiles")
    candidate_mol_id = bundle_data.get("candidate_mol_id", "TAM16")
    poses_per_subpocket = bundle_data.get("poses_per_subpocket", 3)

    groups = []
    unique_frags = sorted(list(set(fragment_id)))
    for f in unique_frags:
        groups.append([i for i, fid in enumerate(fragment_id) if fid == f])

    from .post_anneal import (
        decode_bitstring_to_subpockets,
        stitch_fragments_to_molecule,
        minimize_ligand_in_pocket,
        compute_crystal_rmsd,
    )

    solver_evaluations = []
    best_overall_rmsd = 999.0
    best_overall_solver = None
    best_repaired_bits = None

    for s_res in solver_results:
        raw_bits = s_res["best_bits"]
        repaired_bits, violations = repair_onehot(raw_bits, groups, dG)

        # 3D structure reconstruction from decoded fragment placement coordinates
        decoded = decode_bitstring_to_subpockets(
            [int(b) for b in repaired_bits.tolist()],
            poses_per_subpocket=poses_per_subpocket,
            n_subpockets=len(unique_frags)
        )
        stitched_mol = stitch_fragments_to_molecule(
            decoded,
            variable_coords=coords,
            poses_per_subpocket=poses_per_subpocket,
            candidate_smiles=candidate_smiles,
            fragment_poses_coords=bundle_data.get("fragment_poses_coords"),
            atom_groups=bundle_data.get("atom_groups"),
        )
        relaxed_res = minimize_ligand_in_pocket(stitched_mol, max_steps=50)

        # Heavy-atom RMSD vs authentic crystal structure
        target_ref_pdb = "8TQV" if (candidate_mol_id == "X20403") else "5V3Y"
        actual_rmsd = compute_crystal_rmsd(relaxed_res["minimized_mol"], ref_pdb=target_ref_pdb)

        eval_entry = {
            "solver": s_res["solver"],
            "raw_energy": round(float(s_res.get("best_energy", s_res.get("energy", 0.0))), 3),
            "constraint_violations": int(violations),
            "repaired_rmsd_A": actual_rmsd,
            "rmsd_under_2A_success": bool(actual_rmsd < 2.0),
            "repaired_bits": repaired_bits.tolist(),
        }
        solver_evaluations.append(eval_entry)

        if actual_rmsd < best_overall_rmsd:
            best_overall_rmsd = actual_rmsd
            best_overall_solver = s_res["solver"]
            best_repaired_bits = repaired_bits.tolist()

    ref_pdb_code = "8TQV" if (candidate_mol_id == "X20403") else "5V3Y"
    ref_ligand_code = "JS9" if (candidate_mol_id == "X20403") else "5V8"
    ref_res_A = 2.00 if (candidate_mol_id == "X20403") else 1.98

    summary = {
        "status": "COMPLETED",
        "reference_pdb": ref_pdb_code,
        "reference_ligand_id": ref_ligand_code,
        "reference_resolution_A": ref_res_A,
        "lead_compound": candidate_mol_id,
        "candidate_smiles": candidate_smiles,
        "total_qubo_variables": len(fragment_id),

        "best_solver": best_overall_solver,
        "heavy_atom_rmsd_A": best_overall_rmsd,
        "rmsd_under_2A_success": bool(best_overall_rmsd < 2.0),
        "repaired_solution_bits": best_repaired_bits,
        "solver_evaluations": solver_evaluations,
        "solver_benchmarks": solver_results,
    }

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Stage B9 complete: Best solver %s achieved Heavy-atom RMSD=%.2f A (Success=%s). Metrics saved to %s",
                best_overall_solver, best_overall_rmsd, best_overall_rmsd < 2.0, metrics_dir / "summary.json")
    return summary



def run_cross_docking_benchmark(out_dir: Path) -> Dict[str, Any]:
    """Execute rigorous cross-docking benchmarks across distinct Pks13 receptor conformations.

    Cross-docking protocol:
    1. Cognate 1: TAM16 docked into native wild-type PDB 5V3Y.
    2. Cognate 2: X20403 docked into native cryptic-pocket PDB 8TQV.
    3. Cross-docking 1: X20403 docked into foreign wild-type PDB 5V3Y.
    4. Cross-docking 2: TAM16 docked into foreign cryptic-pocket PDB 8TQV.
    """
    logger.info("Executing Cross-Docking Benchmark across 5V3Y (Closed) and 8TQV (Open)...")
    from .b6_pairs import build_candidate_qubo, load_receptor_pocket_atoms, compute_protein_fragment_contact_potential
    from .solvers.exact import brute_force_qubo
    from .post_anneal import (
        TAM16_SMILES,
        decode_bitstring_to_subpockets,
        stitch_fragments_to_molecule,
        minimize_ligand_in_pocket,
        compute_crystal_rmsd_detailed,
    )

    x20403_smi = "CN(CC1(CC1)COC)C(=O)c2ccc(cc2)CCn3cc(nn3)c4ccc(nc4)c5cc(ccc5OC)OC"
    ref_offset_8tqv = np.array([-8.87, -41.34, 5.63])

    experiments = [
        {"name": "Cognate (TAM16 in 5V3Y)", "cand_id": "TAM16", "smi": TAM16_SMILES, "receptor": "5V3Y", "ref_pdb": "5V3Y"},
        {"name": "Cognate (X20403 in 8TQV)", "cand_id": "X20403", "smi": x20403_smi, "receptor": "8TQV", "ref_pdb": "8TQV"},
        {"name": "Cross-Docking (X20403 in 5V3Y)", "cand_id": "X20403", "smi": x20403_smi, "receptor": "5V3Y", "ref_pdb": "8TQV"},
        {"name": "Cross-Docking (TAM16 in 8TQV)", "cand_id": "TAM16", "smi": TAM16_SMILES, "receptor": "8TQV", "ref_pdb": "5V3Y"},
    ]

    results = []
    cognate_energies = {}

    for exp in experiments:
        qubo_data = build_candidate_qubo(
            candidate_smiles=exp["smi"],
            receptor=exp["receptor"],
            poses_per_subpocket=3,
            n_subpockets=4
        )
        Q = qubo_data["Q"]
        x_exact, e_exact = brute_force_qubo(Q)
        bit_list = [int(b) for b in x_exact.tolist()]

        decoded = decode_bitstring_to_subpockets(bit_list, poses_per_subpocket=3, n_subpockets=4)
        stitched = stitch_fragments_to_molecule(
            decoded,
            variable_coords=qubo_data["coords"],
            poses_per_subpocket=3,
            candidate_smiles=exp["smi"],
            fragment_poses_coords=qubo_data.get("fragment_poses_coords"),
            atom_groups=qubo_data.get("atom_groups"),
        )
        relaxed = minimize_ligand_in_pocket(stitched, max_steps=40)

        # Coordinate transformation offset for cross-receptor frames
        if exp["receptor"] == "8TQV" and exp["ref_pdb"] == "5V3Y":
            trans_offset = -ref_offset_8tqv
        elif exp["receptor"] == "5V3Y" and exp["ref_pdb"] == "8TQV":
            trans_offset = ref_offset_8tqv
        else:
            trans_offset = None

        rmsd_dict = compute_crystal_rmsd_detailed(
            relaxed["minimized_mol"],
            ref_pdb=exp["ref_pdb"],
            receptor_offset=trans_offset
        )

        # Evaluate physical protein pocket contacts and clashes
        pocket_atoms = load_receptor_pocket_atoms(exp["receptor"])
        c_clean = Chem.RemoveHs(relaxed["minimized_mol"]).GetConformer().GetPositions()
        d_prot = np.min(np.linalg.norm(c_clean[:, None, :] - pocket_atoms[None, :, :], axis=-1), axis=-1)
        clashes = int(np.sum(d_prot < 2.0))
        contact_dG = compute_protein_fragment_contact_potential(c_clean, pocket_atoms)

        is_cognate = bool(exp["receptor"] == exp["ref_pdb"])
        if is_cognate:
            cognate_energies[exp["cand_id"]] = float(e_exact)

        delta_vs_cognate = round(float(e_exact) - cognate_energies.get(exp["cand_id"], float(e_exact)), 3)

        results.append({
            "experiment": exp["name"],
            "compound": exp["cand_id"],
            "receptor_pdb": exp["receptor"],
            "reference_crystal_pdb": exp["ref_pdb"],
            "is_cognate": is_cognate,
            "qubo_energy": round(float(e_exact), 3),
            "score_diff_vs_cognate": delta_vs_cognate,
            "receptor_contact_dG": round(float(contact_dG), 2),
            "receptor_clashes": clashes,
            "ligand_strain_energy_kcal_mol": relaxed["ligand_strain_energy_kcal_mol"],
            "minimized_energy_kcal_mol": relaxed["minimized_energy_kcal_mol"],
            "energy_type": "Ligand Intramolecular Strain (MMFF94)",
            "conformer_aligned_rmsd_A": rmsd_dict["conformer_aligned_rmsd_A"],
            "in_pocket_cartesian_rmsd_A": rmsd_dict["in_pocket_cartesian_rmsd_A"],
            "heavy_atom_rmsd_A": rmsd_dict["conformer_aligned_rmsd_A"],
            "success_under_2A": bool(rmsd_dict["conformer_aligned_rmsd_A"] < 2.0),
        })
        logger.info("%s: QUBO E=%.2f, Contact dG=%.2f, Clashes=%d, Aligned RMSD=%.2f A, Pocket RMSD=%.2f A",
                    exp["name"], float(e_exact), contact_dG, clashes,
                    rmsd_dict["conformer_aligned_rmsd_A"], rmsd_dict["in_pocket_cartesian_rmsd_A"])

    cross_data = {
        "status": "COMPLETED",
        "description": "Retrospective Cross-Docking Validation Benchmark with Protein Clashes and Dual RMSD",
        "benchmark_version": "v0.2-causal-brics-physics",
        "assembly_method": "causal_exact_fragment_placement",
        "scoring_function": "atom_typed_vdw_clash_hbond",
        "experiments": results,
    }
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "cross_docking.json", "w", encoding="utf-8") as f:
        json.dump(cross_data, f, indent=2)

    logger.info("Cross-docking benchmarks written to %s", metrics_dir / "cross_docking.json")
    return cross_data


def run_pipeline(raw_csv: str = "data/raw/pks13_compounds.csv", out_dir: str = "data/processed") -> Dict[str, Any]:
    """Executes the complete X-TUBIT computational workflow."""
    start_time = time.perf_counter()
    logger.info("Starting X-TUBIT End-to-End Execution Pipeline...")

    raw_path = Path(raw_csv)
    out_path = Path(out_dir)

    # Stage B1
    candidates_df = run_stage_b1(raw_path, out_path)

    # Stage B2
    run_stage_b2(candidates_df, out_path)

    # Stage B3/B4
    bayes_df = run_stage_b3_b4(candidates_df, out_path)

    # Stage B5
    selected_df = run_stage_b5(bayes_df, out_path)
    top_lead = selected_df.iloc[0] if len(selected_df) > 0 else None
    lead_smi = str(top_lead["smiles_can"]) if top_lead is not None else None
    lead_id = str(top_lead["mol_id"]) if top_lead is not None else "TAM16"

    # Stage B6/B7
    bundle_data = run_stage_b6_b7(out_path, candidate_smiles=lead_smi, candidate_mol_id=lead_id)

    # Stage B8
    solver_results = run_stage_b8(bundle_data, out_path)

    # Stage B9
    summary = run_stage_b9(bundle_data, solver_results, out_path)

    # Cross-Docking Benchmark
    cross_summary = run_cross_docking_benchmark(out_path)
    summary["cross_docking"] = cross_summary

    total_time = time.perf_counter() - start_time
    summary["pipeline_wall_time_s"] = total_time
    logger.info("X-TUBIT Execution successfully completed in %.2f seconds!", total_time)
    return summary


if __name__ == "__main__":
    run_pipeline()
