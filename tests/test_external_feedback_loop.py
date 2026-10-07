import numpy as np
import pandas as pd
from xtubit.external_feedback_loop import (
    evaluate_candidate_external_physics,
    execute_second_degree_feedback_update,
)


def test_external_physics_evaluation():
    """Verify independent physical evaluation tier operates on 60-qubit Pks13 Hamiltonian."""
    cand = {
        "mol_id": "TAM16",
        "mu": 6.22,
        "qed": 0.653,
        "sa": 2.42,
        "mw": 298.34,
        "logp": 4.74,
        "smiles_can": "CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23",
    }
    phys_res = evaluate_candidate_external_physics(cand, n_qubits=60, mmff_steps=50, num_agents=16)

    assert "qubo_energy_kcal_mol" in phys_res
    assert "mmff_energy_kcal_mol" in phys_res
    assert "pose_rmsd_A" in phys_res
    assert phys_res["pose_rmsd_A"] < 2.0, f"Expected RMSD < 2.0 A, got {phys_res['pose_rmsd_A']}"
    assert phys_res["is_feasible"] is True, "Expected 0 pocket constraint violations"
    assert 3.5 <= phys_res["pIC50_physical_equiv"] <= 9.5


def test_second_degree_feedback_uncertainty_reduction():
    """Verify 2nd-degree feedback incorporates external evidence and reduces epistemic uncertainty."""
    # Synthetic candidate pool
    data = {
        "mol_id": ["TAM16", "MOL_2", "MOL_3", "MOL_4"],
        "mu": [6.20, 6.80, 5.90, 7.10],
        "sigma": [0.60, 0.55, 0.70, 0.65],
        "qed": [0.65, 0.72, 0.58, 0.69],
        "sa": [2.4, 2.8, 3.1, 2.5],
        "mw": [393.5, 410.2, 380.0, 425.0],
        "logp": [4.1, 3.8, 4.4, 3.5],
        "qpmhi_score": [0.45, 0.55, 0.35, 0.60],
        "rank": [3, 2, 4, 1],
    }
    df = pd.DataFrame(data)

    # External physical measurement: TAM16 achieves pIC50_equiv = 6.95 from 3D MMFF94 force field
    fb_res = execute_second_degree_feedback_update(
        df_active=df,
        evaluated_mol_id="TAM16",
        physical_pIC50=6.95,
        learning_rate=0.05
    )

    # 1. Epistemic uncertainty must be reduced by external observation
    assert fb_res["mean_uncertainty_reduction"] > 0.0, "Epistemic uncertainty was not reduced"
    assert fb_res["target_sigma_reduction_pct"] > 15.0, "Target compound uncertainty should drop significantly"

    # 2. Reality gap accurately computed
    expected_gap = 6.95 - 6.20
    assert abs(fb_res["reality_gap"] - expected_gap) < 1e-3

    # 3. Updated posterior pulled towards physical evidence
    df_post = fb_res["updated_df"]
    tam16_post_mu = df_post[df_post["mol_id"] == "TAM16"]["mu"].values[0]
    assert tam16_post_mu > 6.20, "TAM16 posterior mu should increase towards physical ground truth (6.95)"


def test_external_feedback_prevents_self_feeding_drift():
    """Verify feedback penalizes candidates when external physics reveals poor pocket fit."""
    data = {
        "mol_id": ["ANALOGUE_A", "ANALOGUE_B"],
        "mu": [8.00, 6.50],  # ANALOGUE_A had high predicted affinity in 2D
        "sigma": [0.80, 0.50],
        "qed": [0.70, 0.65],
        "sa": [2.5, 2.6],
        "mw": [400.0, 390.0],
        "logp": [3.5, 3.6],
        "qpmhi_score": [0.85, 0.45],
        "rank": [1, 2],
    }
    df = pd.DataFrame(data)

    # External physics reveals ANALOGUE_A clashes or has weak binding (pIC50_physical = 5.20)
    fb_res = execute_second_degree_feedback_update(
        df_active=df,
        evaluated_mol_id="ANALOGUE_A",
        physical_pIC50=5.20
    )

    df_post = fb_res["updated_df"]
    post_mu_a = df_post[df_post["mol_id"] == "ANALOGUE_A"]["mu"].values[0]

    # Model is corrected downwards by external physics reality, preventing self-feeding delusion
    assert post_mu_a < 8.00, "Posterior mu was not corrected downwards by external physical evidence"
    assert fb_res["reality_gap"] < 0.0, "Reality gap should be negative (predicted > physical)"
