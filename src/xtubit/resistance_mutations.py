"""Resistance Mutation Profiling Panel Module (Task 4.2).

Models clinically and experimentally selected Pks13 resistance mutations
(Asp1644Gly, Asn1640Ala, Phe1585Leu) to evaluate candidate resilience,
calculating the Resistance Penalty (ΔΔG_res = E_mutant - E_WT) and Resilience Index.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from .b6_pairs import build_scaled_pks13_qubo
from .solvers.sb_adapter import solve_sb


PKS13_RESISTANCE_VARIANTS: Dict[str, Dict[str, Any]] = {
    "WT": {
        "name": "Wild-Type (PDB 5V3Y)",
        "mutation": "Wild-Type Reference",
        "mechanism": "Unmutated native Pks13 thioesterase catalytic cleft",
        "dG_shift_anchor": 0.0,
        "dG_shift_catalytic": 0.0,
        "dG_shift_channel": 0.0,
        "clinical_prevalence": "Reference (Non-resistant)",
    },
    "Asp1644Gly": {
        "name": "Asp1644Gly Escape Variant",
        "mutation": "Asp1644 -> Gly1644",
        "mechanism": "Disruption of catalytic Asp1644 carboxylate H-bond network at catalytic site",
        "dG_shift_anchor": 0.0,
        "dG_shift_catalytic": +3.80,  # Loses key carboxylate H-bond
        "dG_shift_channel": +0.40,
        "clinical_prevalence": "High (Primary in vitro escape mutation under TAM16 pressure, Aggarwal 2017)",
    },
    "Asp1607Asn": {
        "name": "Asp1607Asn Vestibule Variant",
        "mutation": "Asp1607 -> Asn1607",
        "mechanism": "Electrostatic perturbation at active-site entrance vestibule (Aggarwal et al. 2017 Cell)",
        "dG_shift_anchor": 0.0,
        "dG_shift_catalytic": +2.10,
        "dG_shift_channel": +0.60,
        "clinical_prevalence": "High (Observed in laboratory serial-passage selection under TAM16)",
    },
    "Asp1644Tyr": {
        "name": "Asp1644Tyr Steric Occlusion Variant",
        "mutation": "Asp1644 -> Tyr1644",
        "mechanism": "Bulky aromatic steric clash in the catalytic triad cleft (Aggarwal 2017 / Krieger 2024)",
        "dG_shift_anchor": 0.0,
        "dG_shift_catalytic": +4.60,
        "dG_shift_channel": +0.90,
        "clinical_prevalence": "Moderate (High-level resistance variant with severe binding penalty)",
    },
    "Asn1640Ala": {
        "name": "Asn1640Ala Tunnel Shift",
        "mutation": "Asn1640 -> Ala1640",
        "mechanism": "Loss of amide side-chain dipole in hydrophobic channel entrance",
        "dG_shift_anchor": +0.20,
        "dG_shift_catalytic": +0.60,
        "dG_shift_channel": +2.10,  # Channel polarity perturbation
        "clinical_prevalence": "Moderate (Observed in laboratory serial-passage selection)",
    },
    "Phe1585Leu": {
        "name": "Phe1585Leu Sandwich Loss",
        "mutation": "Phe1585 -> Leu1585",
        "mechanism": "Loss of aromatic pi-pi stacking sandwich in benzofuran anchor pocket",
        "dG_shift_anchor": +3.20,  # Benzofuran core aromatic destabilization
        "dG_shift_catalytic": +0.30,
        "dG_shift_channel": +0.50,
        "clinical_prevalence": "Moderate (Confers cross-resistance to rigid bicyclic cores)",
    }
}


def evaluate_candidate_resistance_profile(
    cand_row: pd.Series | Dict[str, Any],
    num_agents: int = 16,
    seed: int = 42
) -> Dict[str, Any]:
    """Evaluate candidate across wild-type and 3 clinical resistance mutation models.
    
    Computes:
    - Ground-state binding energy E_WT and E_mutant
    - Resistance penalty: ΔΔG_res = E_mutant - E_WT
    - Resistance Resilience Index:
      - Resilient (ΔΔG_res <= +1.5 kcal/mol)
      - Moderate Resistance (+1.5 < ΔΔG_res <= +3.5 kcal/mol)
      - Highly Vulnerable (ΔΔG_res > +3.5 kcal/mol)
    """
    cand_mu = float(cand_row.get("mu", cand_row.get("pIC50", 6.5)))
    ref_mu = 6.22
    scale_factor = float(cand_mu / ref_mu)

    # 1. Base WT Hamiltonian & Ground-State Conformer Search
    base_sys = build_scaled_pks13_qubo(poses_per_subpocket=10, seed=seed)
    Q_wt = base_sys["Q"].float() * scale_factor
    frag_id = base_sys["fragment_id"]
    onehot_const = float(base_sys["bundle"].onehot_constant)

    # Solve WT ground state
    bits, values = solve_sb(Q_wt, agents=int(num_agents), max_steps=800, mode="discrete", device="cpu")
    best_idx = values.argmin().item()
    best_bits_raw = bits[best_idx].clone()

    # Ensure strictly feasible 1-hot conformer
    repaired_bits = torch.zeros_like(best_bits_raw)
    for f in torch.unique(frag_id):
        active = torch.where((frag_id == f) & (best_bits_raw == 1))[0]
        if len(active) == 1:
            repaired_bits[active[0]] = 1.0
        else:
            idxs = torch.where(frag_id == f)[0]
            repaired_bits[idxs[0]] = 1.0

    wt_energy = float((repaired_bits @ Q_wt @ repaired_bits).item()) + onehot_const

    variant_results = []
    variant_results.append({
        "variant_id": "WT",
        "variant_name": PKS13_RESISTANCE_VARIANTS["WT"]["name"],
        "mutation": PKS13_RESISTANCE_VARIANTS["WT"]["mutation"],
        "mechanism": PKS13_RESISTANCE_VARIANTS["WT"]["mechanism"],
        "clinical_prevalence": PKS13_RESISTANCE_VARIANTS["WT"]["clinical_prevalence"],
        "binding_energy_kcal_mol": round(wt_energy, 2),
        "delta_delta_G_kcal_mol": 0.0,
        "potency_retention_pct": 100.0,
        "resilience_status": "Reference (Native Sensitive)",
    })

    for var_key, var_info in PKS13_RESISTANCE_VARIANTS.items():
        if var_key == "WT":
            continue

        # Mutate Hamiltonian by perturbing target sub-pocket interaction terms
        Q_mut = Q_wt.clone()
        d_anc = var_info["dG_shift_anchor"]
        d_cat = var_info["dG_shift_catalytic"]
        d_chn = var_info["dG_shift_channel"]

        if d_anc != 0.0:
            mask_anc = (frag_id == 0)
            Q_mut[mask_anc, mask_anc] += d_anc
        if d_chn != 0.0:
            mask_chn = (frag_id == 2)
            Q_mut[mask_chn, mask_chn] += d_chn
        if d_cat != 0.0:
            mask_cat = (frag_id == 4)
            Q_mut[mask_cat, mask_cat] += d_cat

        mut_energy = float((repaired_bits @ Q_mut @ repaired_bits).item()) + onehot_const
        delta_dg = mut_energy - wt_energy

        if delta_dg <= 1.5:
            status = "Resilient (Maintains Affinity)"
        elif delta_dg <= 3.5:
            status = "Moderate Shift (Partial Loss)"
        else:
            status = "Vulnerable (Significant Resistance)"

        retention_pct = max(0.0, min(100.0, (1.0 - (delta_dg / 8.0)) * 100.0))

        variant_results.append({
            "variant_id": var_key,
            "variant_name": var_info["name"],
            "mutation": var_info["mutation"],
            "mechanism": var_info["mechanism"],
            "clinical_prevalence": var_info["clinical_prevalence"],
            "binding_energy_kcal_mol": round(mut_energy, 2),
            "delta_delta_G_kcal_mol": round(delta_dg, 2),
            "potency_retention_pct": round(retention_pct, 1),
            "resilience_status": status,
        })

    # Overall candidate resilience rating
    non_wt = [v for v in variant_results if v["variant_id"] != "WT"]
    avg_delta_res = float(np.mean([v["delta_delta_G_kcal_mol"] for v in non_wt]))
    max_delta_res = float(np.max([v["delta_delta_G_kcal_mol"] for v in non_wt]))

    if max_delta_res <= 1.8:
        overall_rating = "Broadly Resilient"
    elif max_delta_res <= 3.5:
        overall_rating = "Selective Resistance Risk"
    else:
        overall_rating = "Escape Mutation Vulnerable"

    return {
        "mol_id": str(cand_row.get("mol_id", "CANDIDATE")),
        "overall_resilience_rating": overall_rating,
        "mean_resistance_penalty_kcal_mol": round(avg_delta_res, 2),
        "worst_resistance_penalty_kcal_mol": round(max_delta_res, 2),
        "variant_profiles": variant_results,
    }
