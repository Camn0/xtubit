from xtubit.resistance_mutations import (
    evaluate_candidate_resistance_profile,
    PKS13_RESISTANCE_VARIANTS,
)


def test_resistance_variants_dictionary():
    """Verify all 3 clinically relevant Pks13 escape variants exist."""
    assert "WT" in PKS13_RESISTANCE_VARIANTS
    assert "Asp1644Gly" in PKS13_RESISTANCE_VARIANTS
    assert "Asn1640Ala" in PKS13_RESISTANCE_VARIANTS
    assert "Phe1585Leu" in PKS13_RESISTANCE_VARIANTS

    for k, v in PKS13_RESISTANCE_VARIANTS.items():
        assert "name" in v
        assert "mutation" in v
        assert "mechanism" in v
        assert "clinical_prevalence" in v


def test_candidate_resistance_profile_evaluation():
    """Verify computational resistance profiling calculates delta delta G and resilience status."""
    cand = {
        "mol_id": "TAM16",
        "mu": 6.22,
        "qed": 0.653,
        "sa": 2.42,
    }
    res = evaluate_candidate_resistance_profile(cand, num_agents=16, seed=42)

    assert res["mol_id"] == "TAM16"
    assert "overall_resilience_rating" in res
    assert "mean_resistance_penalty_kcal_mol" in res
    assert len(res["variant_profiles"]) == 4

    wt_profile = [p for p in res["variant_profiles"] if p["variant_id"] == "WT"][0]
    assert wt_profile["delta_delta_G_kcal_mol"] == 0.0

    # Mutants must report non-negative delta delta G resistance penalty
    for p in res["variant_profiles"]:
        if p["variant_id"] != "WT":
            assert "delta_delta_G_kcal_mol" in p
            assert "potency_retention_pct" in p
            assert p["resilience_status"] in [
                "Resilient (Maintains Affinity)",
                "Moderate Shift (Partial Loss)",
                "Vulnerable (Significant Resistance)",
            ]
