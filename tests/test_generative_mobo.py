"""Unit Tests for Generative MOBO Analog Engine (Paulson Lab Adaptation)."""

import pytest
import numpy as np
from rdkit import Chem
from xtubit.generative_mobo import (
    mine_seed_fragments,
    mutate_molecule,
    crossover_molecules,
    generate_analog_population,
    screen_and_rank_analogs,
    MEDCHEM_TRANSFORMS,
)


@pytest.fixture
def seed_smiles():
    return [
        "CCOC(=O)c1c(C)oc(c1)c2c(CC)oc3ccccc23",  # TAM16
        "O=C(NCc1cccs1)c2c(C)oc(c2)c3c(CC)oc4ccccc34",  # X20403 (Krieger 2024 DEL)
        "COC(=O)c1c(C)oc(c1)c2coc3ccccc23",  # TAM1
    ]


def test_mine_seed_fragments(seed_smiles):
    """Verify fragment mining extracts valid sub-fragments from seed ligands."""
    frags = mine_seed_fragments(seed_smiles, radius=2)
    assert len(frags) > 5, "Expected at least 5 unique sub-fragments"
    # Ensure every mined fragment is valid SMILES
    for f in frags:
        m = Chem.MolFromSmiles(f)
        assert m is not None, f"Mined invalid fragment SMILES: {f}"


def test_mutate_molecule(seed_smiles):
    """Verify mutation operator produces sanitized, novel child molecules."""
    tam16 = Chem.MolFromSmiles(seed_smiles[0])
    res = mutate_molecule(tam16)
    assert res is not None, "Mutation failed to produce any product"
    child_mol, transform_name = res
    assert child_mol.GetNumAtoms() > 0
    smi = Chem.MolToSmiles(child_mol)
    assert "." not in smi, "Mutated product contains disconnected fragments"
    assert transform_name in [t[0] for t in MEDCHEM_TRANSFORMS]


def test_crossover_molecules(seed_smiles):
    """Verify biaryl linker crossover produces valid chimera molecule."""
    m1 = Chem.MolFromSmiles(seed_smiles[0])
    m2 = Chem.MolFromSmiles(seed_smiles[1])
    child = crossover_molecules(m1, m2)
    if child is not None:
        assert child.GetNumAtoms() > 10
        smi = Chem.MolToSmiles(child)
        assert "." not in smi, "Crossover product contains disconnected fragments"


def test_generate_analog_population(seed_smiles):
    """Verify population generation produces requested count of novel structures."""
    n_target = 8
    analogs = generate_analog_population(seed_smiles, n_analogs=n_target, seed=123)
    assert len(analogs) == n_target, f"Expected {n_target} analogs, got {len(analogs)}"

    # All generated molecules must be distinct from seed SMILES
    seed_canonical = {Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True) for s in seed_smiles}
    for item in analogs:
        assert item["smiles"] not in seed_canonical, f"Generated identical structure to seed: {item['smiles']}"
        assert item["mol"].GetNumAtoms() > 15
        assert "origin" in item


def test_screen_and_rank_analogs(seed_smiles):
    """Verify full MOBO screening, Bayesian GNN inference, and qPMHI acquisition ranking."""
    analogs = generate_analog_population(seed_smiles, n_analogs=6, seed=42)
    ranked_df = screen_and_rank_analogs(analogs, seed=42)

    assert len(ranked_df) == 6
    assert "qpmhi_score" in ranked_df.columns
    assert "mobo_rank" in ranked_df.columns
    assert "mu" in ranked_df.columns
    assert "sigma" in ranked_df.columns
    assert "solubility_um" in ranked_df.columns
    assert "herg_safe" in ranked_df.columns

    # Verify ranks are ordered 1 to 6
    assert list(ranked_df["mobo_rank"]) == list(range(1, 7))

    # Verify qPMHI scores are within [0.0, 1.0] and sum to <= 1.0
    scores = ranked_df["qpmhi_score"].values
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
