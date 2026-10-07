import torch
import numpy as np
from xtubit.b6_pairs import (
    PKS13_SUBPOCKETS,
    PKS13_ADJACENT_SUBPOCKETS,
    generate_scaled_pocket_placements,
    build_scaled_pks13_qubo,
)


def test_pks13_6_subpockets_geometry():
    """Verify 6 pharmacophore sub-pocket regions extracted from PDB 5V3Y."""
    expected_subpockets = ["Anchor", "Linker", "Tunnel", "P1_Cap", "Catalytic_Triad", "Solvent_Front"]
    assert len(PKS13_SUBPOCKETS) == 6

    for name in expected_subpockets:
        assert name in PKS13_SUBPOCKETS
        info = PKS13_SUBPOCKETS[name]
        assert "center" in info and len(info["center"]) == 3
        assert info["radius"] >= 2.0, f"Subpocket {name} radius too small: {info['radius']}"
        assert info["base_dG"] < 0.0, f"Subpocket {name} binding dG should be favorable"


def test_scaled_pocket_placements_60_and_90():
    """Verify scaling to 60 qubits (10 poses/site) and 90 qubits (15 poses/site)."""
    # 60 Qubits
    sys60 = generate_scaled_pocket_placements(poses_per_subpocket=10, seed=12)
    assert sys60["n_vars"] == 60
    assert sys60["dG"].shape == (60,)
    assert sys60["clash"].shape == (60, 60)
    assert sys60["conn"].shape == (60, 60)
    assert sys60["coords"].shape == (60, 3)

    # 90 Qubits
    sys90 = generate_scaled_pocket_placements(poses_per_subpocket=15, seed=12)
    assert sys90["n_vars"] == 90
    assert sys90["dG"].shape == (90,)
    assert sys90["clash"].shape == (90, 90)
    assert sys90["conn"].shape == (90, 90)
    assert sys90["coords"].shape == (90, 3)


def test_clash_and_connectivity_tensors():
    """Verify properties of vector-accelerated clash and connectivity matrices."""
    sys = generate_scaled_pocket_placements(poses_per_subpocket=10, seed=42)
    clash = sys["clash"]
    conn = sys["conn"]

    # Clash matrix is symmetric, non-negative, and has zero diagonal
    assert torch.allclose(clash, clash.T)
    assert torch.all(clash >= 0.0)
    assert torch.all(clash.diagonal() == 0.0)

    # Connectivity matrix is symmetric, non-positive, and has zero diagonal
    assert torch.allclose(conn, conn.T)
    assert torch.all(conn <= 0.0)
    assert torch.all(conn.diagonal() == 0.0)


def test_scaled_qubo_ising_machine_precision_equivalence():
    """Verify symmetry and machine-precision equivalence for scaled 60 and 90 qubit QUBOs."""
    # 60 Qubit system
    res60 = build_scaled_pks13_qubo(poses_per_subpocket=10, seed=42)
    assert res60["Q"].shape == (60, 60)
    assert res60["J"].shape == (60, 60)
    assert res60["validation"]["passed"] is True
    assert res60["validation"]["max_equivalence_error"] < 1e-10

    # 90 Qubit system
    res90 = build_scaled_pks13_qubo(poses_per_subpocket=15, seed=42)
    assert res90["Q"].shape == (90, 90)
    assert res90["J"].shape == (90, 90)
    assert res90["validation"]["passed"] is True
    assert res90["validation"]["max_equivalence_error"] < 1e-10
