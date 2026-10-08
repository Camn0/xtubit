import torch
from xtubit.b7_qubo import build_yanagisawa_qubo
from xtubit.common.math import equivalence_check

def test_equivalence_random_small():
    torch.manual_seed(7)
    n=12
    g=torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5])
    dG=torch.randn(n,dtype=torch.float64)
    c=torch.randint(0,2,(n,n),dtype=torch.float64); c=torch.triu(c,1); c=c+c.T; c.fill_diagonal_(0)
    conn=torch.zeros((n,n),dtype=torch.float64)
    conn[0,2]=conn[2,0]=-1
    b=build_yanagisawa_qubo(dG,c,conn,g)
    err,ok=equivalence_check(b.Q,trials=1000,atol=1e-8)
    assert ok, err


def test_pyqubo_exact_matrix_equivalence():
    """Verify PyQUBO symbolic compilation matches vectorized build_yanagisawa_qubo to machine precision."""
    from xtubit.b7_qubo import build_yanagisawa_pyqubo

    torch.manual_seed(42)
    n = 12
    g = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    dG = torch.randn(n, dtype=torch.float64)
    c = torch.randint(0, 2, (n, n), dtype=torch.float64)
    c = torch.triu(c, 1)
    c = c + c.T
    c.fill_diagonal_(0)
    conn = torch.zeros((n, n), dtype=torch.float64)
    conn[0, 2] = conn[2, 0] = -1.0
    conn[1, 3] = conn[3, 1] = -1.0

    bundle_vec = build_yanagisawa_qubo(dG, c, conn, g, A=1.0, B=5.0, C=5.0, D=25.0)
    bundle_py, py_model = build_yanagisawa_pyqubo(dG, c, conn, g, A=1.0, B=5.0, C=5.0, D=25.0, half_penalty=True)

    max_diff = float((bundle_vec.Q - bundle_py.Q).abs().max())
    assert max_diff < 1e-12, f"Matrix discrepancy between PyQUBO and vectorized: {max_diff}"
    assert abs(bundle_vec.onehot_constant - bundle_py.onehot_constant) < 1e-12


def test_pyqubo_constraint_validation():
    """Verify PyQUBO model correctly tracks and diagnoses subpocket constraint violations."""
    from xtubit.b7_qubo import build_yanagisawa_pyqubo

    n = 6
    g = torch.tensor([0, 0, 1, 1, 2, 2])
    dG = torch.tensor([-2.0, -1.0, -3.0, -0.5, -1.5, -2.5], dtype=torch.float64)
    c = torch.zeros((n, n), dtype=torch.float64)
    conn = torch.zeros((n, n), dtype=torch.float64)

    bundle, model = build_yanagisawa_pyqubo(dG, c, conn, g)

    # Valid one-hot state: exactly 1 bit active per fragment
    valid_sample = {"x_0": 1, "x_1": 0, "x_2": 1, "x_3": 0, "x_4": 0, "x_5": 1}
    dec_valid = model.decode_sample(valid_sample, vartype="BINARY")
    broken_valid = dec_valid.constraints(only_broken=True)
    assert len(broken_valid) == 0, f"Expected 0 broken constraints, got {broken_valid}"

    # Invalid state: fragment 0 has two bits active (x_0=1, x_1=1)
    invalid_sample = {"x_0": 1, "x_1": 1, "x_2": 1, "x_3": 0, "x_4": 0, "x_5": 1}
    dec_invalid = model.decode_sample(invalid_sample, vartype="BINARY")
    broken_invalid = dec_invalid.constraints(only_broken=True)
    assert "onehot_frag_0" in broken_invalid
    assert broken_invalid["onehot_frag_0"][0] is False
