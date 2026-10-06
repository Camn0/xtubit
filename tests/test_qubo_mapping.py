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
