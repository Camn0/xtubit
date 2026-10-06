import itertools
import torch
from xtubit.b7_qubo import build_yanagisawa_qubo

def direct_h(dG,c,conn,g,A=1,B=5,C=5,D=25):
    n=len(dG); out=[]; groups=sorted(set(g.tolist()))
    for x in itertools.product([0.,1.], repeat=n):
        x=torch.tensor(x,dtype=torch.float64)
        e=A*(dG*x).sum()
        e += B*sum(c[i,j]*x[i]*x[j] for i in range(n) for j in range(i+1,n))
        e += C*sum(conn[i,j]*x[i]*x[j] for i in range(n) for j in range(i+1,n))
        for k in groups:
            ids=(g==k).nonzero().flatten()
            z=x[ids].sum()-1
            e += 0.5*D*z*z
        out.append(float(e));
    return out

def test_formula_matches_matrix_up_to_constant():
    torch.manual_seed(1); n=6
    g=torch.tensor([0,0,1,1,2,2])
    dG=torch.randn(n,dtype=torch.float64)
    c=torch.zeros((n,n),dtype=torch.float64); c[0,2]=c[2,0]=1
    conn=torch.zeros_like(c); conn[0,2]=conn[2,0]=-1
    b=build_yanagisawa_qubo(dG,c,conn,g)
    direct=direct_h(dG,c,conn,g)
    got=[float((torch.tensor(x,dtype=torch.float64)@b.Q@torch.tensor(x,dtype=torch.float64)).item()+b.onehot_constant) for x in itertools.product([0.,1.],repeat=n)]
    assert max(abs(a-bv) for a,bv in zip(direct,got)) < 1e-8
