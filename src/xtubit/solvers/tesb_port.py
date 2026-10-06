from __future__ import annotations
import torch


def tesb_port(J, h, tabu=None, n_iter=1000, xi=None, sk=False, batch=1,
              num_tabu=1, dt=1.0, delta=1.0, discrete=False, seed=0):
    """Original X-TUBIT GPU-oriented port based on the public tSB.py semantics.

    This is not the upstream source code. Validate against the pinned upstream
    repository on G1 before using it for experimental results.
    """
    device = J.device
    g = torch.Generator(device=device).manual_seed(int(seed))
    J = J.float(); h = h.float().reshape(-1,1)
    n = J.shape[0]
    if xi is None:
        denom = torch.sum(J*J).sqrt()
        xi = float(0.7*(n-1)**0.5/denom) if sk else float(1.0/J.sum(1).abs().max())
    p = torch.linspace(0,1,int(n_iter),device=device)
    x = 0.01*(torch.rand(n,batch,device=device,generator=g)-0.5)
    y = 0.01*(torch.rand(n,batch,device=device,generator=g)-0.5)
    if tabu is not None:
        tabu = tabu.float().to(device)
    for i in range(int(n_iter)):
        state = torch.sign(x) if discrete else x
        field = J @ state + h
        if tabu is not None and tabu.numel():
            T = tabu.shape[1]
            if T == 1:
                field = field - tabu
            else:
                idx = torch.randint(0,T,(num_tabu,),device=device,generator=g)
                field = field - tabu[:,idx].mean(dim=1,keepdim=True)
        y = y + (-(delta-p[i])*x + xi*field)*dt
        x = x + dt*y*delta
        wall = x.abs() > 1
        x = torch.where(wall, torch.sign(x), x)
        y = torch.where(wall, torch.zeros_like(y), y)
    return torch.sign(x)


def two_stage_tesb(J, h, warm_iter=1000, warm_batch=100, run_iter=9000,
                   run_batch=10, num_tabu=2, seed=0):
    warm = tesb_port(J,h,n_iter=warm_iter,batch=warm_batch,seed=seed)
    return tesb_port(J,h,tabu=warm,n_iter=run_iter,batch=run_batch,
                     num_tabu=num_tabu,seed=seed+1)
