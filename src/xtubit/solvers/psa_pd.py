from __future__ import annotations
import torch


def _sigma(J):
    n = J.shape[0]
    return (((n-1)*J.var(dim=1,unbiased=False)).sqrt()).mean().clamp_min(1e-8)


def spsa(J,h,cycles=1000,batch=256,gamma=0.1,delta=10.0,stall_prop=0.5,nrnd=1.0,tau=1,seed=0):
    device=J.device
    g=torch.Generator(device=device).manual_seed(seed)
    J=J.float(); h=h.float().reshape(-1,1); n=J.shape[0]
    sig=_sigma(J)
    i0=gamma/sig; i0_max=delta/sig
    n_updates=max(1,int(n*(1-stall_prop)))
    s=(torch.randint(0,2,(n,batch),generator=g,device=device)*2-1).float()
    beta=(i0/i0_max)**(tau/max(cycles-1,1))
    step=0
    while float(i0)<=float(i0_max) and step<cycles:
        field=J@s+h
        r=2*torch.rand(n,batch,device=device,generator=g)-1
        proposed=torch.where(torch.tanh(i0*field)+nrnd*r>=0,1.0,-1.0)
        score=torch.rand(n,batch,device=device,generator=g)
        idx=torch.topk(score,n_updates,dim=0).indices
        mask=torch.zeros_like(score,dtype=torch.bool)
        mask.scatter_(0,idx,True)
        s=torch.where(mask,proposed,s)
        i0=i0/beta
        step+=1
    return s


def tapsa(J,h,cycles=1000,batch=256,gamma=0.1,delta=10.0,mean_range=4,nrnd=1.0,tau=1,seed=0):
    from collections import deque
    device=J.device; g=torch.Generator(device=device).manual_seed(seed)
    J=J.float(); h=h.float().reshape(-1,1); n=J.shape[0]
    sig=_sigma(J); i0=gamma/sig; i0_max=delta/sig
    beta=(i0/i0_max)**(tau/max(cycles-1,1))
    s=(torch.randint(0,2,(n,batch),generator=g,device=device)*2-1).float()
    win=deque(maxlen=max(1,int(mean_range)))
    step=0
    while float(i0)<=float(i0_max) and step<cycles:
        win.append(J@s+h)
        field=torch.stack(list(win),dim=0).mean(dim=0)
        r=2*torch.rand(n,batch,device=device,generator=g)-1
        s=torch.where(torch.tanh(i0*field)+nrnd*r>=0,1.0,-1.0)
        i0=i0/beta; step+=1
    return s
