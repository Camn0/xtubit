from __future__ import annotations
import itertools
import torch


def brute_force_qubo(Q: torch.Tensor):
    n=Q.shape[0]
    if n>24:
        raise ValueError("Brute force is limited to N<=24")
    best_e=None; best_x=None
    for bits in itertools.product((0.,1.), repeat=n):
        x=torch.tensor(bits,dtype=Q.dtype,device=Q.device)
        e=float(x@Q@x)
        if best_e is None or e<best_e:
            best_e=e; best_x=x
    return best_x,best_e
