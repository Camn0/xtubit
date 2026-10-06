from __future__ import annotations
import math
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from .common.math import tts_seconds


def repair_onehot(bits, groups, dE):
    x=np.asarray(bits,dtype=int).copy(); violations=0
    for g in groups:
        on=[i for i in g if x[i]>0]
        if len(on)!=1:
            violations+=1
            for i in g: x[i]=0
            x[min(g,key=lambda i:dE[i])]=1
    return x, violations


def heavy_atom_rmsd(pred, ref):
    a=Chem.RemoveHs(pred); b=Chem.RemoveHs(ref)
    return float(rdMolAlign.CalcRMS(a,b))


def summarize_solver_runs(runs, target_p=0.99):
    best=min(r["energy"] for r in runs)
    success=sum(1 for r in runs if r["energy"]==best)/len(runs)
    median_t=float(np.median([r["wall_s"] for r in runs]))
    return {"best_energy":best,"p_success_at_best":success,
            "median_wall_s":median_t,"tts_99":tts_seconds(median_t,success,target_p)}
