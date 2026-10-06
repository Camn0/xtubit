from __future__ import annotations
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_lowest_mmff(smiles: str, n_confs=30, seed=7, max_iters=2000):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError("invalid SMILES")
    p = AllChem.ETKDGv3()
    p.randomSeed = int(seed)
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=p))
    if not conf_ids:
        raise RuntimeError("ETKDG failed to produce a conformer")
    results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=int(max_iters), numThreads=0)
    good = [(cid, float(r[1])) for cid, r in zip(conf_ids, results) if int(r[0]) == 0]
    if not good:
        good = [(cid, float(r[1])) for cid, r in zip(conf_ids, results)]
    best_cid, best_e = min(good, key=lambda t: t[1])
    for cid in list(conf_ids):
        if cid != best_cid:
            mol.RemoveConformer(int(cid))
    return mol, int(best_cid), best_e
