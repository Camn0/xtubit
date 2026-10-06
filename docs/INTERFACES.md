# X-TUBIT module interfaces

## B1

`curate_records(records) -> candidates, labels`

Input: raw SMILES/activity rows.
Output: standardized molecular table and endpoint table.

## B2

`generate_lowest_mmff(smiles, n_confs, seed, max_iters) -> (mol, conf_id, energy)`

## B3

`make_transform(frame_averaging, fa_method)`

`forward_with_frames(batch, model, frame_averaging, crystal_task=False, mode="inference") -> dict`

Expected fields include `hidden_state` when the installed FAENet version exposes it.

## B4

`BayesianGNN(batch) -> pIC50_sample`

`model.kl() -> scalar`

`mc_predict(model, loader, samples) -> (mu, sigma_raw)`

`calibrate_sigma(mu, sigma, y) -> tau`

## B5

`qpmhi_scores(mu, sigma, qed, sa_inv, front, ref, samples, seed) -> probability[n]`

The official qPMHI repository should be wrapped behind the same interface after source audit.

## B6

Placement record:

```python
{
  "pid": str,
  "frag_id": int,
  "subregion_id": int,
  "frag_smiles": str,
  "xyz": float[N,3],
  "attachment": dict,
  "dock_score": float,
  "dE": float,
}
```

## B7

`build_yanagisawa_qubo(...) -> QuboBundle`

`qubo_to_ising(Q) -> J,h,c0`

`validate_qubo(bundle) -> diagnostic`

## B8

Every solver adapter must return:

```python
{
  "spins_or_bits": array,
  "energy_recomputed": float,
  "wall_s": float,
  "seed": int,
  "params": dict,
}
```

The caller, not the solver, owns energy verification.

## B9

`repair_onehot(bits, groups, dE)`

`heavy_atom_rmsd(pred, ref)`

`summary = summarize_solver_runs(runs)`

## Dashboard

Input only materialized result artifacts. The dashboard is a read/annotate surface, not a compute orchestrator.
