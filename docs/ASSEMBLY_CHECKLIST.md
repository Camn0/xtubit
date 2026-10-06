# X-TUBIT Assembly Checklist

## Phase 0 — freeze specification
- [ ] Freeze whether B7 is literal Yanagisawa reward mode or the manuscript's mismatch-penalty adaptation.
- [ ] Freeze pair-count convention (`upper` or `ordered`).
- [ ] Freeze calibration structures before looking at test metrics.
- [ ] Freeze pIC50 endpoint policy and censored-value policy.

## Phase 1 — environment
- [ ] Install RDKit.
- [ ] Install torch + PyG matching the CUDA wheel.
- [ ] Install FAENet and run its symmetry tests.
- [ ] Clone and pin third-party repositories.
- [ ] Build REstretto in a legacy/isolated environment because of Open Babel 2.4.1.

## Phase 2 — mathematical kernel
- [ ] Build Q from synthetic dG/clash/conn/group arrays.
- [ ] Validate Q symmetry.
- [ ] Validate QUBO-to-Ising identity at 1e-8 in float64.
- [ ] Solve N<=24 instances exactly.
- [ ] Verify one-hot penalty sweep.

## Phase 3 — solver references
- [ ] Run upstream TESB G1 example.
- [ ] Run upstream pSA/TApSA/SpSA G1 example.
- [ ] Run X-TUBIT ports with h=0 on same graph.
- [ ] Compare distributions, not a single stochastic run.

## Phase 4 — chemistry benchmark
- [ ] Prepare 2HV5 and receptor/ligand inputs.
- [ ] Run REstretto testdata.
- [ ] Reproduce 2 A subregion placement enumeration.
- [ ] Compare placement counts and score distributions.
- [ ] Build pair terms with UFF.
- [ ] Compare Hamiltonian values to the published benchmark/data where possible.

## Phase 5 — ML pipeline
- [ ] Assemble labels with assay provenance.
- [ ] Scaffold split.
- [ ] Generate conformers.
- [ ] Run FAENet hidden-state extraction.
- [ ] Train 2D baseline and 3D model.
- [ ] Calibrate sigma.
- [ ] Validate qPMHI on a toy problem.

## Phase 6 — Pks13
- [ ] Verify every PDB before counting it.
- [ ] Run 5V3Y end-to-end.
- [ ] Add calibration structures.
- [ ] Run held-out structural tests.
- [ ] Run decoy ranking test.

## Phase 7 — reporting
- [ ] Save metrics and provenance for every seed.
- [ ] Report N, placement count, valid fraction, energy, RMSD, TTS.
- [ ] Separate raw vs repaired validity.
- [ ] State all deviations from literature benchmarks.
- [ ] Do not make wet-lab or clinical efficacy claims.
