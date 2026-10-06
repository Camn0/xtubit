# X-TUBIT Experimental Protocol

## A. Frozen-before-test items

The following must be frozen before looking at held-out Pks13 results:

1. activity endpoint and unit conversion;
2. scaffold split;
3. candidate hard filters;
4. qPMHI reference point;
5. QUBO pair convention;
6. connectivity representation (`reward` vs `mismatch_penalty`);
7. QUBO calibration complex set;
8. solver seed set;
9. primary ranking score for the blind test;
10. RMSD success threshold.

## B. ML experiment matrix

### B1 baseline

2D GNN, no FAENet features.

### B2 representation ablation

Same graph predictor with frozen FAENet hidden_state concatenated to atom features.

### B3 fine-tuned model

Same as B2, unfreeze the final one or two FAENet interaction blocks.

Report:

```text
RMSE
MAE
Gaussian NLL
50/80/90% coverage
median sigma
peak VRAM
seconds/batch
```

Repeat over at least three random seeds if resources permit.

## C. Bayesian uncertainty sweep

```text
KL weight: 0.001, 0.003, 0.01, 0.03, 0.1
MC samples: 20, 50, 100
```

Select the configuration using validation NLL and coverage, never test performance.

## D. qPMHI sweep

```text
S: 256, 512, 1024
q: 5, 10, 20, 40 depending on candidate pool size
prefilter: off, frozen prefilter if required for scale
```

For the paper/PKM result, choose one predeclared q. Use the other q values as sensitivity analysis.

## E. Placement-generation sweep

Start with the literature-inspired configuration:

```text
subregion = 2 x 2 x 2 A
poses_per_fragment_per_subregion = 20
cluster RMSD = 1.0 A
keep only docking scores < 0
```

Then test one controlled sensitivity axis at a time:

```text
10, 20, 40 poses
1.0, 1.5, 2.0 A cluster threshold
2 A vs 3 A subregion
```

Do not mix parameter changes in the main benchmark.

## F. QUBO calibration sweep

Start from:

```text
A=1, B=5, C=5, D=25
```

For the exact Yanagisawa H4 definition, this means the constraint term is `D/2 * (sum x - 1)^2`.

Sweep:

```text
B/C ratio: 0.5, 1, 2
D: 10, 25, 50, 100
```

Only use calibration complexes for the choice.

## G. Solver benchmark

For each case and seed:

```text
SB
TESB
pSA
TApSA
SpSA
```

Recommended common reporting fields:

```text
solver
seed
N
num_fragments
num_placements
wall_s
best_energy
median_energy
energy_gap_to_exact (small N only)
valid_fraction
repair_fraction
strict_valid_fraction
RMSD_raw
RMSD_refined
success_2.0A
success_2.5A
```

For full-size docking, exact optimum is unavailable. Report “best-known energy” and seed attainment rate, not “optimality” unless independently proven.

## H. T0 benchmark protocol

1. Prepare PDB 2HV5.
2. Recover reference ligand and pocket box.
3. Fragment according to the X-TUBIT/BREAKING choice.
4. Enumerate 2 A subregions.
5. Generate placements.
6. Build `dG`, `clash`, `conn`.
7. Build Q.
8. Solve with SB/TESB/pSA variants.
9. Repair only after recording raw validity.
10. Reassemble.
11. Calculate pre-refinement RMSD.
12. Refine with a frozen-protein protocol.
13. Calculate post-refinement RMSD.
14. Compare placement count/energy distributions against the published benchmark and Zenodo data.

## I. Pks13 protocol

### Main test

5V3Y / TAM16.

### Calibration candidates

5V40 / TAM6, 5V41 / TAM5, 5V42 / TAM3 and any additional independently verified complex.

### Additional test candidates

8TQG, 8TQV, 8TR4; verify 8TRY before use.

### Blinding rule

The solver/hyperparameter selection must be locked before reporting a Pks13 test score. A structure used to tune B/C/D is not a clean test structure.

## J. Decoy protocol

For each active, sample multiple inactive/decoy candidates and match approximately on molecular properties.

Recommended matching features:

```text
MW
logP
HBD
HBA
rotatable bonds
formal/net charge
```

Report the exact number of actives, decoys, and hits at each k.

Do not infer statistical significance from a small retrospective set.

## K. Resource accounting

For every heavy stage record:

```text
CPU model
GPU model
GPU memory
CUDA version
RAM
wall time
CPU-hours
peak VRAM
N placements
N pair computations
```

For REstretto, also report the number of subregions and fragment jobs so CPU cost can be reconstructed.

