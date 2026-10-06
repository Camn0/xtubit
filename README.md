# X-TUBIT Build Pack

Build-ready assembly specification and original scaffold for the X-TUBIT PKM-KC 2026 MVP.

## Scope

The target is a reproducible, auditable, decoupled pipeline:

B1 data curation -> B2 conformers -> B3 FAENet 3D representation -> B4 Bayesian GNN -> B5 qPMHI -> B6 fragment placement enumeration -> B7 QUBO/Ising construction -> B8 SB/TESB/pSA variants -> B9 decode/refine/metrics -> HITL dashboard.

This pack does **not** vendor or redistribute the complete source code of external repositories. Instead:
- `third_party/SOURCES.md` records the public projects, URLs, roles, and retrieval commands.
- `scripts/bootstrap_third_party.sh` clones and records exact commits.
- `src/xtubit/` contains original X-TUBIT code, adapters, mathematical implementations, validation harnesses, and integration contracts.
- `docs/BLUEPRINT.md` is the detailed architecture/specification.

## Recommended assembly order

1. Create the environment and clone/pin third-party repositories.
2. Run each third-party project's own examples/tests.
3. Run the internal QUBO/Ising equivalence and exact-solver tests.
4. Port/validate TESB and pSA variants against their Max-Cut references.
5. Build B6 and reproduce the published aldose-reductase T0 case before Pks13.
6. Build B1-B5 and calibrate the model only after the label table is frozen.
7. Run Pks13 retrospective tests, then expose results through the dashboard.

## Important fidelity rule

The published Yanagisawa formulation uses a connectivity **reward** (`conn = -1` when joinable, else 0) and weights `(A,B,C,D)=(1,5,5,25)`. The current manuscript's nonnegative connectivity-penalty notation is therefore an adaptation, not a literal replication. The code supports both modes, but T0 replication should use the reward mode.

## Not research claims

All numeric values in `configs/` tagged as `STARTING_POINT`, `SWEEP`, or `ENGINEERING` are assembly values, not guaranteed optimal hyperparameters. They must be validated on the stated calibration split before being used for the final report.
