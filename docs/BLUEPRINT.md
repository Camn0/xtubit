# X-TUBIT Full Build Blueprint v0.1

## 0. Executive specification

### 0.1 Product

X-TUBIT is an in-silico hardware digital twin for a fragment-based flexible docking optimization problem, embedded in a decoupled computer-aided drug-design pipeline for Pks13-TE inhibitors.

The MVP must deliver:

- a standardized candidate table;
- one selected low-energy 3D conformer per candidate;
- per-atom 3D embeddings from FAENet;
- Bayesian graph predictions of enzymatic Pks13-TE affinity as `(mu, sigma)`;
- qPMHI batch scores and a deterministic top-q selection;
- fragment sets and enumerated rigid fragment placements in the known Pks13-TE pocket;
- a QUBO matrix plus a variable map that can reconstruct every binary variable's chemical meaning;
- Ising parameters `(J,h,c0)`;
- solutions from SB, TESB, and pSA-family baselines;
- constraint validity, energy, RMSD, time-to-solution, and pose-refinement outputs;
- a human-readable Streamlit dashboard that reads artifacts rather than launching heavy jobs directly.

### 0.2 Core design principle

Do not couple the discovery/uncertainty stage directly to the docking optimizer. Every stage writes immutable, versioned artifacts. This permits any downstream block to be rerun without changing upstream results.

### 0.3 System topology

```text
                    OFFLINE / DATA PLANE

 B1                    B2                 B3                  B4
Data curation -> 3D conformer -> FAENet embeddings -> Bayesian GNN
   |                 |                  |                    |
 candidates      conformers           emb/{id}.pt         mu,sigma
   |                 |                  |                    |
   +-----------------+------------------+--------------------+
                                    B5 qPMHI
                                         |
                                  selected candidates
                                         |
                                  B6 fragment poses
                                         |
                                 placements + pair terms
                                         |
                                  B7 QUBO + variable map
                                         |
                              Ising J,h / direct QUBO
                                         |
                    +--------------------+-------------------+
                    |                    |                   |
                   SB                  TESB            pSA/TApSA/SpSA
                    |                    |                   |
                    +--------------------+-------------------+
                                         |
                                      B9 decode
                                         |
                              assemble -> refine -> score
                                         |
                              result artifacts / metrics
                                         |
                                         v
                               Streamlit HITL dashboard
```

### 0.4 Execution domains

Separate the runtime into four environments even when deployed on one machine:

1. `chem`: RDKit, conformer generation, BRICS, property descriptors.
2. `ml`: PyTorch, PyG, FAENet, Bayesian predictor, qPMHI.
3. `opt`: SB/TESB/pSA, exact validation.
4. `ui`: Streamlit, 3D visualization, result browsing.

For a small team, one CUDA environment can host all four, but each CLI should keep imports lazy so a missing optional dependency does not block unrelated blocks.

---

## 1. Artifact contract

### 1.1 Required directory tree

```text
xtubit/
  configs/
    global.yaml
    b1.yaml
    b2.yaml
    b3.yaml
    b4.yaml
    b5.yaml
    b6.yaml
    b7.yaml
    b8.yaml
    b9.yaml
  data/
    raw/
      labels_raw.csv
      pool.smi
      decoys.smi
      pdb/
    processed/
      candidates.parquet
      train.parquet
      val.parquet
      test.parquet
      conformers/
      embeddings/
      placements/
      pair_terms/
      qubo/
      solver_out/
      poses/
      metrics/
  src/xtubit/
  app/
  tests/
  third_party/
```

### 1.2 Immutable artifact principle

Every artifact must contain:

- `run_id`;
- `config_hash`;
- `source_commit` when generated from third-party code;
- `software_versions`;
- random seed(s);
- input artifact hash(es).

A result without provenance metadata is not a reportable result.

### 1.3 Schemas

`candidates.parquet`:

```text
mol_id, smiles_raw, smiles_can, selfies, scaffold, qed, sa,
formal_charge, mw, logp, hbd, hba, rot_bonds, source, source_ref
```

`labels.parquet`:

```text
mol_id, assay_id, endpoint, standard_type, relation,
value, unit, value_nM, pIC50, assay_ref, series, scaffold, split
```

`embeddings/{mol_id}.pt`:

```python
{
  "mol_id": str,
  "z": LongTensor[num_atoms],
  "pos": FloatTensor[num_atoms,3],
  "h_faenet": FloatTensor[num_atoms,d],
  "fa_method": str,
  "software": dict,
}
```

`pred.parquet`:

```text
mol_id, mu, sigma_raw, sigma_cal, model_id, fold, seed
```

`selected.parquet`:

```text
mol_id, qpmhi, mu, sigma_cal, qed, sa, sainv, rank, selection_run
```

`placements/{mol_id}.parquet`:

```text
pid, frag_id, frag_smiles, subregion_id, docking_score,
dE, pose_rmsd_cluster, xyz_json, attachment_json
```

`pair_terms/{mol_id}.npz`:

```text
clash, conn, groups, active_pair_index, metadata
```

`qubo/{mol_id}.pt`:

```python
{
  "Q": FloatTensor[N,N],
  "c0_qubo": float,
  "onehot_constant": float,
  "variable_map": list[dict],
  "construction": dict,
}
```

`solver_out/{mol_id}_{solver}_{seed}.json`:

```text
bits/spins, objective_energy, hamiltonian_energy,
valid, violations, wall_s, peak_vram_mb, solver, seed, params
```

---

## 2. B1 Data curation

### 2.1 Biological endpoint

The training endpoint should be one assay family, preferably enzymatic Pks13-TE IC50. Do not mix whole-cell MIC into the same regression target. Keep assay metadata even if the final regressor uses only `pIC50`.

Define:

```text
pIC50 = 9 - log10(IC50[nM])
```

For an input in micromolar:

```text
pIC50 = 6 - log10(IC50[uM])
```

Store the original unit/value and the transformed value. Never overwrite raw measurements.

### 2.2 Public starting sources

Use:

- Aggarwal et al. 2017 for the TAM inhibitor series and Pks13-TE assay context.
- Krieger et al. 2024 for additional Pks13 inhibitor series and DEL-derived hits.
- BindingDB and ChEMBL as secondary searchable sources.
- PubChem only for compound identity/structure enrichment unless the exact assay provenance is retained.

The first pass should use exact, explicitly enzymatic measurements with equality relation (`=`). Censored values (`<`, `>`) should either be excluded from the regression target or handled in a separate censored-likelihood model. Do not silently turn them into equalities.

### 2.3 Structure standardization

Pipeline:

```text
SMILES parse
  -> cleanup
  -> largest fragment
  -> uncharge if justified
  -> sanitize
  -> canonical isomeric SMILES
  -> descriptors
  -> SELFIES
```

Do not collapse stereochemistry unless the source record is itself achiral/unspecified. A stereoisomer that is biologically meaningful must remain distinct.

### 2.4 Candidate filters

QED and SA are multi-objective descriptors, not training-label filters by default. A better separation is:

- hard chemical validity filter: parse/sanitize, allowed elements, maximum size, formal charge policy;
- soft prioritization objective: QED and `SA_inv`.

This avoids biasing the training distribution by the same objectives later used in qPMHI.

Starting engineering filters:

```yaml
max_heavy_atoms: 80
allowed_elements: [B,C,N,O,F,P,S,Cl,Br,I]
qed_floor: 0.0
sa_ceiling: 10.0
```

Do not claim those are scientifically optimal. They merely prevent pathological input.

### 2.5 Scaffold split

Use Bemis-Murcko scaffolds. All molecules with the same scaffold must remain in the same split.

Preferred split:

```text
train 70-80%
validation 10-15%
test 10-15%
```

The most important invariant is chemical-series separation, not the exact percentage.

If the final labeled set contains fewer than roughly 100 explicit enzymatic labels or fewer than 5 meaningful held-out scaffold groups, switch the report wording to “supporting ranking with uncertainty” rather than a generalizable predictive model. This is an engineering reporting gate, not a published cutoff.

### 2.6 Seed label table

The first smoke-test label table can be assembled from the published TAM values in Aggarwal et al. 2017:

```text
TAM1  0.26 uM
TAM2  0.12 uM
TAM3  0.24 uM
TAM4  0.28 uM
TAM5  0.71 uM
TAM11 19.6 uM
TAM12 0.29 uM
TAM13 0.17 uM
TAM14 35.8 uM
TAM15 2.0 uM
TAM16 0.19 uM
TAM17 0.36 uM
```

These values are useful for unit-conversion tests, but they are far too small for a serious uncertainty model by themselves.

Krieger et al. 2024 gives X20403 at 57 nM in a Pks13-TE enzyme assay. Keep exact assay source metadata when importing it.

---

## 3. B2 3D conformers

### 3.1 Reference implementation

Use RDKit ETKDGv3 followed by MMFF minimization.

```text
SMILES -> add H -> ETKDGv3 -> multiple conformers -> MMFF -> minimum energy conformer
```

Starting values:

```yaml
num_conformers: 30
random_seed: 7
mmff_max_iters: 2000
```

These are only starting points.

### 3.2 Required checks

- sanitize before embedding;
- record failed embeddings;
- verify the conformer count actually generated;
- keep the conformer ID and MMFF energy;
- verify that all coordinates are finite;
- run the procedure twice with the same seed and compare coordinate graphs and energy ordering.

### 3.3 Chemical limitation

Using one lowest-MMFF conformer is a deliberate simplification. It does not imply that the bound conformation is the gas-phase minimum. This uncertainty belongs in the limitation section and motivates a future multi-conformer extension.

---

## 4. B3 FAENet

### 4.1 Public implementation facts

FAENet provides a PyTorch-Geometric `FrameAveraging` transform and `model_forward`. The documentation states that the transform generates `fa_pos`, while `model_forward` performs the frame-aware forward pass. The documented output includes graph-level predictions and a `hidden_state` containing final atom-level representations. Frame averaging options include 3D/2D/DA and stochastic/deterministic/all/SE(3) modes. The package is principally a materials-modeling implementation, so molecular transfer is an explicit adaptation that must be validated.

### 4.2 Data contract

Input:

```text
atomic_numbers [N]
pos [N,3]
batch [N]
```

After transform:

```text
fa_pos
fa_rot
batch
```

Call the FAENet model through `model_forward`, not by directly calling `model(data)` when frame averaging is enabled.

### 4.3 Suggested initialization

```yaml
cutoff: 6.0
hidden_channels: 128
num_filters: 128
num_gaussians: 50
num_interactions: 4
phys_embeds: true
frame_averaging: "3D"
fa_method: "stochastic"
```

These follow the public documentation defaults/representative settings and are not guaranteed to be optimal for drug-like ligands.

### 4.4 Feature extraction

Use `preds["hidden_state"]` from `model_forward` as the per-atom feature if the installed FAENet version returns that field. Confirm the exact shape with a one-molecule smoke test.

If a version does not expose `hidden_state`, then use the last interaction block through a documented public module or a small internal wrapper. Do not depend on fragile private attribute names until the installed commit is pinned.

### 4.5 Fine-tuning strategy

Use three modes:

```text
FA-only-frozen -> train BGNN head
FA+top-blocks -> unfreeze late FAENet layers
FA+full        -> unfreeze all layers if data size permits
```

Ablation required:

```text
2D/BGNN baseline
3D/BGNN + FAENet frozen
3D/BGNN + FAENet fine-tuned
```

### 4.6 Symmetry test

For each selected molecule, construct random rotations `R` and translations `t`:

```text
x' = x R^T + t
```

The final graph-level prediction must satisfy:

```text
|f(x') - f(x)| < tolerance
```

For stochastic FA, compare repeated distributions/means rather than demanding bitwise identity.

---

## 5. B4 Bayesian GNN

### 5.1 Goal

Predict enzymatic pIC50 and a calibrated epistemic uncertainty proxy:

```text
mu(x) = E[pIC50 | x,D]
sigma(x) = Std[pIC50 | x,D]
```

### 5.2 Architecture

Recommended graph:

```text
per-atom RDKit features
       +
per-atom FAENet hidden_state
       |
       v
linear projection to d=128
       |
3 x graph message-passing layers
       |
global add/mean pooling
       |
Bayesian MLP head
       |
pIC50
```

Atom feature minimum:

```text
atomic number
formal charge
degree
aromatic flag
hybridization
ring flag
implicit H count
```

Edge feature minimum:

```text
bond type
conjugation
ring flag
stereo class (optional)
```

### 5.3 Bayesian layer

A modern implementation should use reparameterized Bayesian linear layers rather than pinning the whole project to an old `torchbnn` release. Each weight has mean and log-scale parameters. At forward time:

```text
w = mu_w + sigma_w * epsilon
b = mu_b + sigma_b * epsilon
```

with a diagonal Gaussian prior. The training loss is:

```text
L = MSE + kl_weight * KL(q(w)||p(w)) / N_train
```

Start with:

```yaml
hidden: 128
layers: 3
prior_sigma: 0.1
kl_weight: 0.01
lr: 1e-3
batch: 32
epochs: 300
```

Sweep `kl_weight` across at least `1e-3, 3e-3, 1e-2, 3e-2, 1e-1`.

### 5.4 Monte Carlo inference

For each molecule:

```text
T = 50 stochastic forward passes
mu = mean(y_t)
sigma_raw = std(y_t)
```

Report sensitivity at `T=20,50,100` on the validation set. Do not assume more samples automatically improves calibration.

### 5.5 Calibration

Fit one positive scale factor `tau` on validation:

```text
sigma_cal = tau * sigma_raw
```

Minimize Gaussian NLL:

```text
0.5 * [ ((y-mu)/sigma_cal)^2 + 2 log(sigma_cal) ]
```

Report empirical coverage for nominal 50%, 80%, and 90% intervals.

### 5.6 Fallback

If Bayesian weight sampling is unstable, use MC dropout or a 5-member deep ensemble. State the actual method used. Never call deterministic dropout “Bayesian inference” without qualification.

---

## 6. B5 qPMHI

### 6.1 Objective vector

All objectives must be maximized:

```text
f1 = pIC50
f2 = QED
f3 = SA_inv
```

Recommended normalized accessibility score:

```text
SA_inv = clip((10 - SA)/9, 0, 1)
```

Do not change the reference point after looking at the blind test.

### 6.2 Concept

For each candidate, Monte Carlo-sample the posterior affinity and determine how often that candidate has the maximum hypervolume improvement among the pool.

The candidate-level probability estimate is:

```text
p_i = count(i is argmax HVI) / S
```

The batch is obtained by sorting `p_i` and taking the top q candidates. This is the scalable decomposition used by qPMHI.

### 6.3 Correct dependency ordering

qPMHI belongs after B4 and before docking. It does not need docking oracle calls. The entire candidate pool can be processed once.

### 6.4 Hypervolume implementation

Use a trusted exact hypervolume implementation such as BoTorch's multi-objective utilities for small/medium pools. For large pools, use the official qPMHI implementation after cloning it and benchmark the result against a small exact implementation.

### 6.5 Numerical guardrails

- Replace zero sigma by a small floor, e.g. `1e-6`.
- Ensure all objective directions are maximization.
- Validate Pareto dominance on a toy set.
- Check that candidates with zero HVI do not become selected merely from indexing errors.

### 6.6 Candidate count strategy

If the pool contains tens of thousands of molecules, do not compute a Python nested hypervolume loop over every candidate and every sample. Use the official qPMHI GPU implementation or batch/vectorize the per-candidate Monte Carlo calculation. A prefilter can be used only if it is frozen before final testing.

---

## 7. B6 Fragmentation and placement generation

This is the highest-risk chemical engineering block.

### 7.1 Reference formulation

The benchmark follows Yanagisawa et al.'s fragment-QUBO docking construction.

Four concepts are needed:

```text
protein-fragment score dG_i
placement-pair clash c_ij
placement-pair connection reward conn_ij
one-placement-per-fragment constraint
```

### 7.2 Fragmentation

X-TUBIT's planned MVP uses RDKit BRICS rather than the benchmark paper's Spresso/LigPrep workflow.

For each ligand:

```text
SMILES
 -> BRICS.FindBRICSBonds
 -> FragmentOnBonds(addDummies=True)
 -> fragment graph + attachment metadata
```

You must retain a mapping from each original bond to the two fragment-side attachment atoms. Do not attempt to infer that mapping later from geometry.

### 7.3 Pocket definition

For the main Pks13 test, build a receptor model from the PDB entry and define the docking box from the co-crystal ligand's heavy-atom coordinates.

The crystal ligand coordinates are allowed to define the **search box**. They must not be used as the coordinates of regenerated fragment poses.

### 7.4 Placement enumeration

Published benchmark settings:

```text
subregion size: 2 A x 2 A x 2 A
POSE_RMSD: 1.0 A
POSES_PER_LIG: 20 per fragment per subregion
OUTPUT_SCORE_THRESHOLD: 0.0 kcal/mol
NO_LOCAL_OPT: True
```

The benchmark example used about 343 subregions and produced 3005 placements in the aldose reductase case.

These values should be placed in a config file, not hard-coded.

### 7.5 REstretto

REstretto is the closest public implementation of the fragment-placement engine used by Yanagisawa et al. It is C++ and has an Open Babel 2.4.1 dependency. Use its own testdata as the first B6 smoke test.

The exact names of several paper-specific configuration flags should be checked against the pinned source before use. The repository README exposes the core box, search, grid, receptor, ligand, memory, and output options.

### 7.6 Fallback docking

If REstretto cannot be built, use AutoDock Vina via Python plus Meeko as a documented deviation.

For each fragment/subregion:

```text
prepare receptor PDBQT
prepare fragment PDBQT
set box center and size
compute map
run rigid docking
retain diverse negative-score poses
cluster by RMSD
```

The fallback changes the placement-generation distribution and must be reported.

---

## 8. B6 pair functions using UFF

### 8.1 Required quantities

For each placement pair `(i,j)` belonging to different fragments:

```text
E_NB = interaction energy without covalent bond
E_B  = interaction energy after adding the original covalent bond
```

The exact benchmark method computes these with RDKit UFF and applies:

```text
th_E = 500 kcal/mol
```

### 8.2 Connectivity reward

For fragments that are chemically bonded in the parent molecule:

```text
conn_ij = -1   if E_B <= th_E
          0    otherwise
```

For fragments not bonded in the parent molecule:

```text
conn_ij = 0
```

### 8.3 Clash indicator

```text
clash_ij = 1   if conn_ij == 0 and E_NB > th_E
            0   otherwise
```

This creates a deliberately coarse clash indicator. The 500 kcal/mol threshold is not a molecular mechanics energy cutoff that should be interpreted physically; it is a benchmark engineering rule from a published coarse-placement experiment.

### 8.4 Pair-screening optimization

The benchmark computed all pairs. For a scalable implementation, first screen pairs by a conservative minimum heavy-atom distance, e.g. 8 A, then calculate UFF only for pairs that survive. The distance cutoff is an optimization and not part of the published Hamiltonian. Record it in the config and in the run manifest.

### 8.5 Attachment geometry

This is the hardest B6 detail.

For every BRICS bond retain:

```text
parent bond id
fragment A atom id
fragment B atom id
attachment labels
allowed bond type
expected connection distance range
```

Do not create a bond between two placements merely because their nearest heavy atoms are close. First check that their fragment identities correspond to the same original bond and that the attachment atoms are the correct pair.

---

## 9. B7 QUBO formulation

### 9.1 Published four-term structure

Let `x_i in {0,1}` indicate whether placement i is active. Let `F_k` be the set of placements belonging to fragment k.

Use:

```text
H = A H1 + B H2 + C H3 + D H4
```

where

```text
H1 = sum_i dG_i x_i
H2 = sum_{i<j} clash_ij x_i x_j
H3 = sum_{i<j} conn_ij  x_i x_j
H4 = 1/2 sum_k (sum_{i in F_k} x_i - 1)^2
```

The second binary-only term in the original paper's expanded H4 is zero for valid binary variables and need not be implemented separately when x is already binary.

Published starting weights:

```text
A = 1
B = 5
C = 5
D = 25
```

The relative rule is that `B` and `C` should be similar in scale and `D` should dominate the constraint.

### 9.2 Matrix construction for `E = x^T Q x`

For the upper-triangle pair convention above:

```text
Q_ii = A dG_i - D/2

Q_ij = 1/2 (B clash_ij + C conn_ij)
       + D/2 * I[group(i) = group(j)]     for i != j
```

Then:

```text
E_Hamiltonian(x) = x^T Q x + D/2 * number_of_fragments
```

The constant must be logged.

### 9.3 Why the manuscript must be corrected

The manuscript currently writes the connection term as a nonnegative mismatch penalty `b_ij >= 0`. That is not the literal Yanagisawa reward formulation. For a replication claim, use `conn in {-1,0}`. For the current mismatch-penalty idea, label it as an X-TUBIT adaptation and use a different symbol such as `b_mismatch` so it cannot be confused with the published `conn`.

### 9.4 Pair-counting ambiguity

The paper's notation sometimes writes sums over `(i,j)` rather than explicitly `i<j`. Therefore the implementation supports two modes:

```text
upper: each physical pair enters H2/H3 once
ordered: a symmetric pair is entered twice
```

Use `upper` as the clean internal convention. In T0, compare both against the published Hamiltonian/reference data and document which one reproduces the reported values.

### 9.5 Ising mapping

Let:

```text
x = (s + 1)/2,     s_i in {-1,+1}
Q_o = Q - diag(diag(Q))
```

Then:

```text
J   = -1/2 Q_o
h   = -1/2 Q 1
c0  = 1/2 tr(Q) + 1/2 sum_{i<j} Q_ij
```

and

```text
E = -1/2 s^T J s - h^T s + c0
```

This identity must be numerically checked for random bitstrings on every software change to the matrix builder.

### 9.6 QUBO sparsity

The conceptual Q is dense because the one-hot term connects all placements belonging to the same fragment. Pair interaction terms can be sparse after distance screening.

Store both:

```text
Q_dense.pt      # solver input for moderate N
Q_edges.npz     # sparse audit representation
```

Report `N = total placement variables` in every experiment.

---

## 10. B8 SB solver

### 10.1 Baseline

Use the public `simulated-bifurcation` package. It exposes QUBO and Ising solving through PyTorch and GPU acceleration.

Because solver API conventions and return shapes can change, the adapter must:

1. introspect the installed package version;
2. solve a 2-variable toy QUBO;
3. recompute energy from returned bits using X-TUBIT's own function;
4. reject the adapter if the mapping is inconsistent.

### 10.2 Fair timing

Use the same:

```text
GPU
matrix N
seed set
stop criterion
number of independent trials
```

for every solver.

When measuring CUDA execution:

```python
torch.cuda.synchronize()
t0 = perf_counter()
solve()
torch.cuda.synchronize()
wall = perf_counter() - t0
```

Always record whether startup/model-build time is included.

### 10.3 Recommended starting sweep

```text
agents: 128, 256, 512
max_steps: 2k, 5k, 10k
mode: ballistic, discrete
early_stopping: false for first fairness benchmark
```

The default package early stopping should not be silently inherited if the experiment is intended to compare fixed iteration budgets.

---

## 11. B8 TESB

### 11.1 Public algorithm semantics

The CPU `tSB.py` implementation is the correctness reference because it supports both a linear field and a separate tabu matrix.

Its effective field is:

```text
field = A @ state + h - tabu_term
```

with a pump schedule:

```text
p(t) linearly from 0 to 1
```

and inelastic walls at `|x| > 1`.

Two update modes exist:

```text
ballistic: A @ x
 discrete: A @ sign(x)
```

The tabu list is an `N x T` matrix of prior spin solutions. When more than one tabu vector is stored, the implementation samples/averages tabu columns and subtracts the average field.

### 11.2 Two-stage procedure

Reference pattern:

```text
Stage 1: plain SB warm-up
Stage 2: SB + tabu list generated from warm-up states
```

The exact G1 values in the paper repository (`1000` warm-up steps, `100` batch, then approximately `9000` checking steps and G1-specific `xi`) are benchmark values for G1, not docking defaults.

### 11.3 Docking adapter

Do not use the GPU repository's `h` parameter as a linear field because its meaning differs: in that file `h` is the tabu matrix. Use the CPU semantics to build an independent GPU adapter with a separately named `linear_field` argument.

### 11.4 Parameter sweep

At minimum sweep:

```text
n_iter
xi
num_tabu
warm_batch
number of tabu samples retained
ballistic vs discrete
```

The published G-set values are not transferable to the docking QUBO.

---

## 12. B8 pSA / TApSA / SpSA

### 12.1 Why this baseline exists

The p-bit family provides a probabilistic-spin comparison whose state updates are distinct from oscillator-inspired SB.

### 12.2 Field

Use:

```text
I = J s + h
```

for the X-TUBIT generalized formulation.

### 12.3 pSA schedule

The pSA implementation uses an inverse-temperature-like parameter `I0` and a noise term. The public TApSA and SpSA implementations can be treated as the two reference variants:

- TApSA: time-average the recent local field before the stochastic threshold;
- SpSA: update only a random subset of p-bits while others are stalled.

The public implementation uses a row-variance-derived scale `sigma`, a ramp from `gamma/sigma` to `delta/sigma`, and a tunable `tau`.

### 12.4 X-TUBIT GPU port

The original repositories target Max-Cut and do not natively accept a linear field. The X-TUBIT adapter therefore implements:

```text
I = J @ s + h
```

and then applies the published stochastic update logic.

Correctness criterion: reproduce the public code's Max-Cut behavior with `h=0` before testing a docking matrix.

### 12.5 Starting values

Use public-code defaults as the initial point:

```text
gamma = 0.1
delta = 10
stall_prop = 0.5      # SpSA
mean_range = 4        # TApSA
nrnd = 1
tau = 1
cycles = 1000
```

These are not docking-optimal values. Sweep them.

---

## 13. B9 decoding and chemistry

### 13.1 Spin-to-bit

```text
x = (s + 1)/2
```

### 13.2 Constraint handling

For each fragment:

```text
0 active placements -> choose fallback
1 active placement  -> keep
>1 active placements -> keep a deterministic best placement and record violation
```

For unbiased evaluation, additionally report a strict metric where any violating solution is rejected instead of repaired.

Recommended dual reporting:

```text
raw-valid-fraction
repaired-success-fraction
```

### 13.3 Fragment assembly

Assembly must follow the original fragment bond map, not nearest-neighbor geometry.

Algorithm:

```text
select placement for each fragment
 -> create transformed fragment mols
 -> map attachment atoms
 -> add original bond types
 -> remove dummy atoms
 -> sanitize
```

If sanitization fails, store the failure reason and the assembled unsanitized structure for debugging.

### 13.4 Refinement

Use a clearly separated refinement step. The default MVP can use RDKit MMFF/UFF on the ligand with the receptor fixed, but RDKit alone is not a protein force-field engine. A better optional refinement environment is OpenMM with a frozen receptor and an explicit protein/ligand force-field setup.

Do not compare a pre-refinement QUBO energy in kcal/mol directly to a post-refinement molecular mechanics energy unless the units, force field, and energy definition are identical.

### 13.5 RMSD

For a reference complex:

- use heavy atoms;
- use symmetry-aware atom mapping where appropriate;
- do not silently re-align away a translational/rotational error that the pose generator should have solved if the evaluation intends absolute placement;
- report both raw and symmetry-aware variants if needed.

Success thresholds:

```text
<= 2.0 A  primary X-TUBIT criterion
<= 2.5 A  comparison to the Yanagisawa benchmark
```

### 13.6 Time-to-solution

For target success probability `p*`:

```text
TTS = t_run * log(1-p*) / log(1-p_success)
```

with `p* = 0.99` as a common choice.

If `p_success=0`, TTS is undefined/infinite. Report that rather than assigning an arbitrary cap.

---

## 14. Retrospective validation

### 14.1 T0: aldose reductase

Before Pks13, reproduce the published fragment-QUBO pipeline on PDB 2HV5.

Target sanity values from the publication/presentation:

```text
~4 fragments
~3005 placements
~1.26 A best pose before conventional minimization
~0.27 A after conventional minimization
~74% valid solutions in the reported SQBM+ workflow
```

Do not demand an exact match because the X-TUBIT implementation intentionally differs in fragmentation, conformer generation, molecular mechanics, and optimizer.

### 14.2 Pks13 pose cases

Primary:

```text
5V3Y  TAM16
```

Related structures for calibration/test:

```text
5V40  TAM6
5V41  TAM5
5V42  TAM3
8TQG  X20419
8TQV  X20403 / JS9
8TR4  X20404
```

Treat related TAM structures as related cases, not independent biological replicates.

`8TRY` remains a verification task in the current working inventory and should not enter the final test count until its ligand identity and coordinates are confirmed.

### 14.3 Calibration/test separation

Recommended:

```text
calibration: 5V40, 5V41 (+ any independently verified additional structure)
test:        5V3Y, held-out 2024 structure(s)
```

Do not tune QUBO weights on 5V3Y and then report 5V3Y as an unbiased test.

### 14.4 Blind decoy experiment

Construct a Pks13-specific active/decoy ranking test. Decoys should be property-matched to actives on:

```text
MW
logP
HBD
HBA
rotatable bonds
formal/net charge
```

A DUD-E-style protocol is a reference for property matching, not a proof that the custom Pks13 set is statistically perfect.

Report:

```text
hits@k
hit rate
EF_k = (hits_k/k)/(hits_all/n)
```

Fix the ranking score before viewing the test results.

---

## 15. Experiment registry

Every run gets a UUID-like `run_id`:

```text
YYYYMMDD_solver_model_case_seed
```

Manifest fields:

```yaml
run_id:
git_commit:
config_hash:
data_hashes:
python:
torch:
cuda:
rdkit:
solver_version:
seed:
case:
N_variables:
num_fragments:
```

Store raw stdout/stderr for solver binaries under `logs/`.

---

## 16. Dashboard

### Page 1: Candidates

Columns:

```text
ID | mu | sigma | qPMHI | QED | SA | scaffold | rank
```

Plots:

```text
QED vs mu
SA_inv vs mu
sigma vs mu
Pareto front
```

### Page 2: Docking

Show:

```text
ligand
fragment decomposition
placement count
QUBO size N
solver
best energy
validity
RMSD
TTS
```

### Page 3: Pose

Display protein, reference ligand if available, and predicted ligand. Provide an explicit “raw” versus “refined” toggle.

### Page 4: Solver comparison

Show:

```text
best energy per seed
median wall time
success rate
valid fraction
TTS
```

### HITL log

An accept/reject event must record:

```text
user/action/time/candidate/run_id/reason
```

The UI must never mutate an upstream artifact.

---

## 17. Tests and acceptance gates

### Gate G0 environment

- RDKit import
- PyTorch CUDA visibility
- PyG import
- optional solver imports
- source commit manifest generated

### Gate G1 B1

- 0 invalid output SMILES
- train/test scaffold sets disjoint
- no duplicate standardized molecules across splits

### Gate G2 B2

- deterministic repeat under fixed seed
- no NaN coordinates
- every selected conformer sanitizes

### Gate G3 B3

- random rotation/translation invariance of final prediction
- hidden-state shape stable
- no unexpected frame-count mismatch

### Gate G4 B4

- baseline without FAENet trains
- FAENet variant trains
- calibration coverage logged
- ablation stored under same split

### Gate G5 B5

- toy qPMHI agrees with brute-force candidate probability counts
- reference point fixed
- objective direction correct

### Gate G6 B7

- `Q == Q.T`
- Ising equivalence for all states on N<=20 random instances
- exact optimum match on small random problems
- one-hot states tested against zero/two-hot states under D sweep

### Gate G7 B8

- SB returns interpretable bits
- TESB matches tSB reference on G1 within expected stochastic spread
- TApSA/SpSA match their Max-Cut reference under fixed settings

### Gate G8 B9

- known-good synthetic assembly has zero/self-consistent RMSD
- invalid structures are counted, not dropped silently

### Gate G9 T0

- placement count sanity band
- 2HV5 docking completes
- output can be compared with published target values

---

## 18. Failure modes and remedies

| Failure | Likely cause | Remedy |
|---|---|---|
| FAENet shape mismatch | package version/API drift | pin commit; smoke-test `model_forward` |
| BGNN overfits | tiny label set | freeze FAENet, reduce capacity, ensemble, report ranking-only |
| sigma collapses to 0 | prior/KL too strong/weak or bad scaling | tune KL, standardize target, sigma floor |
| qPMHI too slow | Python HVI loop | vectorize or official qPMHI implementation |
| too many placements | box too large/fragment too flexible | subregion/pose budget sweep, cluster aggressively |
| Q matrix too large | N explosion | sparse pair storage, placement cap, per-fragment top-M |
| solver violates one-hot | finite penalty | sweep D, repair and strict-valid metrics |
| TESB behaves poorly | copied G1 xi | re-scale/sweep xi for each Q |
| pSA sign error | Max-Cut convention mismatch | verify with exact optimum and G1 |
| assembly sanitization failure | dummy/attachment mapping | preserve parent bond map from B6 |
| refinement energy nonsense | inconsistent units/force field | separate metrics and units |

---

## 19. Security/reproducibility

- Never execute untrusted molecular files without sanitization.
- Never let dashboard actions modify model or data artifacts.
- Freeze data snapshots before test runs.
- Hash every generated matrix.
- Save source commit hashes and package lockfile.
- Use a container for the legacy Open Babel 2.4.1 REstretto stack if the host environment cannot coexist with modern RDKit.

---

## 20. Four-month build plan

### Month 1

Environment, third-party demos, B1, B2, FAENet smoke test, data snapshot.

### Month 2

B4, B5, dashboard skeleton, B6/B7 prototype, exact solver harness.

### Month 3

REstretto/placement pipeline, T0, SB/TESB/pSA integration, memory/speed profiling.

### Month 4

Pks13 retrospective cases, blind decoy test, dashboard integration, reporting, screenshots, reproducibility bundle.

The project should be considered MVP-complete when T0 passes and at least one Pks13 structure runs end-to-end from standardized ligand to pose/metrics artifact.

---

## 21. Claims boundary

The system is a computational screening aid. It does not establish biochemical activity, MIC, pharmacokinetics, toxicity, combination efficacy, or clinical utility. Pose RMSD is a retrospective structural metric, not a biological validation. Solver speed gains are only claims relative to the explicitly tested solver configurations unless a conventional docking benchmark is also performed.
