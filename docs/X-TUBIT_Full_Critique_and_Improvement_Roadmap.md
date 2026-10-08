# X-TUBIT — Full Technical, Scientific, Product, and Commercial Critique

**Assessment date:** 2026-10-08  
**Repository assessed:** `Camn0/xtubit`  
**Assessment basis:** the supplied README plus inspection of the current public repository, including the executable pipeline, core modules, tests, experimental protocol, blueprint, and licensing files.

---

# 1. Executive Verdict

## The blunt answer

**X-TUBIT is not worthless.**

It is also **not currently a mature drug-discovery product**, and the present repository is not yet strong enough to justify a large software valuation.

The project has a genuinely interesting research thesis:

> Combine molecular CADD, uncertainty-aware prioritization, discrete flexible docking, QUBO/Ising formulations, and physics-inspired combinatorial solvers into a reproducible "hardware digital twin" workflow.

That is unusual enough to attract a specialist research audience.

However, the current implementation is still best described as:

> **an ambitious research scaffold / prototype with several real mathematical and software components, plus a partially simulated end-to-end demonstration.**

It is **not yet**:

> a validated generalized Pks13-TE docking/screening platform.

That distinction is the central issue.

---

# 2. Overall Scorecard

| Dimension | Current score | Comment |
|---|---:|---|
| Research concept | 7.5/10 | Strong interdisciplinary idea |
| Technical ambition | 8/10 | Unusually broad scope for a small project |
| Exact mathematical kernel | 7/10 | Some good unit tests and algebraic care |
| Chemistry realism | 4/10 | Several stages are still simplified or mocked |
| ML validity | 3/10 | Headline architecture exceeds current pipeline |
| Docking validity | 3/10 | Current main path is not yet generalized flexible docking |
| Solver research value | 6/10 | Interesting, but benchmark discipline still needs work |
| Reproducibility | 5/10 | Protocol exists, execution and provenance lag behind |
| Scientific evidence | 3/10 | Not enough independent benchmark evidence yet |
| Software engineering | 6/10 | Reasonably structured, but prototype-stage |
| Product readiness | 2.5/10 | Not yet a product |
| Market size | 2/10 | Extremely niche |
| Supply scarcity | 7/10 | Few projects have this exact combination |
| Demand | 2/10 | Few buyers need this exact combination today |
| Current sale value | Low thousands | Niche research asset, not enterprise software |
| Upside after validation | High | Can move into research software/service territory |

---

# 3. What X-TUBIT Actually Is

The README presents X-TUBIT as a nine-block pipeline:

```text
Data
  -> conformers
  -> geometric features
  -> Bayesian affinity
  -> active selection
  -> fragment placements
  -> QUBO / Ising
  -> Ising solvers
  -> reconstruction / evaluation
```

This is a good architecture conceptually.

The strongest design decision is the separation between:

1. **chemical / ML prioritization**, and
2. **combinatorial docking optimization**.

The immutable-artifact concept in the blueprint is also good because it makes downstream experimentation repeatable.

The project is especially interesting as a **systems-research platform**, because it is not merely "yet another docking script." It is trying to answer a deeper question:

> Can structured combinatorial optimization formulations and physics-inspired solvers offer a useful advantage for flexible molecular search?

That is a legitimate research question.

The problem is that the current repository has not yet demonstrated the answer.

---

# 4. Strongest Parts of the Project

## 4.1 The problem framing is more interesting than ordinary docking

The combination of:

- molecular representation learning,
- uncertainty estimates,
- multi-objective candidate acquisition,
- fragment-placement combinatorics,
- QUBO/Ising optimization,
- stochastic solver comparison,
- and a human-in-the-loop interface

is legitimately distinctive.

Even if the final solver advantage turns out to be small or nonexistent, the benchmark itself could still be useful.

---

## 4.2 Pks13-TE is a defensible scientific target

This is not an invented target chosen purely to make the project sound impressive.

Pks13 is a biologically important mycolic-acid biosynthesis target, and there is continued medicinal-chemistry work around Pks13/Pks13-TE.

Relevant examples include:

- Aggarwal et al. 2017, including TAM16 and structural work on Pks13-TE.
- Krieger et al. 2024, which expanded the chemical space through DNA-encoded-library screening.
- Subsequent 2025–2026 Pks13 inhibitor studies, including newer chemotypes and efforts addressing liabilities.

This gives X-TUBIT a much better scientific story than a completely arbitrary benchmark target.

---

## 4.3 The repository structure is fairly mature for a prototype

The project already has:

```text
src/
tests/
docs/
configs/
data/
third_party/
app/
```

and the blueprint explicitly defines artifact types.

That matters.

Many academic prototypes are essentially one notebook and a slide deck. X-TUBIT is substantially more organized than that.

---

## 4.4 The QUBO / Ising boundary is one of the strongest parts

The current repository explicitly tests:

- one-hot penalty expansion,
- QUBO symmetry,
- QUBO-to-Ising equivalence.

That is good engineering practice.

The code also documents the convention carefully, which is particularly important because factor-of-two mistakes are common in QUBO implementations.

The `test_qubo_mapping.py` and `test_onehot_formula.py` tests are therefore useful foundations.

---

## 4.5 The experimental protocol is much better than the executable pipeline

`docs/EXPERIMENT_PROTOCOL.md` is actually one of the stronger project assets.

It explicitly calls for:

- scaffold splits,
- frozen hyperparameters,
- multiple seeds,
- validation-based uncertainty calibration,
- decoy controls,
- exact-vs-heuristic comparisons,
- raw vs repaired validity,
- resource accounting,
- best-known-energy language rather than unsupported "optimality" claims.

This is exactly the right direction.

The issue is that the code has not fully caught up with the protocol.

---

# 5. The Biggest Problem: The README and the Executable Pipeline Are Not Yet the Same Thing

This is the most important criticism.

The README describes a sophisticated pipeline involving FAENet and a Bayesian GNN.

The actual end-to-end implementation currently contains a much simpler surrogate.

In `pipeline.py`, B3/B4 trains a Bayesian MLP over:

```text
qed
sa
mw
logp
hbd
hba
rot_bonds
```

rather than running:

```text
3D graph
 -> FAENet
 -> atom-level geometric representation
 -> Bayesian GNN
 -> affinity
```

The repository does contain a FAENet adapter and a Bayesian GNN class, so this is not imaginary work.

But the **main executable path is materially simpler than the architecture advertised in the README**.

### Why this matters commercially

A technical buyer will inspect the code.

If the README says:

> "E(3)-equivariant FAENet + Bayesian GNN"

but the default pipeline actually trains on seven scalar descriptors, the buyer will downgrade trust.

### Recommendation

Change the public positioning until the full path really exists.

Either:

1. make the executable path actually use FAENet + the Bayesian GNN, or
2. call the current stage what it is:

> **Bayesian molecular-property baseline / surrogate prototype**

and explicitly label the FAENet path as an experimental extension.

---

# 6. Critical Issue: Uncertainty Calibration Leakage

The experimental protocol says calibration is supposed to use validation data.

The current pipeline does not fully enforce that.

The current `run_stage_b3_b4()` creates:

```python
X = ...
y = ...
```

from the full dataframe and then trains the Bayesian head on that data.

It then calls:

```python
tau = calibrate_sigma(mu, sigma, y)
```

using the same target values.

That means the uncertainty scaling step is not cleanly separated from training.

## Why this is serious

Uncertainty calibration is one of the features the project uses to justify the "Bayesian / uncertainty-aware" positioning.

A leakage problem here undermines the exact property being advertised.

## Fix

The proper flow should be:

```text
train set
    |
    +--> fit model

validation set
    |
    +--> calibrate tau
    |
    +--> choose hyperparameters

test set
    |
    +--> never used for calibration
```

The cleanest implementation is:

```text
fit model on train
predict validation
fit sigma calibration on validation
freeze model + tau
predict test
evaluate test NLL / coverage / ranking
```

---

# 7. Critical Issue: The Current B6/B7 Pipeline Is Still a Demonstration, Not General Flexible Docking

The current B6/B7 implementation constructs a hand-specified TAM16 example.

It currently uses:

```text
4 fragments
3 candidate placements per fragment
12 binary variables
```

with explicitly written `dG`, clash, and connectivity matrices.

That is useful as a kernel smoke test.

It is **not yet equivalent** to a generalized workflow:

```text
arbitrary ligand
  -> fragment
  -> enumerate real rigid placements
  -> evaluate pocket interactions
  -> compute pairwise conflicts
  -> compute connectivity
  -> QUBO
```

## Why this matters

This is probably the single largest gap between:

> "cool research concept"

and:

> "software somebody can use."

A buyer needs to know that the solution isn't simply specialized to the TAM16 demonstration.

## Fix

The production path needs to accept:

```text
PDB / receptor
SMILES / ligand
fragmentation policy
pocket definition
placement-generation configuration
```

and automatically produce:

```text
placements.parquet
pair_terms.npz
qubo.pt
variable_map.json
```

with no hard-coded compound-specific coordinates or interaction values.

---

# 8. Critical Issue: The RMSD Evaluation Is Not Yet a Clean Crystal-Pose Benchmark

This is one of the most important scientific problems.

The project claims to evaluate pose recovery against PDB 5V3Y.

But `compute_crystal_rmsd()` currently builds the reference from the TAM16 SMILES and generates a new conformer when a reference molecule is not passed.

That is not the same as:

```text
predicted pose
vs.
experimental crystallographic coordinates
```

An independently embedded conformer is simply another computational structure.

## Why this matters

A 1.7 Å RMSD sounds very impressive.

But the scientific meaning depends completely on what the reference coordinates are.

The correct evaluation needs:

```text
PDB 5V3Y crystal ligand coordinates
         |
         v
atom mapping
         |
         v
predicted ligand coordinates
         |
         v
heavy-atom RMSD
```

not:

```text
TAM16 SMILES
 -> new RDKit conformer
 -> compare to another computational conformer
```

## Fix

Extract TAM16 directly from the deposited PDB structure.

Then:

1. normalize the atom mapping;
2. define the exact atom subset;
3. use a reproducible alignment method;
4. compute RMSD against those deposited coordinates;
5. publish the atom mapping alongside the result.

---

# 9. Critical Issue: B9 Currently Uses the Exact Solver Result

In the current pipeline:

```text
results[0] = Exact (Brute Force)
```

and B9 does:

```python
best_solution = solver_results[0]["best_bits"]
```

Therefore the pose reconstruction stage uses the **exact solver output**, not the solution from SB/TESB/pSA.

This is a subtle but major benchmarking problem.

The final output can therefore look like:

> "X-TUBIT recovered the correct pose"

while the result being reconstructed may actually come from brute-force enumeration.

That does not demonstrate the performance of the physics-inspired solver.

## Fix

B9 should explicitly receive:

```text
solver_name
seed
run_id
solution_id
```

and evaluate every relevant solver independently.

For example:

```text
Exact
SB
TESB
pSA
TApSA
SpSA
```

each gets:

```text
raw bits
validity
repaired bits
pose
RMSD
```

Then report them separately.

---

# 10. Critical Issue: "Best Energy" Is Not Enough

The solver benchmark should never be reduced to:

```text
lowest energy observed
```

A stochastic optimizer needs at least:

```text
seed
wall time
best energy
median energy
energy variance
validity
constraint violations
best-known gap
success probability
```

The protocol already recognizes much of this.

The implementation should make the protocol mandatory.

---

# 11. The "Digital Twin" Claim Needs More Precision

The term "digital twin" is attractive, but currently somewhat broad.

A true engineering digital twin generally implies a sufficiently faithful computational counterpart of a defined physical system or process.

Here the project is really doing:

> **software emulation of physics-inspired combinatorial optimization dynamics intended to approximate classes of hardware optimization behavior.**

That is a valid and interesting concept.

But a skeptical reviewer might challenge the word "twin."

## Better phrasing

Something like:

> "GPU-based digital emulation of physics-inspired Ising optimization dynamics"

is harder to attack than claiming equivalence to a physical p-bit/KPO device.

The distinction matters commercially because "digital twin" can imply validated physical-system correspondence.

---

# 12. Solver Research: What Is Good

The solver section is arguably the most interesting research component.

Comparing:

- SB
- TESB
- pSA-family methods
- exact small-instance validation

is scientifically reasonable.

The use of multiple stochastic seeds is also appropriate.

The project is particularly interesting if it asks:

> For molecular QUBOs generated under a reproducible docking construction, which optimizer produces the best energy/validity/runtime tradeoff?

That is a publishable benchmark question.

---

# 13. Solver Research: What Needs Improvement

## 13.1 Validate against upstream references

The repository itself acknowledges that the TESB port is:

> "not the upstream source code"

and should be validated against the pinned repository.

That validation needs to happen before using solver comparisons as evidence.

Otherwise there is an attribution problem:

> Is the observed behavior due to the published algorithm, or due to this independent reimplementation?

---

## 13.2 Compare distributions, not cherry-picked runs

Every solver should be evaluated over the same seed set.

Example:

```text
20 seeds
30 seeds
50 seeds
```

depending on compute budget.

Report:

```text
median
IQR
best
worst
success rate
TTS99
```

not only the winning run.

---

## 13.3 Benchmark classical optimizers too

This is perhaps the biggest missing commercial benchmark.

A biotech user does not care whether:

```text
TESB > pSA
```

in isolation.

They care whether:

```text
TESB / X-TUBIT
    >
their existing docking workflow
```

or at least provides something meaningfully complementary.

The minimum serious external benchmark should include a conventional baseline, such as:

- AutoDock Vina;
- GNINA;
- another appropriate docking package;
- optionally a classical QUBO optimizer / simulated annealing baseline.

The comparison must be like-for-like.

---

# 14. The ML Component Needs a Reality Check

## Current state

There are good building blocks:

- `BayesianLinear`
- `BayesianGNN`
- FAENet adapter
- MC sampling
- sigma calibration helper

But the project currently does not yet demonstrate a strong molecular ML result.

## The fundamental limitation

The Pks13-TE dataset is likely small and heterogeneous compared with the datasets needed to confidently train a high-capacity geometric model.

A 3D geometric network on a tiny target-specific dataset can very easily overfit.

## Better approach

Use:

```text
2D baseline
   |
3D representation
   |
FAENet features
   |
fine-tuned model
```

and make the comparison explicit.

If the geometric model does not outperform the baseline, say so.

That is scientifically valuable.

---

# 15. Do Not Oversell Bayesian Uncertainty

"Bayesian" is not automatically synonymous with:

> reliable uncertainty.

What matters is calibration.

The project should report:

```text
NLL
coverage@50%
coverage@80%
coverage@90%
calibration curves
prediction interval width
error vs sigma
```

And ideally:

```text
Does high sigma actually predict higher error?
```

That is the practical question.

---

# 16. The Pks13 Dataset Is a Bottleneck

This is probably the largest scientific constraint.

A serious target-specific ML model needs:

```text
sufficiently many
+
consistent
+
well-provenanced
+
same-endpoint
```

measurements.

The blueprint correctly warns against mixing:

```text
Pks13 enzyme IC50
```

with:

```text
whole-cell MIC
```

in the same regression target.

Keep that discipline.

The dataset should store:

```text
compound
assay
endpoint
relation
value
unit
source
structure
scaffold
```

and never silently merge incompatible measurements.

---

# 17. Censored Data Handling

The current documentation notes:

```text
< 
>
```

relations.

That is good.

The project should not simply convert:

```text
IC50 < 10 nM
```

into:

```text
IC50 = 10 nM
```

because that distorts the training target.

A future version could use a censored likelihood model.

That would be a meaningful methodological improvement.

---

# 18. Conformer Generation Is Too Simplified for a Strong Product Claim

The current path tends toward:

```text
multiple conformers
 -> MMFF
 -> choose lowest-energy conformer
```

That is fine as an MVP.

But the lowest gas-phase MMFF conformer is not necessarily the bioactive bound conformation.

A serious product should support:

```text
top-k conformers
```

or at minimum:

```text
ensemble of low-energy conformers
```

so the downstream docking stage can account for conformational uncertainty.

---

# 19. The Pocket Is a Major Hidden Assumption

The current architecture assumes a known pocket.

That is acceptable.

But then X-TUBIT should be marketed as:

> **structure-based screening with a predefined binding pocket**

rather than implying general target discovery.

This is not a weakness as long as it is explicit.

---

# 20. The Decoy Benchmark Needs to Be Much Stronger

The current protocol correctly proposes property-matched decoys.

Good.

But to make a convincing screening benchmark, it should report:

```text
number of actives
number of decoys
scaffold overlap policy
MW matching tolerance
logP matching tolerance
HBD/HBA matching tolerance
rotatable-bond tolerance
charge policy
```

Then report:

```text
ROC-AUC
PR-AUC
EF1%
EF5%
EF10%
BEDROC
Top-k recovery
```

For a highly imbalanced virtual-screening problem, early enrichment metrics are particularly important.

---

# 21. Small Retrospective Test Sets Are Dangerous

Using:

```text
TAM16
8TQG
8TQV
8TR4
```

is scientifically interesting.

It is not enough for strong statistical generalization claims.

The correct language is:

> "retrospective structural validation"

rather than:

> "the model generalizes to Pks13."

A small test set can demonstrate a failure or a promising example.

It cannot establish broad predictive performance.

---

# 22. The Current Tests Prove Math, Not Science

The existing tests are useful.

They demonstrate that:

```text
QUBO algebra works
```

under the tested conditions.

They do not demonstrate:

```text
chemical validity
docking validity
ML generalization
solver superiority
```

That distinction should be explicit in the project documentation.

---

# 23. Reproducibility Is Promising but Incomplete

The blueprint proposes excellent provenance fields:

```text
run_id
config_hash
source_commit
software_versions
seed
input artifact hashes
```

But the current implementation does not appear to enforce these artifact contracts everywhere.

This needs to become automatic.

Every result file should be traceable to:

```text
what code
+
what data
+
what config
+
what seed
+
what hardware
```

was used.

---

# 24. Resource Accounting Needs to Become Real

The protocol calls for:

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

This is excellent.

Make it automatic.

For every heavy run, emit:

```json
{
  "hardware": {...},
  "software": {...},
  "input_hashes": {...},
  "timings": {...},
  "memory": {...},
  "placement_count": ...,
  "pair_count": ...,
  "solver": ...,
  "seed": ...
}
```

This is not just academic bookkeeping.

It becomes a commercial differentiator.

---

# 25. Licensing Is a Serious Commercial Red Flag

The README displays an MIT badge.

The actual `LICENSE.txt` is a custom build-pack license describing an assembly scaffold.

The project therefore currently has a **licensing ambiguity**.

A commercial buyer will care about:

```text
What rights am I purchasing?
What third-party code is included?
What code is original?
Which algorithms are merely reimplemented?
Which files are copied?
What licenses govern dependencies?
```

## Fix

Create an explicit license matrix:

| Component | Origin | License | Included? | Modified? |
|---|---|---|---|---|
| X-TUBIT original source | X-TUBIT team | chosen project license | Yes | n/a |
| FAENet adapter | derivative interface code | check source license | Yes | Yes |
| TESB port | independent implementation | chosen project license if clean | Yes | Yes |
| pSA implementation | independent implementation | chosen project license if clean | Yes | Yes |
| REstretto integration | external software | upstream license | No/conditional | Adapter only |
| PyTorch | third party | upstream | Dependency | No |

And decide clearly whether the project is:

```text
MIT
Apache-2.0
dual licensed
commercial + research
```

before offering it for sale.

---

# 26. The README Is Too Aggressive for the Current Evidence

Phrases like:

> "digital twin"

> "hardware emulator"

> "uncertainty-aware affinity prediction"

> "retrospective blind test"

sound like finished scientific capabilities.

Some of those are true at the architecture level.

Some are currently only partially realized in code.

The public README should be rewritten around:

```text
implemented
vs.
validated
vs.
planned
```

This would actually increase credibility.

---

# 27. Recommended README Status System

Use badges such as:

```text
B1 Data: IMPLEMENTED
B2 Conformers: IMPLEMENTED
B3 FAENet: ADAPTER / EXPERIMENTAL
B4 Bayesian GNN: IMPLEMENTED / NOT VALIDATED
B5 qPMHI: IMPLEMENTED / TOY-VALIDATED
B6 Docking placement: PROTOTYPE
B7 QUBO: VALIDATED
B8 Solver ports: PROTOTYPE / UPSTREAM VALIDATION PENDING
B9 RMSD: EXPERIMENTAL
End-to-end Pks13 benchmark: NOT YET VALIDATED
```

This would make the repository look **more serious**, not less.

---

# 28. Commercial Analysis — Supply and Demand

## Supply

The exact combination represented by X-TUBIT is relatively uncommon.

There are many:

- docking systems,
- molecular GNNs,
- active-learning systems,
- Ising optimization projects,
- quantum-inspired optimization papers.

There are far fewer projects combining all of them into one research workflow.

So:

> **Exact-match supply is low.**

But there is an important catch.

---

## Demand

The market for:

> "I specifically need a Pks13-TE QUBO docking digital twin"

is tiny.

This means:

> **Demand for the exact product is also very low.**

Therefore the project lives in the market category:

> **low supply + very low demand + potentially high value for a small number of expert buyers.**

That is an illiquid niche.

---

# 29. Who Would Actually Buy It?

## Most plausible buyers

### A. Academic research lab

They might want:

- a research starting point;
- a student project;
- a solver benchmark;
- a CADD framework;
- a paper-ready experimental platform.

**Probability:** moderate.

---

### B. Quantum / Ising optimization researchers

This is possibly the most natural buyer group.

They may care about:

```text
QUBO construction
protein-ligand optimization
solver benchmarks
real-world combinatorial instances
```

**Probability:** moderate.

---

### C. Small computational-drug-discovery startup

They could care if the project is generalized and validated.

**Probability today:** low.

**Probability after validation:** moderate.

---

### D. Large pharmaceutical company

An outright purchase of the current repository is unlikely.

They would expect:

- validation,
- support,
- integration,
- security,
- reproducibility,
- benchmarking,
- licensing clarity,
- maintenance.

A raw GitHub repository is not enough.

---

# 30. Current Sale Value

## Outright sale — present repository

A realistic range is approximately:

**US$500–3,000**

I would personally consider:

**US$1,500–2,500**

a sensible "serious but still sellable" range for the current research asset.

That assumes the buyer is purchasing:

- source code;
- documentation;
- setup guidance;
- some handoff;
- the research architecture and implementations.

---

# 31. Why It Is Not Worth $20,000+ Yet

A buyer paying that kind of money will ask:

> Can I run it on a new compound?

> Can I reproduce the reported result?

> Can I compare it against Vina/GNINA?

> Is the ML actually using the advertised representation?

> Is the RMSD against true crystal coordinates?

> Are the results independently reproducible?

> Is the licensing clean?

> What is the actual advantage?

Right now, the answer to several of those is still:

> "not fully demonstrated."

That prevents a high valuation.

---

# 32. What Happens After Serious Validation?

Suppose the following becomes true:

```text
2HV5 reproduced
+
real Pks13 blind benchmark
+
multiple structures
+
real decoy screen
+
classical docking baseline
+
multi-seed solver benchmark
+
clean RMSD
+
clean uncertainty calibration
+
full provenance
```

At that point the project is no longer merely:

> "interesting code."

It becomes:

> "validated research software."

A plausible specialist asset value can then move into roughly:

**US$5,000–20,000**

depending on evidence quality, documentation, exclusivity, and buyer fit.

A particularly strong research package could move beyond that.

---

# 33. Productized Version

Once X-TUBIT becomes something that another lab can actually operate without reverse-engineering the repository, the pricing model should change.

Potential product/service formats:

| Offering | Illustrative pricing territory |
|---|---:|
| Academic research license | $1k–5k/year |
| Specialist research package | $3k–15k |
| Pilot screening project | $10k–30k |
| Custom integration | $15k–100k+ |
| Annual biotech license/service | $20k–75k+ |
| Enterprise platform | Potentially much higher |

These are **market-oriented target bands**, not evidence that X-TUBIT currently commands these prices.

---

# 34. The Most Valuable Thing May Not Be the Software

This is strategically important.

The software alone is difficult to sell.

The research result generated by the software may be much more valuable.

The strongest sequence is:

```text
X-TUBIT
    |
    v
validated benchmark
    |
    v
published evidence
    |
    v
novel predicted chemotypes
    |
    v
experimental validation
    |
    v
drug-discovery asset
```

The economic value increases dramatically as you move right.

---

# 35. The Highest-Value Outcome

The highest-value version is not:

> "I built a fancy docking program."

It is:

> "We built and experimentally validated a computational workflow that identifies chemically novel Pks13 inhibitors with reproducible enrichment over established baselines."

That is a completely different commercial proposition.

At that stage you are potentially selling:

```text
software
+
methodology
+
dataset
+
validated benchmark
+
candidate molecules
+
scientific know-how
```

rather than a repository.

---

# 36. Priority Improvement Roadmap

## Priority 0 — Fix scientific credibility issues

Do these first.

### 0.1 Fix B3/B4 data leakage

Use:

```text
train
validation
test
```

strictly.

Calibrate sigma on validation only.

---

### 0.2 Fix RMSD

Use the actual ligand coordinates from:

```text
PDB 5V3Y
```

Do not use an independently generated conformer as the "crystal" reference.

---

### 0.3 Fix solver-to-pose attribution

Do not always use `solver_results[0]`.

Explicitly evaluate each solver's result.

---

### 0.4 Remove hard-coded B6 values

Replace demonstration `dG`, clashes, and connectivity matrices with generated values.

---

### 0.5 Validate solver ports against upstream references

Especially TESB and the pSA variants.

---

# 37. Priority 1 — Make the End-to-End Pipeline Real

The canonical path should become:

```text
SMILES
  |
  v
standardize
  |
  v
conformers
  |
  v
fragmentation
  |
  v
real pocket placements
  |
  v
pairwise energies / clashes / connectivity
  |
  v
QUBO
  |
  v
Ising
  |
  +--> Exact (small N)
  +--> SB
  +--> TESB
  +--> pSA
  |
  v
decode
  |
  v
reconstruct
  |
  v
refine
  |
  v
true PDB RMSD / docking score / validity
```

No hard-coded TAM16 geometry should remain on the critical path.

---

# 38. Priority 2 — Build a Real Benchmark Dataset

Minimum useful package:

```text
2HV5
5V3Y
5V40
5V41
5V42
8TQG
8TQV
8TR4
```

plus a larger set of chemically diverse Pks13-related molecules and appropriate decoys.

For each dataset element preserve:

```text
PDB ID
chain
ligand ID
SMILES
assay
source
scaffold
activity
atom mapping
```

---

# 39. Priority 3 — Compare Against Established Docking

This is essential.

The benchmark should answer:

> Does the combinatorial formulation actually produce anything useful compared with a normal docking workflow?

At minimum test:

```text
X-TUBIT
vs.
AutoDock Vina
vs.
GNINA or another suitable baseline
```

Potential metrics:

```text
ligand pose RMSD
Top-1 success
Top-5 success
Top-10 success
runtime
virtual-screening enrichment
valid fraction
```

---

# 40. Priority 4 — Prove the ML Component Is Useful

Run:

```text
Model A: 2D baseline
Model B: 3D + FAENet frozen
Model C: 3D + FAENet fine-tuned
```

Report:

```text
RMSE
MAE
NLL
coverage
calibration
ranking correlation
```

If FAENet adds no value, remove it.

That is not failure.

It is scientific selection.

---

# 41. Priority 5 — Make qPMHI Earn Its Place

qPMHI should not exist just because it sounds sophisticated.

Demonstrate:

```text
random selection
vs.
single-objective ranking
vs.
Pareto ranking
vs.
qPMHI
```

and ask:

> Which one finds better compounds per experimental batch?

That is the practical question.

---

# 42. Priority 6 — Real Ablation Studies

A compelling paper/product needs ablations.

For example:

```text
No uncertainty
vs.
uncertainty

No qPMHI
vs.
qPMHI

No geometric representation
vs.
FAENet

Rigid docking
vs.
fragment flexible docking

Classical optimizer
vs.
SB
vs.
TESB
vs.
pSA
```

This allows the buyer/reviewer to determine which component actually matters.

---

# 43. Priority 7 — Make the Dashboard Useful

The Streamlit dashboard should answer real questions.

A user should be able to inspect:

```text
compound
predicted pIC50
uncertainty
QED
SA
qPMHI
selected fragments
placement count
solver
energy
constraint validity
RMSD
runtime
```

and click through to:

```text
3D ligand pose
pocket
energy landscape
solver trajectory
```

This is already conceptually described.

The implementation needs to become an actual scientific instrument instead of a presentation layer.

---

# 44. Priority 8 — Build a Reproducibility Command

A strong future interface would look like:

```bash
xtubit run \
  --receptor 5V3Y \
  --ligand TAM16 \
  --config configs/benchmark.yaml \
  --seed 7
```

and automatically produce:

```text
runs/<run_id>/
    manifest.json
    software.json
    hardware.json
    inputs/
    placements/
    qubo/
    solver/
    poses/
    metrics/
    report.html
```

That is a huge increase in usability.

---

# 45. Priority 9 — Add a One-Command Benchmark

This could become the killer demo:

```bash
xtubit benchmark pks13
```

Output:

```text
Pks13 benchmark
---------------

Structures: 8
Actives: ...
Decoys: ...
Solvers: 5

Pose RMSD < 2 Å:
X-TUBIT ...
Vina ...
GNINA ...

Early enrichment:
X-TUBIT ...
Vina ...
GNINA ...

Runtime:
...
```

A buyer can understand that immediately.

---

# 46. Priority 10 — Rewrite the README

The README should be structured as:

```text
What is X-TUBIT?
What is actually implemented?
What has been validated?
What is experimental?
How to reproduce the benchmark?
What does not work yet?
```

Do not hide limitations.

Make the limitations visible.

That increases trust.

---

# 47. Recommended Scientific Positioning

Instead of:

> "Novel hardware digital twin for Pks13 inhibitor discovery"

prefer:

> **"An auditable GPU-based benchmark framework for QUBO/Ising molecular docking and multi-objective Pks13-TE screening."**

Then let the evidence support stronger claims.

A good research project is allowed to be exploratory.

It does not need to pretend to be finished.

---

# 48. Recommended Commercial Positioning

Do not sell it as:

> "AI drug discovery platform."

That puts it directly against mature commercial companies and platforms.

Instead sell it as:

> **"Specialist computational research software for combinatorial molecular docking and physics-inspired optimization."**

The buyer is then:

```text
quantum optimization lab
+
computational chemistry lab
+
academic research group
+
early-stage drug-discovery team
```

rather than:

```text
every pharma company
```

---

# 49. What I Would Sell Today

If the goal is an immediate asset sale, I would package:

```text
X-TUBIT Research Prototype
```

containing:

1. source repository;
2. architecture documentation;
3. benchmark protocol;
4. QUBO/Ising kernel;
5. solver ports;
6. Streamlit interface;
7. setup instructions;
8. known limitations;
9. validation checklist;
10. handover session.

Suggested ask:

> **US$2,500**

Suggested realistic acceptance zone:

> **US$1,500–2,000**

I would only move significantly below that if the goal is simply to recover some value quickly.

---

# 50. What I Would NOT Do

I would not:

- claim that X-TUBIT has demonstrated solver superiority yet;
- claim the current RMSD result proves crystal-pose recovery;
- claim the ML uncertainty is calibrated from an independent validation set;
- imply that the current B6/B7 code is generalized docking;
- market it as a replacement for commercial docking suites;
- claim major biotech companies need it;
- call it clinically validated;
- sell it as a drug-discovery engine before the benchmark is real.

---

# 51. How Fast Could It Sell?

## Current prototype

**Likely sales cycle:**

approximately **2–8 weeks** with direct outreach.

On a generic marketplace:

> potentially never.

The biggest problem is not the price.

It is finding someone who understands why they might want it.

---

## Validated research package

**Likely sales cycle:**

approximately **1–4 months**.

At this stage a professor, computational chemistry group, or specialist startup can justify the purchase.

---

## Productized service

**Likely sales cycle:**

approximately **1–6+ months**.

That is slower because procurement, legal review, data handling, and technical evaluation appear.

But the contract value can be far higher.

---

# 52. "Low Supply, Low Demand" — Final Judgment

Your intuition is essentially correct.

The exact market is:

```text
Supply: LOW
Demand: VERY LOW
Buyer specificity: VERY HIGH
Potential value to matched buyer: MODERATE-HIGH
Liquidity: LOW
```

The important concept is:

> **X-TUBIT is not a mass-market software asset. It is a specialized technical asset.**

That means:

```text
1000 random viewers
```

may produce:

```text
0 buyers
```

while:

```text
20 carefully targeted computational-research groups
```

might produce:

```text
1–3 serious prospects
```

That is the market.

---

# 53. Biggest Risks

## Technical risk

The solver does not outperform standard approaches.

**Impact:** high.

---

## Data risk

The Pks13 dataset is too small/noisy for strong ML claims.

**Impact:** high.

---

## Validation risk

The retrospective benchmark does not generalize.

**Impact:** high.

---

## Product risk

Researchers prefer existing docking workflows.

**Impact:** high.

---

## Commercial risk

The buyer pool is extremely small.

**Impact:** high.

---

## Credibility risk

README claims are stronger than implemented functionality.

**Impact:** very high.

This one is fixable.

---

# 54. Biggest Opportunities

## Opportunity 1 — Publish the benchmark

A clean paper can create credibility far faster than attempting to sell the repository immediately.

---

## Opportunity 2 — Generalize beyond Pks13

Once the QUBO/docking machinery works, the system can be tested on:

```text
kinases
proteases
GPCRs
enzymes
protein-protein interaction pockets
```

The value proposition becomes:

> generalized combinatorial docking research framework.

That is much larger than Pks13 alone.

---

## Opportunity 3 — Use Pks13 as the proof case

The target-specific implementation is actually helpful for establishing a concrete case study.

The architecture becomes:

```text
generic engine
+
Pks13 demonstration
```

instead of:

```text
Pks13-only software
```

---

## Opportunity 4 — Sell the methodology, not only the code

Potential deliverables:

```text
benchmark design
QUBO formulation
solver integration
screening workflow
custom research
```

That is much more defensible than selling source code alone.

---

# 55. Ideal Future Architecture

A stronger architecture would be:

```text
                         X-TUBIT CORE
                              |
        +---------------------+----------------------+
        |                                            |
   MOLECULAR PLANE                            OPTIMIZATION PLANE
        |                                            |
    SMILES / SDF                                QUBO / Ising
        |                                            |
    standardize                                 exact / SB / TESB
        |                                            |
    conformers                                  pSA variants
        |                                            |
    descriptors                                      |
        |                                            |
    FAENet / GNN                                      |
        |                                            |
    uncertainty                                        |
        |                                            |
    qPMHI                                               |
        |                                            |
        +-------------------> docking <---------------+
                                  |
                           pose reconstruction
                                  |
                               refinement
                                  |
                         scientific evaluation
                                  |
                      +-----------+-----------+
                      |                       |
                  benchmark               dashboard
                      |                       |
                  report                   HITL UI
```

---

# 56. Definition of "Done"

I would consider X-TUBIT scientifically ready for serious external evaluation only after all of these pass:

## Chemistry

- [ ] No hard-coded TAM16 placement matrices in production.
- [ ] General ligand fragmentation.
- [ ] Real pocket placement generation.
- [ ] Real pairwise clash/connectivity computation.
- [ ] Valid molecular reconstruction.

## ML

- [ ] True FAENet path.
- [ ] Proper train/validation/test split.
- [ ] No calibration leakage.
- [ ] Uncertainty coverage demonstrated.
- [ ] Baseline ablations.

## Optimization

- [ ] Upstream solver validation.
- [ ] Multi-seed comparison.
- [ ] Exact baseline for small N.
- [ ] Classical optimizer baseline.
- [ ] Reproducible TTS.

## Docking

- [ ] True crystal-coordinate RMSD.
- [ ] Multiple PDB structures.
- [ ] Multiple ligands.
- [ ] Independent blind test.
- [ ] Decoy enrichment.

## Software

- [ ] Clean license.
- [ ] Pinned dependencies.
- [ ] Provenance manifest.
- [ ] Reproducible benchmark command.
- [ ] CI.
- [ ] Example data.
- [ ] Clear implementation-status badges.

---

# 57. The Single Most Valuable Next Experiment

If resources are limited, do not add another ML architecture.

Do this:

## A real benchmark on multiple Pks13 complexes

For each structure:

```text
crystal ligand
      +
property-matched decoys
      |
      +--> X-TUBIT
      +--> Vina
      +--> GNINA / classical baseline
      |
      v
pose accuracy
+
early enrichment
+
runtime
```

Then repeat across multiple seeds for X-TUBIT's stochastic solver components.

That single experiment would tell you whether the project's central thesis has practical merit.

---

# 58. Final Valuation

## Current state

### Scientific asset

**Real, interesting, incomplete.**

### Software asset

**Prototype-quality.**

### Commercial asset

**Niche and illiquid.**

### Current outright value

**~US$500–3,000**

### Suggested optimum asking price today

**~US$2,500**

### Likely reasonable close

**~US$1,500–2,000**

---

## After proper validation

### Scientific asset

**Potentially strong research contribution.**

### Software asset

**Specialist validated research platform.**

### Commercial asset

**Potentially meaningful niche B2B product/service.**

### Plausible value range

**~US$5,000–20,000+**

depending heavily on reproducibility, benchmark quality, novelty, and buyer fit.

---

# 59. Final Bottom Line

The project is **not a dead end**.

Its problem is not:

> "Nobody would ever want this."

Its problem is:

> **"Very few people want this right now, and the current repository does not yet prove enough value to convince those few people to pay much."**

That is an important difference.

The best path is therefore:

```text
Do not immediately maximize sale price.
             |
             v
Fix scientific inconsistencies.
             |
             v
Finish the real end-to-end benchmark.
             |
             v
Show whether X-TUBIT actually beats or complements baselines.
             |
             v
Turn that evidence into a paper / technical report.
             |
             v
Then decide whether to:
    sell it,
    license it,
    offer it as a service,
    or keep it as the foundation of a larger project.
```

If forced to choose between:

```text
sell today for $2k
```

and

```text
spend serious effort validating it first
```

I would choose **validation first**, because this is one of those projects where a relatively modest increase in evidence could increase the economic value by an order of magnitude.

The code already contains enough of the architecture to make that worthwhile.

The main task now is not adding more buzzwords.

It is converting:

> **"interesting architecture"**

into:

> **"measured, reproducible, independently credible result."**

---

# 60. Primary Repository Evidence

- Repository: https://github.com/Camn0/xtubit
- Main pipeline: https://github.com/Camn0/xtubit/blob/main/src/xtubit/pipeline.py
- QUBO implementation: https://github.com/Camn0/xtubit/blob/main/src/xtubit/b7_qubo.py
- Post-annealing / RMSD code: https://github.com/Camn0/xtubit/blob/main/src/xtubit/post_anneal.py
- Experimental protocol: https://github.com/Camn0/xtubit/blob/main/docs/EXPERIMENT_PROTOCOL.md
- Blueprint: https://github.com/Camn0/xtubit/blob/main/docs/BLUEPRINT.md
- Assembly checklist: https://github.com/Camn0/xtubit/blob/main/docs/ASSEMBLY_CHECKLIST.md
- Source/fidelity matrix: https://github.com/Camn0/xtubit/blob/main/docs/SOURCE_FACT_MATRIX.md
- Data checklist: https://github.com/Camn0/xtubit/blob/main/data/README.md
- License file: https://github.com/Camn0/xtubit/blob/main/LICENSE.txt

---

# 61. Key Scientific References

- Aggarwal et al. (2017), *Cell*: Development of a Novel Lead that Targets *M. tuberculosis* Polyketide Synthase 13.
- Krieger et al. (2024), *ACS Infectious Diseases*: Inhibitors of the Thioesterase Activity of *M. tuberculosis* Pks13 Discovered Using DNA-Encoded Chemical Library Screening.
- Yanagisawa et al. (2024), *Entropy*: QUBO Problem Formulation of Fragment-Based Protein-Ligand Flexible Docking.
- Goto et al. (2021), *Science Advances*: High-performance combinatorial optimization based on classical mechanics.
- Onizawa & Hanyu (2024), *Scientific Reports*: Enhanced convergence in p-bit based simulated annealing with partial deactivation.
- Tao et al. (2026), *Communications Physics*: Tabu-Enhanced Simulated Bifurcation for combinatorial optimization.
- Duval et al. (2023), FAENet.
- Ryu et al. (2019), Bayesian graph convolutional network for uncertainty-aware molecular prediction.

---

# 62. One-Sentence Verdict

> **X-TUBIT is a potentially valuable specialist research prototype with a genuinely interesting central idea, but today it is worth a small amount as software and a much larger potential amount as a validated scientific platform.**
