# Task 1.2: Implement Preclinical ADMET & Solubility Surrogate Predictors

## Parent Subtask
**Task 1.2: Implement Preclinical ADMET & Solubility Surrogate Predictors** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 1.2.1: Aqueous Thermodynamic Solubility Predictor (LogS)
- [pending] Sub-sub-subtask 1.2.1.1: Implement Delaney ESOL aqueous solubility model (`calc_esol_logs(mol)`) in `src/xtubit/b1_data.py`.
- [pending] Sub-sub-subtask 1.2.1.2: Calibrate intrinsic solubility threshold targeting $> 50\,\mu\text{M}$ ($\log S > -4.3$).
- [pending] Sub-sub-subtask 1.2.1.3: Write unit tests verifying calculated solubility against known solubility benchmark set (ChEMBL / Delaney).
- [pending] Sub-sub-subtask 1.2.1.4: Add quantitative solubility metric and warning indicator for insoluble compounds ($\log S < -5.0$).

### Sub-subtask 1.2.2: hERG Cardiac Liability Risk Classifier
- [pending] Sub-sub-subtask 1.2.2.1: Build fast Morgan fingerprint QSAR surrogate model predicting hERG channel inhibition ($IC_{50} < 10\,\mu\text{M}$).
- [pending] Sub-sub-subtask 1.2.2.2: Implement `predict_herg_risk(mol)` outputting probability score and binary safety flag.
- [pending] Sub-sub-subtask 1.2.2.3: Test against published hERG datasets (Tox21 / ChEMBL hERG patch clamp).
- [pending] Sub-sub-subtask 1.2.2.4: Integrate hERG safety check into candidate summary metadata.

### Sub-subtask 1.2.3: Mouse Microsomal Metabolic Stability Estimator
- [pending] Sub-sub-subtask 1.2.3.1: Define heuristic and structural alert predictor for metabolic soft spots (benzylic carbons, unhindered esters, unfluorinated aromatics).
- [pending] Sub-sub-subtask 1.2.3.2: Implement microsomal half-life estimator (`predict_microsomal_t12(mol)`) targeting $t_{1/2} > 30\,\text{min}$.
- [pending] Sub-sub-subtask 1.2.3.3: Benchmark TAM16 predicted clearance against Aggarwal et al. (2017) published pharmacokinetic data.
- [pending] Sub-sub-subtask 1.2.3.4: Surface predicted microsomal stability metric in Tab 1 and Tab 2 comparison panels.
