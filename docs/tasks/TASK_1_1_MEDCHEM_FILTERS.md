# Task 1.1: Integrate Strict MedChem Structural Cleanliness Filters

## Parent Subtask
**Task 1.1: Integrate Strict MedChem Structural Cleanliness Filters** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 1.1.1: Implement RDKit-Based PAINS Filter
- [pending] Sub-sub-subtask 1.1.1.1: Compile SMARTS definition library for PAINS-A, PAINS-B, and PAINS-C filters based on Baell & Holloway (2010).
- [pending] Sub-sub-subtask 1.1.1.2: Implement `evaluate_pains(mol)` function returning boolean flag and matched substructure pattern IDs.
- [pending] Sub-sub-subtask 1.1.1.3: Write unit tests covering classic PAINS false-positives (rhodanine, quinone, curcumin) to ensure 100% detection rate.
- [pending] Sub-sub-subtask 1.1.1.4: Integrate PAINS evaluator into `evaluate_single_smiles()` pipeline in `src/xtubit/b1_data.py`.

### Sub-subtask 1.1.2: Implement Brenk Structural Alert Filter
- [pending] Sub-sub-subtask 1.1.2.1: Extract SMARTS queries for Brenk toxicophores (Michael acceptors, alkyl halides, epoxides, hydrazines, nitro groups).
- [pending] Sub-sub-subtask 1.1.2.2: Implement `evaluate_brenk_alerts(mol)` returning list of violated toxicophore classes.
- [pending] Sub-sub-subtask 1.1.2.3: Validate against standard ChEMBL tox reference compounds with known mutagenic alerts.
- [pending] Sub-sub-subtask 1.1.2.4: Attach Brenk alert count to compound metadata dictionary in parquet storage schema.

### Sub-subtask 1.1.3: Add BMS / Abbott Rule-of-Two (Ro2) Checks for Fragment Libraries
- [pending] Sub-sub-subtask 1.1.3.1: Define Rule-of-Two boundaries: $MW \le 300\,\text{Da}$, $\text{cLogP} \le 3.0$, $\text{RotB} \le 3$, $\text{HBD} \le 3$, $\text{HBA} \le 3$.
- [pending] Sub-sub-subtask 1.1.3.2: Implement `evaluate_rule_of_two(mol)` for candidate building blocks in `src/xtubit/b6_pairs.py`.
- [pending] Sub-sub-subtask 1.1.3.3: Write automated test verifying that TAM16 fragment library constituents comply with Ro2 thresholds.
- [pending] Sub-sub-subtask 1.1.3.4: Add Ro2 compliance badge indicator to UI candidate cards in `app/streamlit_app.py`.
