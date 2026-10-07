# X-TUBIT Granular Engineering Tasks Index

This directory contains the immutable, granular task breakdown files derived directly from [`docs/TRANSLATIONAL_ROADMAP.md`](../TRANSLATIONAL_ROADMAP.md).

## Task Hierarchy & Rules
1. **Subtasks:** Exact, unchanged tasks from the translational roadmap.
2. **Sub-subtasks:** Specific functional components.
3. **Sub-sub-subtasks:** Immutable micro-actions.
4. **Status Tags:** Each sub-sub-subtask starts with an exact status tag to its left:
   - `[pending]` — Not yet started
   - `[ongoing]` — Work in progress
   - `[checking quality]` — Implementation under verification / test review
   - `[done checking 1 time]` — Verified and passed primary quality check

---

## Directory Index

### Phase 1: Chemistry Quality & ADMET Hardening
- [`TASK_1_1_MEDCHEM_FILTERS.md`](./TASK_1_1_MEDCHEM_FILTERS.md): Integrate Strict MedChem Structural Cleanliness Filters
- [`TASK_1_2_ADMET_PREDICTORS.md`](./TASK_1_2_ADMET_PREDICTORS.md): Implement Preclinical ADMET & Solubility Surrogate Predictors
- [`TASK_1_3_RETROSYNTHESIS_SCORING.md`](./TASK_1_3_RETROSYNTHESIS_SCORING.md): Real-Time Retrosynthetic Accessibility Scoring

### Phase 2: Structural Physics & QUBO Scalability
- [`TASK_2_1_HAMILTONIAN_SCALING.md`](./TASK_2_1_HAMILTONIAN_SCALING.md): Scale Fragment Docking Hamiltonian to 60–90 Qubits
- [`TASK_2_2_EXPLICIT_DESOLVATION.md`](./TASK_2_2_EXPLICIT_DESOLVATION.md): Incorporate Explicit Desolvation & Water Thermodynamics
- [`TASK_2_3_POST_ANNEALING_FORCEFIELD.md`](./TASK_2_3_POST_ANNEALING_FORCEFIELD.md): Post-Annealing Force-Field Energy Minimization (OpenMM / RDKit MMFF94)

### Phase 3: Digital Annealing & Algorithmic Innovation
- [`TASK_3_1_BLOCK_SIMULATED_BIFURCATION.md`](./TASK_3_1_BLOCK_SIMULATED_BIFURCATION.md): Implement Pharmacophore-Decomposed Block Simulated Bifurcation (Block-tSB)
- [`TASK_3_2_TTS99_AUTOMATED_TUNING.md`](./TASK_3_2_TTS99_AUTOMATED_TUNING.md): Automated TTS99 Scaling & Parameter Tuning

### Phase 4: Pharmacological Dossier & Wet-Lab Handoff
- [`TASK_4_1_WETLAB_SYNTHESIS_DOSSIER.md`](./TASK_4_1_WETLAB_SYNTHESIS_DOSSIER.md): Turnkey Wet-Lab Synthesis Order Dossier
- [`TASK_4_2_RESISTANCE_MUTATION_PANEL.md`](./TASK_4_2_RESISTANCE_MUTATION_PANEL.md): Resistance Mutation Profiling Panel
