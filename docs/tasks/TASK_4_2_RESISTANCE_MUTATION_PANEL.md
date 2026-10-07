# Task 4.2: Resistance Mutation Profiling Panel

## Parent Subtask
**Task 4.2: Resistance Mutation Profiling Panel** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 4.2.1: Model Clinically Relevant Resistant Pks13 Mutants
- [done checking 1 time] Sub-sub-subtask 4.2.1.1: Identify known clinical and laboratory-selected Pks13 escape mutations: Asp1644Gly, Asn1640Ala, Phe1585Leu.
- [done checking 1 time] Sub-sub-subtask 4.2.1.2: Perform computational mutagenesis in PyMOL/RDKit to generate mutant pocket coordinate models.
- [done checking 1 time] Sub-sub-subtask 4.2.1.3: Recalculate interaction matrices $Q_{\text{mutant}}$ and Ising $(J_{\text{mut}}, h_{\text{mut}})$ for each variant pocket.
- [done checking 1 time] Sub-sub-subtask 4.2.1.4: Unit test mutated Hamiltonian building pipeline to ensure matrix consistency.

### Sub-subtask 4.2.2: Cross-Screening Resistance Scorecard & Resilience Index
- [done checking 1 time] Sub-sub-subtask 4.2.2.1: Compute ground-state binding energy delta: $\Delta \Delta G_{\text{res}} = E_{\text{mutant}} - E_{\text{WT}}$.
- [done checking 1 time] Sub-sub-subtask 4.2.2.2: Define Resistance Resilience Index: candidate maintains high affinity if $\Delta \Delta G_{\text{res}} \le +1.5\,\text{kcal/mol}$.
- [done checking 1 time] Sub-sub-subtask 4.2.2.3: Build interactive mutant variant selector in Tab 4 and Tab 2 comparing WT vs. Asp1644Gly binding stability.
- [done checking 1 time] Sub-sub-subtask 4.2.2.4: Render multi-mutant spider chart displaying candidate robustness across all clinical escape variants.
