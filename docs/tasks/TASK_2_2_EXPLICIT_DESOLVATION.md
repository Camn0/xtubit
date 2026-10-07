# Task 2.2: Incorporate Explicit Desolvation & Water Thermodynamics

## Parent Subtask
**Task 2.2: Incorporate Explicit Desolvation & Water Thermodynamics** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 2.2.1: Add Local Desolvation Penalties to Diagonal Hamiltonian
- [pending] Sub-sub-subtask 2.2.1.1: Calculate surface area solvent accessibility (SASA) of the unoccupied Pks13 pocket using Lee-Richards algorithm.
- [pending] Sub-sub-subtask 2.2.1.2: Compute transfer free energy ($\Delta G_{\text{transfer}}$) for burying hydrophobic vs. polar fragment atoms into the catalytic channel.
- [pending] Sub-sub-subtask 2.2.1.3: Add desolvation offset into diagonal $dG_i$ terms: favorable free-energy bonus for hydrophobic displacement; severe penalty for burying un-paired H-bond donors/acceptors.
- [pending] Sub-sub-subtask 2.2.1.4: Unit test $dG$ values against experimental ITC binding thermodynamics of TAM16.

### Sub-subtask 2.2.2: Explicit Displacement of Catalytic Pocket Water Molecules
- [pending] Sub-sub-subtask 2.2.2.1: Identify 4 conserved crystallographic water molecules in Pks13 (PDB 5V3Y) catalytic cleft.
- [pending] Sub-sub-subtask 2.2.2.2: Calculate free energy of displacing each water ($\Delta G_{\text{water}} \approx -1.5$ to $+2.0\,\text{kcal/mol}$).
- [pending] Sub-sub-subtask 2.2.2.3: Modify fragment placement score based on whether candidate successfully displaces unstable waters or preserves stable structural waters.
- [pending] Sub-sub-subtask 2.2.2.4: Render displaced vs. conserved water molecules in Tab 3 3D PyMOL/3Dmol viewer.
