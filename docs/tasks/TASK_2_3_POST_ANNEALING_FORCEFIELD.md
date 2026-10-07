# Task 2.3: Post-Annealing Force-Field Energy Minimization (OpenMM / RDKit MMFF94)

## Parent Subtask
**Task 2.3: Post-Annealing Force-Field Energy Minimization (OpenMM / RDKit MMFF94)** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 2.3.1: Assembly of Continuous Molecular Topology from Decoded Bitstring
- [done checking 1 time] Sub-sub-subtask 2.3.1.1: Write fragment stitcher taking active binary variables ($x_i = 1$) and performing covalent bond formation between adjacent sub-pockets.
- [done checking 1 time] Sub-sub-subtask 2.3.1.2: Check chemical valency, formal charges, and hybridization of the stitched 3D assembly in RDKit.
- [done checking 1 time] Sub-sub-subtask 2.3.1.3: Generate 3D conformer coordinates matching the crystallographic binding pocket orientation.
- [done checking 1 time] Sub-sub-subtask 2.3.1.4: Unit test with known TAM16 bitstring to ensure exact structural reconstruction against 5V3Y co-crystal ligand.

### Sub-subtask 2.3.2: Local Force-Field Relaxation inside Protein Pocket Field
- [done checking 1 time] Sub-sub-subtask 2.3.2.1: Construct OpenMM / RDKit MMFF94 rigid-receptor energy minimization context with pocket atoms within 6 Å frozen.
- [done checking 1 time] Sub-sub-subtask 2.3.2.2: Execute 50 to 100 steps of conjugate gradient minimization on ligand coordinates.
- [done checking 1 time] Sub-sub-subtask 2.3.2.3: Compute post-relaxation interaction energy ($\Delta E_{\text{vdw}} + \Delta E_{\text{elec}}$) and heavy-atom RMSD shift.
- [done checking 1 time] Sub-sub-subtask 2.3.2.4: Save minimized conformer as multi-model SDF block accessible for Tab 3 3D inspection and download.
