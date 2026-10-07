# Task 2.1: Scale Fragment Docking Hamiltonian to 60–90 Qubits

## Parent Subtask
**Task 2.1: Scale Fragment Docking Hamiltonian to 60–90 Qubits** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 2.1.1: Discretize Pks13 Pocket into 6 Sub-pocket Regions
- [done checking 1 time] Sub-sub-subtask 2.1.1.1: Extract 3D coordinates from PDB 5V3Y defining 6 pharmacophore sub-sites: Anchor, Linker, Tunnel, P1 Cap, Catalytic Triad (Ser1533/Asp1644), Solvent Front.
- [done checking 1 time] Sub-sub-subtask 2.1.1.2: Define coordinate center and spherical boundary radii for each sub-pocket in `src/xtubit/b6_pairs.py`.
- [done checking 1 time] Sub-sub-subtask 2.1.1.3: Generate conformer library of 10 to 15 building blocks per sub-pocket site ($6 \times 15 = 90$ total binary variables).
- [done checking 1 time] Sub-sub-subtask 2.1.1.4: Map binary variables $x_0 \dots x_{89}$ with one-hot constraints $\sum_{i \in F_k} x_i = 1$ for $k \in \{0 \dots 5\}$.

### Sub-subtask 2.1.2: Compute Pairwise Steric Clash and Connectivity Matrices
- [done checking 1 time] Sub-sub-subtask 2.1.2.1: Implement vector-accelerated inter-fragment atomic distance calculation in PyTorch.
- [done checking 1 time] Sub-sub-subtask 2.1.2.2: Populate $B_{ij}$ clash matrix (penalty for interatomic distance $< 2.0\,\text{Å}$).
- [done checking 1 time] Sub-sub-subtask 2.1.2.3: Populate $C_{ij}$ connectivity matrix (reward for correct covalent attachment geometry across neighboring sub-sites).
- [done checking 1 time] Sub-sub-subtask 2.1.2.4: Validate symmetry, diagonal zeroing, and positive definiteness of scaled $Q$ matrix ($90 \times 90$).
