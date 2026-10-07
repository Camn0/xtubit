# Task 3.1: Implement Pharmacophore-Decomposed Block Simulated Bifurcation (Block-tSB)

## Parent Subtask
**Task 3.1: Implement Pharmacophore-Decomposed Block Simulated Bifurcation (Block-tSB)** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 3.1.1: Partition Hamiltonian into Domain Sub-graphs
- [pending] Sub-sub-subtask 3.1.1.1: Partition 60+ qubit matrix into Core Anchor sub-matrix ($Q_{\text{core}}$) and Peripheral Substituent sub-matrix ($Q_{\text{periph}}$).
- [pending] Sub-sub-subtask 3.1.1.2: Compute cross-coupling terms ($Q_{\text{cross}}$) connecting core scaffolds to linkers and tail groups.
- [pending] Sub-sub-subtask 3.1.1.3: Validate that sub-matrix partition preserves global energy sum: $E_{\text{tot}} = x_c^T Q_{\text{core}} x_c + x_p^T Q_{\text{periph}} x_p + 2 x_c^T Q_{\text{cross}} x_p$.
- [pending] Sub-sub-subtask 3.1.1.4: Unit test partitioning logic on 12-qubit and 60-qubit synthetic test matrices.

### Sub-subtask 3.1.2: Two-Stage Alternating Block-Coordinate Bifurcation
- [pending] Sub-sub-subtask 3.1.2.1: Implement Stage 1: Run discrete tSB on Core Anchor variables with peripheral variables initialized to zero field.
- [pending] Sub-sub-subtask 3.1.2.2: Implement Stage 2: Freeze the chosen core variable, inject its field into $h_{\text{periph}} = Q_{\text{cross}}^T x_{\text{core}}$, and run tSB on peripheral variables.
- [pending] Sub-sub-subtask 3.1.2.3: Benchmark Block-tSB time-to-solution against monolithic tSB and Exact brute-force.
- [pending] Sub-sub-subtask 3.1.2.4: Integrate Block-tSB solver into `src/xtubit/solvers/tesb_port.py` and register in pipeline stage B8.
