# X-TUBIT Translational Roadmap: Pharmacological Value & High-Quality Lead Optimization

## Executive Summary

To deliver genuine value to medicinal chemists and pharmacologists, computational drug discovery platforms must bridge the gap between abstract algorithmic scoring and wet-lab realities. Simple 2D property filters and toy-sized docking models are routinely dismissed by pharmaceutical teams as "low-value noise" because they frequently propose molecules that cannot be synthesized, suffer from metabolic instability, or fail in live biological assays.

This document establishes the **translational objectives**, **domain quality criteria**, and **prioritized TODO engineering tasks** required to elevate X-TUBIT into a high-trust, decision-support platform for tuberculosis lead optimization targeting *Mycobacterium tuberculosis* Pks13 (PDB: `5V3Y`).

---

## 1. Domain Premise & Clinical Context

- **Target Biology:** *Mycobacterium tuberculosis* Polyketide Synthase 13 Thioesterase Domain (**Pks13-TE**, PDB ID: `5V3Y`, 1.98 Å crystallographic resolution).
- **Function:** Pks13 catalyzes the final condensation step synthesizing α-alkyl β-hydroxy mycolic acids, the essential structural core of the mycobacterial outer cell envelope. Its inhibition results in rapid bacterial lysis and sterilizing lethality against multidrug-resistant (MDR) and extensively drug-resistant (XDR) strains.
- **Reference Lead:** **TAM16** (Aggarwal et al., *Nature Medicine* 2017). Co-crystallized benzofuran carboxamide lead binding with 1.34 Å heavy-atom RMSD.
- **Translational Bottlenecks of TAM16:**
  1. *Metabolic Clearance:* The benzofuran core and ester/amide linkages are susceptible to hepatic microsomal oxidation and esterase hydrolysis.
  2. *Lipophilicity / Solubility Balance:* High cLogP (~4.2) requires formulation surfactants; improvements in aqueous solubility without loss of Pks13 pocket affinity are essential.
  3. *Mutational Resilience:* Emergence of resistant mutations in the catalytic channel (e.g., Asp1644, Asn1640) requires analogues that engage backbone contacts rather than mutable side chains.
  4. *Freedom-to-Operate (FTO):* Identification of novel, patentable chemotypes with bioisosteric core replacements.

---

## 2. Why Most Computational Platforms Fail Wet-Lab Scrutiny

Medicinal chemists frequently discard purely computational platforms due to five failure modes:

| Failure Mode | Chemist's Criticism | Real-World Consequence |
| :--- | :--- | :--- |
| **Non-Synthesizable "Franken-Molecules"** | *"AI gave me a compound with 5 chiral centers and no retrosynthetic route."* | Quoted at $20,000+ by CROs; rejected by synthesis team; zero wet-lab progression. |
| **Pan-Assay Interference (PAINS)** | *"This molecule has a reactive quinone / rhodanine core."* | False positive in optical/enzymatic assays; unspecific covalent protein aggregation. |
| **Grease-Ball Artifacts** | *"Algorithm just added greasy aromatic rings to maximize lipophilic contact."* | High predicted affinity, but precipitation in water (cLogP > 5.5); inactive in mouse PK. |
| **Rigid Crystal False Positives** | *"The model assumed rigid atoms and missed severe side-chain steric clashes."* | Zero binding affinity in isothermal titration calorimetry (ITC) assays. |
| **Lack of Actionable Ordering** | *"It gave me a SMILES string, but no building block catalog numbers or reaction steps."* | Requires weeks of manual retrosynthetic planning before ordering starting materials. |

---

## 3. High-Value Objectives for Medicinal Chemists & Pharmacologists

### Objective 1: Synthetically Accessible Reaction-Driven Exploration (Make-on-Demand)
- Confine structural permutations to **robust medicinal chemistry reactions** (e.g., amide Schotten-Baumann, Suzuki-Miyaura cross-coupling, Buchwald-Hartwig amination, reductive amination).
- Restrict fragment choices to building blocks stocked in commercial on-demand catalogs (e.g., Enamine REAL Space, Mcule, Chemspace).

### Objective 2: Comprehensive Preclinical ADMET Profiling
- Expand multi-parametric optimization beyond standard Lipinski/Veber rules to include:
  - **Metabolic Stability:** Predicted human and mouse liver microsomal clearance ($CL_{\text{int}}$).
  - **Cardiotoxicity Filter:** hERG potassium channel inhibition ($IC_{50} > 10\,\mu\text{M}$ requirement).
  - **Permeability:** Predicted Caco-2 / PAMPA membrane permeability to ensure intracellular penetration through the thick mycobacterial cell envelope.
  - **Mutagenicity / Tox Alerts:** Automated flagging of Ames mutagenicity and genotoxic alerts (Derek-style structural filters).

### Objective 3: Scaled Digital Annealing of Physical Fragment Assembly
- Scale the Ising/QUBO Hamiltonian from the 12-variable proof-of-concept benchmark to a **60–120 variable physical docking model**:
  - 6 distinct Pks13 pocket sub-sites (Sub-pocket 0 Core, Sub-pocket 1 Linker, Sub-pocket 2 Lipophilic Channel, Sub-pocket 3 P1 Hydrophobic Cap, Sub-pocket 4 Catalytic Triad Ser1533/Asp1644, Sub-pocket 5 Solvent Exit).
  - 10–20 high-diversity fragment conformers per sub-pocket.
- Include directional hydrogen bonding and explicit desolvation free-energy terms ($\Delta G_{\text{desolv}}$).

### Objective 4: Resistance & Multi-Conformation Screening
- Cross-dock candidates against both the wild-type Pks13 crystal structure (PDB: `5V3Y`) and simulated resistance mutant pockets (e.g., Asp1644Gly, Asn1640Ala) to quantify resistance resilience scores.

### Objective 5: Turnkey Synthesis Dossier & CRO Procurement
- Generate downloadable wet-lab dossiers complete with:
  - IUPAC names, CAS numbers, and Enamine building block IDs.
  - 2-to-3 step forward synthetic schemes with reaction conditions.
  - Recommended biochemical assay protocol ($IC_{50}$ on recombinant Pks13) and cell-based assay protocol ($MIC_{90}$ on *M. tuberculosis* H37Rv).

---

## 4. Prioritized Engineering Roadmap & TODO Tasks

### Phase 1: Chemistry Quality & ADMET Hardening (High Priority)

- [ ] **Task 1.1: Integrate Strict MedChem Structural Cleanliness Filters**
  - Implement RDKit-based **PAINS (Pan-Assay Interference Compounds)** filter (Filter-it / Baell rules).
  - Implement **Brenk structural alert filter** (flagging toxicophores, reactive Michael acceptors, alkyl halides, hydrazines).
  - Add **BMS / Abbott Rule-of-Two (Ro2)** lead-likeness checks for fragment libraries ($MW \le 300$, $\text{cLogP} \le 3$, $\text{RotB} \le 3$).

- [ ] **Task 1.2: Implement Preclinical ADMET & Solubility Surrogate Predictors**
  - Integrate fast QSAR / GNN predictors for:
    - Aqueous thermodynamic solubility ($\log S$, targeting $> 50\,\mu\text{M}$).
    - hERG cardiac liability risk (binary classification with $> 85\%$ accuracy).
    - Mouse microsomal metabolic stability ($t_{1/2} > 30\,\text{min}$).
  - Display these metrics alongside QED and SA in Tab 1 and Tab 2.

- [ ] **Task 1.3: Real-Time Retrosynthetic Accessibility Scoring**
  - Replace static heuristics with **SCScore (Synthetic Complexity Score)** or open-source retrosynthetic tree validation (AiZynthFinder / ASKCOS API integration).
  - Filter out any candidate that cannot be assembled in $\le 4$ standardized synthetic steps from commercial starting materials.

---

### Phase 2: Structural Physics & QUBO Scalability (Core Computational Depth)

- [ ] **Task 2.1: Scale Fragment Docking Hamiltonian to 60–90 Qubits**
  - Discretize the Pks13 binding cleft into 6 well-defined pharmacophore regions:
    1. *Scaffold Anchor:* Catalytic pocket base.
    2. *Central Linker:* Amide/bioisostere channel.
    3. *Lipophilic Tunnel:* Interacting with Phe1585, Tyr1663.
    4. *P1 Cap:* Hydrophobic pocket near Ala1534.
    5. *Catalytic Contact:* H-bond network with Ser1533 and Asp1644.
    6. *Solvent Front:* Solubilizing extension zone.
  - Compute pairwise steric clash ($B$) and connectivity ($C$) matrices for 15 fragment candidates per site.

- [ ] **Task 2.2: Incorporate Explicit Desolvation & Water Thermodynamics**
  - Add local desolvation penalty terms ($\Delta G_{\text{desolv}}$) into the diagonal $dG$ coefficients: hydrophobic fragment placement into deep lipophilic pockets receives favorable desolvation bonuses, while burying un-paired polar groups incurs severe penalties.

- [ ] **Task 2.3: Post-Annealing Force-Field Energy Minimization (OpenMM / RDKit MMFF94)**
  - Automatically feed the decoded QUBO fragment assembly into local gradient relaxation (50 steps of conjugate gradient) inside the static protein field to eliminate micro-clashes and calculate refined interaction energies.

---

### Phase 3: Digital Annealing & Algorithmic Innovation

- [ ] **Task 3.1: Implement Pharmacophore-Decomposed Block Simulated Bifurcation (Block-tSB)**
  - For expanded 60+ qubit systems, implement block-coordinate relaxation:
    - Step 1: Solve Core Anchor + Catalytic Triad via tSB.
    - Step 2: Freeze core coordinates and anneal peripheral Tail + Cap fragments.
  - Benchmarks convergence speed vs. monolithic tSB and classical solvers.

- [ ] **Task 3.2: Automated TTS99 Scaling & Parameter Tuning**
  - Automatically calibrate bifurcation parameters ($\xi$, pump schedule $dt$, tabu vector weights) based on the spectral radius of the interaction matrix $J$.

---

### Phase 4: Pharmacological Dossier & Wet-Lab Handoff (Translational Utility)

- [ ] **Task 4.1: Turnkey Wet-Lab Synthesis Order Dossier**
  - In Tab 5, generate an exportable PDF/CSV **Preclinical Synthesis Package** containing:
    - Target 2D structure, SMILES, InChIKey.
    - Reagent catalog links (Enamine, Mcule, Sigma-Aldrich).
    - Predicted physical properties ($MW$, $\text{cLogP}$, $TPSA$, $SA$).
    - Predicted Pks13 binding mode figure (3D pocket snapshot).
    - Standardized bioassay protocol: Pks13 fluorogenic esterase assay conditions (50 mM Tris-HCl, pH 7.5, 0.01% Triton X-100, substrate: 4-methylumbelliferyl heptanoate).

- [ ] **Task 4.2: Resistance Mutation Profiling Panel**
  - Allow pharmacologists to toggle the target protein between Wild-Type (5V3Y) and mutant variants (e.g., Asp1644Gly, Ser1533Ala) to immediately see if candidate binding stability drops.

---

## 5. Success Metrics for Platform Maturity

| Stage | Milestone Criteria | Target Timeline |
| :--- | :--- | :--- |
| **Stage 1 (Current)** | Verified 12-qubit benchmark, MPO Pareto exploration, 2D/3D inspection, and active library curation. | Completed |
| **Stage 2 (Filter Hardening)** | Strict PAINS/Brenk exclusion, ADMET risk scoring, and Enamine building block alignment. | Milestone 1 (2 Weeks) |
| **Stage 3 (Physics Scaling)** | 60-qubit Pks13 sub-pocket Hamiltonian, desolvation penalties, MMFF post-docking relaxation. | Milestone 2 (4 Weeks) |
| **Stage 4 (Wet-Lab Handoff)** | 1-click CRO synthesis order dossier, retrosynthesis trees, resistance mutation profiling. | Milestone 3 (6 Weeks) |
