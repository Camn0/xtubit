# Task 1.3: Real-Time Retrosynthetic Accessibility Scoring

## Parent Subtask
**Task 1.3: Real-Time Retrosynthetic Accessibility Scoring** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 1.3.1: Replace Static Heuristics with Synthetic Complexity Score (SCScore)
- [pending] Sub-sub-subtask 1.3.1.1: Integrate Coley et al. SCScore or fast open-source neural complexity surrogate into `src/xtubit/qpmhi.py`.
- [pending] Sub-sub-subtask 1.3.1.2: Calibrate SCScore scale (1.0 = trivial starting material, 5.0 = highly complex natural product).
- [pending] Sub-sub-subtask 1.3.1.3: Write automated unit tests evaluating SCScore for simple fragments vs. complex multi-ring polycycles.
- [pending] Sub-sub-subtask 1.3.1.4: Update qPMHI objective function to substitute or weight SCScore alongside Ertl SA score.

### Sub-subtask 1.3.2: Standardized Forward Synthetic Step Count Constraint
- [pending] Sub-sub-subtask 1.3.2.1: Implement forward step-estimator checking commercial building block availability via Enamine Building Block SMILES catalog.
- [pending] Sub-sub-subtask 1.3.2.2: Establish hard constraint rejecting candidates requiring $> 4$ synthetic transformation steps.
- [pending] Sub-sub-subtask 1.3.2.3: Validate step-count filter against TAM16 3-step synthesis published in Nature Medicine (2017).
- [pending] Sub-sub-subtask 1.3.2.4: Render synthetic route step count and primary reaction type badge in UI candidate dossier.
