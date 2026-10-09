# Third-party source inventory

Use the clone script to retrieve these projects. Do not copy the full third-party sources into the X-TUBIT package; keep them under `third_party/` and record commit hashes.

| Component | Public source | Role | Status in X-TUBIT |
|---|---|---|---|
| Simulated Bifurcation | https://github.com/bqth29/simulated-bifurcation-algorithm | SB baseline/QUBO solver | adapter |
| TESB | https://github.com/Tao-qubit/Tabu-Enhanced-Simulated-Bifurcation | tabu-enhanced SB reference | reimplementation/adapter |
| pSA | https://github.com/nonizawa/pSA | CPU correctness reference for pSA/TApSA/SpSA | port |
| GPU-pSAv | https://github.com/nonizawa/GPU-pSAv | GPU kernel reference | optional legacy path |
| qPMHI | https://github.com/PaulsonLab/Generative_MOBO_qPMHI | official qPMHI acquisition implementation | adapter / audit source |
| FAENet | https://github.com/vict0rsch/faenet | 3D frame averaging and graph embedding | adapter |
| OCP training code | https://github.com/RolnickLab/ocp | FAENet training examples | reference |
| REstretto | https://github.com/akiyamalab/restretto | fragment placement/docking engine | external configuration adapter |
| COFFEE-PRESC | https://github.com/akiyamalab/coffee-presc | fragment decomposition/docking auxiliary | optional |
| Pi-Stacking | https://github.com/alebeneventi/Pi-Stacking | alternate QUBO docking reference | optional |
| PyQUBO | https://github.com/recruit-communications/pyqubo | symbolic QUBO/Ising compilation and constraint tracking | adapter / validation reference |

## Public data/paper sources

- Pks13 TAM series: Aggarwal et al. 2017, Cell, “Development of a Novel Lead that Targets M. tuberculosis Pks13”.
- Pks13 DEL series: Krieger et al. 2024, ACS Infectious Diseases 10(5):1561-1575.
- Pks13 main structure: RCSB PDB 5V3Y, TAM16, 1.98 Å.
- Related structures: 5V40, 5V41, 5V42; 8TQG, 8TQV, 8TR4; verify 8TRY before use.
- Yanagisawa et al. 2024, Entropy 26(5):397, fragment-QUBO docking, PDB 2HV5 benchmark.
- Yanagisawa benchmark dataset: Zenodo DOI 10.5281/zenodo.10889782.
- DUD-E: property-matched decoy benchmark, Journal of Medicinal Chemistry 2012.
- PubChem PUG REST: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial

## What was actually extracted from public documentation during blueprint preparation

FAENet docs expose:
- `FrameAveraging(frame_averaging, fa_method)`;
- 3D/2D/DA frame-averaging choices;
- `model_forward(batch, model, frame_averaging, mode=..., crystal_task=...)`;
- `hidden_state` as a final atom-level representation;
- symmetry evaluation utilities.

TESB public CPU implementation exposes:
- `SB(A, h=0, tabu=None, K=1, delta=1, dt=1., sigma=1., M=2, n_iter=1000, xi=None, sk=False, batch_size=1, num_tabu=1, device='cpu')`;
- separate linear field and tabu list;
- ballistic/discrete updates;
- row-sum or SK-style coupling scaling;
- linear pump schedule;
- inelastic boundary handling.

pSA public code exposes:
- pSA, TApSA, SpSA scripts;
- cycle/trial/tau/noise options;
- TApSA mean-range control;
- SpSA stall proportion control.

REstretto public README exposes:
- `atomgrid-gen`, `conformer-docking`;
- `INNERBOX`, `OUTERBOX`, `BOX_CENTER`, `SEARCH_PITCH`, `SCORING_PITCH`, `MEMORY_SIZE`, `RECEPTOR`, `LIGAND`, `OUTPUT`, `GRID_FOLDER`;
- build dependency on Open Babel 2.4.1.

qPMHI public paper/repository was identified, but its full GitHub source was not fully auditable through the current web fetch. The clone script therefore treats the repository as the authoritative source to inspect locally.
