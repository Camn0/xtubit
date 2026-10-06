# Source/Fidelity Matrix

| Claim/component | Source basis | Fidelity target | X-TUBIT action |
|---|---|---|---|
| Pks13 TAM-series enzymatic labels | Aggarwal 2017 | source values | import with assay provenance |
| Pks13 DEL hits | Krieger 2024 | source values | import only exact assay rows |
| 5V3Y | RCSB PDB | exact structure record | download/checksum |
| FAENet transform/API | public docs/repo | API-compatible | adapter |
| FAENet molecular transfer | not originally the target task | adaptation | fine-tune + ablation |
| qPMHI | Muthyala et al. 2025/2026 + public repo | acquisition-compatible | wrapper + toy verification |
| fragment-QUBO terms | Yanagisawa 2024 | replication on T0 where possible | reward-mode implementation |
| BRICS fragmentation | X-TUBIT engineering choice | adaptation | record deviation |
| RDKit conformers | X-TUBIT engineering choice | adaptation | ETKDGv3 + MMFF |
| REstretto | public same-lab placement code | placement-compatible | run/build |
| TESB | Tao et al. 2026 + public repo | algorithm-compatible | independent port + G1 validation |
| pSA/TApSA/SpSA | Onizawa & Hanyu 2024 + public repo | algorithm-compatible | independent port + G1 validation |
| SB | public PyTorch package | API-compatible | wrapper + toy exact tests |
| refinement | X-TUBIT engineering choice | not a literature replication | separate metric stage |

## Status labels

`VERIFIED`: directly supported by source/docs reviewed.

`ADAPTED`: same conceptual method, but X-TUBIT changes an implementation detail.

`ENGINEERING`: chosen for the MVP and must be benchmarked.

`UNVERIFIED`: do not use as evidence until checked.
