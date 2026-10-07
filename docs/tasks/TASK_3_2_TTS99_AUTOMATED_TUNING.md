# Task 3.2: Automated TTS99 Scaling & Parameter Tuning

## Parent Subtask
**Task 3.2: Automated TTS99 Scaling & Parameter Tuning** (from `docs/TRANSLATIONAL_ROADMAP.md`)

---

### Sub-subtask 3.2.1: Spectral Radius Calibration of Bifurcation Parameter $\xi$
- [pending] Sub-sub-subtask 3.2.1.1: Compute maximum eigenvalue $\lambda_{\max}$ and spectral norm of interaction matrix $J$ using PyTorch power iteration.
- [pending] Sub-sub-subtask 3.2.1.2: Set theoretical optimal Kerr coupling scale: $\xi^* = \frac{c}{\sqrt{N} \cdot \|J\|_2}$.
- [pending] Sub-sub-subtask 3.2.1.3: Validate that automatic $\xi^*$ avoids chaotic divergence while maintaining fast non-linear bifurcation.
- [pending] Sub-sub-subtask 3.2.1.4: Unit test adaptive scaling across varying problem sizes ($N = 12, 24, 60, 90$).

### Sub-subtask 3.2.2: Adaptive Pump Time Step and Tabu Weight Scheduling
- [pending] Sub-sub-subtask 3.2.2.1: Calibrate adiabatic pump schedule $dt$ as a function of condition number $\kappa(J)$.
- [pending] Sub-sub-subtask 3.2.2.2: Implement dynamic tabu weight decay: penalize recent local basins strongly while decaying older history exponentially.
- [pending] Sub-sub-subtask 3.2.2.3: Measure Success Probability ($P_{\text{success}}$) and $\text{TTS}_{99}$ confidence times over 100 random restarts.
- [pending] Sub-sub-subtask 3.2.2.4: Log optimal parameter configurations to `data/processed/solver_out/tuning_summary.json`.
