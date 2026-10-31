# Research Outputs

Research outputs are retained for reproducibility but do not define supported
package behavior.

- [`solver_studies/`](solver_studies/): dated advection-reaction, diffusion,
  combined ADR, and solver/preconditioner investigations.
- Guiding-center research (`../../../guiding-center-cg-hdg/docs/README.md`):
  equilibrium construction, convergence studies, solver experiments and saved data.
- `run_logs/` (local, untracked): raw benchmark logs and variability studies;
  the July 2026 AMGX and raw-CUDA findings notes are dated records in
  [`solver_studies/`](solver_studies/).

Paths under `artifacts/` and `run_outputs/` quoted in these studies, marked
"local, untracked", name the machine-local evidence directories where the
measurements were produced. Both are Git-ignored and not distributed with the
repository.
