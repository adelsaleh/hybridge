# ADR closed-loop stress campaign

The stationary ADR stress comparison on the square and the annulus: the `trap`, `cross` and
`orthogonal` closed-loop cases at the main severity level, p = 6, 50k and 100k triangle targets,
every iterative solver against the PyPardiso reference (60 solver comparisons plus separate
profiles). On the closed-loop cases only ASM/BJ with polynomial preconditioning and pMG-AMG
converge (orthogonal case), AMGX never, and all are slower than PyPardiso; these are the stress cases
of the dossier's phase 2 (`campaigns/h100_solvers`).

| | |
|---|---|
| Status | complete (local GPU and CPU; `adr_closed_loop_stress_main`, `_main_v2`, the strong Numba h = 150k runs and the square BSR 2M short runs) |
| Launcher | `run_closed_loop_stress.py` (plan, `--prepare-only`, `--execute`); worker `closed_loop_stress_worker.py` (isolated solver tree); `closed_loop_stress_pardiso.py` (the PyPardiso reference through `scripts/advection_diffusion_reaction/diagnostics/check_cached_adr_pardiso.py`) |
| Output directory | `run_outputs/solver_studies/adr_closed_loop_stress_main` (15 GB), `adr_closed_loop_stress_main_v2`, `adr_closed_loop_stress_strong_numba_h150k` (55 GB), `adr_square_stress_strong_numba_h150k`, `adr_square_bsr_2m_short*` |
| Records | `docs/research/solver_studies/adr_closed_loop_stress_runner.md`, `docs/research/solver_studies/adr_solver_comparison_2026_09_17.md` |
| Tests | `tests/test_closed_loop_stress.py`, `tests/test_square_stress.py`, `tests/test_stress_pardiso_selection.py` |

## Commands

Plan (reads the configuration and prints JSON; no output directory, mesh or solve):

```bash
.venv/bin/python -B campaigns/adr_stress/run_closed_loop_stress.py --help
.venv/bin/python -B campaigns/adr_stress/run_closed_loop_stress.py --output run_outputs/solver_studies/adr_closed_loop_stress_main
```

Execution (starts solves) wraps the same command in the CUDA 13 launcher with `--execute`:

```bash
HYBRIDGE_CUDA13_ROOT=/path/to/cuda-13 HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13 HYBRIDGE_AMGX_INSTALL_ROOT=../AMGX-install-cuda13 scripts/gpu/run_cuda13.sh .venv/bin/python -B campaigns/adr_stress/run_closed_loop_stress.py --output run_outputs/solver_studies/adr_closed_loop_stress_main --execute
```
