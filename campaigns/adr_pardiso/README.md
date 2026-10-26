# ADR PyPardiso direct-solve campaign

Retests every archived ADR report system with repeated multithreaded PyPardiso solves on the CPU:
thread autotuning, warm-up and repeated factorizations and solves, sequential workers guarded by
memory and timeout. The default invocation is a read-only inventory; `--execute` writes a separate
output directory. It never rebuilds, reassembles, reruns iterative solvers or edits the reports. Its
timings are the direct-solver reference of the ADR solver comparison and of the dossier's phase 2.

| | |
|---|---|
| Status | complete (2026-09-22, local CPU, 16 threads verified) |
| Launcher | `run_adr_pardiso_campaign.py`; worker `adr_pardiso_worker.py`; `adr_pardiso_inventory.py` (the archived systems); `adr_pardiso_tuning.py` (thread autotuning) |
| Output directory | `run_outputs/solver_studies/adr_scaling_2026_09_17/pardiso_all_2026_09_22` (default), `pardiso_tuned_2026_09_22` |
| Records | `docs/research/solver_studies/adr_pardiso_campaign.md` |
| Tests | `tests/test_adr_pardiso_campaign.py`, `tests/test_stress_pardiso_selection.py` |

## Commands

Inventory (read-only) and the tuned run (starts CPU factorizations; no CUDA wrapper needed):

```bash
.venv/bin/python -B campaigns/adr_pardiso/run_adr_pardiso_campaign.py --help
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B campaigns/adr_pardiso/run_adr_pardiso_campaign.py --autotune-threads --threads 1 2 4 6 8 12 16 24 --tuning-repeats 3 --repeats 5
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u -B campaigns/adr_pardiso/run_adr_pardiso_campaign.py --output run_outputs/solver_studies/adr_scaling_2026_09_17/pardiso_tuned_2026_09_22 --autotune-threads --threads 1 2 4 6 8 12 16 24 --tuning-repeats 3 --warmup 1 --repeats 5 --execute
```

## Additional saved output directories

- [pardiso_tuned_2026_09_22](records/pardiso_tuned_2026_09_22/DATA.md).
