# Solver tuning

Poisson and transport solver studies on captured or replayed guiding-center systems: backend
comparisons along one bounded trajectory, AMGX BSR smoothing and p = 0 coarse-solver tuning, saved
hybrid hierarchy CSR/BSR feasibility, overlapping-patch (ASM) smoothing, direct solves with PyPardiso
(and the one SciPy LU exception), repeated Poisson solves, hierarchy product measurements, the
FP32/FP64 precision benchmark and the host solver stage comparison. Their evidence is recorded under
`docs/research/solver_studies/` (file names carry the dates). None of them starts a campaign; the
fastest configurations they established are written into `runtime/fast_profiles.py`.

| Script | Role | Evidence |
|---|---|---|
| `benchmark_poisson_backends.py` | Matched-state Poisson backend comparison along one bounded Euler trajectory (`--order {4,5,6}`). | `docs/development/plans/face_block_hp_multigrid.md` |
| `tune_poisson_bsr_smoothing.py`, `amgx_bsr_smoothing.py` | Fully BSR AMG smoothers on shared systems from 2–5 Euler steps; configuration and algebraic checks. | tests |
| `tune_poisson_p0_amgx.py` | Parameter-only p = 0 AMGX study inside the native Poisson solver. | `docs/development/plans/face_block_hp_multigrid.md` |
| `benchmark_hybrid_hierarchy_bsr.py`, `hierarchy_product_benchmark.py` | Saved hybrid systems, exported hierarchy, BSR feasibility and preallocated product measurements. | `hybrid_hierarchy_bsr_feasibility_2026_09.md` |
| `replay_poisson_bsr.py`, `replay_poisson_asm.py`, `asm_patches.py` | Replay captured Poisson systems through the device AMGX wrapper; element ASM as the finest smoother. | tests |
| `benchmark_poisson_lu.py`, `benchmark_repeated_poisson.py` | Factor once and reuse (PyPardiso, 16 threads, verified); cached solves on fixed sources. | `iter_repeated_poisson_2026_09_24.md` |
| `benchmark_scipy_lu_gpu.py` | The explicitly requested SciPy LU → GPU experiment (an exception to the PyPardiso-only rule). | `iter_scipy_lu_gpu_experiment.md` |
| `run_precision_benchmark.py` | The complete pipeline at explicit process precision (FP32 against FP64). | `docs/development/fp32_guiding_center.md` |
| `benchmark_host_solver_stages.py` | Stage-matched host solver study (SciPy BiCGSTAB+ILU and PyPardiso shadow solvers). | tests |
| `analyze_poisson_timings.py` | Offline summary of existing Poisson logs (no HYBRIDGE import). | `guiding_center_poisson_timings_2026_09.md` |
| `audit_solver_convergence.py` | Iterative-solver audit of run records (iterations, residual over target, retries). | none; kept as a tool |

`asm_patches.py` and `amgx_bsr_smoothing.py` are libraries also used by
`scripts/diffusion_reaction/compare_cuda_bsr_csr.py`. The Poisson p-multigrid degree study of
2026-10-17 (`campaigns/gc_poisson_pmg_degrees`) drives `scripts/diffusion_reaction/benchmark_poisson_pmg_degrees.py`.

## Outputs

`run_outputs/solver_studies/<study>` (for example `iter_repeated_poisson_20260924`,
`iter_scipy_lu_gpu_20260924`, `iter_h003_lu_gpu_20260925`), and `run_outputs/article/poisson_pmg_degrees_20261017`
for the degree study.

## Commands

Options only (every script accepts `--help`; the replays and benchmarks solve when started):

```bash
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.benchmark_poisson_backends --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.tune_poisson_bsr_smoothing --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.tune_poisson_p0_amgx --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.benchmark_hybrid_hierarchy_bsr --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.replay_poisson_bsr --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.replay_poisson_asm --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.benchmark_poisson_lu --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.benchmark_repeated_poisson --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.benchmark_scipy_lu_gpu --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.run_precision_benchmark --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.benchmark_host_solver_stages --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.analyze_poisson_timings --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.audit_solver_convergence --help
```

Offline analyses (no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.analyze_poisson_timings --output-dir run_outputs/solver_studies/poisson_timings run_outputs/article/diocotron_order_20261016/runs/d8_C1_L2_p6_timings.jsonl
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.audit_solver_convergence --windows 0:20,20:70 --json run_outputs/solver_studies/audit.json run_outputs/article/diocotron_order_20261016/runs/d8_C1_L2_p6.jsonl
```

## Tests

`tests/test_amgx_bsr_smoothing.py`, `tests/test_amgx_bsr_dilu.py`, `tests/test_amgx_bsr_interpolation.py`,
`tests/test_amgx_asm_cycle.py`, `tests/test_asm_patches.py`, `tests/test_poisson_lu_benchmark.py`,
`tests/test_lu_benchmark_monitor.py`, `tests/test_cupy_triangular.py`, `tests/test_pcgf_initial_guess.py`,
`tests/test_precision.py`, `tests/test_guiding_center_host_solver_benchmark.py`.
