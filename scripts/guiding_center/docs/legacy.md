# Legacy tools and reference items

What each item that is no longer part of a current study is, where it was last used, and why it
stays. Nothing here is imported by the runtime, the studies or the campaigns, except where noted.

| Item | What it is | Last use | Why it stays |
|---|---|---|---|
| `reference/legacy_ark3/` | The archived monolithic runner with the Kennedy–Carpenter IMEX-ARK3 stepper, restored from the snapshot taken before the runtime split; `provenance.json` records the source and restored SHA-256 of every file; it still imports `diagnostics/diocotron_reference.py` and `diagnostics/diocotron_diagnostics.py`, which therefore stay in `diagnostics/`. | `docs/research/solver_studies/ark3_wiring_audit_2026_09.md`; `run_configs/guiding_center/*legacy_ark3*.args`; `tests/test_guiding_center_legacy_ark3.py` | Independent comparison runner for the current ARK3 wiring; frozen, excluded from every path rewrite so its hashes stay valid. |
| `reference/dolfinx_torsion_guiding_center_supg.py` | The independent DOLFINx CG+SUPG guiding-center run from a torsion-designed band (needs the `diocotron-fenicsx` environment). | `docs/research/torsion_initialized_equilibrium/stationarity_report_2026_08_20.md`; the D7 SUPG counterpart lives in `projects/diocotron/comparisons/discharge_supg.py` | Method-independent reference for the imported-equilibrium runs. |
| `viewers/smoke_holoviz.py`, `viewers/animate_holoviz.py` | Static and animated checks of the runner's HoloViz viewer (no PDE). | `docs/backends/holoviz.md` | The only way to check the GPU viewer without a run. |
| `studies/solver_tuning/benchmark_scipy_lu_gpu.py` | The explicitly requested SciPy LU → GPU experiment, an exception to the PyPardiso-only rule. | `docs/research/solver_studies/iter_scipy_lu_gpu_experiment.md` (2026-09-24/25) | Its tests (`test_lu_benchmark_monitor`, `test_cupy_triangular`) cover shared pieces. |
| `studies/solver_tuning/benchmark_host_solver_stages.py` | Stage-matched host solver study of the nonlinear Gaussian annulus. | `tests/test_guiding_center_host_solver_benchmark.py` | Host reference of the solver stage costs. |
| `studies/solver_tuning/audit_solver_convergence.py` | Iterative-solver audit of run records. | no document references it | Still the quickest audit of a campaign's solver health; delete if unused by the next campaign. |
| `studies/solver_tuning/hierarchy_product_benchmark.py` | Preallocated native-AMGX and cuSPARSE product measurements. | `docs/research/solver_studies/hybrid_hierarchy_bsr_feasibility_2026_09.md` (through `benchmark_hybrid_hierarchy_bsr.py`) | Sibling of the feasibility study. |
| `studies/recovered_field/check_oversampled_recovery_order.py` | Experimental oversampled recovery (not a production recovery API). | `docs/recovered_field.md` | Evidence for the recovery-order discussion. |
| `campaigns/gc_diocotron_shell/run_campaigns*.sh` (v1–v10) | The successive versions of the first diocotron campaign chain; every later version superseded the previous one. | `run_outputs/article/diocotron` (complete) | Provenance of the published diocotron benchmarks; see the campaign README. |

Historical commands in `artifacts/`, `run_outputs/` and `docs/research/*.json` keep their old module
paths; translate them with the path map in [`../README.md`](../README.md#path-map-reorganization-of-2026-10-26).
