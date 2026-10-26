# Guiding-center scripts

Run commands from the repository root. All six time schemes use the same case
CLI, `run_guiding_center_cases.py`; its AMGX launcher forwards to that CLI:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases --help
HYBRIDGE_PYTHON="$PWD/.venv/bin/python" scripts/guiding_center/run_local_amgx_cases.sh --help
```

The CLI selects a preset, applies overrides, and starts the shared runner. It
contains no time-stepping equations. Guiding-center evolution composes the
package diffusion and advection HDG solvers, so its temporal schemes belong
in [`time_schemes/`](time_schemes/), alongside case policy and orchestration.
The `hybridge` package supplies the HDG solvers, assembly, field/trace operations,
and shared diagnostics and I/O. The canonical, tested commands of every case
family, study and campaign are in [`example_runs.md`](example_runs.md); keep it
current in the same change as any script, option, preset or campaign change.

## Layout

| Directory | Contents |
|---|---|
| `cases/` | Case definitions, curated run presets, band seeds and imported equilibria. |
| `time_schemes/` | SI Euler, predictor-corrector, SI BDF2, SI BDF3, H1/H2-BDF3, IMEX-ARK3, and their common registry and stage data. |
| `runtime/` | CLI arguments, configuration, stepper construction, execution, diagnostics, plotting policy, terminal logging; the shared step observers (`observers.py`), device housekeeping (`device.py`), the kernel-cache guard (`kernel_cache.py`), Poisson-tau recovery (`poisson_recovery.py`) and the fastest device profiles. |
| `diagnostics/` | Shared models and readers imported by the runtime and the studies: diocotron diagnostics and reference spectra, the magnetron reference and the run-record readers (`run_records.py`). Nothing study-specific lives here. |
| `studies/<study>/` | One directory per study with its drivers, analysis and figure scripts and a README giving the exact commands, output directories and follow-ups (table below). |
| `viewers/` | HoloViz smoke and animation checks of the runner's viewer. |
| `docs/` | The scheme, recovered-field, exact-drift, showcase, diagnostic-history, plotting and legacy notes (table below). |
| `reference/` | The archived IMEX-ARK3 runner (`legacy_ark3/`, frozen with hashes) and the independent DOLFINx CG/SUPG torsion case; see [`docs/legacy.md`](docs/legacy.md). |
| `paths.py` | `REPO_ROOT`, `AMGX_CONFIGS`, `CACHE`, `RUN_OUTPUTS`, `ARTICLE_OUTPUTS` and `NATIVE_AMGX` for every script; no script computes `parents[N]` for the repository root any more (the `sys.path` guards of directly started scripts are the one exception, since they run before this module is importable). |

Campaign launchers, their records and the shared launch protocol live in the
repository's [`campaigns/`](../../campaigns/README.md); a study driver never
starts a campaign itself.

## Studies

| Study | Drivers (start solves) | Analysis and figures (no solves) | README |
|---|---|---|---|
| Convergence (article Section 5, Appendices B–C; the H100 campaign `campaigns/gc_article_convergence`) | `run_guiding_center_temporal_convergence`, `run_poisson_degree_study`, `exact_drift_transport_rates`, `steady_transport_velocity_rates`, `snapshot_manufactured_solution` | `make_convergence_tables`, `plot_convergence_study`, `guiding_center_temporal_comparison` | [`studies/convergence/`](studies/convergence/README.md) |
| Diocotron (disk and annulus benchmarks, the D8 order study and its D8-S smoke) | `run_diocotron_benchmark` | `analyze_diocotron`, `analyze_diocotron_campaign`, `analyze_diocotron_order`, `analyze_diocotron_pairs`, `diocotron_order_errors`, `plot_diocotron_*`, `render_diocotron_movie`, `preflight_diocotron` | [`studies/diocotron/`](studies/diocotron/README.md) |
| Magnetron (D7 vane magnetron, the collisional discharge, E9 CG-versus-HDG Poisson timing) | `run_magnetron`, `run_poisson_cg_vs_hdg` | `magnetron_convergence`, `magnetron_order_efficiency`, `plot_magnetron`, `plot_discharge`, `write_magnetron_tables` | [`studies/magnetron/`](studies/magnetron/README.md) |
| Band evolution (E7: control and seeded evolution of imported equilibrium bands, Section 7.2) | `run_band_evolution` | `analyze_band_evolution`, `plot_band_evolution_article`, `equilibrium_transfer_check` | [`studies/band_evolution/`](studies/band_evolution/README.md) |
| Equilibrium preservation (bounded CPU checks of archived equilibria) | `run_equilibrium_preservation` | `plot_equilibrium_preservation` | [`studies/equilibrium_preservation/`](studies/equilibrium_preservation/README.md) |
| Recovered field (bounded matrix diagnostics of the electric-field recovery order) | none | `check_field_recovery_order`, `check_oversampled_recovery_order`, `compare_field_recovery` | [`studies/recovered_field/`](studies/recovered_field/README.md) |
| Solver tuning (Poisson backends, AMGX smoothing, saved-system replays, direct solves, precision, host stages) | `benchmark_*`, `tune_*`, `replay_*`, `hierarchy_product_benchmark`, `run_precision_benchmark` | `analyze_poisson_timings`, `audit_solver_convergence`, `asm_patches`, `amgx_bsr_smoothing` | [`studies/solver_tuning/`](studies/solver_tuning/README.md) |

## Documentation

| Note | Content |
|---|---|
| [`docs/schemes.md`](docs/schemes.md) | Schemes and tested presets; the Poisson-tau fallback for every scheme. |
| [`docs/recovered_field.md`](docs/recovered_field.md) | BDF2 with reduced-order Poisson and recovered electric field; the electric-field convergence diagnostic. |
| [`docs/exact_drift.md`](docs/exact_drift.md) | Exact drift for reference runs. |
| [`docs/showcase_cases.md`](docs/showcase_cases.md) | Positive guiding-center turbulence; horseshoe, ITER and Pac-Man Euler gas. |
| [`docs/diagnostic_histories.md`](docs/diagnostic_histories.md) | Matplotlib diagnostic histories and their display options. |
| [`docs/plotting.md`](docs/plotting.md) | Plotting ownership between `hybridge.io` and the scripts. |
| [`docs/legacy.md`](docs/legacy.md) | What each legacy tool and `reference/` item is, where it was last used, and why it is kept. |

## Common entry points

The examples below only display options; invoking a runner without `--help`
may perform solves or time integration.

```bash
# Saved-system hybrid hierarchy CSR/BSR feasibility study
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.benchmark_hybrid_hierarchy_bsr --help

# Replay existing Poisson captures
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.replay_poisson_bsr --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.replay_poisson_asm --help

# Temporal comparisons and precision benchmarks
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence --help
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.run_precision_benchmark --help

# Analyze existing output and inspect a diocotron case
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.analyze_poisson_timings --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.preflight_diocotron --help
```

Use the [CUDA launcher](../gpu/run_cuda13.sh) and the repository's configured
runtime environment for GPU work. No native library or Python package rebuild
is needed for this directory layout.

## Imports and saved studies

Presets come from `scripts.guiding_center.cases.guiding_center_presets`.
Time-stepping classes come from `scripts.guiding_center.time_schemes`.
They reuse `hybridge` helpers to compose the existing HDG solvers.
Programmatic study drivers import
`run_guiding_center_case` from `scripts.guiding_center.runtime.runner`, and
snapshot/result types from `scripts.guiding_center.runtime.models`. They reuse
the same execution path as the user CLI. Tests and study drivers contain
validation and study orchestration, not separate scheme implementations.
A study imports shared code from `runtime/`, `cases/` and `diagnostics/`, never
from another study.

Repository imports, current commands and documentation links use the current
locations. Historical files under `artifacts/`, `run_outputs/` and
`docs/research/*.json` retain their original source paths and hashes. When
adapting an archived command, translate its module path with the map below.

## Path map (reorganization of 2026-10-26)

| Old path | New path |
|---|---|
| `scripts/guiding_center/benchmarks/campaign_protocol.py` | `campaigns/common/protocol.py` |
| `scripts/guiding_center/poisson/poisson_recovery.py` | `scripts/guiding_center/runtime/poisson_recovery.py` |
| `scripts/guiding_center/diagnostics/analyze_band_evolution.py` | `scripts/guiding_center/studies/band_evolution/analyze_band_evolution.py` |
| `scripts/guiding_center/diagnostics/equilibrium_transfer_check.py` | `scripts/guiding_center/studies/band_evolution/equilibrium_transfer_check.py` |
| `scripts/guiding_center/diagnostics/plot_band_evolution_article.py` | `scripts/guiding_center/studies/band_evolution/plot_band_evolution_article.py` |
| `scripts/guiding_center/benchmarks/run_band_evolution.py` | `scripts/guiding_center/studies/band_evolution/run_band_evolution.py` |
| `scripts/guiding_center/benchmarks/exact_drift_transport_rates.py` | `scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py` |
| `scripts/guiding_center/benchmarks/guiding_center_temporal_comparison.py` | `scripts/guiding_center/studies/convergence/guiding_center_temporal_comparison.py` |
| `scripts/guiding_center/benchmarks/make_convergence_tables.py` | `scripts/guiding_center/studies/convergence/make_convergence_tables.py` |
| `scripts/guiding_center/benchmarks/plot_convergence_study.py` | `scripts/guiding_center/studies/convergence/plot_convergence_study.py` |
| `scripts/guiding_center/benchmarks/run_guiding_center_temporal_convergence.py` | `scripts/guiding_center/studies/convergence/run_guiding_center_temporal_convergence.py` |
| `scripts/guiding_center/benchmarks/run_poisson_degree_study.py` | `scripts/guiding_center/studies/convergence/run_poisson_degree_study.py` |
| `scripts/guiding_center/benchmarks/snapshot_manufactured_solution.py` | `scripts/guiding_center/studies/convergence/snapshot_manufactured_solution.py` |
| `scripts/guiding_center/benchmarks/steady_transport_velocity_rates.py` | `scripts/guiding_center/studies/convergence/steady_transport_velocity_rates.py` |
| `scripts/guiding_center/diagnostics/analyze_diocotron.py` | `scripts/guiding_center/studies/diocotron/analyze_diocotron.py` |
| `scripts/guiding_center/diagnostics/analyze_diocotron_campaign.py` | `scripts/guiding_center/studies/diocotron/analyze_diocotron_campaign.py` |
| `scripts/guiding_center/diagnostics/analyze_diocotron_order.py` | `scripts/guiding_center/studies/diocotron/analyze_diocotron_order.py` |
| `scripts/guiding_center/diagnostics/analyze_diocotron_pairs.py` | `scripts/guiding_center/studies/diocotron/analyze_diocotron_pairs.py` |
| `scripts/guiding_center/diagnostics/diocotron_order_errors.py` | `scripts/guiding_center/studies/diocotron/diocotron_order_errors.py` |
| `scripts/guiding_center/diagnostics/plot_diocotron_growth.py` | `scripts/guiding_center/studies/diocotron/plot_diocotron_growth.py` |
| `scripts/guiding_center/diagnostics/plot_diocotron_highmode.py` | `scripts/guiding_center/studies/diocotron/plot_diocotron_highmode.py` |
| `scripts/guiding_center/diagnostics/plot_diocotron_injection.py` | `scripts/guiding_center/studies/diocotron/plot_diocotron_injection.py` |
| `scripts/guiding_center/diagnostics/plot_diocotron_ionization.py` | `scripts/guiding_center/studies/diocotron/plot_diocotron_ionization.py` |
| `scripts/guiding_center/diagnostics/plot_diocotron_smooth.py` | `scripts/guiding_center/studies/diocotron/plot_diocotron_smooth.py` |
| `scripts/guiding_center/diagnostics/plot_diocotron_zg.py` | `scripts/guiding_center/studies/diocotron/plot_diocotron_zg.py` |
| `scripts/guiding_center/diagnostics/preflight_diocotron.py` | `scripts/guiding_center/studies/diocotron/preflight_diocotron.py` |
| `scripts/guiding_center/diagnostics/render_diocotron_movie.py` | `scripts/guiding_center/studies/diocotron/render_diocotron_movie.py` |
| `scripts/guiding_center/benchmarks/run_diocotron_benchmark.py` | `scripts/guiding_center/studies/diocotron/run_diocotron_benchmark.py` |
| `scripts/guiding_center/benchmarks/equilibrium_preservation.md` | `scripts/guiding_center/studies/equilibrium_preservation/README.md` |
| `scripts/guiding_center/diagnostics/plot_equilibrium_preservation.py` | `scripts/guiding_center/studies/equilibrium_preservation/plot_equilibrium_preservation.py` |
| `scripts/guiding_center/benchmarks/run_equilibrium_preservation.py` | `scripts/guiding_center/studies/equilibrium_preservation/run_equilibrium_preservation.py` |
| `scripts/guiding_center/diagnostics/magnetron_convergence.py` | `scripts/guiding_center/studies/magnetron/magnetron_convergence.py` |
| `scripts/guiding_center/diagnostics/magnetron_order_efficiency.py` | `scripts/guiding_center/studies/magnetron/magnetron_order_efficiency.py` |
| `scripts/guiding_center/diagnostics/plot_discharge.py` | `scripts/guiding_center/studies/magnetron/plot_discharge.py` |
| `scripts/guiding_center/diagnostics/plot_magnetron.py` | `scripts/guiding_center/studies/magnetron/plot_magnetron.py` |
| `scripts/guiding_center/benchmarks/run_magnetron.py` | `scripts/guiding_center/studies/magnetron/run_magnetron.py` |
| `scripts/guiding_center/benchmarks/run_poisson_cg_vs_hdg.py` | `scripts/guiding_center/studies/magnetron/run_poisson_cg_vs_hdg.py` |
| `scripts/guiding_center/diagnostics/write_magnetron_tables.py` | `scripts/guiding_center/studies/magnetron/write_magnetron_tables.py` |
| `scripts/guiding_center/diagnostics/check_field_recovery_order.py` | `scripts/guiding_center/studies/recovered_field/check_field_recovery_order.py` |
| `scripts/guiding_center/diagnostics/check_oversampled_recovery_order.py` | `scripts/guiding_center/studies/recovered_field/check_oversampled_recovery_order.py` |
| `scripts/guiding_center/diagnostics/compare_field_recovery.py` | `scripts/guiding_center/studies/recovered_field/compare_field_recovery.py` |
| `scripts/guiding_center/poisson/amgx_bsr_smoothing.py` | `scripts/guiding_center/studies/solver_tuning/amgx_bsr_smoothing.py` |
| `scripts/guiding_center/diagnostics/analyze_poisson_timings.py` | `scripts/guiding_center/studies/solver_tuning/analyze_poisson_timings.py` |
| `scripts/guiding_center/poisson/asm_patches.py` | `scripts/guiding_center/studies/solver_tuning/asm_patches.py` |
| `scripts/guiding_center/diagnostics/audit_solver_convergence.py` | `scripts/guiding_center/studies/solver_tuning/audit_solver_convergence.py` |
| `scripts/guiding_center/benchmarks/benchmark_host_solver_stages.py` | `scripts/guiding_center/studies/solver_tuning/benchmark_host_solver_stages.py` |
| `scripts/guiding_center/poisson/benchmark_hybrid_hierarchy_bsr.py` | `scripts/guiding_center/studies/solver_tuning/benchmark_hybrid_hierarchy_bsr.py` |
| `scripts/guiding_center/poisson/benchmark_poisson_backends.py` | `scripts/guiding_center/studies/solver_tuning/benchmark_poisson_backends.py` |
| `scripts/guiding_center/poisson/benchmark_poisson_lu.py` | `scripts/guiding_center/studies/solver_tuning/benchmark_poisson_lu.py` |
| `scripts/guiding_center/poisson/benchmark_repeated_poisson.py` | `scripts/guiding_center/studies/solver_tuning/benchmark_repeated_poisson.py` |
| `scripts/guiding_center/poisson/benchmark_scipy_lu_gpu.py` | `scripts/guiding_center/studies/solver_tuning/benchmark_scipy_lu_gpu.py` |
| `scripts/guiding_center/poisson/hierarchy_product_benchmark.py` | `scripts/guiding_center/studies/solver_tuning/hierarchy_product_benchmark.py` |
| `scripts/guiding_center/poisson/replay_poisson_asm.py` | `scripts/guiding_center/studies/solver_tuning/replay_poisson_asm.py` |
| `scripts/guiding_center/poisson/replay_poisson_bsr.py` | `scripts/guiding_center/studies/solver_tuning/replay_poisson_bsr.py` |
| `scripts/guiding_center/benchmarks/run_precision_benchmark.py` | `scripts/guiding_center/studies/solver_tuning/run_precision_benchmark.py` |
| `scripts/guiding_center/poisson/tune_poisson_bsr_smoothing.py` | `scripts/guiding_center/studies/solver_tuning/tune_poisson_bsr_smoothing.py` |
| `scripts/guiding_center/poisson/tune_poisson_p0_amgx.py` | `scripts/guiding_center/studies/solver_tuning/tune_poisson_p0_amgx.py` |
| `scripts/guiding_center/diagnostics/animate_holoviz.py` | `scripts/guiding_center/viewers/animate_holoviz.py` |
| `scripts/guiding_center/diagnostics/smoke_holoviz.py` | `scripts/guiding_center/viewers/smoke_holoviz.py` |
| `scripts/guiding_center/benchmarks/run_observers.py` | `scripts/guiding_center/runtime/observers.py` (absorbed, with the `Observers`, `SnapshotSaver` and `MovieObserver` classes that were private to `run_diocotron_benchmark`) |
| `_release_device_memory` (three copies) | `scripts/guiding_center/runtime/device.py` (`release_device_memory`) |
| `guiding_center_temporal_comparison.kernel_cache_only` | `scripts/guiding_center/runtime/kernel_cache.py` (still re-exported from the comparison module) |
| `plot_diocotron_highmode._records/_series`, `plot_diocotron_ionization._image/_raster` | `scripts/guiding_center/diagnostics/run_records.py` |

## Adding a study

1. Create `studies/<name>/` with a one-line `__init__.py` and a `README.md`
   stating the purpose, the article section or dossier phase it feeds, the
   drivers with their exact commands (`--help` and `--dry-run` forms first),
   the output directory (`run_outputs/article/<study>` for article work,
   `run_outputs/<area>/<study>` otherwise), the analysis and figure commands,
   the tests and the open follow-ups.
2. Put anything two studies need in `runtime/` or `diagnostics/`; a study never
   imports another study's modules.
3. Add the commands to `example_runs.md`; `tests/test_guiding_center_example_runs.py`
   checks every module, option, preset, case and `@run_configs` file it names.
4. Campaign launchers go to `campaigns/<campaign>/` with their README and records
   (conventions in `campaigns/README.md`).
5. Update the `CODEMAP.md` rows of every file added, moved or removed.
