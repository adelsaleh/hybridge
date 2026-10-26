# Diocotron study

The diocotron instability on the disk and the annulus: the smooth m = 9 ring (D1 ladders, D2, D4),
the Zoni–Güçlü truncated ring (D3), the Gaussian rings m = 64 and 128 (high modes), central
injection and ionization sources, the annulus against the semi-Lagrangian reference, and the D8
order study (p = 5 to 9 at fixed resolution, SI-BDF2, raw and compatible drift, with and without
the KKT positivity projection) with its D8-S smoke campaign that selected the fastest correct
device preset per degree. Growth rates are modal-plateau fits against the dispersion relation of
the sharp annulus; field displays use m = 128 only, with colour bars that include the negative
density.

| Script | Role |
|---|---|
| `run_diocotron_benchmark.py` | The driver of every benchmark (`--benchmark annulus-sl|disk-smooth|disk-zg|disk-highmode|disk-injection|disk-ionization`); `--dry-run` prints the configurations without meshing or solving. Starts solves otherwise. |
| `analyze_diocotron.py` | Growth, positivity, invariants and solver cost audit from the logs. |
| `analyze_diocotron_campaign.py` | Growth-rate tables of the campaign summaries (`--latex`). |
| `analyze_diocotron_order.py` | `select` and `report` of the D8-S smoke (fastest correct preset per degree, `presets_selected.json`, `smoke_report.md`). |
| `analyze_diocotron_pairs.py` | Cost per phase, reference-free accuracy and cross-mesh differences of a few runs of one D8 study. |
| `diocotron_order_errors.py` | Cross-mesh L2 errors of saved fields against a reference run. |
| `plot_diocotron_growth.py`, `plot_diocotron_smooth.py`, `plot_diocotron_zg.py`, `plot_diocotron_highmode.py`, `plot_diocotron_injection.py`, `plot_diocotron_ionization.py` | Article figures and tables of each benchmark family. |
| `render_diocotron_movie.py` | Supplementary MP4 movies from saved snapshots (`gc-hdg-article/supplementary/movies/render_movies.sh` calls it). |
| `preflight_diocotron.py` | Static projection and resolution audit of a case. |

Shared models (reference spectra, mode diagnostics, record readers) are in `scripts/guiding_center/diagnostics/`.

## Outputs

| Campaign | Directory | Status |
|---|---|---|
| Shell campaigns of the article's disk and annulus benchmarks (D0–D4, injection, ionization, movies) | `run_outputs/article/diocotron` | complete; launchers in `campaigns/gc_diocotron_shell` |
| D8-S smoke (preset selection, p = 1..9 then 5..9) | `run_outputs/article/diocotron_order_smoke_20261016` | complete (`SMOKE_DONE`); `campaigns/gc_d8s_smoke` |
| D8 order study (phases I and X done; P2a/P1/P2b/R/T planned) | `run_outputs/article/diocotron_order_20261016` | partial (`D8_I_DONE`, `D8_X_DONE`); `campaigns/gc_d8_diocotron_order` |

## Commands

Options only, then dry runs (no mesh, no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --benchmark disk-smooth --orders 6 --dts 0.05 --final-time 70 --schemes si-bdf2 --velocities raw --output-dir run_outputs/article/diocotron/smooth --dry-run
.venv/bin/python -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --benchmark disk-zg --orders 6 --dts 0.05 --schemes imex-ark3 --velocities raw --output-dir run_outputs/article/diocotron/zg --dry-run
```

One D8 run (the launcher's form; `--poisson-candidate`/`--transport-candidate` name the presets of
`runtime/fast_profiles.py` selected by the smoke, `--synchronize-step-timing` makes step times exact,
`--enstrophy-budget` records the enstrophy balance every step):

```bash
.venv/bin/python -u -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --benchmark disk-highmode --modes 128 \
  --orders 6 --band-mesh-sizes 0.004 --mesh-size 0.02 --dts 0.0125 --final-time 70 --schemes si-bdf2 \
  --velocities compatible --compatible-potential postprocessed --poisson-tau 128000 \
  --poisson-candidate amgx-cheb-csr --transport-candidate fused+csr --poisson-order-offset 0 \
  --synchronize-step-timing --enstrophy-budget --movie-interval 0 --epsilon 1e-4 --diagnostics-every 10 \
  --snapshot-times 30,40,45,50,55,60,70 --poisson-solver-atol 1e-14 \
  --output-dir run_outputs/article/diocotron_order_20261016/runs --prefix d8 --run-tag C1_L2_p6_dt0.0125 --verbosity 2 --dry-run
```

Add `--density-positivity kkt --density-positivity-points quadrature+lattice` for the KKT variant and
`--volume-quad-1d 14` for the collapsed volume rule exact for the advection term at p = 9.

Analysis and figures (no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_order select run_outputs/article/diocotron_order_smoke_20261016
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_order report run_outputs/article/diocotron_order_smoke_20261016
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_pairs --runs run_outputs/article/diocotron_order_20261016 --output-dir run_outputs/article/diocotron_order_20261016/pairs
.venv/bin/python -m scripts.guiding_center.studies.diocotron.diocotron_order_errors --times 30,50,70 --fields density,potential --output run_outputs/article/diocotron_order_20261016/errors_p6_vs_p9.json run_outputs/article/diocotron_order_20261016/runs/d8_C1_L2_p6 run_outputs/article/diocotron_order_20261016/runs/d8_C1_L3_p9
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_campaign --output run_outputs/article/diocotron/summary --latex ../gc-hdg-article/tables/diocotron_growth.tex
.venv/bin/python -m scripts.guiding_center.studies.diocotron.plot_diocotron_highmode --output-dir ../gc-hdg-article/figures --stem diocotron_highmode
.venv/bin/python -m scripts.guiding_center.studies.diocotron.render_diocotron_movie --work-dir run_outputs/article/movies --threads 4 --duration 42 --title 'Smooth ring, $m=9$' --output ../gc-hdg-article/supplementary/movies/diocotron_smooth_m9.mp4 run_outputs/article/diocotron/smooth_m9_snapshots.npz
```

## Tests

`tests/test_diocotron_benchmark.py`, `tests/test_diocotron_annulus.py`, `tests/test_raster_movie_recorder.py`,
`tests/test_diagnostic_time_series.py`.

## Follow-ups

- Finish the D8 phases P2a, P1, P2b, R, T and P3 (`campaigns/gc_d8_diocotron_order`), then the D8 analysis
  (cross-mesh errors, growth-rate errors, cost tables and figures).
- Generalize the D8-S launcher into a reusable kernel-smoke runner (`campaigns/common/kernel_smoke.py`).
