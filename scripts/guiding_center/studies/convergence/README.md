# Convergence study

Manufactured convergence of the guiding-center scheme: the k = 8π travelling wave `rho_helm_wave`
and the orbiting vortex, temporal ladders per scheme, spatial ladders per degree on structured
(aligned, misaligned, lattice), Gmsh and unstructured meshes, the drift variants (raw q = p,
raw q = p + 1, compatible drift, exact drift), the steady transport rates of nine discrete drifts
and the raised-degree Poisson study. It feeds the article's Section 5 and Appendices B–C
(tables `tab:alignment`, `tab:v2-accuracy`, `tab:v2-cost`, `tab:potential-error`, `tab:efficiency`,
`tab:steady-drift`, figures `fig:work-precision` and the spatial/temporal convergence figures).
Errors are measured against the exact solution at t = T with a 2p + 2 points-per-direction rule
(`--error-quad-1d`).

| Script | Role |
|---|---|
| `run_guiding_center_temporal_convergence.py` | Manufactured temporal (`--study manufactured`), spatial (`--study spatial`) and fixed-space vortex-gas sensitivity (`--study vortex-gas`) runs; writes one CSV per study. Starts solves. |
| `run_poisson_degree_study.py` | Raised-order Poisson study (q = p + s) with `--drift raw|compatible|compatible-primal|exact`, per mesh family, with `--checks` (dt/2 on the finest mesh), `--combine-only` tables and the GPU-memory wait (`--min-free-gpu-mib`). Starts solves. |
| `exact_drift_transport_rates.py` | Density convergence with the exact drift at the quadrature points (no Poisson solve). Starts solves. |
| `steady_transport_velocity_rates.py` | Steady HDG transport rates for different discrete drifts on Gmsh meshes. Starts solves. |
| `snapshot_manufactured_solution.py` | One manufactured run saved as raster snapshots for figures. Starts a solve. |
| `make_convergence_tables.py` | LaTeX tables from the CSV files (temporal, spatial, alignment, Poisson degree, steady, efficiency). No solve. |
| `plot_convergence_study.py` | Figures from the CSV files (spatial, temporal, work-precision, wave figure). No solve. |
| `guiding_center_temporal_comparison.py` | Matched-time vortex-gas sensitivity library used by the temporal driver and its tests. |

## Outputs

| Study | Directory (symlink `run_outputs/article` → `gc-hdg-article/run_outputs`) | Status |
|---|---|---|
| Poisson degree study, structured and Gmsh families | `run_outputs/article/convergence_poisson_degree` | complete (`all-done`) |
| Unstructured Gmsh meshes, raw and compatible drift | `run_outputs/article/convergence_unstructured` | complete (`all-done`, 2026-10-26) |
| Exact-drift rates | `run_outputs/article/convergence_exact_drift` | complete (`aligned-done`) |
| H100 campaign (every GPU convergence run, accurate errors) | `run_outputs/campaigns/gc_article_convergence` locally, `$WORK` on Jean Zay | prepared, not started; see `campaigns/gc_article_convergence/README.md` |

The local launchers of the finished studies are kept in `campaigns/gc_conv_poisson_degree`,
`campaigns/gc_conv_unstructured` and `campaigns/gc_conv_exact_drift` with their records.

## Commands

Options only:

```bash
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence --help
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --help
.venv/bin/python -m scripts.guiding_center.studies.convergence.exact_drift_transport_rates --help
.venv/bin/python -m scripts.guiding_center.studies.convergence.steady_transport_velocity_rates --help
.venv/bin/python -m scripts.guiding_center.studies.convergence.snapshot_manufactured_solution --help
.venv/bin/python -m scripts.guiding_center.studies.convergence.make_convergence_tables --help
.venv/bin/python -m scripts.guiding_center.studies.convergence.plot_convergence_study --help
```

Poisson degree study (the local campaign of `run_outputs/article/convergence_poisson_degree`;
`--dry-run` lists the runs, the other commands start solves and wait for 16 GiB of free device memory):

```bash
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families misaligned,aligned --checks --dry-run
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families misaligned,aligned --checks --min-free-gpu-mib 16384
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families gmsh --orders 2,3,4,5,6 --checks --min-free-gpu-mib 16384
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --combine-only --families misaligned,aligned,gmsh --orders 2,3,4,5,6
```

Unstructured meshes (lattice table, raw drift with q = p and q = p + 1, compatible drift, then the combined tables):

```bash
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --lattice --families unstructured --output-dir run_outputs/article/convergence_unstructured
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families unstructured --orders 2,3,4 --checks --offsets 0,1 --output-dir run_outputs/article/convergence_unstructured --min-free-gpu-mib 16384
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families unstructured --orders 2,3,4 --checks --drift compatible --offsets 0 --output-dir run_outputs/article/convergence_unstructured --min-free-gpu-mib 16384
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --combine-only --families unstructured --orders 2,3,4 --output-dir run_outputs/article/convergence_unstructured
```

Exact-drift rates (one (p, h) per process, then the combined table):

```bash
.venv/bin/python scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py --orders 2 --mesh-sizes 0.1 \
  --case-param 'kx=8*pi' --case-param 'ky=-8*pi' --case-param U=1 --case-param A=0.028134884879909 --final-time 0.0625 \
  --dt-table run_outputs/article/convergence_exact_drift/dt_table.json --output run_outputs/article/convergence_exact_drift/runs/misaligned_p2_h0.1.csv
.venv/bin/python scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py --combine run_outputs/article/convergence_exact_drift/runs/misaligned_p2_h0.1.csv
```

Steady transport rates and the manufactured snapshot:

```bash
.venv/bin/python scripts/guiding_center/studies/convergence/steady_transport_velocity_rates.py --drifts averaged --orders 1,2,3,4 --mesh-sizes 0.2,0.1,0.05,0.025,0.0125 --output run_outputs/article/convergence_poisson_degree/steady/steady_drift_averaged.csv
.venv/bin/python -u scripts/guiding_center/studies/convergence/snapshot_manufactured_solution.py --order 3 --mesh-size 0.005 --dt 0.00027533039647577095 --final-time 0.0625 \
  --case-param 'kx=8*pi' --case-param 'ky=8*pi' --case-param U=1 --case-param A=0.028134884879909 \
  --backend device --device-profile bsr --mesh-family structured --poisson-order-offset 0 --resolution 2000 --output run_outputs/article/convergence_poisson_degree/snapshots/wave_p3.npz
```

Temporal and spatial manufactured runs (`--dry-run` prints the configurations):

```bash
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence --study manufactured --scheme both --orders 2,3,4 --dry-run
HYBRIDGE_PRECISION=float64 .venv/bin/python -m scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence --study spatial --scheme both --orders 2,3,4 --backend device --output-dir run_outputs/article/convergence_poisson_degree/spatial
```

Tables and figures (no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.convergence.make_convergence_tables --temporal run_outputs/article/convergence_poisson_degree/temporal.csv --output-dir ../gc-hdg-article/tables
.venv/bin/python -m scripts.guiding_center.studies.convergence.plot_convergence_study --spatial run_outputs/article/convergence_poisson_degree/spatial.csv --output-dir ../gc-hdg-article/figures
```

The H100 campaign commands (`list`, `plan`, `run --dry-run`, `verify-meshes`) are in
[`campaigns/gc_article_convergence/README.md`](../../../../campaigns/gc_article_convergence/README.md).

## Tests

`tests/test_guiding_center_exact_drift.py`, `tests/test_guiding_center_temporal_comparison.py`,
`tests/test_hdg_scalar_norms.py`, `tests/test_guiding_center_cases.py`.

## Follow-ups

- The GPU-lock probe of `run_poisson_degree_study.py` (`--gpu-lock`, `--exclusive-gpu`, `--ignore-gpu-lock`)
  duplicates `campaigns/common/protocol.py` with a different lock mode; unify them in `campaigns/common/gpu_lock.py`.
- The identity audit of the article's `tab:identities` (`run_identity_audit.py`) is planned here.
