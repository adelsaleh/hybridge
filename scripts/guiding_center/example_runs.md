# Guiding-center example runs

Run these commands from the repository root after activating the project environment.

## Euler vortex gas

Signed Euler gas on the disk retains its earlier, faster Poisson settings:
direct `p -> 0` multigrid, order-2 Chebyshev with one pre/post sweep, the original
`diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json` hybrid fallback, and no
extended Poisson retry ladder. The original tolerances are `rtol=1e-11` and
`atol=1e-12`. PCGF residual safeguards and transport recovery remain enabled.
Positive turbulence and shaped-domain gases retain the stronger Poisson policy
described below, with FGMRES/`MULTICOLOR_DILU` reserved for the final attempt.

### Unit disk

#### IMEX-ARK3 with runtime overrides

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr.args \
  --verbosity 3 \
  --dt 1.0 --num-steps 500 \
  --mesh-size 0.0068 --minimum-triangles 150000 \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-every 50 --plot-backend pyvista \
  --poisson-tau-retry-factor 2 --poisson-tau-max-retries 4
```

#### SI Euler with runtime overrides

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_si_euler_p6_h0068_dt005_t50.args \
  --dt 0.01 --num-steps 700 \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-every 50 --plot-backend pyvista \
  --poisson-tau-retry-factor 2 --poisson-tau-max-retries 8
```

#### Predictor-corrector with runtime overrides

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_predictor_corrector_p6_h0068_dt005_t50.args \
  --num-steps 500 \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-backend pyvista \
  --poisson-tau-retry-factor 2 --poisson-tau-max-retries 8
```

#### SI-BDF2

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_si_bdf2_p6_h0068_dt005_t50.args
```

#### SI-BDF3

Identical to the SI-BDF2 preset (mesh, `dt=0.05`, solvers, Poisson policy)
except for the integrator: one transport and one Poisson solve per step with
third-order extrapolated drift. Step 1 is Richardson-extrapolated SI Euler
(three transport and two Poisson solves), step 2 is SI-BDF2. The BDF3 timestep
has not been qualified for this case.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_si_bdf3_p6_h0068_dt005_t50.args
```

### Shaped geometries

#### Horseshoe — IMEX-ARK3, 360 vortices

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_horseshoe_gas_imex_ark3_p6_150k_t50.args \
  --verbosity 3 --plot-diagnostics
```

#### ITER — IMEX-ARK3, 5,760 vortices

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_iter_gas_imex_ark3_p6_300k_t50.args \
  --verbosity 3 --plot-diagnostics
```

#### ITER — SI-BDF2, 11,520 vortices

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_iter_gas_si_bdf2_p6_300k_t50.args \
  --verbosity 3 --plot-diagnostics
```

#### Pac-Man — IMEX-ARK3, 360 vortices

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_pacman_gas_imex_ark3_p6_150k_t50.args \
  --verbosity 3 --plot-diagnostics
```

## Positive turbulence

These SI-BDF2 examples retain DG p=6 and use the spatial resolutions already
selected for the corresponding signed gases. They use `dt=0.005`, ten times
smaller than the signed-gas BDF2 examples, because the positive gas has
uncancelled circulation and bulk rotation. These are conservative user-run
starting values, not a time-step qualification.

Positivity diagnostics always audit the projected initial density. With BDF2,
accepted endpoints are audited at `--diagnostics-every 10`, here every 0.05
time units; no positivity limiter or clipping is applied.

Both BDF2 presets opt into the robust Poisson policy. Native FB-HP-MG-PCG uses
the halved p-hierarchy, fourth-order Chebyshev smoothing, two symmetric
pre/post sweeps, a strengthened p=0 cycle, and a 1000-iteration default cap.
Positive turbulence and shaped-domain gas presets use a Poisson absolute tolerance of
`1e-10`, retaining relative tolerance `1e-11`: the checked target is
`max(1e-10, 1e-11 * ||rhs||_2)`. This gives `1e-10` for the reported ITER
initial RHS, rather than `4.1e-12`. The same absolute tolerance is passed to
every AMGX fallback; transport tolerances are unchanged. Override it with
`--poisson-solver-atol` if needed.
Native true refreshes and convergence checks use the original assembled
matrix, so a transformed residual just inside tolerance cannot prematurely
end the solve: PCGF continues from the current trace when that check fails.
If the native attempt still fails, the solver passes its
single best true-residual checkpoint to strong coefficient-exact hybrid
BSR/CSR PCGF, using the block-compatible `CHEBYSHEV` smoother of order four
with scalar-row L1 Jacobi and symmetric 2+2 sweeps. A zero-start retry of that
hierarchy is followed by scalar-CSR
PCGF, then at most two PCGF residual corrections. Only after those failures is
one scalar-CSR FGMRES/`MULTICOLOR_DILU` solve attempted. Every endpoint is
accepted against the original assembled Poisson matrix, including native solves.
The native backend uses the flexible PCGF recurrence (the existing
`fb-hp-mg-pcg` option name is unchanged). A true/recursive residual gap exceeding
10% of the true norm restarts its search direction. Six true-residual checks
without 1% improvement in the best norm end a stalled native attempt and hand
its best checkpoint to the fallback; this never relaxes the acceptance target.
Native PCGF keeps one best vector ranked by the original-matrix residual at
true checks (every 10 iterations, on apparent convergence, and at exit); the
AMGX retry wrapper keeps the best candidate between attempts.
Internal AMGX iterates are not exposed by the current binding. Scalar residual
logs remain available; no vector history is stored.
At verbosity 3 native rows are emitted during iteration, and AMGX uses a
flushed print callback so progress reaches both terminal and log immediately.
This policy is an unqualified robustness setting for
these user-run turbulence cases, not a new mesh/time-step qualification.

### Unit disk — SI-BDF2, h=0.0068, dt=0.005, T=50

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_si_bdf2_p6_h0068_dt0005_t50_raw_cuda_bsr.args \
  --positivity-diagnostics \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-every 100 \
  --diagnostics-prefix positive_turbulence_disc_si_bdf2_p6_h0068_dt0005_t50 \
  --verbosity 3
```

### Unit disk — SI-BDF3, h=0.0068, dt=0.005, T=50

Same mesh, initial data, timestep, robust Poisson policy and positivity
auditing as the SI-BDF2 preset; only the integrator differs. The Richardson
startup step and BDF3 itself are not positivity preserving, and no limiter
is applied, so watch the reported negative parts.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_si_bdf3_p6_h0068_dt0005_t50_raw_cuda_bsr.args \
  --positivity-diagnostics \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-every 100 \
  --diagnostics-prefix positive_turbulence_disc_si_bdf3_p6_h0068_dt0005_t50 \
  --verbosity 3
```

### ITER — FFT-generated positive blobs, SI-BDF2

11,520 positive blobs on a cached `2048 x 4096` FFT grid, with DG p=6,
`h=0.014`, `dt=0.005`, and `T=50`. The approximate initializer preserves the
empty wall band; GPU speedup remains unmeasured.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_iter_fft_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr.args \
  --positivity-diagnostics \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-every 1 \
  --diagnostics-prefix positive_turbulence_iter_fft_si_bdf2_p6_h014_dt0005_t50 \
  --verbosity 3
```

### ITER — FFT-generated positive blobs, SI-BDF3

The SI-BDF2 FFT preset above with the SI-BDF3 integrator; mesh, FFT grid,
timestep and solvers are unchanged.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_iter_fft_si_bdf3_p6_h014_dt0005_t50_raw_cuda_bsr.args \
  --positivity-diagnostics \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-every 1 \
  --diagnostics-prefix positive_turbulence_iter_fft_si_bdf3_p6_h014_dt0005_t50 \
  --verbosity 3
```

### ITER — FFT blobs, SI-BDF2 with Holoviz over SSH X11 forwarding

Run from your current `ssh -Y` terminal with its original `DISPLAY` and
`XAUTHORITY`. The launcher creates and checks a fresh `NV-GLX` compatibility
proxy, raises the stack limit to 32 MiB, and removes the proxy when the run exits.

```bash
env LD_LIBRARY_PATH=/path/to/cuda-13/lib64 \
  CUDA_PATH=/path/to/cuda-13 \
  .venv/bin/python -m hybridge.io.holoviz_ssh -- \
  .venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_iter_fft_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr.args \
  --positivity-diagnostics \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-backend holoviz \
  --plot-every 1 \
  --diagnostics-prefix positive_turbulence_iter_fft_si_bdf2_p6_h014_dt0005_t50 \
  --verbosity 3
```

## Gaussian-annulus diocotron

### m=64 — IMEX-ARK3, dt=0.5, T=400

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m64_ark3_p6_h008_dt005_t70.args \
  --verbosity 3 --dt 0.5 --num-steps 800 \
  --mesh-size 0.0068 --minimum-triangles 150000
```

### m=64 — IMEX-ARK3 with Holoviz over SSH X11 forwarding

Uses a fresh `NV-GLX` compatibility proxy from the current `ssh -Y` session.
Keep the original SSH `DISPLAY` and `XAUTHORITY`. The run retains `dt=0.5`,
`T=400`, and `h=0.0068`.

```bash
env LD_LIBRARY_PATH=/path/to/cuda-13/lib64 \
  CUDA_PATH=/path/to/cuda-13 \
  .venv/bin/python -m hybridge.io.holoviz_ssh -- \
  .venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m64_ark3_p6_h008_dt005_t70.args \
  --verbosity 3 --dt 0.5 --num-steps 800 \
  --mesh-size 0.0068 --minimum-triangles 150000 \
  --plot-backend holoviz \
  --plot-every 1
```

### m=64 — SI Euler, dt=0.5, T=400

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m64_si_euler_p6_h0068_dt05_t400.args
```

### m=64 — Predictor-corrector, dt=0.5, T=400

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m64_predictor_corrector_p6_h0068_dt05_t400.args
```

### m=64 — SI-BDF2, dt=0.5, T=400

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m64_si_bdf2_p6_h0068_dt05_t400.args
```

### m=64 — SI-BDF3, dt=0.5, T=400

Same mesh, `fast` Poisson policy and retries as the SI-BDF2 preset; only the
integrator differs. At this large `dt` the BDF3 stability margin is untested.

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m64_si_bdf3_p6_h0068_dt05_t400.args
```

### m=64 — SI-BDF2 with Holoviz over SSH X11 forwarding

Uses a fresh `NV-GLX` compatibility proxy from the current `ssh -Y` session.
Keep the original SSH `DISPLAY` and `XAUTHORITY`. The run retains `dt=0.5`,
`T=400`, and `h=0.0068`. This preset selects the `fast` Poisson policy:
`p=6 -> p=0`, order-1 smoothing, and one scalar AMG V-cycle per application.
Residual tolerances and robust recovery remain enabled.

```bash
env LD_LIBRARY_PATH=/path/to/cuda-13/lib64 \
  CUDA_PATH=/path/to/cuda-13 \
  .venv/bin/python -m hybridge.io.holoviz_ssh -- \
  .venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m64_si_bdf2_p6_h0068_dt05_t400.args \
  --verbosity 3 --mesh-size 0.0068 --dt 0.1 \
  --plot-backend holoviz \
  --plot-every 5
```

### m=128 — IMEX-ARK3, dt=0.025, T=20

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m128_ark3_p6_h008_dt005_t70.args \
  --verbosity 3 --dt 0.025 --num-steps 800 \
  --mesh-size 0.0068 --minimum-triangles 150000
```

### m=64 — quiet SI-BDF2, plots every 5 steps

```bash
python scripts/guiding_center/run_guiding_center_cases.py \
  @run_configs/guiding_center/diocotron_gaussian_m64_si_bdf2_p6_h0068_dt01_t400_fast.args
```

Derived from the existing m=64 SI-BDF2 preset with the same mesh,
orders, tolerances, fast Poisson preconditioner and retry policies, using dt=0.1
and 4,000 steps to T=400. Disables
field/positivity/modal diagnostics, diagnostic plots, timing files, solver
iteration histories and routine solver output. Density/potential plotting keeps
its existing settings and updates every 5 steps. Native Poisson periodic
true-residual refreshes are disabled; initial, convergence-candidate and final
checks remain, as does independent physical acceptance. Krylov stopping norms
remain active in both solvers. This is minimal optional checking, not an
unchecked fixed-iteration solve. Removing periodic refreshes can affect iteration
counts or recovery behavior; throughput has not been benchmarked.

Use `--poisson-true-residual-every 10 --poisson-residual-history` to restore native
Poisson monitoring, `--amgx-residual-history` for AMGX histories, and
`--diagnostics-every 10 --diocotron-diagnostics --positivity-diagnostics` to request
field diagnostics. `--verbosity 3 --record-timings` restores detailed output.

The local AMGX source audit used `AMGX-hdg-cuda13`, as selected by
`AMGX-build-cuda13/CMakeCache.txt`. In `src/solvers/solver.cu`, memory usage queries
and residual table formatting are guarded by `print_solve_stats`; CUDA timing
events/synchronization by `obtain_timings`; history copies by `store_res_history`.
All are off in this preset and its retries. The divergence guard uses existing
host residual scalars and performs an extra matrix action only on suspected
failure. In `bicgstab_solver.cu`, `pbicgstab_solver.cu` and `pcgf_solver.cu`, the
remaining norms implement stopping checks; the BiCGStab early-exit path also
refreshes the final residual. These are retained. No AMGX source change or rebuild
is required. The CLI retains its terminal capture for errors; field diagnostics
and timing files are not created. Plot frames may still be saved in headless mode.

The quiet `dt01_t400_fast` preset now selects Holoviz and records a compressed
H.264 MP4 at 20 playback fps to
`outputs/movies/diocotron_gaussian_m64_si_bdf2_p6_h0068_dt01_t400_fast.mp4`
(relative to the working directory). It records the initial display and every
5th step, including the displayed labels, without intermediate PNG files.
Recording is optional: append `--no-save-movie` to retain only the live display;
`--save-movie` enables it again. Override the destination with
`--movie-path outputs/movies/my_run.mp4` and playback speed with `--movie-fps 30`.
The destination is replaced on a new run. Closing the viewer stops capture;
normal viewer cleanup finalizes the encoder. Recording retains every requested
frame and adds framebuffer readback and CPU H.264 encoding work, but does not
enable field diagnostics. Install the `holoviz` extra for the bundled encoder.

## Studies

One block per study, in the order of `README.md`: options first, then dry runs, then the commands
that start solves (marked **starts solves**), then analysis and figures (no solve). Every command
below is checked by `tests/test_guiding_center_example_runs.py` (module present, options accepted by
the script's parser, presets, cases and `@run_configs` files present). Each study's README gives the
context, the output directories and the follow-ups.

### Convergence (`studies/convergence/`)

Options and dry runs:

```bash
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence --help
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence --study manufactured --scheme both --orders 2,3,4 --dry-run
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --help
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families misaligned,aligned --checks --dry-run
```

Poisson degree study and unstructured meshes (**starts solves**; each command waits for 16 GiB of free device memory):

```bash
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families misaligned,aligned --checks --min-free-gpu-mib 16384
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families gmsh --orders 2,3,4,5,6 --checks --min-free-gpu-mib 16384
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --combine-only --families misaligned,aligned,gmsh --orders 2,3,4,5,6
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --lattice --families unstructured --output-dir run_outputs/article/convergence_unstructured
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families unstructured --orders 2,3,4 --checks --offsets 0,1 --output-dir run_outputs/article/convergence_unstructured --min-free-gpu-mib 16384
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --families unstructured --orders 2,3,4 --checks --drift compatible --offsets 0 --output-dir run_outputs/article/convergence_unstructured --min-free-gpu-mib 16384
.venv/bin/python -m scripts.guiding_center.studies.convergence.run_poisson_degree_study --combine-only --families unstructured --orders 2,3,4 --output-dir run_outputs/article/convergence_unstructured
```

Exact-drift rates, steady rates and the manufactured snapshot (**starts solves**):

```bash
.venv/bin/python scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py --orders 2 --mesh-sizes 0.1 \
  --case-param 'kx=8*pi' --case-param 'ky=-8*pi' --case-param U=1 --case-param A=0.028134884879909 --final-time 0.0625 \
  --dt-table run_outputs/article/convergence_exact_drift/dt_table.json --output run_outputs/article/convergence_exact_drift/runs/misaligned_p2_h0.1.csv
.venv/bin/python scripts/guiding_center/studies/convergence/exact_drift_transport_rates.py --combine run_outputs/article/convergence_exact_drift/runs/misaligned_p2_h0.1.csv
.venv/bin/python scripts/guiding_center/studies/convergence/steady_transport_velocity_rates.py --drifts averaged --orders 1,2,3,4 --mesh-sizes 0.2,0.1,0.05,0.025,0.0125 --output run_outputs/article/convergence_poisson_degree/steady/steady_drift_averaged.csv
.venv/bin/python -u scripts/guiding_center/studies/convergence/snapshot_manufactured_solution.py --order 3 --mesh-size 0.005 --dt 0.00027533039647577095 --final-time 0.0625 \
  --case-param 'kx=8*pi' --case-param 'ky=8*pi' --case-param U=1 --case-param A=0.028134884879909 \
  --backend device --device-profile bsr --mesh-family structured --poisson-order-offset 0 --resolution 2000 --output run_outputs/article/convergence_poisson_degree/snapshots/wave_p3.npz
```

Tables and figures (no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.convergence.make_convergence_tables --temporal run_outputs/article/convergence_poisson_degree/temporal.csv --output-dir ../gc-hdg-article/tables
.venv/bin/python -m scripts.guiding_center.studies.convergence.plot_convergence_study --spatial run_outputs/article/convergence_poisson_degree/spatial.csv --output-dir ../gc-hdg-article/figures
```

### Diocotron (`studies/diocotron/`)

Options and dry runs (no mesh, no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --benchmark disk-smooth --orders 6 --dts 0.05 --final-time 70 --schemes si-bdf2 --velocities raw --output-dir run_outputs/article/diocotron/smooth --dry-run
.venv/bin/python -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --benchmark disk-zg --orders 6 --dts 0.05 --schemes imex-ark3 --velocities raw --output-dir run_outputs/article/diocotron/zg --dry-run
.venv/bin/python -m scripts.guiding_center.studies.diocotron.preflight_diocotron --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron --help
```

One D8 run with the launcher's options (**starts a solve** without `--dry-run`):

```bash
.venv/bin/python -u -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --benchmark disk-highmode --modes 128 \
  --orders 6 --band-mesh-sizes 0.004 --mesh-size 0.02 --dts 0.0125 --final-time 70 --schemes si-bdf2 \
  --velocities compatible --compatible-potential postprocessed --poisson-tau 128000 \
  --poisson-candidate amgx-cheb-csr --transport-candidate fused+csr --poisson-order-offset 0 \
  --synchronize-step-timing --enstrophy-budget --movie-interval 0 --epsilon 1e-4 --diagnostics-every 10 \
  --snapshot-times 30,40,45,50,55,60,70 --poisson-solver-atol 1e-14 \
  --output-dir run_outputs/article/diocotron_order_20261016/runs --prefix d8 --run-tag C1_L2_p6_dt0.0125 --verbosity 2 --dry-run
```

Analysis and figures (no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_order select run_outputs/article/diocotron_order_smoke_20261016
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_order report run_outputs/article/diocotron_order_smoke_20261016
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_pairs --runs run_outputs/article/diocotron_order_20261016 --output-dir run_outputs/article/diocotron_order_20261016/pairs
.venv/bin/python -m scripts.guiding_center.studies.diocotron.diocotron_order_errors --times 30,50,70 --fields density,potential --output run_outputs/article/diocotron_order_20261016/errors_p6_vs_p9.json run_outputs/article/diocotron_order_20261016/runs/d8_C1_L2_p6 run_outputs/article/diocotron_order_20261016/runs/d8_C1_L3_p9
.venv/bin/python -m scripts.guiding_center.studies.diocotron.analyze_diocotron_campaign --output run_outputs/article/diocotron/summary --latex ../gc-hdg-article/tables/diocotron_growth.tex
.venv/bin/python -m scripts.guiding_center.studies.diocotron.plot_diocotron_highmode --output-dir ../gc-hdg-article/figures --stem diocotron_highmode
.venv/bin/python -m scripts.guiding_center.studies.diocotron.plot_diocotron_growth --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.plot_diocotron_smooth --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.plot_diocotron_zg --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.plot_diocotron_injection --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.plot_diocotron_ionization --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.render_diocotron_movie --work-dir run_outputs/article/movies --threads 4 --duration 42 --title 'Smooth ring, $m=9$' --output ../gc-hdg-article/supplementary/movies/diocotron_smooth_m9.mp4 run_outputs/article/diocotron/smooth_m9_snapshots.npz
```

### Magnetron (`studies/magnetron/`)

Options, one D7 v2 configuration (`--dry-run` builds only the meshes; **starts a solve** without it) and E9 (**starts solves**):

```bash
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_magnetron --help
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_magnetron \
  --anode-geometry vanes --vane-width 0.26 --wave-amplitudes 0.05 --wave-speeds 0.25 --hub-edge 0.025 \
  --emission-top 0.6 --refine-band 0.55,1.35 --schemes si-bdf2 --final-time 40 \
  --hall-parameter 30 --electron-temperature 0.01 --emission-period 20 --emission-phase 5 \
  --density-backend device --density-positivity-points quadrature+lattice+dense --velocities compatible \
  --poisson-order-offset 0 --curvature-points 3000 --density-rtol 1e-10 \
  --snapshot-times 10,20,30,40 --prefix d7v2 --verbosity 1 \
  --orders 6 --dts 0.05 --mesh-size 0.03 --band-mesh-size 0.015 --corner-mesh-size 0.001 --cathode-mesh-size 0.003 \
  --density-positivity none --movie-interval 0 --output-dir run_outputs/article/magnetron_d7_production_v2/dry/p6_s1 --dry-run
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_poisson_cg_vs_hdg --help
.venv/bin/python -m scripts.guiding_center.studies.magnetron.run_poisson_cg_vs_hdg --orders 6,7 --output-dir run_outputs/article/e9_cg_vs_hdg
```

Analysis, tables and figures (no solve):

```bash
.venv/bin/python -m scripts.guiding_center.studies.magnetron.magnetron_order_efficiency --reference run_outputs/article/magnetron_d7_production_v2/runs/ref_p6_s0p36/d7v2.jsonl --runs run_outputs/article/magnetron_d7_production_v2/runs/p6_s1/d7v2.jsonl run_outputs/article/magnetron_d7_production_v2/runs/p4_s0p72/d7v2.jsonl --stem d7_ladder --times 10,20,30,40
.venv/bin/python -m scripts.guiding_center.studies.magnetron.magnetron_convergence --runs run_outputs/article/magnetron_d7_production_v2/runs/p6_s2/d7v2.jsonl run_outputs/article/magnetron_d7_production_v2/runs/p6_s1/d7v2.jsonl --parameter mesh --label 'mesh scale' --stem d7_mesh --times 10,20,30,40
.venv/bin/python -m scripts.guiding_center.studies.magnetron.write_magnetron_tables --ladder d7_ladder.json --dt d7_dt.json --ladder-table ../gc-hdg-article/tables/magnetron_ladder.tex --dt-table ../gc-hdg-article/tables/magnetron_dt.tex
.venv/bin/python -m scripts.guiding_center.studies.magnetron.plot_magnetron --runs run_outputs/article/magnetron_d7_production_v2/runs/p6_s1/d7v2.jsonl --stem magnetron --times 10,20,30,40
.venv/bin/python -m scripts.guiding_center.studies.magnetron.plot_discharge --run run_outputs/article/magnetron_d7_production_v2/runs/p6_s1/d7v2.jsonl --stem discharge --times 10,20,30,40
```

### Band evolution (`studies/band_evolution/`)

Options and the E7 run form (**starts solves** without `--dry-run`):

```bash
.venv/bin/python -m scripts.guiding_center.studies.band_evolution.run_band_evolution --help
.venv/bin/python -u -m scripts.guiding_center.studies.band_evolution.run_band_evolution \
  --geometries star --kinds control,seed --modes 'star=16;uglypacman=16;iter=18;horseshoe=20' --edge-fraction 0.05 \
  --order 6 --dt 0.05 --final-time 200 --initial-projection-quad-1d 24 \
  --poisson-solver-atol 5e-14 --poisson-solver-rtol 5e-11 \
  --output-dir run_outputs/article/e7_band_evolution_20261019/runs --prefix e7 --skip-existing --verbosity 1 --dry-run
```

Transfer check, analysis and figures (no time stepping):

```bash
.venv/bin/python -m scripts.guiding_center.studies.band_evolution.equilibrium_transfer_check --help
.venv/bin/python -m scripts.guiding_center.studies.band_evolution.analyze_band_evolution --runs run_outputs/article/e7_band_evolution_20261019/runs --output run_outputs/article/e7_band_evolution_20261019/analysis --prefix e7
.venv/bin/python -m scripts.guiding_center.studies.band_evolution.plot_band_evolution_article --runs run_outputs/article/e7_band_evolution_20261019/runs --article ../gc-hdg-article --prefix e7
```

### Equilibrium preservation (`studies/equilibrium_preservation/`)

Options; the smoke driver (**starts solves**) is in the study README.

```bash
.venv/bin/python -m scripts.guiding_center.studies.equilibrium_preservation.run_equilibrium_preservation --help
.venv/bin/python -m scripts.guiding_center.studies.equilibrium_preservation.plot_equilibrium_preservation --help
```

### Recovered field (`studies/recovered_field/`)

Bounded matrix diagnostics; `NUMBA_DISABLE_JIT=1` is mandatory (no kernel compilation):

```bash
NUMBA_DISABLE_JIT=1 .venv/bin/python -m scripts.guiding_center.studies.recovered_field.check_field_recovery_order --help
NUMBA_DISABLE_JIT=1 .venv/bin/python -m scripts.guiding_center.studies.recovered_field.check_field_recovery_order --density-order 3 --subdivisions 2 4 8
NUMBA_DISABLE_JIT=1 .venv/bin/python -m scripts.guiding_center.studies.recovered_field.check_oversampled_recovery_order --help
NUMBA_DISABLE_JIT=1 .venv/bin/python -m scripts.guiding_center.studies.recovered_field.compare_field_recovery --density-order 3
```

### Solver tuning (`studies/solver_tuning/`)

Options (the replays and benchmarks solve when started without `--help`) and the offline analyses:

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
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.analyze_poisson_timings --output-dir run_outputs/solver_studies/poisson_timings run_outputs/article/diocotron_order_20261016/runs/d8_C1_L2_p6_timings.jsonl
.venv/bin/python -m scripts.guiding_center.studies.solver_tuning.audit_solver_convergence --windows 0:20,20:70 --json run_outputs/solver_studies/audit.json run_outputs/article/diocotron_order_20261016/runs/d8_C1_L2_p6.jsonl
```

### Viewers (`viewers/`)

```bash
.venv/bin/python -B scripts/guiding_center/viewers/smoke_holoviz.py --help
.venv/bin/python -B scripts/guiding_center/viewers/animate_holoviz.py --help
```

## Campaigns

Campaign launchers, their records and the shared protocol live in `campaigns/` (index and
conventions in `campaigns/README.md`). The article convergence campaign (H100-ready) prints its task
list and plan without solving:

```bash
.venv/bin/python campaigns/gc_article_convergence/campaign.py list
.venv/bin/python campaigns/gc_article_convergence/campaign.py plan --shards 8
.venv/bin/python campaigns/gc_article_convergence/campaign.py run --shard 0/8 --dry-run
.venv/bin/python campaigns/gc_article_convergence/campaign.py verify-meshes
```

### Local campaign records and task lists

The campaign index records partial and completed stages. These commands print plans; they do not start runs.

```bash
.venv/bin/python campaigns/gc_d8_diocotron_order/launcher.py --plan
.venv/bin/python campaigns/gc_d8s_smoke/launcher.py --plan
.venv/bin/python campaigns/gc_e7_band_evolution/launcher.py --plan
.venv/bin/python campaigns/gc_e7_five_sites/launcher.py --plan
.venv/bin/python campaigns/gc_e7_lumps/launcher.py --plan
.venv/bin/python campaigns/gc_poisson_pmg_degrees/launcher.py --plan
.venv/bin/python campaigns/gc_diocotron_shell/todo_20261015/launcher.py --list
```

Start/resume/stop commands and output overrides are in each campaign README. The D7 shell stages can
be syntax-checked with `bash -n`; only production v2 offers `DRY=1` configuration inspection.

### Planned GENCI phases

The assembly, H100/H200 solver and turbulence task matrices live in
`campaigns/h100_assembly/`, `campaigns/h100_solvers/` and `campaigns/h100_turbulence/`.
Their combined cluster launchers remain to be built. Inspect the existing drivers without solving:

```bash
.venv/bin/python scripts/gpu/benchmark_advection_tsle_tensor.py --help
.venv/bin/python scripts/gpu/benchmark_poisson_assembly.py --help
.venv/bin/python campaigns/adr_unified/run_adr_unified_campaign.py --help
.venv/bin/python campaigns/adr_pardiso/run_adr_pardiso_campaign.py --help
.venv/bin/python campaigns/adr_stress/run_closed_loop_stress.py --help
.venv/bin/python -m scripts.guiding_center.studies.diocotron.run_diocotron_benchmark --help
```
