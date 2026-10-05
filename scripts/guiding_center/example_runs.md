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
env LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  CUDA_PATH=/usr/local/cuda-13.0 \
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
env LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  CUDA_PATH=/usr/local/cuda-13.0 \
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
env LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  CUDA_PATH=/usr/local/cuda-13.0 \
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
