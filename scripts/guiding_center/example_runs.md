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

### ITER — SI-BDF2, h=0.014, dt=0.005, T=50

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr.args \
  --positivity-diagnostics \
  --diagnostics-every 10 --plot-diagnostics \
  --plot-every 100 \
  --diagnostics-prefix positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50 \
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

### m=128 — IMEX-ARK3, dt=0.025, T=20

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/diocotron_gaussian_m128_ark3_p6_h008_dt005_t70.args \
  --verbosity 3 --dt 0.025 --num-steps 800 \
  --mesh-size 0.0068 --minimum-triangles 150000
```
