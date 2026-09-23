# H2-BDF3 guiding-center integration

Select `--time-scheme h2-bdf3` in the existing guiding-center runner. H2 is
the two-transport-solve hybrid. It predicts the endpoint using extrapolated
drift BDF3, recomputes the Poisson field, then corrects the endpoint with BDF3.
There is no nonlinear iteration or explicit transport residual in the default
algorithm. H1 and H2 share startup, stage timing and history management in
`scripts/guiding_center/time_schemes/hybrid_bdf3.py`.

## Fixed-step update

Keep three accepted densities and their physical, unscaled Poisson drifts,
newest first. Let `A(beta)` be the mass-inverted upwind-HDG transport operator:

```text
alpha = 6 dt / 11
source = (18 rho_n - 9 rho_previous + 2 rho_older) / 11
beta_extrapolated = 3 beta_n - 3 beta_previous + beta_older

(I + alpha A(beta_extrapolated)) rho_predictor = source
Poisson(rho_predictor) -> beta_predictor

(I + alpha A(beta_predictor)) rho_next = source
Poisson(rho_next) -> beta_next
```

Both transport stages use the same BDF3 source and scaling. The second stage
replaces the first endpoint; it does not advance through another timestep.
Only `rho_next` and `beta_next` enter accepted history. Boundary data use the
endpoint time in both solves. The existing standard upwind stabilization and
`zero-flux` or eliminated transport boundaries are supported.

For smooth semidiscrete dynamics and exact history, extrapolated drift has
`O(dt^3)` endpoint error. Its contribution to the density equation is multiplied
by `alpha = O(dt)`, so the predictor has an `O(dt^4)` local density error.
Updating its Poisson drift and correcting preserves the `O(dt^4)` local defect
and third-order temporal accuracy under the usual stability assumptions.
This is a fixed-timestep method. Higher order and the additional correction do
not establish an unconditional stability guarantee or a CFL limit.

## Startup and costs

The default `--h2-startup si-euler-extrap3` uses the same two third-order startup
steps as [H1-BDF3](h1_bdf3.md#startup-and-accuracy): independent SI-Euler paths
with 1, 2 and 3 substeps, combined as `rho_next = Y_1/2 - 4 Y_2 + 9 Y_3/2`.
The startup endpoint density is projected to the configured trace basis;
the reported transport result belongs to the last SI-Euler microstep. From
step 3 onward, the accepted trace is the actual BDF3 corrector trace.

| Phase | Transport solves/step | Poisson solves/step | Explicit residuals/step |
|---|---:|---:|---:|
| Initialization | 0 | Existing initial Poisson | 0 |
| First two steps, default startup | 6 | 7 | 0 |
| Regular H2-BDF3 steps | 2 | 2 | 0 |

`--h2-startup ssprk3` retains the explicit startup option for later comparisons.
It evaluates one initial residual and three residuals plus three Poisson solves
per startup step. It inherits SSPRK3's explicit timestep restriction. Regular
H2 steps still perform no residual evaluations. Counts above are stage solves;
additional iterative retry attempts remain visible in the usual solver logs.

## Guesses, caching and output

- The first transport guess is `3 rho_n - 3 rho_previous + rho_older`, projected
  to the configured transport trace basis at the endpoint time.
- Predictor Poisson starts from the latest accepted potential trace.
- Corrector transport starts from the actual predictor transport trace at the
  same endpoint, copied before solver buffers can be reused.
- Final Poisson starts from the same-time predictor potential trace.
- Startup uses the nearest available stage for each guess, preferring the most
  recently evaluated state when distances tie within floating-point rounding.
- Histories own their storage and are committed only after all stages and
  device work complete successfully. Ordinary AMGX retries retain the provided
  guess or the best current iterate; correction solves retain their existing
  correction-vector policy.

`HDGTraceWorkspace` reuses package field-to-trace projection, mesh/space device
mirrors and cached reference tables, without allocating explicit residual work
arrays. The H1 residual evaluator extends this same workspace. Trace projection
is primed during initialization. Poisson retains its fixed operator, factors
and hierarchy according to the existing solver configuration. Transport reuses
geometry, reference tensors, topology and supported solver allocations, while
refreshing matrix values and dependent factors for each changed drift. The
two H2 transport matrices are different; their numeric factors cannot be
blindly reused. The initial stages warm runtime kernels and libraries; regular
steps are the relevant window for throughput measurements.

H2 uses the existing verbosity levels 0–3, terminal tee, diagnostic CSV/JSONL
and every-step timing CSV/JSONL. Stage labels identify `BDF3 predictor`,
`BDF3 corrector`, and `accepted endpoint`. Stage counts, initial-guess times,
solver metrics and aggregate timings include both transport and both Poisson
stages. Additional keys include `h2_bdf3_startup`, `h2_bdf3_startup_method`,
`h2_bdf3_history_count` and `hybrid_bdf3_setup_time`.

## User qualitative run

From the repository root, with the same CUDA/AMGX environment as the existing
runs:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr.args \
  --verbosity 3
```

This matches the heavy Euler vortex-gas disk: `h=0.008`, `p=6`, Poisson
`tau=1000`, `dt=0.005`, `T=50`, native FB-HP-MG Poisson, AMGX BSR transport,
and Holoviz/diagnostics every 0.5 time units. For a preliminary run, append
`--num-steps 100 --diagnostics-prefix h2_bdf3_gas_preliminary`. At least three
steps are needed to reach H2 after startup. Append `--dry-run` to inspect
settings without solving.

Implementation checks cover nonlinear fourth-order local defect, exact stage
counts, same-endpoint initial guesses, reused-buffer ownership, atomic history
on stage failure, and terminal/log parity at all four verbosity levels. A short
20-step manufactured run agrees between host and GPU to relative L2 error
below `1e-9` in density and potential. The heavy H2 qualitative run and its
convergence/CFL study are left for the user and the subsequent analysis.
