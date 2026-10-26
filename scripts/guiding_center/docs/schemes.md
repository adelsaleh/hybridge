# Schemes and tested presets

The time-scheme registry drives both `--time-scheme` choices and execution. Every
scheme has its own module; SI BDF2 and predictor-corrector reuse SI Euler's
stage-solve and accepted-history handling, and SI BDF3 extends SI BDF2. All
return the same `GuidingCenterStep` result. H1/H2 share their existing BDF3
histories and third-order startup. ARK3 retains its frozen transport operator.
All seven schemes support bounded Poisson-tau recovery.

SI BDF3 is the plain one-solve analogue of SI BDF2: one linear transport solve
with third-order extrapolated drift, then the endpoint Poisson solve,

```text
(I + 6 dt/11 A(3 v_n - 3 v_(n-1) + v_(n-2))) rho_(n+1)
    = (18 rho_n - 9 rho_(n-1) + 2 rho_(n-2)) / 11.
```

The temporal algebra is `hybridge.core.time_integration.bdf3_transport_data`. Its
startup keeps third order: step 1 is Richardson-extrapolated SI Euler
(`2 E(dt/2) - E(dt)`: three transport and two Poisson solves), step 2 is SI BDF2.
A plain Euler/BDF2 ramp would limit the global order to two. The extrapolated
first step is not positivity preserving. Recovered drift
(`--transport-electric-field postprocessed`, `--poisson-order-offset -1`) works
as for SI BDF2. Per-step records add `bdf3_startup`, `bdf3_startup_method`
(`si-euler-extrap2`, `si-bdf2`, or none) and `transport_time_order`. Canned
scalar checks in `tests/test_guiding_center_bdf3.py` measure global order 3. No
PDE timestep-stability qualification has been made; BDF3 is only A(alpha)-stable.

| `--time-scheme` | Scheme module | Tested vortex-gas preset |
|---|---|---|
| `si-euler` | `si_euler.py` | `euler_vortex_gas_si_euler_p6_h008_dt005_t50_raw_cuda_bsr` |
| `predictor-corrector` | `predictor_corrector.py` | `euler_vortex_gas_predictor_corrector_p6_h008_dt005_t50_raw_cuda_bsr` |
| `si-bdf2` | `si_bdf2.py` | `euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr` |
| `si-bdf3` | `si_bdf3.py` | `euler_vortex_gas_si_bdf3_p6_h0068_dt005_t50` |
| `h1-bdf3` | `h1_bdf3.py` | `euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr` |
| `h2-bdf3` | `h2_bdf3.py` | `euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr` |
| `imex-ark3` | `imex_ark3.py` | `euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr` |

These are the existing tested presets and response files. Their timesteps,
startup policies, stabilization and solver settings are preserved. The H1/H2
and ARK3 rows use `dt=0.005`, `poisson_tau=1000`, and 10,000 steps to `T=50`;
the lower-order rows use their existing `dt=0.05` configurations. SI BDF3
presets mirror the corresponding SI-BDF2 presets field for field, except the
scheme and output prefix: `euler_vortex_gas_si_bdf3_p6_h0068_dt005_t50`,
`positive_turbulence_si_bdf3_p6_h0068_dt0005_t50_raw_cuda_bsr`,
`positive_turbulence_iter_fft_si_bdf3_p6_h014_dt0005_t50_raw_cuda_bsr` and
`diocotron_gaussian_m64_si_bdf3_p6_h0068_dt05_t400` (see `example_runs.md`). Host accuracy
and diocotron presets remain available through `--list-presets`, which now
shows each preset's scheme and timestep.

Inspect a tested configuration without running it:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_h2_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr.args \
  --dry-run
```

The same CLI accepts a positional preset name or `--preset NAME`. Use the
scheme's own preset to retain its tested configuration; `--time-scheme` changes
only the algorithm and does not silently replace the other preset settings.
Precision selection still happens before numerical imports. No rebuild is
required for this Python refactor.


## Poisson-tau fallback for every scheme

SI Euler, predictor-corrector, SI BDF2, SI BDF3, H1-BDF3, H2-BDF3, and IMEX-ARK3 all
retry numerical transport failures through the same backend-independent tau
policy. This includes direct-factorization failures and iterative nonconvergence,
as well as diagnosed active trace-rank loss. Capacity, configuration, and
unrelated programming errors propagate without increasing tau.

The default is `--poisson-tau-retry-factor 2 --poisson-tau-max-retries 4`.
No additional flags are needed in existing commands. A failed transport stage
backtracks to the Poisson source, boundary time, and owning trace that supplied
its drift. Recovery doubles tau, invalidates stale Poisson and transport
operators/preconditioners, repeats that Poisson solve, and replays the whole
unaccepted timestep. Accepted densities and time remain unchanged on exhaustion.
The increased tau persists after success; `--poisson-tau-max-retries 0` disables
recovery.

BDF2 rebuilds its accepted flux history; SI BDF3 rebuilds both accepted flux
levels; H1 rebuilds the Poisson-derived residual
history; H2 rebuilds its drift history. Startup and predictor/corrector stages
also participate. IMEX-ARK3 rebuilds its frozen operator and all stage residuals.
The shared replay orchestration is in `time_schemes/recovery.py`; tau policy and
failure classification are in `runtime/poisson_recovery.py`.

Retries record their old/new tau, failure, and Poisson checkpoint immediately in
`<diagnostics-prefix>_poisson_tau_recovery.jsonl`, including on exhaustion.
Completed and failed retry work is included in step recovery diagnostics. A tau
change changes the discrete Poisson operator and should be accounted for when
comparing accuracy or conservation across runs.
