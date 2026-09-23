# H1-BDF3 guiding-center integration

Select `--time-scheme h1-bdf3` in the guiding-center runner, or
`--scheme h1-bdf3` in the temporal-convergence runner. This is the
one-transport-solve hybrid. No native source or build-system change is needed.

## Fixed-step update

Let `F(rho) = -M^{-1} R(beta(rho), rho)` be the complete semidiscrete upwind-HDG
transport RHS, with the standard Poisson flux rotated into the drift. Keep
three accepted densities and their accepted-field residuals, newest first.

```text
rho_predictor = rho_n + dt/12 * (23 F_n - 16 F_previous + 5 F_older)
Poisson(rho_predictor) -> beta_predictor
source = (18 rho_n - 9 rho_previous + 2 rho_older) / 11
(M + 6 dt/11 R(beta_predictor)) rho_next = M source
Poisson(rho_next) -> beta_next
F_next = -M^{-1} R(beta_next, rho_next)
```

Here `R(beta)` denotes the spatial linear operator with the algebraic trace
constraint eliminated. `UpwindHDGTransportResidual` performs that elimination
with independent face mass solves. For each interior face, the weak constraint
is

```text
sum_sides integral_face mu * (tau - beta.n) * trace(rho)
    = sum_sides integral_face mu * tau * rho_side,
tau = abs(beta.n).
```

Both element-side velocities, orientations and the configured trace basis are
retained. Boundary traces follow the existing nodal/modal projection convention;
`zero-flux` removes the entire boundary flux. Inactive faces use zero traces.
Active faces with fewer distinct inflow quadrature nodes than trace DOFs are
rejected before the dense solve: LU can otherwise return finite, enormous
coefficients for a rank-deficient constraint. This check reuses the transport
inflow diagnostic; it is not a full condition-number estimate. No artificial
regularization is applied. Custom stabilization and penalty boundaries are not supported by H1.

The accepted residual is recomputed with `beta_next` and its consistent trace.
The trace from the transport solve used `beta_predictor` and cannot substitute
for this algebraic elimination. Accepted-state output retains the actual
transport trace; the residual's trace is internal after startup. During startup,
accepted density and trace belong to the extrapolated state; the reported
transport result is the last SI-Euler substep, as distinguished in stage logs.

## Startup and accuracy

The default `--h1-startup si-euler-extrap3` uses third-order extrapolation of
SI-Euler paths for the first two steps. Each path starts independently from the
same accepted density and drift and uses `m=1,2,3` equal substeps over `dt`:

```text
for each m in (1,2,3):
    start from rho_n, beta_n
    repeat m times:
        (I + dt/m A(beta_current)) rho_substep = rho_current
        Poisson(rho_substep) -> beta_substep
    save endpoint Y_m
rho_next = 1/2 Y_1 - 4 Y_2 + 9/2 Y_3
Poisson(rho_next); cache F_next
```

The extrapolation weights cancel the `1/m` and `1/m^2` terms of the SI-Euler
endpoint error expansion. Every substep remains a linear transport solve;
boundary data use its actual endpoint time. Branch states own their storage,
and both solvers select their nearest available stage as the initial guess.
Only the extrapolated endpoint enters BDF3 history. This initializer damps stiff
frozen transport modes without an explicit startup CFL restriction, but neither
it nor the full nonlinear scheme has an unconditional stability guarantee.

`--h1-startup ssprk3` retains the original explicit initializer for comparisons:

```text
y1 = rho_n + dt F_n
Poisson(y1); f1 = F(y1) at t+dt
y2 = 3/4 rho_n + 1/4 (y1 + dt f1)
Poisson(y2); f2 = F(y2) at t+dt/2
rho_next = 1/3 rho_n + 2/3 (y2 + dt f2)
Poisson(rho_next); cache F_next
```

| Phase | Global transport solves/step | Poisson solves/step | New residuals/step |
|---|---:|---:|---:|
| Initialization | 0 | Existing initial Poisson | 1 |
| First two steps, SI-Euler extrap3 (default) | 6 | 7 | 1 |
| First two steps, SSPRK3 (explicit option) | 0 | 3 | 3 |
| H1-BDF3 | 1 | 2 | 1 |

The original full-step SSPRK3 startup corrupted the heavy p=6 vortex-gas
histories at `dt=0.01` before the first BDF3 solve: its saved BDF3 source ranged
from roughly `-1.39e4` to `1.63e4`, compared with initial vorticity near `[-9,10]`.
A completed `dt=0.005` run also developed large overshoots. Those old runs do not
validate H1 stability. The AB3 predictor still imposes a limitation after startup.
Constant timestep and sufficiently smooth semidiscrete dynamics are required
for the formal third-order argument. A changed timestep requires different
AB/BDF coefficients and history handling; this runner uses a fixed timestep.

Static tests check an `O(dt^4)` one-step defect against independent exact history.
They also reverse a small stationary HDG solve to verify that the residual and
implicit transport use the same spatial operator at p=2 and p=6. Real host
Helmholtz-wave runs with 5, 10, 20, 40 and 80 steps to T=0.1 produced temporal
self-convergence rates 2.85, 3.04 and 2.89 with the revised startup. This checks
temporal order on a fixed mesh. The actual GPU pipeline gives rates 3.04 and
2.89, and a matched host/GPU trajectory agrees to 4e-13 in relative density L2.
The manufactured GPU check uses Schur-LU caching for its nonzero Dirichlet data;
compact Schur-Cholesky RHS caching is restricted to zero Dirichlet data.

## Initial guesses and caches

- Predictor Poisson starts from the most recent accepted potential trace.
- Transport starts from the AB3 density projected to the configured trace basis
  at the endpoint time.
- Accepted-endpoint Poisson starts from the same-time predictor potential trace.
- Startup transport and Poisson select the closest available stage time; ties
  within floating-point rounding prefer the most recently evaluated stage. All retained traces are detached from solver buffers.
- Ordinary H1 AMGX retries use the endpoint guess or the best current endpoint
  iterate. Residual-correction retries solve for a correction around that iterate.
- Accepted density/residual history is committed only after all endpoint work
  succeeds and owns its storage. No predictor enters this history.

The residual reuses the package's mesh/space device mirrors, trace orientation
tables, mass inverse, boundary-data and field-to-trace projection helpers.
Face maps, quadrature contractions and scratch buffers persist for the fixed
space. Trace projection is primed during initialization. The dense advection
reference tensor is now cached per device, alongside the existing sparse tensor
and CSR/BSR topology caches. Velocity-dependent face masses and transport matrix
values are refreshed. Poisson retains its existing matrix, factor and solver
reuse policy.

Timings include `explicit_residual_setup_time`, `explicit_residual_time`,
`explicit_residual_count`, `h1_bdf3_startup`, stage counts and stage initial-guess
times, AB3/source construction and endpoint trace projection. Every Poisson
stage retains its solver metrics, label, stage time, guess time and wall time.
Aggregate timings and transfer costs count each stage once. GPU stage boundaries
are synchronized. Initial setup and the first
transport solve can still incur cold library/kernel setup; use later H1 steps
for warmed throughput comparisons. No artificial global warmup solves are added.

## Terminal output and logs

H1 uses the existing guiding-center runner and the same verbosity mapping:

| `--verbosity` | Terminal and terminal-log detail |
|---|---|
| 0 | Quiet, except errors |
| 1 | Compact accepted-state summaries and final run/output summary |
| 2 | Accepted-state diagnostics, stage labels and solver phase logs |
| 3 | Backend assembly/reconstruction timings, native/AMGX iteration tables, stage/guess times and H1 residual/projection timings |

The existing terminal tee captures Python and native output. The same diagnostic
CSV/JSONL and every-step timing CSV/JSONL files are written at every verbosity.
Startup labels and stage counts distinguish the two initializers. Accepted-state physics
still follows `diagnostics_every`; every-step timings include all startup and
BDF3 stages. No independent H1 logger or verbosity scale is introduced.

## User launch commands

From the repository root, a small host configuration is available as:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/h1_bdf3_host_accuracy.args
```

Append `--dry-run` to inspect its settings without solving. The heavy disk
trial matches the prior vortex gas at `h=0.008`, `p=6`, with Poisson `tau=1000`,
`dt=0.005`, `T=50` (10000 steps), native FB-HP-MG Poisson, AMGX BSR transport,
and Holoviz output every 0.5 time units:

```bash
.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_h1_bdf3_p6_h008_dt0005_t50_raw_cuda_bsr.args --verbosity 3
```

Use the same CUDA/AMGX environment as your existing runs. The smaller initial
trial timestep has passed an 80-step heavy-mesh check with the revised
initializer; it is not a verified long-time stability limit. Append timestep and step-count overrides after the
response file. Run at least three steps to reach the BDF3 phase; two steps
exercise only startup. Qualitative heavy runs precede the later diagnostic,
convergence and CFL study.


## Short heavy-mesh GPU checks

On 113894 triangles, p=6, tau=1000, the revised initializer completed 80 steps
at dt=0.005 (T=0.4), 80 at dt=0.01 (T=0.8), and 30 at dt=0.05 (T=1.5). Relative
energy changes were +1.64e-6, -2.38e-5 and -1.43e-2 respectively. Different end
times prevent an equal-time accuracy comparison. In particular, the largest
timestep's 1.43% energy loss should not be mistaken for an accurate solution.
The full T=50 trajectory remains untested.

Independent residual checks of accepted raw-CUDA transport solutions on this
mesh gave relative density-equation defects from 4.5e-14 to 3.1e-11. Both Poisson
stages reused their operator, local factors and multigrid hierarchy. At dt=0.01,
the warmed median coupled step was 0.424 s, including 0.117 s transport, 0.280 s
for both Poisson stages and 0.0228 s residual work. Poisson work is the largest
cost in this measured configuration. This is not a cross-scheme benchmark.
