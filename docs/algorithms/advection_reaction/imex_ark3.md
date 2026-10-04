# IMEX-ARK3 guiding-center integration

Select `--time-scheme imex-ark3` in the existing guiding-center runner.
The implementation is `scripts/guiding_center/time_schemes/imex_ark3.py`, using the paired
[SUNDIALS ARK324L2SA ERK/DIRK tables](https://sundials.readthedocs.io/en/latest/arkode/Butcher_link.html)
for Kennedy--Carpenter ARK3(2)4L[2]SA. It starts directly from the accepted
initial density and Poisson field, without multistep startup.

## Split and stage equations

Let `F(rho) = -A(beta(rho)) rho` be the mass-inverted spatial HDG residual.
Freeze the physical drift at the accepted beginning of each step:

```text
I_n(rho) = -A(beta_n) rho
E_n(rho) = F(rho) - I_n(rho)
gamma = 1767732205903 / 4055673282236
c = (0, 2 gamma, 3/5, 1)
Y_1 = rho_n, I_1 = F(rho_n), E_1 = 0

for i = 2, 3, 4:
    S_i = rho_n + dt sum_{j<i} (aI_ij I_j + aE_ij E_j)
    (Id + gamma dt A(beta_n)) Y_i = S_i
    Poisson(Y_i) -> beta_i
    F_i = -A(beta_i) Y_i
    I_i = (Y_i - S_i) / (gamma dt)
    E_i = F_i - I_i

rho_next = rho_n + dt sum_i b_i F_i
embedded_difference = dt sum_i (b_i - bhat_i) F_i
Poisson(rho_next) -> beta_next
F_next = -A(beta_next) rho_next  # retained for the next step
```

The identical explicit/implicit output weights permit the `F_i` combination.
The accepted density generally differs from `Y_4`: the explicit last tableau
row is not the output weight vector. Consequently the accepted endpoint needs
its own Poisson evaluation. The accepted density trace comes from the spatial
HDG residual at that endpoint; the stored last transport result is stage 4.
Stage and endpoint boundary values are evaluated at their actual times.

`A` includes volume transport, both element-side upwind contributions,
`abs(beta.n)` stabilization, and the consistently eliminated algebraic trace.
Applying `A(beta_i-beta_n)` would give a different method. Timestep-dependent
condensed trace matrices are not the spatial density operator either.
Recovering `I_i` from the solved stage equation avoids three additional frozen
residual evaluations. Tight independently checked linear residuals matter
because division by `gamma dt` amplifies stage solve errors.

For smooth semidiscrete dynamics the pair has third-order primary and
second-order embedded accuracy. The implicit tableau is A- and L-stable.
The field correction remains explicit, and the upwind operator is only
piecewise smooth as face normal velocities change sign. Neither third order
nor the implicit stability property establishes an unrestricted nonlinear
CFL or long-time invariant bound.

## Cost, guesses and caching

Each step has **three transport solves and four Poisson solves**, plus four
spatial residual evaluations. Initialization adds the usual initial Poisson
solve and one residual. Iterative retries are recorded separately. Poisson's
actual wall time remains part of the comparison.

- All three transport systems share `gamma dt` and `beta_n`. The first stage
  assembles and condenses once; the next two refresh only source/boundary RHS
  data and reconstruct the density.
- `AdvectionReactionHDGOptions(cache_operator=True)` enables the reusable
  package facility. Use `set_source(source, boundary_condition=...)` for RHS
  updates. Changing beta, reaction, space or options invalidates the operator.
  In-place coefficient edits must be followed by their setter or `clear_cache()`.
- NumPy retains local inverses, trace lifts, reduced matrix, scaled matrix and
  iterative preconditioner. Raw CUDA fused CSR/BSR retains LU factors, pivots,
  trace lifts and trace-response columns. Existing triangular solves, source
  moments, boundary projection, orientation and reconstruction helpers are reused.
- Raw CUDA retains AMGX allocations across timesteps. Numerical setup refreshes
  for a new frozen operator; subsequent stages reuse the same matrix upload
  and setup. Retry configurations have separate retained solver instances.
- Geometry, reference tensors, sparsity, device mirrors and factor allocations
  persist across steps. Raw factor caching adds element LU/pivot/lift storage:
  approximately 1.2 GiB on the 113,894-triangle, p=6 disk, in addition to the
  existing local response. Device error norms use the shared scalar HDG Gram
  and transfer only scalar results.
- Every stage starts from the closest available trace in time, preferring the
  latest evaluation on a tie. In particular, stages 3 and 4 use stage 2's trace
  rather than blindly using the immediately preceding evaluation. Endpoint
  Poisson starts from stage 4's potential trace at the same time.
- Stage traces and accepted fields own their storage. Failure leaves the last
  accepted state intact. No partially completed step enters accepted history.

The initial cache implementation supports NumPy and fused raw-CUDA transport,
standard upwind stabilization, no trace ordering, and eliminated/zero-flux
boundaries where supported by the backend. It rejects unsupported cache paths
early. Existing H1/H2/BDF2 configurations retain their prior defaults.

## Rank loss and Poisson tau recovery

The CUDA residual uses CuPy's batched device solve for the small face systems.
`UpwindHDGTraceRankError` derives from NumPy's exception class for compatibility;
that name does not select a host solver. The explicit trace constraint requires
at least p+1 distinct inflow quadrature nodes on each active interior face.
The rank guard is retained; no pseudoinverse or diagonal perturbation is used.

IMEX-ARK3 retries numerical implicit transport-solve failures and confirmed
explicit trace-rank failures automatically:

1. Save the density, time and owning potential trace of the Poisson evaluation
   that supplied the failing velocity. A frozen implicit-stage failure points
   back to the accepted-start Poisson state, not the latest unrelated stage.
2. Multiply Poisson stabilization by `--poisson-tau-retry-factor` (default 2).
   The public `with_options(stabilization=...)` API clears the old Poisson
   numerical operator, factors and native/AMGX hierarchy. Mesh, reference and
   device-space caches remain reusable.
3. Repeat that exact Poisson source/boundary/time, warm-started with its saved
   trace; reconstruct and check the new drift. If rank loss remains, double
   again, up to `--poisson-tau-max-retries` (default 4) in this unaccepted step.
4. Rebuild the accepted-start field and replay the whole unaccepted ARK step
   under the new tau. This avoids mixing old/new Poisson maps in one tableau.
   The first transport stage refreshes its frozen matrix and the other two
   reuse it. Discarded stage traces remain nearest-time warm-start candidates;
   their density/residual values never enter the accepted update.

The increased tau persists into subsequent steps. Initial-state and endpoint
rank failures also use this policy. Numerical transport-solver failures do not
require proof of rank loss. Allocation/capacity, configuration, and unrelated
programming errors do not trigger tau changes. A diagnosed frozen-transport
rank loss from an existing failure snapshot is eligible too. Exhaustion raises
the original solve error with a recovery note and leaves the accepted state untouched.
Set `--poisson-tau-max-retries 0` to disable recovery.

All completed retry solves contribute to stage counts, iterations and timing
sums; failed-operation wall time is also retained. Rejected results keep only
solver diagnostics, releasing their field/matrix/factor arrays. Every-step
records include `poisson_tau`, retry events, failed residual counts and rejected
stage counts. Events are written immediately to
`<diagnostics-prefix>_poisson_tau_recovery.jsonl`, including on exhaustion, and
printed at verbosity 1--3. Verbosity 0 remains quiet.

Increasing tau changes the discrete Poisson field; it is not guaranteed to
restore trace rank. The finite retry cap bounds this experiment. ARK stages
are consistent within a retried step, but stabilization changes must still be
accounted for in convergence and invariant comparisons. This recovery policy
also applies to SI Euler, predictor-corrector, SI BDF2, and both hybrid BDF3
schemes, including their startup stages. Their replay refreshes all cached
Poisson-derived flux or residual histories at the new tau; see the
[guiding-center runner documentation](../../../scripts/guiding_center/README.md#poisson-tau-fallback-for-every-scheme).

## Diagnostics and user run

Verbosity levels 0--3, the full terminal tee, CSV/JSONL physics diagnostics and
CSV/JSONL every-step timings are inherited from the existing runner. Records
include every stage's time, nearest-time guess, iterative solver diagnostics,
operator reuse flags, aggregate costs, and embedded absolute/relative L2 error.
`imex_ark3_embedded_error_l2` measures the primary/embedded density difference;
`imex_ark3_embedded_error_relative` divides by the primary density L2 norm.
These are diagnostic estimates; the runner does not automatically change dt.

From `/home/adelsaleh/src/hybridge`, this machine's CUDA/AMGX environment is:

```bash
export CUDA_PATH=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/home/adelsaleh/src/AMGX-build-cuda13:/home/adelsaleh/src/AMGX-install-cuda13/lib:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HDGFEM_PRECISION=float64

.venv/bin/python -m scripts.guiding_center.run_guiding_center_cases \
  @run_configs/guiding_center/euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr.args \
  --verbosity 3
```

The qualitative preset uses Euler vortex gas on the disk, h=0.008, p=6,
Poisson tau=1000, dt=0.005 to T=50, native FB-HP-MG Poisson, AMGX BSR
transport, and Holoviz/diagnostics every 0.5 time units. To try only 100 steps,
append `--num-steps 100 --diagnostics-every 1 --plot-every 10
--diagnostics-prefix imex_ark3_gas_preliminary`. Use `--dry-run` to inspect
settings without solving. Long qualitative stability remains user-tested.
