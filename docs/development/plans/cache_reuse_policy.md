# Cache Reuse Policy For AR, DR And ADR Solvers

Date: 2026-11-02. Requested by the user. Status: proposed; nothing here is implemented
except where the "Today" table says so. Tracked in `TODO.md`, "Reusable Solver Classes Under
Unsteady Updates".

## Principles

1. **Classify what changed, keep what does not depend on it.** Every cached artifact declares
   the inputs it depends on (table below). An update invalidates only the artifacts depending on
   the changed input, for advection-reaction (AR), diffusion-reaction (DR) and ADR alike.
2. **The default is the best qualified configuration.** A new `cache_policy="auto"` retains what
   recorded evidence shows is faster at numerical parity and what fits the memory budget. The
   decision is made once per operator, from degree, mesh size, equation and available memory, and
   is recorded in the solve details.
3. **The user stays in control.** Every explicit option keeps its meaning and overrides `auto`:
   `cache_local_factors`, `cache_operator`, `cache_local_solvers`, `reuse_static_coefficients`,
   `numba_reuse_local_columns`, `amgx_reuse`, `pardiso_reuse_analysis`,
   `HYBRIDGE_PRIMAL_CACHE_MAX_BYTES`. `cache_policy="none"` reproduces the uncached paths for
   benchmarks and diagnostics.
4. **Exact reuse is automatic; inexact reuse is opt-in.** `auto` changes no converged solution
   beyond the repeat floor. A stale preconditioner (`amgx_reuse="preconditioner"`) is the only
   inexact reuse and stays explicit, with its refresh rules. Policies that are not at parity,
   today `schur-cholesky` and the stored local inverse, are not selected by `auto`.
5. **Every result reports what was reused.** Per artifact: created, reused or refreshed, and
   bytes. A solve never allocates a cache silently beyond the budget.

## What each artifact depends on

"Local system" below means the condensed element matrix `S = S_diff(kappa, tau_d) + S_adv(beta, r, tau_a)`;
AR has only the second term and no flux unknown.

| Artifact | Depends on | Per-element size | Equations |
|---|---|---|---|
| Reference tensors, geometry, quadrature tables | mesh, degree, basis, quadrature | shared or small | all |
| Sparsity pattern, trace reduction maps, side maps | mesh, degree, trace basis, boundary set, format | global | all |
| Workspaces and output buffers, overwritten in place | array shapes | global | all |
| Tuned kernel selection | device, structural workload identity, policy; not sampled coefficient classes, which a changing coefficient can flip | none | all |
| AMGX objects (`amgx_reuse="solver"`), PARDISO analysis | config, size, sparsity | global | all |
| Previous reduced trace as initial guess | sizes only; always valid as a guess | global | all |
| Prepared diffusion tensor, diffusion stabilization | `kappa`, `tau_d`, geometry | small | DR, ADR |
| Factored tensor mass `M_kappa` | `kappa`, geometry | up to `4*NEL^2` reals | DR, ADR |
| Stored `S_diff` (not cached today) | `kappa`, `tau_d`, geometry | `NEL^2` reals | DR, ADR |
| Schur LU factors | all coefficients and all stabilization | `NEL^2` reals, `NEL` ints | all |
| Trace responses of the local solve | all coefficients and all stabilization | `NEL*3*NTR` reals | all |
| Source solution of the local solve | the above and the source | `NEL` reals | all |
| Reduced trace matrix values | all coefficients and all stabilization | global | all |
| AMG hierarchy values | reduced matrix; may lag when opted in | global | all |
| DR p+1 primal operator, scalar `kappa` | mesh, `kappa` | P,Q form `d*(2*NEL+1)` or inverse form `d*(d+1)` reals, `d` the p+1 size | DR |
| ADR primal recovery system | all coefficients and stabilization | per call today | ADR |
| Flux recovery tables (RT, L2-closest) | mesh, degree, trace basis; not `tau` | small | all |

## What to keep for each kind of change

| Change | Keep | Recompute | Typical use |
|---|---|---|---|
| **RHS only**: source or boundary values | Everything else | RHS, source solution | IMEX-ARK3 stages, repeated Poisson solves |
| **Reaction only**: `r`, for example a new `dt` | Structure, `kappa` data, `S_diff`, p+1 operator of DR | `S_adv`, LU, responses, matrix values | Heat equation, variable `dt` |
| **Advection only**: `beta` | Structure, `kappa` data, `M_kappa` factors, `S_diff`, `tau_d` | `S_adv`, `tau_a` if it depends on `beta`, LU, responses, matrix values, ADR recovery | SI-BDF2 density transport |
| **Stabilization only**: scalar `tau` retry | Structure, `kappa` data, flux recovery tables | LU, responses, matrix values | Tau recovery |
| **Diffusion only**: `kappa` | Structure | Every `kappa` artifact, LU, responses, matrix values | Nonlinear or time-dependent diffusion |
| **All PDE coefficients** | Structure only: the first six rows, buffers overwritten in place | Everything else, within the operator's lifetime | Nonlinear iterations, moving parameters |
| **Mesh, degree, trace basis, boundary set or format** | Tuned records only | Everything | Adaption, remeshing |

Three consequences are easy to get wrong:

- **Partial reuse does not save the factorization.** A change in `beta` or `r` is generally
  full rank in the element block, so `S` is refactorized. What partial reuse saves is sampling,
  `kappa` preparation, the mass factorization and the diffusive contraction. In AR almost nothing
  coefficient-specific survives, so its savings are structure, buffers and solver objects.
- **Local data is worth keeping even for a single solve per operator.** Assembly already solves
  every local column. Retaining the factors or responses through reconstruction avoids a second
  factorization whenever the operator changes each step. Only the storage is shared between
  operators, by overwriting it.
- **Stale AMG setup is a separate decision.** Replacing coefficients in an old hierarchy is allowed
  only under `amgx_reuse="preconditioner"`, whatever the change class.

## What `auto` selects

Retention by priority when the budget is short, from the measurements below. Each rung is dropped
only if the next would not fit:

1. Structure, buffers and tuned selection: always.
2. Schur LU factors: always when they fit. Without them a p=9 step was 28% slower.
3. p+1 primal operator: inverse form when the P,Q form does not fit, per-call solve only as a last resort.
4. Trace responses: when they fit. They add 55-150% to the local storage and gave 1-3% on larger meshes.
5. P,Q form of the primal operator over the inverse form: only at low degree, where it is measurably faster.

One shared budget replaces the separate 25% limit of the primal cache: a fraction of device memory
for all retained artifacts, overridable, charged in the rung order above. Host paths use the same
order with their own budget.

## Evidence so far

`docs/campaigns/poisson_element_cost/FINDINGS.md`, section 8, raw-CUDA Poisson on one H200 NVL,
float64: wave rows at p=2-9, N=1,474-1,478,818, AMGX, compatible and raw drift. This covers DR with
`kappa=1` only, change class "RHS only". Holding the tuned selection across changing coefficients is
also measured for ADR (`adr-selection-cost` branch, `docs/campaigns/adr_selection_cost/FINDINGS.md`,
uncommitted, 2026-11-02: selection cost only). Everything else about AR, ADR and the partial classes
above is a design from the dependency table, not yet a measurement.

## Today

| Solver | Coefficient update | Defaults and gaps against this policy |
|---|---|---|
| `AdvectionReactionHDGSolver` | `set_beta`, `set_reaction`, `set_boundary_condition` clear the operator; only retry AMGX solvers survive; `set_source` keeps it under `cache_operator` | `cache_local_solvers=False`, `cache_operator=False`; pattern, maps and buffers are rebuilt when `beta` changes |
| `DiffusionReactionHDGSolver` | `set_reaction` clears everything; RHS updates keep the operator on eligible paths; scalar `tau` retries keep flux recovery | `cache_local_factors="none"`, so a user who sets nothing refactorizes in reconstruction |
| `AdvectionDiffusionReactionHDGSolver` | Static `kappa` data and pattern survive `set_beta` and `set_reaction` under identity-keyed `reuse_static_coefficients=True`; factored mass reloads | `cache_local_factors="schur-lu+mass"`; no `S_diff` retention, no raw-CUDA responses; `numba_reuse_local_columns=False` on host |
| Research drivers | `cache_operator` set from the time scheme; policy spelled per profile | Should inherit `auto` and stop encoding policy |

## Acceptance

- Equality with a fresh solver at the repeat floor for each row of the change table, on host and
  raw-CUDA, with AR, DR and ADR, nodal and modal traces, covering the matching open items of the
  [unsteady solver validation plan](unsteady_solver_validation.md).
- A test per change class asserting which artifacts were kept and which refreshed, through the
  reported details, including mutation of external arrays followed by `clear_cache()`.
- Recorded timing and memory per equation, degree and mesh for each `auto` rung before it becomes
  a default, with setup, cold, repeated and warm-started solves reported separately.
- Explicit options override `auto`, and `cache_policy="none"` matches today's uncached timings.
- Budget tests: forced small budgets drop rungs in order and keep parity.
