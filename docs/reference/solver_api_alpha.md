# Early-Alpha Solver API

This document defines the supported Python solver surface for the early alpha.
It freezes names and behavioral contracts within the bounded scope below. It
does not claim that every backend, polynomial order, trace basis, coefficient
type, or host/device combination is supported.

## Supported Imports

Application code should import these symbols from `hdgfem`:

```python
from hdgfem import (
    AdvectionDiffusionReactionHDGOptions,
    AdvectionDiffusionReactionHDGSolver,
    AdvectionDiffusionReactionResult,
    AdvectionDiffusionReactionTimings,
    AdvectionReactionHDGOptions,
    AdvectionReactionHDGSolver,
    AdvectionReactionResult,
    AdvectionReactionTimings,
    DiffusionReactionAssemblyResult,
    DiffusionReactionHDGOptions,
    DiffusionReactionHDGSolver,
    DiffusionReactionResult,
    DiffusionReactionTimings,
    LinearSolveConvergenceError,
    LinearSolveError,
    SolveResult,
    SolveStatus,
    solve_global_system,
    solve_advection_diffusion_reaction_hdg,
    solve_advection_reaction_hdg,
    solve_diffusion_reaction_hdg,
)
```

The HDG objects are also available from `hdgfem.solvers`. Module-oriented code
should use `hdgfem.solvers.advection_reaction`,
`hdgfem.solvers.diffusion_reaction`, or
`hdgfem.solvers.advection_diffusion_reaction`. The reusable solver classes are
the primary API for repeated solves. The three `solve_*_hdg` functions remain
supported for one-shot calls and compatibility. Pure diffusion-reaction flux
postprocessing uses the same canonical
`flux_postprocess_space="l2_closest"|"RT_projection"` selector.
The default `l2_closest` path retains the full
`[P_{p+1}]^2` constrained minimum-distance recovery;
`RT_projection` reconstructs the unique member of
`[P_p]^2 + x P_p` from numerical `P_p(F)` normal moments
and raw `[P_{p-1}]^2` interior moments. The RT solve supports
`postprocessing_backend="numba"|"cupy"`; scalar primal recovery
and the full-space flux recovery remain host Numba. The compatibility spellings
`full-p-plus-1` and `rt-p` remain accepted.

Stationary combined ADR
supports variable scalar and elliptic tensor diffusion with NumPy, Numba, and FP64 raw CUDA (p=0--6; tensor postprocessing disabled)
assembly/reconstruction. Numba reports per-element structural path counts in
`diffusion_structure`; raw CUDA reports these counts too. For raw CUDA tensors
select `hdg_postprocess="none"`; NumPy/Numba additionally permit `"flux"`.
Primal postprocessing still requires constant isotropic diffusion.
The [Numba ADR guide](../backends/numba_adr.md) documents exact structural
dispatch and the incidence-wise sampled normal-diffusivity stabilization.
Stationary combined ADR
currently supports full-boundary Dirichlet elimination; its default
stabilizations are `abs(beta.n)` for upwind advection and
`kappa/L_Omega` for positive constant scalar diffusion. The old
`(p+1)^2*kappa/h_F` rule is an explicit legacy comparison mode. ADR assembly and reconstruction
are independently selectable across the available host paths; Raw CUDA assembly
currently requires Raw CUDA reconstruction. The default total-flux postprocessor is selected by
`flux_postprocess_space="l2_closest"` and uses the full
`[P_{p+1}]^2` constrained minimum-distance method.
`flux_postprocess_space="RT_projection"` selects the Raviart--Thomas moment
reconstruction. Both total-flux variants and coupled primal recovery support
`postprocessing_backend="numba"|"cupy"`. `auto` selects CuPy after Raw CUDA
reconstruction and Numba after host reconstruction. With Raw CUDA,
`materialize_host_solution=False` retains the trace and mixed solution as CuPy
arrays and all returned fields as lazy device-backed `DGField` objects, including
raw/total flux and degree-`p+1` recovery. Accessing a field's `.coeffs` explicitly
downloads it. The default `True` eagerly materializes returned host results.
Explicit Numba postprocessing requires host materialization and is rejected with
`False` before assembly. See the [ADR device recovery contract](../backends/adr_device_postprocessing.md)
for tested scope and transfer accounting. Legacy
`full-p-plus-1` and `rt-p` spellings remain compatibility aliases. The ADR
host default is nonsymmetric `pypardiso`. The linear-solve result, status,
exceptions, and dispatcher are also available from `hdgfem.linalg`.

The optional host direct backend is selected through the same dispatcher:

```python
from hdgfem.linalg import clear_pypardiso_cache, solve_global_system

result = solve_global_system(rows, cols, data, rhs, size, solver="pypardiso")
clear_pypardiso_cache()
```

`"pardiso"` is an alias for the general real-matrix path.
`"pypardiso-spd"` and `"pardiso-spd"` select real-SPD `mtype=2` after a
symmetry check and upper-triangle conversion. Positive definiteness is a caller
and discretization contract. The optional import remains lazy, calls are
serialized around pypardiso's process-global solvers, and the normalized
physical-residual acceptance contract is evaluated against the original full,
unscaled matrix.

`hdgfem.solvers.adv_rea` and `hdgfem.solvers.diff_rea` remain importable
compatibility shims for the full-name implementation modules. Their lower-level assembly
helpers and the aliases `adv_rea_hdg_solv` and `diff_rea_hdg_solve` are not
part of the frozen
package-level surface. Decision: neither these aliases nor the functional
`return_=(...)` tuple interface will be deprecated or removed during alpha.
After a replacement and deprecation release are named, both interfaces remain
available for at least one complete documented transition release; removal can
occur no earlier than the following release.

## Object And Update Semantics

- Option, timing, result, and assembly-result dataclasses are frozen. Arrays,
  device objects, and dictionaries stored inside them are not deep-copied or
  made immutable.
- Solver constructors store the supplied `DGSpace`, problem inputs, and options.
  Problem/coefficient objects are held by reference.
- `with_options(**overrides)` persists valid overrides and conservatively clears
  cached solve artifacts. Unknown option names raise `TypeError`.
- Boundary conditions accept only callables `g(x, y)` and real scalar
  constants. Scalar constants are normalized to constant callables before
  backend dispatch. `DGField` and future `HDGTraceField` boundary inputs are
  rejected until their trace projection/interpolation semantics are designed.
  Advection `boundary_mode="zero-flux"` instead requires
  `boundary_condition=None` and rejects all supplied boundary data.
- Complete problem arguments passed to `solve(...)` replace the stored problem.
  Partial problem bundles raise `ValueError`. Advection zero-flux mode is the
  documented exception: its problem bundle requires no boundary argument (or
  an explicit `None`).
- Other option keywords passed to `solve(...)` persist on the solver. The
  explicit `initial_guess=` argument applies only to that call and is not stored
  in the option dataclass.
- A successful `solve(...)` returns the canonical result object and stores the
  same object plus its principal artifacts on the solver instance. Result
  arrays are owned by the result: a later solve on the same solver never
  overwrites them. `result.field` is always the solution field; for
  device-resident solves it is a lazy device-backed `DGField`
  (`field.coefficients_materialized` reports whether host coefficients exist).
- Without an explicit `initial_guess=`, reusable advection and diffusion solves
  warm-start from the solver's previous reduced trace.
- Reusable solvers are context managers; leaving a `with` block calls
  `close()`, releasing persistent device state, and never suppresses an
  exception.
- Without an `options` object, constructors fill options that the selected
  solver or assembly admits only one value for, unless passed explicitly:
  diffusion `solver="fb-hp-mg-pcg"` implies `trace_basis="legendre-modal"`
  and `scale_system=False`; raw-CUDA, CuPy or Numba diffusion assembly implies
  `boundary_mode="eliminate"`; raw-CUDA advection with
  `boundary_mode="zero-flux"` implies `raw_local_assembly="fused"`. A diffusion
  constructor given `source` and `boundary_condition` without `reaction` uses a
  zero reaction.
- The cache-reset, coefficient-setter, and space-setter methods exposed by each
  class retain their documented invalidation behavior. Call `clear_cache()`
  after mutating a stored coefficient object externally.
- Advection source, beta, reaction, and boundary setters clear the exposed
  matrix/RHS/solution artifacts. Diffusion reaction updates clear the operator;
  with eliminated boundaries and operator caching enabled, diffusion source and
  boundary updates retain the eligible operator and clear only RHS/solution
  state.
- Diffusion `with_options(stabilization=<finite real scalar>)`, with no other
  overrides, retains compatible raw-CUDA flux-recovery references and device
  geometry factors. The DGSpace, trace space, recovery variant, and active CUDA
  device must match. Tau-dependent operators, local diffusion factors, solver
  hierarchies, and solution state still reset. Explicit `clear_cache()` and
  other option updates discard this recovery cache. See the
  [recovery cache contract](../backends/raw_cuda.md#flux-only-recovery-and-scalar-tau-retries)
  for validation commands and scope.
- Contract tests construct every advertised solve-capability row. Actual
  NumPy/Numba advection and diffusion solves additionally exercise per-call warm
  starts and original-system residual acceptance; optional backend execution is
  qualified only by its dedicated runtime lane.

Numba diffusion supports persistent `cache_local_factors="schur-lu"` and
`"schur-cholesky"` with identity diffusion, eliminated boundaries, and operator
caching enabled. These policies use the reusable solver's existing source,
boundary, reaction, options and space invalidation contract. See the
[host Schur cache guide](../backends/numba_diffusion.md) for supported scope,
checks, storage and timings. The default remains `"none"`.

## Results And Functional Compatibility

The default return from each canonical functional solver is its corresponding
result dataclass. The legacy `return_=(...)` tuple-selection interface remains
supported throughout alpha and through the transition policy above; unknown
return keys raise `ValueError`. New code
should consume the result object instead of depending on tuple position.

`DiffusionReactionHDGSolver.assemble_global_matrix()` returns
`DiffusionReactionAssemblyResult`. Advection's current
`assemble_trace_system()` diagnostic returns `AdvectionReactionResult` with no
global solve; no separate advection assembly-result type is promised yet.
Combined ADR returns `AdvectionDiffusionReactionResult`; its `field`, diffusive
`flux`, `total_flux`, trace, stabilization samples, timings, and optional
`postprocessed_field`/`postprocessed_flux` are the supported result view.

## Failure Contract

The supported exception categories are:

- `TypeError` for unknown option names and coefficient types rejected by a
  selected backend;
- `ValueError` for incomplete problem bundles and invalid option values;
- `RuntimeError` when solving without a complete stored problem or when a
  configured solve fails;
- `NotImplementedError` when a valid feature is unavailable in the selected
  backend.

Unsupported assembly/solve/reconstruction combinations raise
`hdgfem.runtime.errors.UnsupportedBackendConfigurationError`, a stable
`NotImplementedError` subclass. It was previously exported from the removed
`hdgfem.backends` package; the support table and validators are in
`hdgfem.solvers.capabilities`. Its message identifies the rejected backend
combination, gives an actionable alternative, and links to
`docs/reference/backend_capabilities.md`. Backend preflight runs before coefficient
sampling, optional-runtime imports, raw-CUDA launch setup, or matrix assembly.
Unknown global solver names remain `ValueError`.

Completed linear solves follow `docs/reference/solver_convergence_contract.md`. A native
success code is insufficient: finite solver-system and original-system true
residuals must meet their targets. Rejected solves return a normalized
`SolveResult` or raise `LinearSolveConvergenceError`, according to
`raise_on_nonconvergence`; the exception carries that result in `.result`.
`backend_info` preserves the native status while `info` remains the normalized
integer compatibility field.

AMGX/CUDA out-of-memory errors instead raise `LinearSolveCapacityError`,
regardless of `raise_on_nonconvergence`. They stop the retry sequence, close
failed owned native resources, and expose the failed phase and available
memory counters. See the
[capacity failure contract](solver_convergence_contract.md#amgx-capacity-failures)
for the fields, cleanup rules, and tested scope.

## Alpha Change Policy

Additive fields and methods may be introduced during alpha. Renames, removals,
return-type changes, or semantic changes to this documented surface require a
release-note entry and a compatibility path or explicit deprecation period.
Backend support and numerical defaults may change from measured evidence and
must be documented separately.
