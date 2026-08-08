# Early-Alpha Solver API

This document defines the supported Python solver surface for the early alpha.
It freezes names and behavioral contracts within the bounded scope below. It
does not claim that every backend, polynomial order, trace basis, coefficient
type, or host/device combination is supported.

## Supported Imports

Application code should import these symbols from `hdgfem`:

```python
from hdgfem import (
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
    solve_advection_reaction_hdg,
    solve_diffusion_reaction_hdg,
)
```

The HDG objects are also available from `hdgfem.solvers`. Module-oriented code
should use `hdgfem.solvers.advection_reaction` or
`hdgfem.solvers.diffusion_reaction`. The reusable solver classes are the primary
API for repeated and unsteady solves. The two `solve_*_hdg` functions remain
supported for one-shot calls and compatibility. The linear-solve result, status,
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
  same object plus its principal artifacts on the solver instance.
- The cache-reset, coefficient-setter, and space-setter methods exposed by each
  class retain their documented invalidation behavior. Call `clear_cache()`
  after mutating a stored coefficient object externally.
- Advection source, beta, reaction, and boundary setters clear the exposed
  matrix/RHS/solution artifacts. Diffusion reaction updates clear the operator;
  with eliminated boundaries and operator caching enabled, diffusion source and
  boundary updates retain the eligible operator and clear only RHS/solution
  state.
- Contract tests construct every advertised solve-capability row. Actual
  NumPy/Numba advection and diffusion solves additionally exercise per-call warm
  starts and original-system residual acceptance; optional backend execution is
  qualified only by its dedicated runtime lane.

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
`hdgfem.backends.UnsupportedBackendConfigurationError`, a stable
`NotImplementedError` subclass. Its message identifies the rejected backend
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

## Alpha Change Policy

Additive fields and methods may be introduced during alpha. Renames, removals,
return-type changes, or semantic changes to this documented surface require a
release-note entry and a compatibility path or explicit deprecation period.
Backend support and numerical defaults may change from measured evidence and
must be documented separately.
