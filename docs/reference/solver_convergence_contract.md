# Solver Convergence Contract

This document defines the backend-neutral linear-solve acceptance contract for
the early alpha. It applies to the supported SciPy, PyPardiso, PETSc, Cupyx,
host PyAMGX, and raw-CUDA-to-PyAMGX paths. Supported assembly/solve
combinations remain defined by `docs/reference/backend_capabilities.md`.

## Public Surface

Application code may import the normalized result and exceptions from
`hdgfem`:

```python
from hdgfem import (
    LinearSolveConvergenceError,
    LinearSolveError,
    SolveResult,
    SolveStatus,
    solve_global_system,
)
```

`SolveStatus` is one of `converged`, `not-converged`, `stagnated`, or
`non-finite`. `SolveResult.info` is the compatibility integer: zero means
accepted and nonzero means rejected. `backend_info` preserves the native
backend code or status string.

## Acceptance Rule

A solve is accepted only when all of these conditions hold:

1. The backend reports success.
2. The solution contains only finite values.
3. Solver-system residual diagnostics are finite and meet their target.
4. Original, unscaled `A x = b` residual diagnostics are finite and meet
   their target.

This prevents a native success code from hiding an inaccurate solution. For a
row-scaled solve, `solver_*` fields describe the scaled system and
`physical_*` fields describe the original system. Both targets must pass. The
target is `max(atol, rtol * ||b||)`. Setting both tolerances to zero disables
target comparison, but finite values and backend success are still required.

The normalized diagnostic fields are:

- `backend`, `backend_info`, `status`, `converged`, and
  `failure_reason`;
- `iteration_count`, whose exact counting convention remains backend-specific;
- `solution_is_finite`, `solver_residual_is_finite`, and
  `physical_residual_is_finite`;
- `solver_residual_target_met` and `physical_residual_target_met`;
- `solver_residual_norm`, `solver_residual_target`,
  `physical_residual_norm`, and `physical_residual_target`;
- `stagnated` and the last at most 64 values in `residual_history`, when the
  backend exposes iteration residuals.

## Failure Behavior

Invalid tolerances, iteration limits, shapes, or non-finite matrix/RHS/initial
guess values fail before backend setup with `TypeError` or `ValueError`.

With `raise_on_nonconvergence=False`, a completed but rejected solve returns a
`SolveResult` with `converged=False`. With
`raise_on_nonconvergence=True`, it raises
`LinearSolveConvergenceError`. The exception's `result` attribute carries
the same normalized diagnostics. Native backend exceptions remain chained when
a normalized convergence exception is raised after bounded retries.

AMGX retry sequences are capped at eight total attempts. Every completed
attempt is validated independently; a finite result with the smallest physical
residual is retained for diagnostics. Reusable PyAMGX objects close after
setup/solve failures, one-shot objects are destroyed in `finally`, and PETSc
Mat/Vec/KSP objects are destroyed in reverse ownership order.

## Stagnation Scope

Stagnation is classified when an available recent residual history fails to
improve over the configured window and the residual target is unmet. SciPy
callbacks and raw AMGX solves feed this shared classifier.

PyAMGX exposes residual history only after its blocking `solve` call returns.
The current contract therefore detects and reports AMGX stagnation after an
attempt; it does not interrupt a stalled native solve early. Chunked restarts
or a nonblocking native interface require separate convergence and performance
evidence before becoming a default.

## Test Scope

`tests/test_solver_convergence_contract.py` covers normalized direct and
iterative results, native-success/true-residual disagreement, independent
physical targets, non-finite input rejection, bounded history and stagnation,
stable exceptions, PETSc cleanup ownership, AMGX attempt bounds, and retry
terminal behavior without requiring optional device runtimes.

This is bounded contract evidence, not exhaustive numerical parity across
every backend, matrix class, order, mesh, preconditioner, or runtime.
