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
    LinearSolveCapacityError,
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

### AMGX Capacity Failures

AMGX/CUDA allocation failures raise `LinearSolveCapacityError`, including
when `raise_on_nonconvergence=False`. The direct device CSR/BSR retry path
stops at the failing attempt; it does not try another configuration, zero
guess, or direct fallback after OOM. An OOM on the primary attempt therefore
performs exactly one attempt. If an ordinary convergence failure precedes
OOM, the earlier attempt remains in the log and no later retry runs. Ordinary
convergence failures retain the configured bounded retry policy.

This handling is shared by `PyAMGXCsrDeviceSolver` and the host-to-AMGX
`solve_pyamgx_csr` adapter. It covers configuration/resource/object creation,
matrix upload, setup and synchronization, solution allocation, vector upload,
iteration, solution download, and cached coefficient replacement. The original
backend exception remains chained. The capacity exception exposes:

- `backend`: `"pyamgx-device"`;
- `phase`: the operation that failed, such as `"matrix upload"`,
  `"solver setup"`, `"solver iteration"`, or `"coefficient replacement"`;
- `memory["amgx"]`: available process-wide live/reserved and peak allocation
  counters from the pinned PyAMGX fork, in bytes;
- `memory["device"]`: whole-device `used_bytes`, `free_bytes`, and
  `total_bytes`, sampled through CUDA;
- `amgx_attempt_count` and `amgx_attempts` on the bounded retry path. Its
  terminal log entry contains `terminal_capacity_failure=True` and `phase`.

Memory is sampled before destroying owned native objects. Missing or failing
optional counters are reported as unavailable; diagnostic failures never
replace the capacity error. These are AMGX process counters and whole-device
usage, not per-solver attribution.

Cleanup attempts every owned native destroy even if an earlier destroy fails,
and preserves the capacity exception. Failed reusable solvers close their
handles. Shared resources remain live while another solver owns them; failed
first acquisition resets the resource manager so a subsequent call can
acquire resources again. A destroy failure does not prove that the native
library reclaimed that allocation.

The retry wrapper retains no full matrix backup on host or device. CSR/BSR
views use the assembled coefficient storage, and left/symmetric scaling
restores it after an AMGX failure. BSR supports left scaling only. If the CUDA
runtime also rejects restoration, the original capacity failure remains
terminal and `matrix_restore_error` records the secondary failure; the caller
must discard that assembly rather than reuse potentially scaled coefficients.

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

`tests/test_amgx_capacity.py` injects failures at the native API boundaries
using two-row CSR and face-BSR stand-ins. It exercises the real Python
adapter/retry/ownership code, with device allocations, native AMGX calls,
and scaling kernels replaced by host doubles. It checks exactly one primary
OOM attempt with both values of `raise_on_nonconvergence`, later-retry OOM,
cached replacement and fixed-operator failures, cleanup after partial
construction and failing destroys, another live solver's shared resources,
original exception/phase preservation, optional memory counters, and absence
of full matrix copies or downloads. Scaling checks verify orchestration and
restoration; they do not qualify CUDA scaling kernels. The suite belongs to
the `host-fast` lane.

Reproduce the focused contract checks from the repository root without native
compilation or simulation:

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  tests/test_amgx_capacity.py tests/test_solver_convergence_contract.py
```

Recorded on 2026-09-25: **162 passed in 0.79 s**, with no skips or warnings.

The supplemental command with the same environment and
`tests/test_documentation_structure.py tests/test_alpha_test_matrix.py
tests/test_amgx_fixed_cycles.py` produced 22 passes and five existing
documentation failures: an extra local documentation directory, generated
LaTeX/PDF artifacts, missing research-artifact links, and missing package
docstrings. The checker and documents with broken links are unchanged from
`HEAD`; the modified backend files introduce no new missing docstrings, and
all local links in this change's four documents resolve. These repository-wide
documentation issues are outside this capacity-handling qualification.

This qualifies Python failure handling with deterministic fault injection.
It does not assert a GPU memory ceiling, reproduce native allocator exhaustion,
or rerun the 807,453-triangle guiding-center case. Native resource-stability
and larger-case evidence remain owned by the
[guiding-center failure regression](../../TODO.md#driver-time-integration-and-validation).

This is bounded contract evidence, not exhaustive numerical parity across
every backend, matrix class, order, mesh, preconditioner, or runtime.
