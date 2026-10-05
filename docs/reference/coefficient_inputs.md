# Coefficient Inputs

This reference defines coefficient-input semantics for the HDG solver APIs. It prevents accidental host materialization and distinguishes mathematical coefficient meaning from backend residency. Exact support limits remain governed by [`backend_capabilities.md`](backend_capabilities.md). The per-solver/backend user forms,
stabilization policies, callable vectorization rules, and lowering behavior are
listed in [`coefficient_stabilization_matrix.md`](coefficient_stabilization_matrix.md).

## Coefficient Forms

### Analytic PDE coefficients

Analytic source, reaction, diffusion, advection, boundary, or stabilization data are Python callables that describe the exact PDE coefficient. NumPy and CuPy assembly paths may sample supported callable coefficients directly at quadrature points; accepted forms remain coefficient- and backend-specific. Both NumPy and CuPy sample callable advection stabilization on element-side face quadrature, while Numba requires callable stabilization to be projected first and raw-CUDA currently requires the default upwind value. See [`advection_boundary_stabilization.md`](advection_boundary_stabilization.md). This is not the same abstraction as a projected DG coefficient field: direct analytic assembly computes quadrature integrals from the exact callable samples, while a projected DG field first replaces the callable by its L2 projection in the chosen DG space.

Callables remain outside `DGField`. Use them when the backend can sample them directly and exact quadrature sampling is desired.

### Projected DG coefficient fields

`DGField` and `VectorDGField` are the canonical discrete coefficient inputs. A projected field created with `space.project_callable(...)` represents a polynomial/table approximation to an analytic coefficient in that DG space. Its `coefficient_kind` is `"projected"`. NumPy and CuPy evaluate DG stabilization by contracting its coefficients with the face-basis reference tables belonging to that field's `DGSpace`; they do not route a `DGField` through analytic callable or physical point-location evaluation.

Use projected fields when a backend requires tables, when repeat solves should reuse the same discretized coefficient, or when projected-coefficient semantics are desired intentionally.

### Element-local coefficients

`ElementCoefficient(function, mesh, components=1, name=...)` describes a
coefficient that is only known elementwise, for example a pointwise quotient of
DG fields such as `Gamma/max(n, n_floor)` or a term containing the elementwise
gradient of a DG field. `function(reference_points, *, xp, t=None)` returns the
values at the same reference-triangle points on every element, shape `(K, n)`
or `(K, n, components)`, as NumPy (`xp=numpy`) or CuPy (`xp=cupy`) arrays.
Because the values are element-local, the value an element sees on a face is
its own value at reference face points: one evaluator supplies volume samples,
per-incidence face samples (which may jump across a face) and samples on the
degree-`p+1` postprocessing quadrature, and nothing is projected into the
solution space. Use `field_values_at_ref(field, points, device=...)` and
`field_gradient_at_ref(...)` to evaluate DG fields inside such a function
without host round trips.

ADR accepts `ElementCoefficient` for `source`, `reaction` and the two-component
`beta` on the NumPy, Numba and raw-CUDA assembly paths and in both total-flux
and primal postprocessing. The raw-CUDA path calls the function with
`xp=cupy`, so the samples stay on the device; a function that raises
`TypeError` for CuPy input falls back to host evaluation and upload. Instances
are deliberately not callable, so they are never mistaken for `(x, y)` laws.
Other solver families do not accept them yet.

### Compiled pointwise coefficients (host Numba)

`pointwise_coefficient(function, mesh_or_space, *, fields=(), gradients=(),
params=(), time=0., name=...)` compiles `function` (or a tuple of component
functions) with Numba `cfunc` and returns a `PointwiseCoefficient`, an
`ElementCoefficient` that every ADR consumer above accepts. The function takes
`(x, y, t)`, or `(x, y, t, v)` where `v` holds the values of `fields`, then the
`(d/dx, d/dy)` pair of each field in `gradients`, then `params`. Sampling runs
in the parallel kernels of `hybridge/core/pointwise_kernels.py`, which are compiled
once per signature and cached on disk, so new functions never recompile them;
each function is itself cached when it is defined in a file (closures over
numbers included, keyed by their captured values). `at_time(t)` rebinds the
time and `bind(fields=..., gradients=..., params=..., time=...)` the point data
without recompiling, as long as the layout of `v` stays the same; a time
stepper compiles once and rebinds every step.

`pointwise_law(function, *, params=(), time=0., name=...)` compiles the same
kind of function into a `PointwiseLaw`, a plain `(x, y)` callable accepted
wherever a coefficient law is, for example the components of a diffusion
tensor (`v` then holds only `params`).

A compiled function may call other compiled functions only through module
globals: define them at module level with `@njit(cache=True)`. A closure that
captures a compiled function misses Numba's cache in every process, so it
recompiles each run and the cache keeps growing.
`scripts/n_gamma/compiled.py` shows the pattern: its coefficient functions
call the generated scalar evaluators of `scripts/n_gamma/cases/forcing_numba.py`.

Only NumPy and `math` code compiles. SciPy functions, Python objects and other
libraries raise `TypeError` asking to project the coefficient first
(`space.project_callable(...)`), or to pass the plain callable, which is
sampled with NumPy on the host. Functions reading numeric or array module
globals (other than `math`/NumPy constants such as `pi`) raise `ValueError`:
Numba freezes globals and its cache does not notice later changes, so pass such
values through `params`, `fields`, `t` or a closure over numbers. Functions defined outside a
file (REPL, notebook, `exec`) compile with a warning and without the disk cache.
Evaluation is host-only; the raw-CUDA path needs projected coefficients.

Precomputed device arrays are also accepted on the raw-CUDA ADR path: a CuPy
source of shape `(K, el_dof)` is taken as element moments and `(K, nq)` as
volume-quadrature values (moments win when both shapes coincide, for example
p=2 with the default 6-point rule, so prefer an `ElementCoefficient` or
`hybridge.hdg.condensation.source_moments_from_values` for values), and a CuPy reaction of shape
`(K, nq)` passes through unchanged.

### Lazy zero and constant DG fields

`space.zeros(...)` and `space.constant(value, ...)` create exact DG fields with metadata but no full coefficient table. Their `constant_value` property is the authoritative fast-path fact. Accessing `.coeffs` or `.asarray()` materializes the full host table.

Backends should prefer `constant_value` over `is_zero` when possible, because `is_zero` may need to inspect a nonconstant table. For exact constants, assembly should use reference moments or mass matrices instead of expanding a per-element coefficient table.

### Host and device coefficient tables

`DGField.coeffs` always means host NumPy coefficients. This is deliberate. NumPy, Numba, postprocessing, transfer, plotting, and result-inspection utilities may call `.coeffs` when they are producing or consuming host arrays.

CuPy-backed fields can be constructed without a host table through `CupyDGSpace.field(...)` or `CupyDGSpace.project_callable(...)`. These return ordinary `DGField` objects with a cached device coefficient table. CuPy backends should use `as_cupy_coefficients(field, cspace)` or `as_cupy_vector_coefficients(field, cspace)` instead of `field.coeffs`; those helpers reuse cached device storage, upload host-born fields only when necessary, and download nothing unless the caller explicitly requests `.coeffs`.

## Backend Support Matrix

| Backend path | Analytic callables | DGField tables | Lazy zero/constant | Device-backed DGField |
| --- | --- | --- | --- | --- |
| NumPy assembly | Sample directly on host | Host `.coeffs` by design | Fast paths where available | Downloads through `.coeffs` if used |
| Numba assembly/reconstruction | Rejected | Host tables/descriptors only | Compact host descriptors for zero/constant | Downloads through `.coeffs` if used |
| CuPy assembly | Sample directly on device when callable is CuPy-compatible | `as_cupy_coefficients` | Fast paths without table materialization | Native path |
| Raw-CUDA diffusion | Callable source can become device `source_rhs`; reaction remains zero-only for now | Device tables or prepared device RHS as required | Zero reaction checked through metadata/device data | Native for accepted table inputs |
| Raw-CUDA advection-reaction | Rejected for strict raw paths | Projected `DGField`/`VectorDGField` required | Currently table-driven, with zero/constant descriptors deferred | Native once accessed via CuPy helpers |

## `.coeffs` Materialization Audit

The remaining package-code `.coeffs` uses fall into these categories:

- Core `DGField`/`VectorDGField` methods: intentional host operations such as `values`, gradients, algebra, packing with `as_component_first`, and explicit `asarray`. These define the host semantics of the public objects.
- NumPy reference assembly helpers (`hybridge.core.mass`, `hybridge.hdg.matrices`, `hybridge.transport.local_numpy`, `hybridge.mixed.local_numpy`): intentional host assembly. Source/reaction/mass helpers use scalar or `constant_value` fast paths before table access where that avoids unnecessary materialization.
- Numba backend adapters: intentional host table/descriptors. Numba is CPU-side and should not depend on device preparation.
- CuPy backend adapters: device paths now use `as_cupy_coefficients` or `as_cupy_vector_coefficients`; the only remaining `.coeffs` fallback is inside `as_cupy_coefficients` when uploading a host-born nonconstant field to the active device.
- Host diffusion postprocessing, mesh transfer, plotting, and explicit result inspection intentionally materialize host arrays. Device-capable scalar-error evaluation, field combinations, and solver-result extraction instead use resident coefficients/traces when available; host-only reductions such as `DGField.integral()` and `DGField.min_max()` remain explicit materialization boundaries.
- Tests and example scripts: intentional inspection or comparison of host coefficient arrays.

Current decision: no additional secondary utility needs a new zero/constant fast path beyond the existing host assembly and CuPy helper paths. Future GPU-specific utilities should avoid direct `.coeffs` access and use backend coefficient accessors instead.
