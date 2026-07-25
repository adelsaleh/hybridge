# Coefficient Input API Notes

This note records the coefficient-input semantics used by the HDG solver APIs after the coefficient cleanup. It is intended to prevent accidental host materialization and to keep backend-specific restrictions explicit.

## Coefficient Forms

### Analytic PDE coefficients

Analytic source, reaction, diffusion, advection, boundary, or stabilization data are Python callables that describe the exact PDE coefficient. NumPy and CuPy assembly paths may sample these callables directly at quadrature points. This is not the same abstraction as a projected DG coefficient field: direct analytic assembly computes quadrature integrals from the exact callable samples, while a projected DG field first replaces the callable by its L2 projection in the chosen DG space.

Callables remain outside `DGField`. Use them when the backend can sample them directly and exact quadrature sampling is desired.

### Projected DG coefficient fields

`DGField` and `VectorDGField` are the canonical discrete coefficient inputs. A projected field created with `space.project_callable(...)` represents a polynomial/table approximation to an analytic coefficient in that DG space. Its `coefficient_kind` is `"projected"`.

Use projected fields when a backend requires tables, when repeat solves should reuse the same discretized coefficient, or when projected-coefficient semantics are desired intentionally.

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
- NumPy assembly helpers in `hdgfem.assembly`: intentional host assembly. Source/reaction/mass helpers use scalar or `constant_value` fast paths before table access where that avoids unnecessary materialization.
- Numba backend adapters: intentional host table/descriptors. Numba is CPU-side and should not depend on device preparation.
- CuPy backend adapters: device paths now use `as_cupy_coefficients` or `as_cupy_vector_coefficients`; the only remaining `.coeffs` fallback is inside `as_cupy_coefficients` when uploading a host-born nonconstant field to the active device.
- Diffusion postprocessing, transfer, diagnostics, plotting, and result comparisons: intentional host operations. These utilities produce host fields, host norms, host plots, or host transfers, so materialization is expected.
- Tests and example scripts: intentional inspection or comparison of host coefficient arrays.

Current decision: no additional secondary utility needs a new zero/constant fast path beyond the existing host assembly and CuPy helper paths. Future GPU-specific utilities should avoid direct `.coeffs` access and use backend coefficient accessors instead.
