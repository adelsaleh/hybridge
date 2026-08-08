# Backend And Residency Capabilities

This is the authoritative early-alpha support matrix for the public
advection-reaction and diffusion-reaction solver APIs. A listed row is a
supported assembly/solve/reconstruction combination. Backend experiments not
listed here remain research interfaces and carry no alpha compatibility
guarantee.

`host -> device -> host` means the public API assembles or owns host data,
uploads it for the sparse solve, and returns a host solution. `device -> host`
means device assembly is followed by an intentional host materialization.
`device (optional host copy)` means device data remain authoritative and are
downloaded only for an explicitly host-dependent operation. Only
rows whose active phases say `device`, or say `device (optional host copy)`
with host materialization disabled, are fully device resident.

The Python source of truth is
`hdgfem.backends.capabilities.BACKEND_CAPABILITIES`. The generated block below
is checked by `tests/test_backend_capabilities.py`.

<!-- BEGIN GENERATED CAPABILITY MATRIX -->
| Equation | Operation | Assembly | Sparse solve | Assembly residency | Solve residency | Reconstruction | Boundary modes | Trace bases | Notes |
|---|---|---|---|---|---|---|---|---|---|
| advection-reaction | assemble | numpy | none | host | none | none | penalty, eliminate | legacy-lagrange, legendre-modal | - |
| advection-reaction | assemble | numba | none | host | none | none | penalty, eliminate, zero-flux | legacy-lagrange, legendre-modal | - |
| advection-reaction | assemble | cupy | none | device (optional host copy) | none | none | penalty, eliminate | legacy-lagrange, legendre-modal | COO values and RHS remain on-device unless host assembly diagnostics are requested. |
| advection-reaction | assemble | raw-cuda | none | device -> host | none | none | eliminate, zero-flux | legacy-lagrange, legendre-modal | Assembly-only diagnostics materialize the reduced system on the host. |
| advection-reaction | solve | numpy | scipy | host | host | host | penalty, eliminate | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numpy | pypardiso | host | host | host | penalty, eliminate | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numpy | petsc | host | host | host | penalty, eliminate | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numpy | cupyx | host | host -> device -> host | host | penalty, eliminate | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numpy | amgx | host | host -> device -> host | host | penalty, eliminate | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numba | scipy | host | host | host | penalty, eliminate, zero-flux | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numba | pypardiso | host | host | host | penalty, eliminate, zero-flux | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numba | petsc | host | host | host | penalty, eliminate, zero-flux | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numba | cupyx | host | host -> device -> host | host | penalty, eliminate, zero-flux | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | numba | amgx | host | host -> device -> host | host | penalty, eliminate, zero-flux | legacy-lagrange, legendre-modal | - |
| advection-reaction | solve | cupy | scipy | device (optional host copy) | host | device (optional host copy) | penalty, eliminate | legacy-lagrange, legendre-modal | Cupyx stays device-native with no preconditioner, a device operator, or device ILU(1); host solvers and host ILU export materialize the trace system. |
| advection-reaction | solve | cupy | pypardiso | device (optional host copy) | host | device (optional host copy) | penalty, eliminate | legacy-lagrange, legendre-modal | Cupyx stays device-native with no preconditioner, a device operator, or device ILU(1); host solvers and host ILU export materialize the trace system. |
| advection-reaction | solve | cupy | petsc | device (optional host copy) | host | device (optional host copy) | penalty, eliminate | legacy-lagrange, legendre-modal | Cupyx stays device-native with no preconditioner, a device operator, or device ILU(1); host solvers and host ILU export materialize the trace system. |
| advection-reaction | solve | cupy | cupyx | device (optional host copy) | device (optional host copy) | device (optional host copy) | penalty, eliminate | legacy-lagrange, legendre-modal | Cupyx stays device-native with no preconditioner, a device operator, or device ILU(1); host solvers and host ILU export materialize the trace system. |
| advection-reaction | solve | cupy | amgx | device (optional host copy) | host -> device -> host | device (optional host copy) | penalty, eliminate | legacy-lagrange, legendre-modal | Cupyx stays device-native with no preconditioner, a device operator, or device ILU(1); host solvers and host ILU export materialize the trace system. |
| advection-reaction | solve | raw-cuda | scipy | device -> host | host | device (optional host copy) | eliminate, zero-flux | legacy-lagrange, legendre-modal | The reduced matrix is downloaded before non-AMGX solves. |
| advection-reaction | solve | raw-cuda | pypardiso | device -> host | host | device (optional host copy) | eliminate, zero-flux | legacy-lagrange, legendre-modal | The reduced matrix is downloaded before non-AMGX solves. |
| advection-reaction | solve | raw-cuda | petsc | device -> host | host | device (optional host copy) | eliminate, zero-flux | legacy-lagrange, legendre-modal | The reduced matrix is downloaded before non-AMGX solves. |
| advection-reaction | solve | raw-cuda | cupyx | device -> host | host -> device -> host | device (optional host copy) | eliminate, zero-flux | legacy-lagrange, legendre-modal | The reduced matrix is downloaded before non-AMGX solves. |
| advection-reaction | solve | raw-cuda | amgx | device | device | device (optional host copy) | eliminate, zero-flux | legacy-lagrange, legendre-modal | Direct device AMGX and compatible CuPy-to-Cupyx solves are fully device-resident. |
| diffusion-reaction | assemble | numpy | none | host | none | none | eliminate | legacy-lagrange, legendre-modal, bernstein | - |
| diffusion-reaction | assemble | numba | none | host | none | none | eliminate | legacy-lagrange, legendre-modal | - |
| diffusion-reaction | assemble | cupy | none | device -> host | none | none | eliminate | legacy-lagrange, legendre-modal | Identity diffusion and scalar stabilization only. |
| diffusion-reaction | assemble | raw-cuda | none | device -> host | none | none | eliminate | legacy-lagrange, legendre-modal | Identity diffusion and scalar stabilization only. |
| diffusion-reaction | solve | numpy | scipy | host | host | host | penalty, eliminate | legacy-lagrange, legendre-modal, bernstein | Bernstein is supported without HDG postprocessing. |
| diffusion-reaction | solve | numpy | pypardiso | host | host | host | penalty, eliminate | legacy-lagrange, legendre-modal, bernstein | Bernstein is supported without HDG postprocessing. |
| diffusion-reaction | solve | numpy | petsc | host | host | host | penalty, eliminate | legacy-lagrange, legendre-modal, bernstein | Bernstein is supported without HDG postprocessing. |
| diffusion-reaction | solve | numpy | cupyx | host | host -> device -> host | host | penalty, eliminate | legacy-lagrange, legendre-modal, bernstein | Bernstein is supported without HDG postprocessing. |
| diffusion-reaction | solve | numpy | amgx | host | host -> device -> host | host | penalty, eliminate | legacy-lagrange, legendre-modal, bernstein | Bernstein is supported without HDG postprocessing. |
| diffusion-reaction | solve | numba | scipy | host | host | host | eliminate | legacy-lagrange, legendre-modal | - |
| diffusion-reaction | solve | numba | pypardiso | host | host | host | eliminate | legacy-lagrange, legendre-modal | - |
| diffusion-reaction | solve | numba | petsc | host | host | host | eliminate | legacy-lagrange, legendre-modal | - |
| diffusion-reaction | solve | numba | cupyx | host | host -> device -> host | host | eliminate | legacy-lagrange, legendre-modal | - |
| diffusion-reaction | solve | numba | amgx | host | host -> device -> host | host | eliminate | legacy-lagrange, legendre-modal | - |
| diffusion-reaction | solve | cupy | amgx | device | device | device (optional host copy) | eliminate | legacy-lagrange, legendre-modal | Identity diffusion and scalar stabilization; HDG postprocessing may materialize reconstruction data on host. |
| diffusion-reaction | solve | raw-cuda | amgx | device | device | device | eliminate | legacy-lagrange, legendre-modal | Requires CSR, identity diffusion, scalar stabilization, and no HDG postprocessing. |
<!-- END GENERATED CAPABILITY MATRIX -->

## Enforcement

Unsupported rows raise
`hdgfem.backends.UnsupportedBackendConfigurationError`, a stable subclass of
`NotImplementedError`. The message identifies the equation, operation,
assembly backend, sparse-solver family, reason, and this document. Invalid
option values and unknown solver names remain `ValueError`.

Preflight runs before coefficient sampling, local assembly, optional-backend
imports, raw-CUDA launch selection, and sparse-solver setup. Optional runtime
availability is checked later only for a supported row that actually needs that
runtime.

## Executable Contract Evidence

- Every published solve row is constructed through its reusable solver class in
  `tests/test_backend_capabilities.py`; this is configuration coverage, not
  optional-runtime numerical parity.
- NumPy and Numba reusable advection/diffusion solves exercise per-call initial
  guesses, true physical residuals, and documented cache invalidation/reuse.
- The GPU release lane asserts one host-matrix upload and one solution download
  for host-assembled Cupyx, and zero `cp.asnumpy` full-array downloads for
  raw-CUDA advection/diffusion CSR-to-AMGX and compatible CuPy-to-Cupyx
  advection solves without host materialization.

## Important Limits

- CuPy advection reconstructs directly from a device trace and returns
  `field_device`/`trace_device`; host `field`/`trace` copies are made only when
  `materialize_host_solution=True`. Local inverses and element-boundary blocks
  are recomputed for reconstruction and retained only when explicitly cached.
- CuPy hands device COO/RHS directly to Cupyx with no preconditioner, an
  existing device operator, or device ILU(1). Host solvers, host ILU export,
  matrix diagnostics, and `materialize_host_system=True` intentionally
  materialize the trace system.
- Raw-CUDA advection with non-AMGX sparse solvers downloads the reduced system;
  Cupyx then uploads it again. This is supported mixed residency, not a direct
  device pipeline.
- Raw-CUDA advection with `boundary_mode="zero-flux"` requires
  `raw_local_assembly="fused"`.
- Raw-CUDA diffusion full solves are exposed through
  `DiffusionReactionHDGSolver`, require direct CSR-to-AMGX, and do not support
  HDG postprocessing in the solver call.
- Diffusion `assemble_global_matrix()` always returns host arrays, including
  when CuPy or raw-CUDA performed the assembly.
- Numba diffusion uses eliminated boundary trace degrees of freedom. Request
  `boundary_mode="eliminate"` explicitly.
- The two production trace bases are `legacy-lagrange` and `legendre-modal`.
  NumPy diffusion additionally has a bounded Bernstein solve/assembly contract
  without HDG postprocessing.
  Bernstein is not a beta production requirement, so no Numba or raw-CUDA
  expansion is planned for beta; the existing bounded NumPy contract remains
  supported and broader work can be reconsidered from user demand.
