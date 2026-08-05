# Backend Module Map

The public solver API separates assembly, global sparse inversion, and
reconstruction. Supported combinations and transfer boundaries are defined in
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).
Backend modules remain implementation details unless exported from `hdgfem`.

## Solver Modules

- `hdgfem.solvers.advection_reaction` owns the advection-reaction
  implementation and public module API.
- `hdgfem.solvers.diffusion_reaction` owns the diffusion-reaction
  implementation and public module API.
- `hdgfem.solvers.adv_rea` and `hdgfem.solvers.diff_rea` are compatibility
  shims retained by the documented alpha API contract.

## Backend Modules

| Module | Responsibility |
|---|---|
| `hdgfem.backends.capabilities` | Support-table lookup and preflight validation. |
| `hdgfem.backends.numpy` | NumPy assembly and reconstruction adapters. |
| `hdgfem.backends.numba` | Table-driven Numba host assembly and reconstruction. |
| `hdgfem.backends.cupy` | Generic CuPy mirrors, Cupyx sparse solves, device ILU, and PyAMGX resource adapters. |
| `hdgfem.backends.raw_cuda` | Shared raw-CUDA launch policy and validation. |
| `hdgfem.backends.advection_cuda` | CUDA advection orchestration, reconstruction, and direct device-CSR-to-AMGX solve. |
| `hdgfem.backends.advection_raw_cuda` | Raw-CUDA advection assembly and reconstruction kernels. |
| `hdgfem.backends.diffusion_cupy` | CuPy diffusion assembly and postprocessing helpers. |
| `hdgfem.backends.diffusion_raw_cuda` | Raw-CUDA diffusion assembly, cached operator/RHS, reconstruction, and postprocessing kernels. |

The former `cupy_*_gpu4` and abbreviated equation backend modules were internal
prototype names and were removed before the first alpha. The hard-coded fused
tensor Test 7 adapter and kernels now live under
`scripts/diffusion_reaction/experiments/` and are not installed.

## Naming And Ownership Rules

- Use full equation names in production modules; standard algorithm names such
  as CUDA, AMGX, HDG, SCC, ILU, and GS may remain abbreviated.
- Name modules by equation and execution role, not prototype generation number.
- Keep experiments under `scripts/<equation>/experiments/`, with focused tests
  for reusable numerical logic.
- Keep optional imports lazy. Importing `hdgfem` must not require CUDA, AMGX,
  PETSc, PARDISO, Gmsh, or DOLFINx.
- Do not infer residency from a filename. Use the checked capability record and
  transfer instrumentation.
- Further splitting of generic CuPy resources, sparse-solver adapters, and
  reconstruction remains an internal ownership improvement, not a reason to
  reintroduce historical names.
