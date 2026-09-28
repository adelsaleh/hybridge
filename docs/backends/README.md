# Backend Module Map

The public solver API separates assembly, global sparse inversion, and
reconstruction. Supported combinations and transfer boundaries are defined in
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).
Backend modules remain implementation details unless exported from `hdgfem`.

## Guides

- [`numba_adr.md`](numba_adr.md): variable scalar and tensor ADR diffusion,
  structural fast paths and normal-diffusivity stabilization.
- [`numba_diffusion.md`](numba_diffusion.md): fused host diffusion assembly,
  persistent Schur-LU/Cholesky factors, cache contracts and phase benchmarks.

- [`holoviz.md`](holoviz.md): optional GPU plotting for guiding-center runs,
  device sampling, explicit image saving, and static smoke checks.
- [`cuda_execution.md`](cuda_execution.md): CUDA runner entry points, matrix
  formats, sparse-solve handoff, and direct-CSR timing anchor.
- [`amgx_classical_bsr.md`](amgx_classical_bsr.md): hybrid classical-AMG
  hierarchy with a BSR fine operator, scalar setup expansion, and the
  pure-BSR successor contract.
- [`bsr_amgx_dependency_map.md`](bsr_amgx_dependency_map.md): authoritative
  ownership and dependency map for direct BSR assembly, AMGX hybrid/pure-BSR
  hierarchies, HDGFEM face-block p/h multigrid, cuSPARSE, and remaining custom
  kernels.
- [`face_dense_gpu.md`](face_dense_gpu.md): experimental fixed-slot face-block
  GMRES path with block-Jacobi/ASM and polynomial preconditioning, including
  its precise CuPy/cuBLAS ownership and independence from AMGX.
- [`face_hp_mg_pcg.md`](face_hp_mg_pcg.md): supported p=4--6 native
  face-BSR Poisson hierarchy, cache/fallback contract, guiding-center policy,
  and periodic raw-CUDA `RT_p` electric-field postprocessing.
- [`raw_cuda.md`](raw_cuda.md): raw-CUDA kernel ownership, launch policy,
  discontinuous-advection audit, and diffusion setup ownership.

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
| `hdgfem.backends.amgx_errors` | Shared AMGX/CUDA capacity classification, memory diagnostics, and native-object cleanup. |
| `hdgfem.backends.raw_cuda` | Shared raw-CUDA launch policy and validation. |
| `hdgfem.backends.advection_cuda` | CUDA advection orchestration, reconstruction, and direct device-CSR-to-AMGX solve. |
| `hdgfem.backends.advection_raw_cuda` | Raw-CUDA advection assembly and reconstruction kernels. |
| `hdgfem.backends.diffusion_cupy` | CuPy diffusion assembly and postprocessing helpers. |
| `hdgfem.backends.diffusion_raw_cuda` | Raw-CUDA diffusion assembly, cached operator/RHS, reconstruction, and postprocessing kernels. |
| `hdgfem.backends.diffusion_rt_postprocess_raw_cuda` | Raw-CUDA per-element `RT_p` diffusion-flux moment reconstruction. |

The former `cupy_*_gpu4` and abbreviated equation backend modules were
internal prototype names and were removed before the first alpha. The
hard-coded fused tensor Test 7 adapter and kernels now live under
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
- Keep dated performance conclusions in [`../research/`](../research/), not in
  maintained backend guidance.

- [ADR device postprocessing](adr_device_postprocessing.md): CuPy recovery, optional host materialization, and transfer-accounted parity.
