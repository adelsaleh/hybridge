# Backend Module Map

The public solver API separates assembly, global sparse inversion, and
reconstruction. Supported combinations and transfer boundaries are defined in
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).
Backend modules are implementation details unless exported from `hybridge` or
`hybridge.solvers`. Internal module paths carry no compatibility guarantee: the
package reorganization moved them without re-export shims.

## Guides

- [`numba_adr.md`](numba_adr.md): variable scalar and tensor ADR diffusion,
  structural fast paths and normal-diffusivity stabilization.
- [`numba_diffusion.md`](numba_diffusion.md): fused host diffusion assembly,
  persistent Schur-LU/Cholesky factors, cache contracts and phase benchmarks.
- [`adr_device_postprocessing.md`](adr_device_postprocessing.md): CuPy ADR
  recovery, optional host materialization, and transfer-accounted parity.
- [`holoviz.md`](holoviz.md): optional GPU plotting for guiding-center runs,
  device sampling, explicit image saving, and static smoke checks.
- [`cuda_execution.md`](cuda_execution.md): CUDA runner entry points, matrix
  formats, sparse-solve handoff, and direct-CSR timing anchor.
- [`amgx_classical_bsr.md`](amgx_classical_bsr.md): hybrid classical-AMG
  hierarchy with a BSR fine operator, scalar setup expansion, and the
  pure-BSR successor contract.
- [`bsr_amgx_dependency_map.md`](bsr_amgx_dependency_map.md): authoritative
  ownership and dependency map for direct BSR assembly, AMGX hybrid/pure-BSR
  hierarchies, HYBRIDGE face-block p/h multigrid, cuSPARSE, and remaining custom
  kernels.
- [`face_dense_gpu.md`](face_dense_gpu.md): experimental fixed-slot face-block
  GMRES path with block-Jacobi/ASM and polynomial preconditioning, including
  its precise CuPy/cuBLAS ownership and independence from AMGX.
- [`face_hp_mg_pcg.md`](face_hp_mg_pcg.md): supported p=4--6 native
  face-BSR Poisson hierarchy, cache/fallback contract, guiding-center policy,
  and periodic raw-CUDA `RT_p` electric-field postprocessing.
- [`raw_cuda.md`](raw_cuda.md): raw-CUDA kernel ownership, launch policy,
  discontinuous-advection audit, and diffusion setup ownership.

## Package Layering

```text
runtime → core → cases → linalg → hdg → {transport, mixed} → solvers → diagnostics → io
```

- A module imports only from its own layer or from layers to its left.
  Function-level (lazy) imports count.
- `transport` and `mixed` share a layer and never import each other. Code they
  both need belongs in `hdg/` or lower.
- The package root `hybridge/__init__.py` is the public facade over every layer.
- `tests/test_package_layering.py` enforces the rule with an empty
  allowed-violation list.

| Layer | Contents |
|---|---|
| `runtime` | Optional-dependency gates (`optional`: `require_cupy`, `require_pyamgx`, `asnumpy`, `njit`/`prange` fallbacks), `precision`, `logging`, `terminal`, `errors` (`UnsupportedBackendConfigurationError`), `devices`, `threads`, `benchmarking`. |
| `core` | Mesh, space, basis, quadrature, fields, transfer, adaptivity, pointwise coefficients, generic mass matrices (`mass`), L2 projection (`projection`), and CuPy mirrors of meshes, spaces and trace spaces (`device`). |
| `cases` | Analytic coefficient sets and initial profiles. |
| `linalg` | Solve dispatch (`system`), `results`, `reduction`, `direct`, `iterative`, orderings, host preconditioners, `face_dense`, `sparse_pattern`, `failure_snapshot`; `amgx/` (device solver, host solver, config, errors), `gpu/` (CuPy sparse views and scaling, Cupyx solves, face-dense GMRES stack, Legendre face-BSR), `multigrid/` (face-block hp-MG). |
| `hdg` | Equation-independent HDG: static condensation (host and device), coefficient sampling (host and device), advection τ/γ policies (`stabilization`), trace maps, reference tables, shared NumPy trace blocks (`matrices`), Gram operators, Numba LU/Cholesky and trace helpers (`numba_common`), and `cuda/` (`launch`, `raw_source`, `pattern`). |
| `transport` | First-order HDG: advection-reaction. |
| `mixed` | Mixed HDG: diffusion-reaction and advection-diffusion-reaction. |
| `solvers` | Public solver modules, `capabilities`, device pipelines, compatibility shims. |
| `diagnostics` | `errors`, `solver`, `guiding_center`; public names re-exported from the package. |
| `io` | Plotting, rasters, Holoviz, movies, records, time series. |

## Operator Families

The package organizes HDG code by discretization family, then by stage
(coefficients, local operator, condensation/assembly, reconstruction,
postprocessing), then by backend.

- **Transport (first-order) HDG**, `hybridge/transport/`: advection-reaction.
  There is no flux unknown; face weights come from the upwind τ/γ policies in
  `hdg/stabilization`.
- **Mixed (second-order) HDG**, `hybridge/mixed/`: diffusion-reaction (DR) and
  advection-diffusion-reaction (ADR). Both use local unknowns `[u, q_x, q_y]`
  with `q = -κ∇u`, the same block layout and signs, and τ in the same three
  places. DR is ADR with β = 0 whenever τ_adv(β = 0) = 0, which holds for the
  upwind family. DR and ADR modules sit side by side (`numba` / `adr_numba`,
  `local_numpy` / `adr_numpy`) and share the building blocks listed below.
  Diffusion τ policies (`GlobalLengthDiffusion`, domain lengths,
  `resolve_diffusion_stabilization`) live in `mixed/stabilization`;
  `mixed/coefficients.normalize_diffusion_stabilization` normalizes τ inputs.

## Backend Ownership

Module paths are relative to `hybridge/`. Solver modules dispatch to these
backends after `solvers/capabilities` preflight validation.

### Transport (advection-reaction)

| Backend | Modules | Responsibility |
|---|---|---|
| NumPy | `transport/local_numpy`, `hdg/matrices`, `core/mass`, `hdg/condensation` | Reference local advection and boundary matrices, trace-stabilization blocks, condensation, and reconstruction. |
| Numba | `transport/numba` → `transport/numba_kernels`, `transport/numba_local_kernels` | Fused projected assembly for penalty, eliminate and zero-flux boundaries, ordered block COO, and reconstruction. |
| CuPy | `transport/cupy`, `hdg/condensation_device` | Device assembly, boundary elimination, and reconstruction. |
| Raw CUDA | `transport/cuda` → `transport/raw_cuda`, `transport/tsle_bsr` | Device orchestration and RHS updates; fused solve-and-emit COO/CSR/BSR kernels and reconstruction; TSLE-BSR (`raw_local_assembly="split3"`). |
| Support | `transport/residual`, `transport/diagnostics` | Semidiscrete upwind residual without a global solve; transport constraint checks and failed-system inspection. |

### Mixed (diffusion-reaction and ADR)

| Stage / backend | DR | ADR |
|---|---|---|
| Coefficients | `mixed/coefficients` (κ kinds), `mixed/stabilization` (τ_diff) | Same, plus `mixed/adr_preparation` (`prepare_adr_data`) and `mixed/coefficients_device` (raw CUDA) |
| NumPy | `mixed/local_numpy` | `mixed/adr_numpy` (through `mixed/local_numpy`) |
| Numba | `mixed/numba` → `mixed/numba_kernels` | `mixed/adr_numba` → `mixed/adr_numba_kernels`, `mixed/numba_diffusion_mass` |
| CuPy | `mixed/cupy` | none (raw CUDA only) |
| Raw CUDA | `mixed/raw_cuda/identity` (identity κ), with assembly wrappers in `mixed/cupy` | `mixed/raw_cuda/tensor` (κ kinds 0–6), `mixed/raw_cuda/adr_operator` |
| Device pipeline | `solvers/diffusion_device` | `solvers/advection_diffusion_reaction_device` |
| Face-dense storage | `mixed/face_dense`, `solvers/diffusion_face_dense` | – |
| Flux postprocessing | `mixed/postprocess/flux`; CuPy RT via `flux_cupy`; raw CUDA via `rt_raw_cuda` and `flux_recovery_raw_cuda` | `mixed/postprocess/total_flux`; CuPy via `flux_cupy` |
| Primal postprocessing | `mixed/postprocess/flux` (host Numba) | `mixed/postprocess/total_flux`; CuPy assembly via `primal_raw_cuda` |

`mixed/postprocess/flux_recovery` holds the reference maps used by the cached
raw-CUDA recovery. Primal recovery remains two methods: DR solves a
Stenberg-type problem on q_h with cached κ-independent factors; ADR solves a
local mixed problem on the recovered total flux.

### Shared DR/ADR building blocks

| Building block | Location | Users |
|---|---|---|
| Mixed local inverse | `mixed/local_numpy.mixed_local_inverse` | DR `local_solvers_numpy`, ADR `adr_numpy`. Takes the equation-specific `u` block; uses the closed-form scalar Schur complement for identity κ. |
| Mixed trace assembler | `mixed/local_numpy.assemble_mixed_trace_system` | DR `assemble_diffusion_trace_system`, ADR `adr_numpy`. τ for DR; `tau_total` and γ = τ_total − β·n for ADR. |
| Numba condensation and column solve | `mixed/numba_common` (`_finish_diffusion_condensation`, `_solve_mixed_columns`) | `mixed/numba_kernels`, `mixed/adr_numba_kernels`. Only the `u`-row construction differs (DR exact projected tables, ADR sampled). |
| Flux-recovery kernels | `mixed/postprocess/numba_kernels` | RT_p (`solve_rt_flux_postprocess_kernel`) is one kernel for both. `l2_closest` shares `_project_flux_to_post` and `_apply_min_distance_correction` between `solve_flux_min_distance_postprocess_kernel` (ADR, host-sampled gaps) and `solve_diffusion_flux_min_distance_postprocess_kernel` (DR, gaps in registers). |
| Cooperative CUDA LU | `hdg/cuda/raw_source.RAW_COOP_LU_FACTOR` | `transport/raw_cuda`, `mixed/raw_cuda/identity`, and `mixed/raw_cuda/tensor` (through `RAW_COOPERATIVE_SOLVES`). |

DR keeps its specializations: identity-κ closed forms, exact projected
reaction and τ tables, cached Schur-LU/Cholesky factors, RHS-only reuse, the
compact CuPy warp kernels, and the raw identity kernels with face-BSR and
`fb-hp-mg-pcg` integration. Replacing the DR kernels wholesale by the ADR
kernels would remove these fast paths.

### Linear algebra and device infrastructure

| Module | Responsibility |
|---|---|
| `solvers/capabilities` | Support table (`BACKEND_CAPABILITIES`, `get_backend_capability`) and preflight validators. |
| `linalg/system` | Global trace-system assembly and solve dispatch. |
| `linalg/amgx/device_solver` | `PyAMGXCsrDeviceSolver` and `solve_reduced_system_amgx_device`: device CSR/BSR AMGX solves, retries, and shared PyAMGX resources for every family. |
| `linalg/amgx/host`, `linalg/amgx/config`, `linalg/amgx/errors` | Host PyAMGX solves, AMGX configuration loading, capacity classification and native-object cleanup. |
| `linalg/gpu/sparse`, `linalg/gpu/cupyx`, `linalg/gpu/cupyx_device` | Device CSR/BSR views and row/symmetric scaling, Cupyx Krylov and ILU, device-resident Cupyx solves. |
| `linalg/gpu/legendre_face_bsr`, `linalg/multigrid/` | Legendre face-BSR operators and the face-block hp-multigrid PCG. |
| `hdg/cuda/launch` | Raw-CUDA block-size policy and validation; kernel compilation with the dynamic shared-memory opt-in. |
| `hdg/cuda/pattern` | Reduced CSR/BSR pattern builder shared by AR, DR and ADR. |
| `hdg/cuda/raw_source` | Shared CUDA source: trace orientation, cooperative LU and column solves, checked warp LU. |
| `core/device` | CuPy mirrors (`CupyDGSpace`, `as_cupy_space`, `as_cupy_trace_space`) and device coefficient upload helpers. |

## Solver Modules

- `hybridge.solvers.advection_reaction`, `hybridge.solvers.diffusion_reaction` and
  `hybridge.solvers.advection_diffusion_reaction` own the public module APIs and
  stage orchestration.
- `hybridge.solvers.adv_rea` and `hybridge.solvers.diff_rea` are compatibility
  shims retained by the documented alpha API contract.
- Unsupported combinations raise
  `hybridge.runtime.errors.UnsupportedBackendConfigurationError`.

The hard-coded fused tensor Test 7 adapter and kernels live under
`scripts/diffusion_reaction/experiments/` and are not installed.

## Naming And Ownership Rules

- Place a module by family, then stage, then backend. Code used by both
  families belongs in `hdg/`; equation-independent linear algebra belongs in
  `linalg/`.
- Use full equation names in production modules; standard algorithm names such
  as CUDA, AMGX, HDG, SCC, ILU, and GS may remain abbreviated.
- Name modules by execution role, not prototype generation number.
- Keep experiments under `scripts/<equation>/experiments/`, with focused tests
  for reusable numerical logic.
- Keep optional imports lazy. Importing `hybridge` must not require CUDA, AMGX,
  PETSc, PARDISO, Gmsh, or DOLFINx.
- Do not infer residency from a filename. Use the checked capability record and
  transfer instrumentation.
- Keep dated performance conclusions in [`../research/`](../research/), not in
  maintained backend guidance.
