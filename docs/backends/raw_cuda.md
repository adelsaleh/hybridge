# Raw-CUDA Backend Notes

This document records raw-CUDA implementation ownership and launch policy.
Current support is defined only by
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).

## Kernel Ownership

The fused raw-CUDA element kernels perform the expensive local work:

- construct equation-specific local operators from prepared coefficient data;
- factor and solve the local condensed systems;
- eliminate known trace columns;
- emit reduced COO entries, direct scalar CSR values, or direct dense face-BSR blocks;
- accumulate the reduced right-hand side on device;
- reconstruct local fields with the matching equation kernel.

Reference tensors, projected/source moments, compact boundary data, topology,
and reduction maps are prepared before the element launch. Reusable solver
objects cache matrix- and graph-dependent data when their invalidation contract
allows it.

## Launch Policy

Public solvers and runners use `raw_block_size="auto"`. The established fused
kernels resolve that option before kernel selection and cache-key construction;
kernels receive an integer block size. TSLE-BSR is the exception: its three
stages retain `"auto"` until the backend selects an independent launch size for
each stage.

| Equation | Polynomial order | Recommended block size |
|---|---:|---:|
| Advection-reaction | `p <= 6` | 32 |
| Advection-reaction | `p = 7` | 64 |
| Advection-reaction | `p = 8, 9` (experimental) | 128 |
| Diffusion/Poisson | `p <= 2` | 32 |
| Diffusion/Poisson | `p = 3, 4` | 64 |
| Diffusion/Poisson | `p = 5, 6` | 128 |

These are policy defaults covered by launch-resolution and parity tests.
The p=8--9 fused recommendation is additionally backed by the CUDA-13 SM75
block sweep below; it remains experimental rather than a cross-architecture
claim. Explicit `1`, `32`, `64`, and `128` overrides remain available for
benchmark reproduction.
Block size 1 is a serial correctness baseline, not a performance
recommendation. Orders outside the policy fail explicitly rather than silently
choosing a launch shape.

## Tri-Stage Local Elimination BSR (`split3`)

`raw_local_assembly="split3"` selects **TSLE-BSR**, implemented in
`hdgfem.backends.advection_tsle_bsr`. It preserves the fused advection HDG
discretization and separates only the execution schedule:

1. `advection_tsle_build` constructs each local operator `A_e`, unsolved trace
   and source columns `[B_e, f_e]`, and the face `tau/gamma` tables.
2. `advection_tsle_solve` performs the cooperative scaled-pivot LU and applies
   it to every trace/source column. The deprecated `safe` LU path is not used.
3. `advection_tsle_scatter_bsr` forms the trace lifts, condensed Schur blocks
   and RHS, eliminates boundary columns, and writes direct face BSR. Unique
   off-diagonal blocks use plain stores; the two element contributions to each
   diagonal face block use the fused Schur-plus-mass atomic update.

The operator, solved response, and face-flux arrays live in a persistent
`RawAdvectionTsleWorkspace`. A reusable solver preserves their identities while
the device, mesh size, polynomial dimensions, and quadrature shape are fixed;
coefficient updates overwrite values without reallocating the workspace. BSR
is the only supported output, `raw_lu_mode="coop"` is required, p=1--7 is the
qualified scope, and p=8--9 is available as an explicitly experimental path.

For `raw_block_size="auto"`, TSLE screens candidates 32, 64, 128, and 256 for
each stage independently on at most 32,768 elements, then retimes the best two
per stage on the full mesh before caching the winning triple by device and
discrete shape. Subsequent assemblies in the process skip tuning.
Level-3 solver logging prints the three stage times, selected blocks, device
total, workspace GiB, and cold/reused tune state in two lines. Levels 2 and 4
retain the full candidate/micro-timing dump.

On the Quadro RTX 6000 CUDA-13 qualification mesh (157,280 triangles,
exact matrix/RHS/local-response parity), CUDA-event kernel medians were seven
alternating hot samples for p=1--7 and five for experimental p=8--9:

| p | Legacy fused | Legacy TSLE | Gain | Modal fused | Modal TSLE | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.664 ms | 5.636 ms | 0.5% | 5.857 ms | 5.817 ms | 0.7% |
| 2 | 13.347 ms | 13.343 ms | 0.0% | 13.625 ms | 13.612 ms | 0.1% |
| 3 | 27.214 ms | 27.230 ms | -0.1% | 28.004 ms | 28.023 ms | -0.1% |
| 4 | 54.815 ms | 54.899 ms | -0.2% | 55.525 ms | 55.591 ms | -0.1% |
| 5 | 121.766 ms | 122.156 ms | -0.3% | 122.992 ms | 123.639 ms | -0.5% |
| 6 | 241.261 ms | 243.599 ms | -1.0% | 243.584 ms | 247.125 ms | -1.4% |
| 7 | 534.697 ms | 474.323 ms | **12.7%** | 552.716 ms | 476.477 ms | **16.0%** |
| 8 | 1261.866 ms | 857.723 ms | **47.1%** | 1269.227 ms | 864.241 ms | **46.9%** |
| 9 | 1918.382 ms | 1757.016 ms | **9.2%** | 1984.066 ms | 1769.417 ms | **12.1%** |

The gate therefore keeps `fused` as the production `p <= 6` choice. TSLE
remains explicit at p=7 despite its measured gain, and p=8--9 remains
experimental. The high-order fused kernel uses 128 threads and moves its Schur
lift rows into block-shared storage: compiler profiles report zero local-memory
bytes for fused and all TSLE stages. Fused dynamic shared memory is 40,948 B at
p=8 and 56,604 B at p=9, below the target GPU's 65,536 B opt-in limit.

TSLE's extra persistent workspace is material: 1.732 GiB at p=6, 2.679 GiB at
p=7, 3.969 GiB at p=8, and 5.676 GiB at p=9 on this mesh. Full-mesh finalist
autotuning cost about 25.7 s at p=8 and 53.6 s at p=9 in the modal runs. The
corresponding measured cold-cost break-even is roughly 62 and 253 repeated
assemblies, respectively; use the high-order path only when that cost and
workspace are acceptable.

Reproduce the assembly-only sweep with:

```bash
scripts/gpu/run_cuda13.sh .venv/bin/python \
  scripts/gpu/benchmark_advection_tsle_bsr.py \
  --case disk-tangent --mesh-size 0.0068 --orders 1,2,3,4,5,6,7 \
  --trace-bases legacy-lagrange --warmups 2 --repeats 7
scripts/gpu/run_cuda13.sh .venv/bin/python \
  scripts/gpu/benchmark_advection_tsle_bsr.py \
  --case disk-tangent --mesh-size 0.0068 --orders 8,9 \
  --trace-bases legacy-lagrange --warmups 2 --repeats 5
```

Use `--trace-bases legendre-modal` for the second production orientation and
`--case test2` for the legacy nonzero-boundary manufactured problem. Focused
parity, autotune reuse, workspace identity, tangent-assembler, and explicit
p=9/256-thread TSLE scratch tests live in `tests/test_advection_tsle_bsr.py`.
The latter is also clean under CUDA-13 memcheck and racecheck.

## Discontinuous Advection Audit

CuPy assembly stores `beta . n` per element side. Raw-CUDA precomputed,
fused COO, fused CSR, and fused face-BSR paths preserve those side values through local
coupling, stabilization, and trace-mass construction. Interior sides are
emitted independently and summed into the global trace row; no face averaging
is introduced.

Fused reconstruction is element-local and uses the element's own projected
advection coefficient. It does not reconstruct an averaged interior-face
coefficient. The stateful raw-CUDA path retains the device workspace for the
local response ``A_e^-1 [B_e, f_e]``; each coefficient update overwrites that
workspace during assembly, and a small raw kernel applies it to the converged
trace. On the 157,280-triangle p=6 guiding-center case this reduced warm
reconstruction from about 0.212 s to 0.0021 s without changing the approximately
0.239 s operator assembly.

Validation anchors include:

- `tests/test_advection_reaction_numba.py` for side-weighted trace mass;
- `tests/test_cupy_backend.py` for CuPy/raw-CUDA matrix and RHS parity,
  modal traces, and COO/CSR/BSR equivalence;
- `tests/test_advection_reaction_conservation.py` for global conservation and
  no-through-flow boundaries.

## Diffusion Setup Ownership

The raw diffusion kernel owns local Schur construction, local solves, boundary
elimination, reduced emission, and RHS accumulation. It does not materialize
the pure-CuPy path's large `local_lhs`, `solved_el_bd`, `solved_src`,
`trace_blocks`, or `faces` intermediates.

Prepared outside the hot kernel:

| Data | Preparation |
|---|---|
| Source moments | CuPy quadrature/projection or a reusable prepared table |
| Boundary trace | Compact device interpolation/projection and edge mapping |
| Reference derivatives and face tensors | Small order-dependent device tables |
| COO reduction maps | Host-origin maps copied to device |
| CSR pattern and block positions | Raw-CUDA pattern kernels and device prefix operations |
| Mesh/reference mirrors | One-time host-to-device construction for the CuPy space |

For a fixed Schur-Cholesky Poisson operator, source updates take a compact
RHS-only path. Same-space DG source moments are formed exactly as coefficient
tables times the reference mass matrix, without a coefficient-to-quadrature
round trip. Setup retains the scalar Cholesky factor and scalar trace response
``S_e^-1 B_e`` but releases the redundant dense mixed coupling, element-boundary,
and trace-flux tensors. One warp per element then fuses the source triangular
solve, flux recovery, and reduced-face atomic scatter; a second compact kernel
reconstructs ``(u,q_x,q_y)`` from the persistent source solution and trace
response. Homogeneous Dirichlet trace data is reused without a host zero test.
The cached BSR values, face graph, local factors, and multigrid hierarchy are not
rebuilt or uploaded. Timing details expose `raw.assembly.rhs_only`,
`cached_rhs.source_moments`, `cached_rhs.fused_solve_flux_scatter`, and
`cupy.reconstruction.compact`. On the 157,280-triangle p=6 case, warm RHS and
reconstruction medians fell from about 0.121/0.134 s to 0.0293/0.0070 s.

The remaining optimization direction is to cache all order-dependent reference
tables, keep compact boundary storage, generalize table-driven coefficients,
and eliminate COO-only host map construction where measurements justify it.

## Tangent Advection And Diffusion Flux Postprocessing

`AdvectionReactionHDGSolver.assemble_tangent_boundary_raw_cuda_bsr()` is the
assembly-only entry point for fused zero-normal-flux face BSR. It excludes
boundary trace unknowns and retains both element-side contributions on interior
faces. Distinct off-diagonal face blocks have one element owner and are stored
directly. Diagonal Schur values are staged in dead coefficient-cache storage
and combined with the same element's tangent trace mass before one atomic
addition; the two elements adjacent to an interior edge remain the only
diagonal writers. Normal device solves resolve `raw_matrix_format="auto"` to
BSR; explicit CSR/COO selections remain available, and assembly diagnostics
use COO.

The p=6 Dubiner reference contraction remains dense. Two compressed variants
passed matrix, RHS, reconstruction, and conservation parity, but the more
regular paired-direction variant increased the 157,280-triangle kernel median
from 234.358 ms to 259.794 ms. Reusing dead local-LU shared storage for trace
lifts also regressed the measured p=6 kernel, so the cache-efficient per-thread
lift array remains active.

## Flux-Only Recovery And Scalar Tau Retries

Diffusion `hdg_postprocess="flux"` with raw-CUDA recovery uses
`hdgfem.backends.diffusion_flux_recovery_raw_cuda` for both `RT_projection`
and `l2_closest`. It applies cached reference lifts to resident local fields
and full traces. L2-closest recovery additionally uses per-element geometry
Cholesky factors. The current stabilization enters the dynamic flux jump;
the reference maps, uploaded tables, geometry, and factors are tau-independent.

`DiffusionReactionHDGSolver.with_options(stabilization=<finite real scalar>)`
preserves these recovery data when stabilization is the only override and
flux-only raw-CUDA recovery remains active. `postprocessing_backend="auto"`
is supported when it resolves to raw CUDA. The retained cache must match the
DGSpace identity, trace-space identity/basis, recovery variant, and active CUDA
device. A retry from a built-in scalar stabilization policy such as
`global_length` to its increased scalar value follows the same rule.

| State after a scalar tau retry | Behavior |
|---|---|
| Recovery reference maps, postprocessing space, uploaded tables, geometry and L2 metric factors | Retained with the same object/buffer identities |
| Diffusion operators, Schur-LU/Schur-Cholesky assembly factors, native and AMGX hierarchies | Invalidated; owned solver contexts are closed |
| Accepted solution, traces, RHS and exposed assembly artifacts | Cleared |
| Other host postprocessing factors | Discarded |

Other option updates, non-scalar stabilization updates, explicit `clear_cache()`
or `close()`, and space/mesh or reaction replacement clear the solver-owned
recovery cache. Source and boundary setters continue to preserve it when
operator caching with eliminated boundaries is enabled. Mesh geometry is
treated as fixed during reuse; in-place geometry mutation is outside this
contract. Device mismatch checks are tested with controlled device IDs;
numerical GPU parity is scoped to one active CUDA device.

The focused host command exercises metadata compatibility, repeated tau
updates, native/AMGX resource cleanup, the guiding-center retry controller,
and the existing small-matrix recovery reference:

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  tests/test_diffusion_recovery_cache.py tests/test_diffusion_flux_recovery_maps.py \
  -k 'not test_cuda_recovery'
```

On 2026-09-25 this command passed 48 checks, with the 6 opt-in CUDA cases
deselected. The new host cache tests are included in `host-fast`.
Another 16 focused option/cache and test-manifest checks passed; changed
documentation links, Python syntax parsing, and `git diff --check` also passed.

The GPU checks compare solver-owned recovery after two scalar tau
changes against fresh device recovery and the host reference, using changed
local fields and traces on skew triangles. Degrees 0, 2, and 5, both production
trace bases, and both recovery variants are covered. During reuse, the checks
reject host uploads/downloads and reference/geometry-factor rebuilding, verify
device-buffer identity, and require device-only recovered fields. These tests
are included in `gpu-smoke`.

```bash
HDGFEM_RUN_CUDA_RECOVERY_TESTS=1 NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m pytest -q tests/test_diffusion_recovery_cache_cuda.py \
  tests/test_diffusion_flux_recovery_maps.py::test_cuda_recovery_matches_host_and_reuses_geometry
```

On 2026-09-25 all 18 GPU checks passed in 5.38 seconds with no skips or
warnings. CuPy runtime compilation was explicitly authorized; CPU Numba JIT
remained disabled. The checks used prescribed fields without PDE solves or
time integration. Whole-driver retry replay, performance sweeps, and
multi-device execution remain outside this cache-lifetime contract.

## Failure And Validation Policy

Raw-CUDA requests are preflighted before assembly. Unsupported combinations
must raise a capability error or take an explicitly documented compatibility
path; they must not silently change coefficient semantics, trace basis, matrix
format, or convergence requirements.

The required parity and smoke lanes are documented in
[`../development/alpha_test_matrix.md`](../development/alpha_test_matrix.md).

## Tensor ADR assembly and reconstruction

Stationary ADR supports FP64 elliptic tensor diffusion at p=0--6 with both
`legacy-lagrange` and `legendre-modal` traces. Select `assembly_backend="raw-cuda"`,
`solver="amgx"`, and `hdg_postprocess="none"` for tensors. `raw_matrix_format`
accepts COO, direct CSR (default), and native face BSR; `raw_block_size="auto"`
selects 32/64/128 threads at p<=2/4/6. No BSR scalarization is needed.

For assembly without a sparse solve, use
`prepare_adr_data(..., dense_local_matrices=False)` followed by
`assemble_projected_adr_trace_operator_raw_cuda(..., matrix_format="bsr")` in
`hdgfem.backends.advection_diffusion_reaction_raw_cuda`. The returned operator
contains compressed graph metadata, exact per-element diffusion classifications,
and preparation/upload/graph/JIT/kernel timings. Its `assembly` contains the
device sparse arrays and RHS. `reconstruct_projected_adr_local_unknowns_raw_cuda`
accepts this operator and a full device trace, including boundary coefficients,
and returns device `[u,qx,qy]` coefficients plus timings.

The seven exact tensor paths share Numba's classifications. Constant tensors use
reference mass inverses; variable isotropic/diagonal tensors use scalar
Cholesky; coupled symmetric/general tensors use Cholesky/pivoted LU. Shared
storage never exceeds 48 KiB, and local failures are checked before returning.
Reconstruction retains coefficient uploads and rebuilds local factors with the
same algebra. Diffusion's extracted cooperative helpers preserve its generated
source and launch/workspace choices.

`tau_diff` may vary along a face, including through a same-mesh DG field or
`tau(x,y,*,element,local_face,normal,t=None)` law. Both incidences are sampled
independently. Original laws are retained; recovery never interpolates an
incompatible quadrature-only table. Tensor postprocessing remains unsupported
in raw CUDA. See the [implementation plan](../development/plans/raw_cuda_adr_tensor.md)
and [qualification report](../research/solver_studies/raw_cuda_adr_tensor_2026_09_28.md).
