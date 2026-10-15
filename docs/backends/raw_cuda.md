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

Kernel source lives with its family: `hybridge/transport/raw_cuda.py` and
`hybridge/transport/tsle_bsr.py` (advection-reaction),
`hybridge/mixed/raw_cuda/poisson.py` and `poisson_source.py` (pure Poisson,
dispatched by `identity.py`, which retains the serial reference),
`hybridge/mixed/raw_cuda/tensor.py` (tensor ADR), and the recovery kernels in
`hybridge/mixed/postprocess/`. The shared trace-orientation and cooperative LU
source is `hybridge/hdg/cuda/raw_source.py`, the launch policy is
`hybridge/hdg/cuda/launch.py` (legacy/ADR) or `hdg/cuda/tuning.py` (Poisson),
and the reduced CSR/BSR pattern builder is
`hybridge/hdg/cuda/pattern.py`.

Reference tensors, projected/source moments, compact boundary data, topology,
and reduction maps are prepared before the element launch. Reusable solver
objects cache matrix- and graph-dependent data when their invalidation contract
allows it.

## Launch Policy

Public solvers and runners use `raw_block_size="auto"`. Poisson retains this
option until its backend independently tunes assembly, RHS refresh, and
reconstruction. TSLE-BSR also chooses its three launch sizes in the backend.
Other fused paths resolve a fixed launch policy before selecting kernels.

| Equation | Polynomial order | Recommended block size |
|---|---:|---:|
| Advection-reaction | `p <= 6` | 32 |
| Advection-reaction | `p = 7` | 64 |
| Advection-reaction | `p = 8, 9` (experimental) | 128 |
| Pure Poisson | `p = 0..9` | Device/resource screening and bounded timing |
| ADR | `p <= 2` | 32 |
| ADR | `p = 3, 4` | 64 |
| ADR | `p = 5..9` | 128 |

These are policy defaults covered by launch-resolution and parity tests.
The p=8--9 fused recommendation is additionally backed by the CUDA-13 SM75
block sweep below; it remains experimental rather than a cross-architecture
claim. Explicit `1`, `32`, `64`, and `128` overrides remain available for
benchmark reproduction. Poisson additionally accepts `256`. An explicit
Poisson block size disables tuning and chooses a deterministic compact layout.
Block size 1 retains the serial baseline through p=6. Poisson p=7..9 requires
a cooperative launch; orders above p=9 fail explicitly.

Tensor ADR retains single-thread assembly and reconstruction in COO, CSR, and
BSR: LU/Cholesky reductions use the active lanes, and column solves use a
serial fallback below one warp. This also supports cached Schur and mass
factors; full-warp launches keep the warp column solves.

## Tri-Stage Local Elimination BSR (`split3`)

`raw_local_assembly="split3"` selects **TSLE-BSR**, implemented in
`hybridge.transport.tsle_bsr`. It preserves the fused advection HDG
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

The cooperative pure-Poisson path supports p=0..9, both production trace bases,
and direct COO, scalar CSR, and face BSR. BSR remains the optimization priority;
CSR and COO use the same local algebra and emit their own entries without a
BSR conversion. Diagonal blocks receive each incident element's Schur and face
mass contributions. Boundary columns contribute only to the reduced RHS.

`CupyDGSpace` caches reference contractions `D_a M^-1 D_b`,
`F_f M^-1 D_a`, derivative tables, and oriented trace lifts. Geometry combines
them into one scalar Schur matrix per element. The compact layout holds that
factor and a bounded column batch in shared memory; a second layout retains
the two directional coupling matrices when resources and timings favor it.
No full-mesh mixed matrix or trace-flux tensor is constructed.

All three local-factor policies work at high order. `none` refactors on demand;
`schur-lu` retains the scalar LU and pivots; `schur-cholesky` retains the scalar
factor, scalar trace response, and source solution. LU checks pivots and
Cholesky checks symmetry, coupling adjointness, and positive pivots. Compact
Cholesky reconstruction needs no shared factor. Source-only solutions remain
separate from the boundary lift, so zero → nonzero → changed → zero Dirichlet
updates do not double count boundary data. Cached CSR/BSR format metadata,
matrix storage, and factors remain intact across these updates.

The arbitrary-order dense CuPy mixed and scalar-Schur/cuBLAS paths remain
independent references. `tests/test_poisson_high_order_cuda.py` compares their
matrix, RHS, scalar field, and both flux components directly with raw kernels,
and adds NumPy/Numba references, boundary/source updates, and public AMGX
CSR/BSR solves with scaling on/off. FP64 is the qualification precision;
the kernels retain the package's precision specialization. ADR, high-order
postprocessing, and the native p=4..6 multigrid gate are unchanged.

### Poisson tuning and resource limits

For each unseen workload, `auto` queries the visible GPU, screens shared
memory, thread and register limits, then tests at most four configurations on
at most 2,048 representative elements. Candidate ordering uses architecture
properties including FP32:FP64 throughput; CUDA-event timings choose the
winner. Two warmups and five timed samples follow a numerical check against
the conservative layout. Trial launches use isolated buffers, and cache hits
do not allocate trial scratch. Trials include local solves and isolated response
writes; the standalone full-mesh measurements also include sparse scatter.

The versioned JSON cache lives under `$XDG_CACHE_HOME/hybridge/cuda-tuning`
(default `~/.cache/hybridge/cuda-tuning`), with
`HYBRIDGE_CUDA_TUNING_CACHE_DIR` as an override. Keys include GPU identity and
resources, driver/runtime/CuPy versions, kernel-source hash, precision, element
and trace bases, order, quadrature size, phase, format, factor policy, boundary
state, and a mesh-size bucket. Writes are atomic; malformed or unavailable
disk caches fall back to measurement and process memory. `--refresh` replaces
matching records. `--thorough` sweeps every legal candidate with ten samples.

For FP64, the conservative factor-plus-one-column layout requires:

| p | Element DOFs | Trace block size | Shared bytes |
|---:|---:|---:|---:|
| 7 | 36 | 8 | 11,572 |
| 8 | 45 | 9 | 17,680 |
| 9 | 55 | 10 | 25,984 |

These layouts fit 48/64/99/160 KiB resource tiers. The larger retained-coupling
variants are pruned against the actual allocation limit, including compiler
static shared storage and register limits. Reports include compiler resource
attributes, modeled occupancy, candidate errors, rejected variants, first-call
and JIT/tuning wall time, and warmed phase samples. Modeled occupancy is not an achieved
hardware-counter measurement.

Run the assembly-only tuner without a global solve or AMGX build:

```bash
.venv/bin/python scripts/gpu/tune_poisson_raw_cuda.py \
  --orders 7,8,9 --mesh-sizes 16,32 \
  --trace-bases legacy-lagrange,legendre-modal --formats coo,csr,bsr \
  --cache-policies none,schur-lu,schur-cholesky \
  --output artifacts/poisson_raw_cuda_tuning.json
```

`--include-legacy` measures the preceding cooperative assembly wherever its
shared layout fits, and checks sparse matvec parity. Cholesky construction is
fused with assembly; reports label the combined cost. JSON and a flushed
progress log are written beside each other. For Nsight counters, the existing
`capture_raw_assembly_profiles.py` harness recognizes `poisson_local` and p≤9;
its `cache` phase now measures assembly plus Cholesky construction.

### Poisson qualification (2026-10-14)

FP64 checks cover orders 0..9, both production trace bases, direct COO/CSR/BSR,
and all three cache policies. Matrix/RHS comparisons use the arbitrary-order
CuPy mixed solve or scalar-Schur/cuBLAS implementation; reconstruction checks
all three mixed components. The orthonormal-element suite also compares with
NumPy/Numba. Degree-nine Bernstein and hierarchical element bases have
additional normwise CuPy checks (matrix/RHS ≤5e-11, reconstruction ≤2e-10).
Changing source and Dirichlet data, cache identity, persistent tuning, and
CSR/BSR AMGX solves with scaling enabled/disabled are covered separately.

CUDA-13 memcheck and racecheck report zero errors/hazards for the final p=9
BSR assembly/reconstruction policies and isolated automatic-tuning trials.
The broader direct COO/CSR/BSR and changing-boundary checks also passed the
sanitizers before the final reconstruction workspace reduction. A thorough
p=9 sweep validated 44 assembly configurations and four each for cached RHS
and reconstruction, with no numerical rejections on this device.

The final review preserved the public format defaults (CSR for AMGX, BSR
for native multigrid) and verified source/boundary reuse and operator
invalidation. Its focused checks passed 301 tests, including reservation and
source-provenance tooling; the maintained comparison runner also passed a
p=0,2,9 smoke check across its five raw variants. Logs are retained in
`artifacts/poisson_finalization_checks.log` and
`artifacts/poisson_comparison_smoke.log`.

The following uncontended measurements used an NVIDIA RTX PRO 5000 Blackwell
(SM 12.0, 110 SMs, 99 KiB opt-in shared memory per block), CuPy 14.2.0 and
CUDA runtime 13.2. Device work was serialized with the shared workspace GPU
lock; clocks were not locked. These are seven-sample CUDA-event medians after
two warmups, for 2,048 affine triangles and modal face BSR:

| p | New assembly, no factor cache | Previous layout | Cached Cholesky RHS | Cached Cholesky reconstruction |
|---:|---:|---:|---:|---:|
| 0 | 0.046 ms | 0.058 ms | 0.030 ms | 0.025 ms |
| 2 | 0.164 ms | 0.162 ms | 0.054 ms | 0.034 ms |
| 4 | 0.554 ms | 0.928 ms | 0.128 ms | 0.045 ms |
| 6 | 2.226 ms | 3.232 ms | 0.260 ms | 0.060 ms |
| 7 | 5.387 ms | 21.950 ms | 0.459 ms | 0.115 ms |
| 8 | 10.026 ms | 67.281 ms | 0.564 ms | 0.135 ms |
| 9 | 16.431 ms | 103.541 ms | 1.154 ms | 0.163 ms |

The previous layout was invoked through the benchmark override beyond its
former p≤6 public guard only where its shared allocation fit; sparse matvec
parity was checked for every comparison. The low-order near tie at p=2 is not
a claimed speedup. Selected p=7/8/9 assembly layouts used 39,700 / 26,752 /
36,904 shared bytes, 56 / 68 / 68 registers per thread, zero compiler-reported
local-memory bytes, and modeled occupancy 33.3% / 25.0% / 16.7%. These are
single-device assembly measurements, not global-solver or simulation speedups.

Workspace artifacts: `artifacts/poisson_tuning_uncontended.json` (512/2,048
elements and all factor policies), `artifacts/poisson_tuning_formats.json`
(p=7..9, both trace bases, direct COO/CSR/BSR),
`artifacts/poisson_tuning_thorough.json`, and
`artifacts/poisson_final_device_checks.log`. The JSON preserves every timed
sample, selected/rejected configuration, resources, and first-use costs.
Earlier overlapping measurements are excluded from this table.

The [100,000-triangle Poisson comparison](../research/solver_studies/poisson_raw_cupy_2026_10_14.md)
records a separate same-day baseline against arbitrary-order CuPy
Schur-Cholesky: direct COO/CSR/BSR assembly, cached RHS and reconstruction,
and all raw cache policies. Its matched p=2,4,6 follow-up gives 2.91x, 3.95x,
and 2.25x warm CSR assembly gains, versus 2.41x, 1.75x, and 1.40x at p=7,8,9.
These use complete wrapper wall times; the previous-raw-layout table above
measures kernels. Lower-order gains are not strictly monotonic, and raw
cached RHS refresh is slower than CuPy at p=2,4.

The maintained comparison runner reserves the shared GPU campaign lock,
waits for other compute jobs, and records exact sources and timing samples:

```bash
.venv/bin/python scripts/gpu/benchmark_poisson_assembly.py \
  --orders 2,4,6 --nx 250 --ny 200 --warmups 1 --repeats 3 \
  --output artifacts/poisson_100k_low_orders.json
```

It writes JSON, a flushed log and a source archive. `--dry-run` performs no
device work; `--gpu-lock` and `--wait-timeout` control reservation. This runner
acquires the lock internally and must not be wrapped in the same `flock`.

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
`hybridge.mixed.postprocess.flux_recovery_raw_cuda` for both `RT_projection`
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
HYBRIDGE_RUN_CUDA_RECOVERY_TESTS=1 NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 \
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

Raw-CUDA ADR assembly and reconstruction accept elliptic tensor diffusion at
p=0--9 with both `legacy-lagrange` and `legendre-modal` traces. Select
`assembly_backend="raw-cuda"`, `solver="amgx"`, and `hdg_postprocess="none"`
for the assembly/reconstruction path. `raw_matrix_format` accepts COO, direct
CSR (default), and native face BSR; `raw_block_size="auto"` selects 32/64/128
threads at p<=2/4/9. The p=7--9 FP64 extension passed qualification on
2026-10-15: 508 ADR checks and clean memcheck/racecheck. See the
[dated evidence](../research/solver_studies/raw_cuda_adr_high_order_2026_10_15.md)
and [higher-order implementation](../development/plans/raw_cuda_adr_tensor.md#higher-order-extension-2026-10-14).

Native p=8 BSR solves require an explicit compatible `amgx_config`: the current
AMGX fork's default `MULTICOLOR_DILU` preconditioner has no specialization for
9-component blocks. The qualification uses FGMRES with `JACOBI_L1` and
`jacobi_l1_scalar_rows_for_blocks=1`, retaining BSR storage. This restriction
does not affect CSR/COO assembly, BSR assembly, or local reconstruction.

For assembly without a sparse solve, use
`prepare_adr_data(..., dense_local_matrices=False)` followed by
`assemble_projected_adr_trace_operator_raw_cuda(..., matrix_format="bsr")` in
`hybridge.mixed.raw_cuda.adr_operator`. The returned operator
contains compressed graph metadata, exact per-element diffusion classifications,
and preparation/upload/graph/JIT/kernel timings. Its `assembly` contains the
device sparse arrays and RHS. `reconstruct_projected_adr_local_unknowns_raw_cuda`
accepts this operator and a full device trace, including boundary coefficients,
and returns device `[u,qx,qy]` coefficients plus timings.

The seven exact tensor paths share Numba's classifications. Constant tensors use
reference mass inverses; variable isotropic/diagonal tensors use scalar
Cholesky; coupled symmetric/general tensors use Cholesky/pivoted LU. Through
p=6, shared storage retains its original 48 KiB limit. At p=7--9, coupling
entries are recomputed to save shared memory, and the batch width (at most
eight columns) fits the device's opt-in shared-memory limit. If a variable mass
factor still does not fit, it uses the requested persistent mass cache directly
or processes element chunks with at most 64 MiB of temporary device scratch.
The operator does not retain this temporary scratch. Assembly timings report
the selected shared bytes/limit, external mass, scratch bytes, and launch count.

Oversized quadrature still raises `TensorWorkspaceError` (an
`UnsupportedBackendConfigurationError` and `ValueError`) before upload,
compilation or launch, naming the order, diffusion kind, quadrature sizes and
largest fitting volume rule. Local factorization failures are checked before
returning. The p<=6 FP64 layout fits the default and listed overintegrated rules
except for general variable tensors at p=6 (at most 143 volume points); see the
[shared-memory requirements](../research/solver_studies/raw_cuda_adr_tensor_shared_memory_2026_09_29.md)
and `scripts/advection_diffusion_reaction/diagnostics/tensor_shared_memory_budget.py`.

All three factor-cache policies apply through p=9. `none` rebuilds local factors
during reconstruction, `schur-lu` retains the scalar Schur LU and pivots, and
`schur-lu+mass` (the default) also retains variable diffusion mass factors.
Static mass factors can be reused across assemblies while source, reaction,
advection, and boundary data change. Reconstruction consumes the full trace and
keeps its `[u,qx,qy]` coefficients on the device.

`tau_diff` may vary along a face, including through a same-mesh DG field or
`tau(x,y,*,element,local_face,normal,t=None)` law. Both incidences are sampled
independently. Original laws are retained; recovery never interpolates an
incompatible quadrature-only table. Tensor primal and both total-flux recoveries
are qualified separately through p=6 with CuPy; primal assembly uses a fused
CUDA kernel. See the
[device postprocessing guide](adr_device_postprocessing.md) for qualification. See the [implementation plan](../development/plans/raw_cuda_adr_tensor.md)
and [qualification report](../research/solver_studies/raw_cuda_adr_tensor_2026_09_28.md).
