# GPU Assembly of the Condensed HDG Systems

This page describes how hybridge assembles the statically condensed (trace)
systems on the GPU for the three equation families, and every implementation
variant: what each one computes, where its code lives, which switches select
it, and how the automatic policy chooses. Measurements, timings and tuned
defaults live in one folder per test architecture:

| Architecture | Folder | Content |
|---|---|---|
| NVIDIA H200 NVL, sm_90 (FP64-rich, FP64 tensor cores) | [`h200_sm90/`](h200_sm90/README.md) | the October 2026 study: every path, experiments, timings, defaults |
| NVIDIA RTX PRO 5000 Blackwell, sm_120 (FP64 1/64) | [`rtx_pro_5000_sm120/`](rtx_pro_5000_sm120/README.md) | September-October 2026 baselines and H200 comparison |
| NVIDIA Quadro RTX 6000, sm_75 | [`quadro_rtx6000_sm75/`](quadro_rtx6000_sm75/README.md) | July 2026 studies and the origin of the launch recommendations |

A new test architecture gets its own folder with the same structure: platform
table, per-family results (AR, DR, ADR, CuPy assemblers), defaults on that
architecture, open items, and links to the raw records.

The formulations themselves are derived in
[`../diffusion_reaction/assembly.tex`](../diffusion_reaction/assembly.tex) and
[`../advection_diffusion_reaction/assembly.tex`](../advection_diffusion_reaction/assembly.tex);
supported interfaces remain governed by
[`../../reference/backend_capabilities.md`](../../reference/backend_capabilities.md).

> Integration status (2026-10-31). The H200 branches described below are
> consolidated into local H200 development and reconciled with the research
> extraction. Bracketed branch labels identify the historical source of each
> result. The shared runtime tuner now uses a 30-second/eight-candidate budget
> for Poisson and tensor local solves; see the current
> [architecture policy](../../development/architecture_tuning.md) for coverage
> and qualification limits.

## 1. What every assembly computes

All three families use the same static condensation. On each triangle `K`
the local unknowns `x_K` (the scalar, and for mixed forms the flux) satisfy

```
A_K x_K = B_K lambda + f_K ,
```

where `lambda` collects the trace coefficients of the three faces
(`3*NTR` columns, `NTR = p+1` per face) and `f_K` is the source column. One
assembly therefore has to, per element:

1. **build** the local operator `A_K` (volume quadrature or reference-tensor
   contraction, face terms, stabilization) and the right-hand sides
   `[B_K | f_K]` (`NCOLS = 3*NTR + 1` columns);
2. **solve** `A_K R_K = [B_K | f_K]` (a batched dense factorization and
   `NCOLS` solves; `NEL = (p+1)(p+2)/2` unknowns per scalar block, 55 at p=9);
3. **condense**: form the face rows `S_K = L_K R_K` (lift rows `L_K` times the
   local response) and the trace-trace terms;
4. **scatter** the `3*NTR x 3*NTR` element contribution into the reduced
   system (COO, scalar CSR, or face-block BSR with `NTR x NTR` blocks), with
   Dirichlet (or zero-flux) traces eliminated into the right-hand side;
5. optionally keep the local response `R_K` (FP64, `E x NEL x NCOLS`) or the
   factors so that **reconstruction** of the local fields from the solved
   trace needs no refactorization.

The variants differ in how these five steps are split into kernels, where the
intermediate per-element arrays live (registers/shared memory vs. global
memory), and which hardware units do the dense algebra (FP64 vector pipes vs.
FP64 tensor cores, DMMA).

Per-element sizes that matter for every variant (p=9): `NEL=55`, `NTR=10`,
`NCOLS=31`; the default volume rule has `NQ=400` points at p=9 (324 at p=8,
Dunavant rules of 25-42 points up to p=7). The mixed ADR flux block doubles
the local size (`2*NEL = 110` for the inverse-diffusion mass).

## 2. Advection-reaction (AR, upwind HDG transport)

Entry point: `assemble_reduced_system_cuda` (`hybridge/transport/cuda.py`)
with `backend="raw-cuda"` and `raw_local_assembly in {"fused", "split3",
"tensor", "auto"}`; solver wrapper `AdvectionReactionHDGSolver`
(`hybridge/solvers/advection_reaction.py`). Output is BSR (face blocks) or CSR.
Upwind variants: plain upwind and the production *conflict-averaged upwind*
(`positive_transport_options`), which averages the normal flux on faces where
both sides see outflow and adds a gauge on faces with zero flux on both sides.
Zero-flux boundaries (`zero_boundary_flux`) remove boundary faces.

### 2.1 `fused` — one kernel per element, everything on the fly

`_RAW_FUSED_TEMPLATE` in `hybridge/transport/raw_cuda.py`. One thread block
owns one element: it builds `A_K` and `[B_K|f_K]` in shared memory, factors
with the cooperative pivoted LU (`raw_lu_mode="coop"`; `"safe"` is a serial
fallback), solves, condenses and scatters directly into the output, and
optionally writes the local response. No per-element workspace is kept in
global memory, so its footprint is only inputs + output (+ the optional
response copy). Block sizes 32/64/128 (recommendation 32 for p<=6, 64 at p=7,
128 at p=8-9). Volume operator: Dunavant quadrature at p<=7 (`NQ` 25-42), the
dense reference-tensor contraction at p=8-9 (the sparse Dubiner form is
disabled at density 0.49-0.50 > 0.40), chosen by `use_volume_quadrature`; a
pointwise (quadrature-point) advection field is supported.

### 2.2 `split3` — TSLE-BSR: build / solve / scatter kernels

`hybridge/transport/tsle_bsr.py`. Three kernels with a per-element workspace
`E x (NEL^2 + NEL*NCOLS + 6*NQF)` reals in global memory
(`split3_workspace_bytes`, 38 KB per element at p=9): **build** writes `A_K`
and the response columns, **solve** runs the cooperative LU in place, and
**scatter** forms and scatters the condensed face rows. Each kernel is
autotuned separately (`raw.tsle.{build,solve,scatter}.block_size`). Optional
**volume pre-pass** (`HYBRIDGE_RAW_SPLIT3_VOLUME`, `hybridge/transport/tsle_volume_gemm.py`):
at p=8-9 the dense volume operator becomes one element-batched GEMM
`X[E x K] * Y[K x NEL^2]` (`X` = per-element coefficient rows, `Y` = reference
tensor), either the hand-written DMMA kernel (`dmma`, inline PTX
`mma.sync.aligned.m8n8k4.row.col.f64`) or cuBLAS (`cublas`); the build kernel
then only adds the face terms (`RAW_VOLUME_PRELOADED`). Default `inline`.

### 2.3 `tensor` — tensorized TSLE with library contractions

`hybridge/transport/tsle_tensor.py`
(`assemble_projected_advection_trace_system_eliminated_tsle_tensor`). The
local algebra is rewritten as batched tensor contractions against fixed
reference tables, so the build and condense steps run on the FP64 tensor
cores (cuTENSOR `contraction="cutensor"`, or cuBLAS GEMMs `"cublas"`):

| Stage | Work | Engine |
|---|---|---|
| prepare | FP64 face weights `tau_w`, `gamma_w` (upwind, conflict averaging, gauge) and per-element coefficient rows | small custom kernels |
| build | `A_K = X_K * T_operator`, trace columns `B_K`, source column `f_K` | cuTENSOR / cuBLAS (FP64 DMMA) |
| solve | `A_K R_K = [B_K | f_K]` | `local_solver`: `coop`, `cublas`, `magma`, `fused` |
| condense | lift rows `L_K`, Schur rows `S_K = L_K R_K`, trace masses | cuTENSOR / cuBLAS |
| scatter | orientation signs, boundary elimination, gauge, BSR accumulation | custom kernel |

Local solvers:

* `coop` — the TSLE cooperative kernel (block per element), any precision.
* `cublas` — `getrfBatched`/`getrsBatched`.
* `magma` — MAGMA `dgetrf_batched` + `dgetrs_batched`
  (`hybridge/linalg/gpu/magma_batched.py`); `gesv_batched` is identical
  because MAGMA's small fused path only covers `nrhs=1` / `n<=32`.
* `fused` **[ar-batched-lu]** — `hybridge/linalg/gpu/fused_batched_solve.py`.
  One block per element holds the augmented `[A_K | B_K f_K]` in registers on a
  `TR x TC` thread grid (cyclic row/column split, no spills) and eliminates it
  with Gauss-Jordan and partial pivoting without moving rows (the pivot is the
  largest |a| among rows not yet used, found with a warp-shuffle reduction).
  `R_K` is written directly in the layout the condense contraction reads (no
  getrs, no transposes, `A_K` left intact), and zero/non-finite pivots are
  counted in an `info` buffer. Variants: v1 (two barriers per step), v2
  (packed-key full-warp pivot search), v3 (one barrier per step: row k+1 is
  published as the guessed pivot and republished on a miss). Tuned
  `DEFAULT_CONFIGS` (TR, TC, min blocks/SM, variant): n=28 `(8,4,8,3)`, n=36
  `(8,8,4,2)`, n=45 `(8,8,3,3)`, n=55 `(8,8,2,3)`. FP64 only.

Engines default to `default_tensor_engines()`: cuTENSOR when it loads, else
cuBLAS; local solver `auto` measures same-precision cuBLAS, MAGMA, cooperative and
fused candidates on bounded actual systems and persists the choice. Explicit
engine overrides retain the historical paths. Overrides: `HYBRIDGE_RAW_TENSOR_CONTRACTION`,
`HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER`.

**Element chunking [ar-batched-lu].** All stage buffers except the face
weights are sized for one *chunk* of elements and stages build -> scatter loop
over chunks (`tensor_chunk_elements`: `HYBRIDGE_RAW_TENSOR_CHUNK_ELEMENTS`,
else equal chunks within `HYBRIDGE_RAW_TENSOR_CHUNK_GIB`, 16 GiB by default,
i.e. one pass up to about 270k elements at p=9). The face weights stay
mesh-wide because the conflict-averaged upwind and the gauge read the
neighbouring element across each face, which may be in another chunk; the
coefficient-row and scatter kernels take an element offset. Without chunking
the stage buffers alone are about 64 KB per element at p=9 (128 GB at 2M
elements); with it they are bounded (17.6 GB at 2M). The optional FP64
response copy for reconstruction (13.6 KB per element at p=9) remains
mesh-wide.

Tensor inputs: DG-field advection in the solution space (or a compatible
space), DG or scalar reaction and source; quadrature/pointwise coefficients
and cross-space advection fields are rejected (`NotImplementedError`), and
`auto` then keeps split3/fused. BSR output and `raw_lu_mode="coop"` only.

### 2.4 Automatic choice (`raw_local_assembly="auto"`)

Not BSR, a non-coop LU mode, or a cached operator gives `fused`. Otherwise
`select_pipeline` (`hybridge/transport/pipeline_tuning.py`) times every
equivalent pipeline (fused, split3, and tensor when the advection field is in
the solution space) whose workspace fits a quarter of the free device memory,
on a sample of the actual elements, and stores the fastest for reuse. There is
no degree threshold: at `p = 5` split3 can win.

The degree rule below is the earlier static policy. It survives as
`resolve_raw_local_assembly` (`hybridge/solvers/capabilities.py`), which the
solver no longer calls:

1. Not BSR, a non-coop LU mode, or a cached operator -> `fused`.
2. `tensor` on compute capability 9.0 when the inputs allow it, `p >=
   RAW_TENSOR_MIN_ORDER` (6; **5 on [ar-batched-lu]**), and the tensor workspace
   (`tensor_workspace_bytes`, chunk-aware on [ar-batched-lu]) fits
   `RAW_TENSOR_WORKSPACE_FRACTION = 0.25` of free device memory, or
   **[ar-batched-lu]** at `p >= 8` is no larger than the split3 workspace that
   step 3 would take without a memory check.
3. `split3` at `p >= RAW_SPLIT3_MIN_ORDER = 8`, or at p=6-7 when its workspace
   fits 25% of free memory.
4. `fused` otherwise.

## 3. Diffusion-reaction (DR, mixed HDG Poisson)

### 3.1 `poisson_local` — production raw-CUDA kernel

`hybridge/mixed/raw_cuda/poisson.py` and `poisson_source.py` (dispatched
from `identity.py`). One kernel source specialized at compile time:

* `PHASE` 0 assembly, 1 right-hand-side refresh (new source, same operator),
  2 reconstruction;
* `FORMAT` COO / CSR / BSR (CSR/BSR accumulate with atomics);
* local factor policy `none`, `schur-lu` or `schur-cholesky` (the Schur
  complement of the mixed system is SPD); with a cached factor, RHS refresh
  and reconstruction skip the factorization (Cholesky reconstruction at p=9
  takes 1.35 ms vs. 14.1 ms with the LU cache and 30 ms without, 100k
  triangles);
* `BATCH` columns of the `3*NTR+1` right-hand sides per pass, `RETAIN`
  (keep the coupling block in shared memory between batches);
* launch threads 32/64/128/256.

Pivoted LU: `checked_warp_lu_source` (warp 0 finds the pivot, the block does
the rank-1 update, two barriers per step); Cholesky: an in-file loop with
three barriers per step. Triangular solves use one thread per right-hand-side
column.

**Launch tuning** (`hybridge/hdg/cuda/tuning.py`, `configurations()` in
`poisson.py`): candidates are screened for legality (shared memory, registers
x threads), then timed on up to 2,048 sampled elements with isolated outputs.
The default tuner checks a conservative baseline plus ranked legal candidates
within 30 seconds and eight candidates; the
thorough tuner (`--thorough` in `scripts/gpu/tune_poisson_raw_cuda.py`) permits
all 32-48, still subject to the configured time cap. Results are cached in `HYBRIDGE_CUDA_TUNING_CACHE_DIR`
(`~/.cache/hybridge/cuda-tuning`) keyed by device, workload (order, bases,
format, policy, volume points, `boundary_nonzero`, log2 mesh-size bucket) and
source version. On GPUs with full-rate FP64 (FP32:FP64 <= 4) the assembly
candidates are ranked by the measured H200 priors first
(`_fp64_rich_assembly_priors`): all columns in one batch, 32 threads up to
NEL=15, 64 at NEL=21, 128 up to NEL=45, 256 at NEL=55, retained coupling up to
NEL=28, the 2x-thread neighbour, and 256 threads x 16 retained columns (the
p>=7 Cholesky winner); such records are tagged `candidate_priors`. Other GPUs
keep the generic ranking (retained coupling only when FP32:FP64 >= 16).

### 3.2 Legacy kernels

`assemble_diffusion_raw_coop[_bsr]` / `reconstruct_diffusion_raw_coop`
(`_legacy=True`, LU only, one column per pass at p>=7) and the serial
thread-per-element `assemble_diffusion_raw[_csr]` (block size 1, p<=6). Kept
as references; 7-13x slower than `poisson_local` at p>=7.

### 3.3 CuPy reference

`assemble_projected_diffusion_trace_system_eliminated_cupy`
(`hybridge/mixed/cupy.py`): batched matmuls, `cupy.linalg.cholesky`
(cuSOLVER) and `dtrsmBatched`. 2.3-4x slower than `poisson_local` on the H200.

## 4. Advection-diffusion-reaction (ADR, mixed HDG)

### 4.1 `assemble_adr_tensor` — production raw-CUDA kernel

`hybridge/mixed/raw_cuda/tensor.py` (`kernel_source`,
`assemble_tensor_operator`; entry `assemble_projected_adr_trace_operator_raw_cuda`
in `adr_operator.py`). One block per element; one CUDA source specialized by
`NEL, NTR, NQ, NFQ` and switches. Per element:

1. **inverse-diffusion mass** `G_ab = int kappa^{-1}_ab phi_i phi_j` for
   variable tensors (diffusion *kinds* 3-6: variable isotropic, diagonal,
   symmetric, full): point coefficients `jac*w*kappa^{-1}`, one product per
   symmetric `(i,j)` pair, then
   * kinds 3-5: packed Cholesky (one NEL block, two NEL blocks, or one
     `2*NEL` block), diagonal stored as reciprocals;
   * kind 6 (full tensor): either a warp-pivoted `2*NEL` LU (`factor_mass`) or,
     **by default at p>=7**, the explicit inverse `G^{-1}` (in-place
     Gauss-Jordan with partial pivoting, three barriers per step; rows
     exchanged on `G` become columns exchanged on `G^{-1}`);
   * constant kinds 0-2 use the reference `M^{-1}` and a 2x2 tensor;
2. **local matrix** `A = reaction - advection + face terms` by quadrature,
   register-blocked (`QUAD_COLS` columns per thread);
3. **Schur complement** `S = A + (N - D) G^{-1} D`, built in batches of
   `RAW_BATCH_COLS` (<= 8) columns; each batch applies `G^{-1}` (triangular
   solves, or with the explicit inverse one block-wide product
   `apply_mass_inverse`); `S` is factored with a pivoted LU (`factor_schur`);
4. **output columns** (`3*NTR` trace columns + source): right-hand side,
   `G^{-1}` application, coupling, Schur solve (warp per column), gradient,
   second `G^{-1}` application, weighted face traces, BSR/CSR/COO scatter.

Layouts and switches:

| Switch | Values | Meaning |
|---|---|---|
| layout (`tensor_execution_workspace`) | p<=6: 48 KiB layout with `N-D` stored; p>=7: *compact* (`N-D` recomputed) with opt-in shared memory; *external* mass factor in global memory when it does not fit | `HYBRIDGE_RAW_ADR_TENSOR_MASS=auto/shared/external` forces a p>=7 layout |
| non-compact at p>=7 | `N-D` kept in shared memory | used by the wide launch at p=9 |
| `cache_local_factors` | `none`, `schur-lu`, `schur-lu+mass` (default) | reconstruction reuses the Schur LU and the mass factor/inverse; `mass_factors=` reuses the mass across assemblies |
| block size | `auto`: 32 (p<=2), 64 (p<=4), 128 (p<=9); **sm_90, variable kinds**: 256 at p=8, 512 at p=9 with the non-compact layout and `--maxrregcount=128` (`wide_variable_launch`); explicit 1/32/64/128 | block 1 is a serial diagnostic path |
| `HYBRIDGE_RAW_ADR_MASS_INVERSE` | `auto` (= on), `on`, `off` | explicit kind-6 inverse at p>=7 with the mass in shared memory (`with_mass_inverse`, +`2*NEL*batch` shared reals) |
| `HYBRIDGE_RAW_ADR_PHASE_CLOCK` | `1` | diagnostic build: thread 0 charges `clock64` cycles to 16 phases (`TENSOR_PHASES`), reported as `raw.phase.<name>` |

`reconstruct_adr_tensor` is the same source with `RAW_RECONSTRUCT` (one output
column from the solved trace); with automatic sizing it reuses the assembly's
launch.

### 4.2 `assemble_adr_raw` (legacy)

`adr_operator.py`: one thread per element, used only for block size 1, CSR,
scalar diffusion, dense preparation and p<=6. `reconstruct_adr_raw` is not
reached by the public reconstruction path.

### 4.3 Vendored CuPy ADR assembler

`vendor/adr_gmres/hdgfem` (`AdvDiffReaHDG`, used by the unified ADR campaign).
Two routes, both returning the same host arrays:

* **host** route: coefficients, local matrices, topology and gather maps are
  built with NumPy on the host (single-threaded BLAS under
  `OPENBLAS_NUM_THREADS=1`, Python loops for the face topology), copied to the
  GPU for the batched inverse and condensation, and copied back;
* **device** route **[cupy-adr-device]** (`hdgfem/backends/cupy_adr_assembly.py`,
  default there): coefficients sampled on the GPU (host fallback for
  callables that reject CuPy arrays), local matrices as batched GEMMs against
  mesh-independent reference tables, in-place cuBLAS batched `getrf/getri`,
  face topology and gather map from a GPU sort, RHS and Dirichlet elimination
  on the GPU, element chunks within `HDGFEM_CUPY_ADR_WORKSPACE_GIB` (2 GiB),
  and host pages prefaulted by worker threads while the GPU computes.
  `HDGFEM_CUPY_ADR_ROUTE=host` keeps the old route.

## 5. Where the time goes (summary of the H200 study)

* Dense per-element algebra with no reuse across elements (volume
  contractions, Schur products) belongs on the FP64 tensor cores as
  element-batched GEMMs/contractions; on the H200 that is what makes the
  tensor AR path 2-4x faster than the raw kernels from p=5.
* Batched small LU/solves are latency-bound (barriers, pivot searches,
  dependent shuffles), not FP64-bound: the fused register Gauss-Jordan solve is
  1.3-3.4x faster than MAGMA, and replacing repeated warp triangular solves by
  one explicit inverse plus block-wide products halves the ADR p=7-9 time.
* Kernels that keep one element per block are occupancy-starved at high
  order (one or two blocks per SM from shared memory); more threads per
  element, not more blocks, is the remedy there.
* Launch tuning priors from FP64-starved GPUs (Blackwell RTX) are wrong for
  FP64-rich ones (H200); quick tuning must rank the right candidates first.

The numbers behind each statement are in [`h200_sm90/`](h200_sm90/README.md);
the FP64-limited Blackwell behaves differently (see
[`rtx_pro_5000_sm120/`](rtx_pro_5000_sm120/README.md)).
