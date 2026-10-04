# CUDA Execution Paths

This document explains the CUDA-oriented assembly and sparse-solve pipeline.
The supported equation, coefficient, basis, order, and reconstruction
combinations are defined by
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).
This page describes architecture and runner usage; it is not a second support
matrix.

## Environment

CUDA and AMGX are optional dependencies. On the qualified CUDA-13 workspace,
use the checked launcher so compiler, AMGX, runtime, and math-library paths are
set together:

```bash
scripts/gpu/run_cuda13.sh .venv/bin/python \
  scripts/gpu/run_advection_reaction_cuda.py --help
```

The canonical `/tmp` aliases and advanced root overrides are documented in
[`../getting_started/forked_amgx_stack.md`](../getting_started/forked_amgx_stack.md).

Package imports remain lazy: importing `hdgfem` does not require CuPy, CUDA,
or PyAMGX.

## Standalone Runners

- `scripts/gpu/run_advection_reaction_cuda.py` exercises CuPy and raw-CUDA
  advection-reaction assembly, Cupyx/PyAMGX solves, and reconstruction.
- `scripts/gpu/run_diffusion_reaction_cuda.py` exercises the corresponding
  diffusion-reaction paths.
- `scripts/gpu/sweep_cuda_hdg.py` runs controlled basis, trace, quadrature,
  and matrix-format comparisons.
- `scripts/gpu/benchmark_advection_tsle_bsr.py` compares fused and three-stage
  face-BSR assembly without including AMGX solve time.
- `scripts/gpu/benchmark_advection_tsle_tensor.py` compares split3 with the
  experimental tensorized TSLE configurations on the guiding-center showcase
  transport step (assembly only).

The standalone runners are diagnostic and benchmarking entry points. Normal
application workflows should use the public solver classes or maintained case
drivers.

## Assembly And Solve Pipeline

| Path | Assembly storage | Solve handoff | Role |
|---|---|---|---|
| CuPy | Vectorized device arrays followed by CuPy COO/CSR construction | Direct device Cupyx for compatible preconditioners; explicit host staging for host solvers/preconditioners | Device reference and parity path |
| Raw-CUDA COO | Reduced COO triplets and RHS emitted by raw kernels | CuPy COO-to-CSR, then Cupyx or PyAMGX | Inspectable correctness path |
| Raw-CUDA CSR | Values emitted directly into a known reduced CSR pattern | Direct device CSR view | Preferred large-run path |
| Raw-CUDA face BSR | Dense trace-face interaction blocks emitted directly into a block pattern | Native FB-HP-MG-PCG or direct PyAMGX BSR upload | Default eligible device format; CSR/COO remain explicit comparison/diagnostic paths |
| Direct CSR-to-AMGX | Existing device `indptr`, `indices`, and values | PyAMGX device upload/view | Avoids host staging and COO conversion |

Raw-CUDA CSR/BSR pattern construction and block-position maps remain on device.
The direct paths avoid both global COO-to-CSR reconstruction and large
per-element condensed tensor materialization.

The hybrid direct-BSR path and its classical-AMG hierarchy lifecycle are
documented in [`amgx_classical_bsr.md`](amgx_classical_bsr.md). Its retained
fine operator is BSR, while transfer and coarse operators remain scalar CSR.
For supported p=4--6 Legendre-modal Poisson systems, the default repeated-solve
path is instead HDGFEM's native [`FB-HP-MG-PCG`](face_hp_mg_pcg.md).

## Advection-Reaction Modes

The fused cooperative path builds local matrices from projected coefficients,
performs local factorization and solves, eliminates known boundary columns, and
emits the reduced operator. The default `raw_lu_mode=None` selects the
cooperative LU (`coop`) for fused and split3 assembly. The `precomputed`
local-assembly path and the explicit `safe` LU mode remain compatibility and
debugging references until they are retired (see `TODO.md`).

The explicit `raw_local_assembly="split3"` selector chooses TSLE-BSR, which
separates build, cooperative LU/solve, and Schur/BSR scatter so each stage can
autotune its launch shape. It preserves the same HDG algebra, requires BSR and
cooperative pivoted LU, and retains persistent intermediate workspaces.
`raw_local_assembly="auto"` (implied for zero-flux raw-CUDA transport and used
by the guiding-center GPU presets) selects TSLE-BSR from p=8 on and fused
below, falling back to fused when face-BSR output, cooperative LU, or an
operator-cache-free solve is unavailable. On an RTX PRO 5000 Blackwell
(117k-triangle star mesh, zero-flux transport, bit-identical matrices) TSLE was
24% faster than fused at p=8, 44% at p=9, and within 4% for p=4--7. Its
workspaces cost about `E*(NEL**2 + 6*NQF)*8` bytes more than fused, roughly 17
and 25 GB per million triangles at p=8 and p=9. See [`raw_cuda.md`](raw_cuda.md)
for earlier timings and the full support/reuse contract.

### Experimental tensorized TSLE

`hdgfem/transport/tsle_tensor.py` keeps the split3 algebra and stages but
replaces the build and condensation kernels with library contractions
(cuTENSOR or cuBLAS GEMMs against static reference tables) and offers three
batched local LU solvers: the TSLE cooperative kernel, cuBLAS
`getrf/getrsBatched`, and MAGMA batched LU. The working precision is FP32 or
FP64. Face tables and coefficient rows are always formed in FP64, and the
face-BSR values accumulate in FP64. The module is not selectable through
`raw_local_assembly` yet.

Optional dependencies: cuTENSOR comes from the `cutensor-cu13` wheel, loaded
through `hdgfem.runtime.optional.require_cutensor` because CuPy 14 does not
preload `libcutensorMg`. MAGMA (2.10 or newer for CUDA 13 and sm_120) is
located through `HDGFEM_MAGMA_LIBRARY` or `HDGFEM_MAGMA_ROOT`
(`hdgfem/linalg/gpu/magma_batched.py`).

### Advection Assembly Configuration Guide (baseline: RTX PRO 5000 Blackwell)

These tables describe one machine. They are a baseline for comparing other
GPUs (H100 trials first), not portable defaults. The machine, driver, CUDA,
and library versions, the raw rows, the predictions to test on FP64-capable
GPUs, and the H100 reproduction steps are recorded in
[`../research/solver_studies/advection_assembly_baseline_2026_10_03.md`](../research/solver_studies/advection_assembly_baseline_2026_10_03.md).

The three implementation classes differ in how much dense work runs in
HDGFEM's own kernels:

- **Native:** HDGFEM raw kernels only. `fused` uses one kernel per element;
  `split3` (TSLE-BSR) uses build, cooperative-LU, and scatter kernels.
- **Hybrid:** library contractions with HDGFEM's cooperative LU (tensorized
  TSLE with `local_solver="coop"`), or split3 with a library LU in stage 2.
- **Pure library:** library contractions and library LU (tensorized TSLE with
  `local_solver="magma"` or `"cublas"`). HDGFEM code is limited to the FP64
  face-weight and coefficient-row kernels and the BSR scatter, which encode
  mesh topology, upwinding, and boundary elimination.

Baseline machine: NVIDIA RTX PRO 5000 Blackwell (sm_120, 110 SMs, 48 GB,
FP64 = 1/64 of FP32), driver 580.126.09, CUDA 13.0, CuPy 14.2.0, cuBLAS
13.0.2, cuTENSOR 2.8.1, MAGMA 2.10.0. Problem: the 117k-triangle
guiding-center star mesh, one BDF2 transport step, zero-flux boundaries,
conflict-averaged upwind, and face-BSR output, measured on 2026-10-03. Values
are median CUDA-event assembly times in ms, excluding pattern construction
and AMGX. Reproduce them with `scripts/gpu/benchmark_advection_tsle_tensor.py`
(`--native fused,split3` for the native columns).

**FP64 on the baseline machine (its production precision):**

| p | fused (native) | split3 (native) | split3 + MAGMA LU (hybrid, est.) | tensor + coop LU (hybrid) | tensor + MAGMA LU (pure library) | Fastest on the baseline machine |
|---|---:|---:|---:|---:|---:|---|
| 4 | 17.0 | 17.0 | 16.7 | 20.3 | 19.4 | native (`auto` → fused) |
| 5 | 32.7 | 32.3 | 37.7 | 32.3 | 37.3 | native |
| 6 | 63.0 | 60.8 | 63.8 | 65.2 | 66.8 | native |
| 7 | 118.7 | 114.6 | 130.8 | 115.5 | 129.5 | native |
| 8 | 286.8 | 217.4 | 205.5 | 212.1 | **198.6** | pure library (-9% vs split3) |
| 9 | 699.4 | 388.8 | 321.7 | 366.9 | **298.5** | pure library (-23% vs split3) |

The tensor columns use cuTENSOR contractions; cuBLAS GEMMs were 13--43%
slower in FP64. "est." sums measured split3 build and scatter times with the
standalone MAGMA solve time; that path is not implemented yet, because MAGMA
needs a column-major response. At p=8 and p=9 the tensorized workspace is
about 1.7 times split3's: 5.2 and 7.1 GiB against 3.0 and 4.2 GiB at 117k
triangles.

**FP32 on the baseline machine, a throughput proxy for FP64 on FP64-capable GPUs:**

| p | fused (native) | split3 (native) | tensor + coop LU (hybrid) | tensor + library LU (pure library) | Projected for FP64-capable GPUs (verify on H100) |
|---|---:|---:|---:|---:|---|
| 4 | **2.79** | 3.73 | 5.14 (cuTENSOR) | 3.93 (cuTENSOR + MAGMA) | native fused |
| 5 | **5.83** | 8.07 | 6.43 (cuBLAS) | 8.83 (cuTENSOR + cuBLAS LU) | native fused |
| 6 | 11.5 | 15.8 | **11.1** (cuBLAS) | 12.1 (cuBLAS + MAGMA) | fused, hybrid, or library (within 9%) |
| 7 | 19.3 | 29.5 | **17.3** (cuBLAS) | 24.5 (cuTENSOR + cuBLAS LU) | hybrid (fused +11%) |
| 8 | 45.9 | 62.2 | **30.9** (cuBLAS) | 33.8 (cuTENSOR + MAGMA) | hybrid; pure library within 9% |
| 9 | 84.5 | 107.6 | 49.7 (cuBLAS) | **46.9** (cuTENSOR + cuBLAS LU) | pure library |

The fused and split3 columns run in the `HDGFEM_PRECISION=float32` package
mode, and the contraction engine is named in parentheses. The native ranking
reverses between precisions. In FP32, fused beats split3 at every order and
runs 5.5--8.3x faster than in FP64; in FP64 on this card, split3 is 24% and 44%
faster than fused at p=8 and p=9. The `auto` rule (split3 from p=8) is
therefore specific to FP64-limited GPUs; the proxy never ranks split3 first.
Once the build is tensorized, the local LU dominates (55--65% of the time). The library LUs are weakest at
`NEL` = 36 (p=7): about 16--18 ms against 9.5 ms for the cooperative kernel.
The proxy should understate the library contractions on GPUs whose cuBLAS and
cuTENSOR use FP64 tensor cores, but not MAGMA's batched LU, which runs on the
ordinary FP64 cores. FP64 also moves twice the bytes of FP32. FP32 itself is
not a production option here: against FP64 split3, the end-to-end BSR matrix
error grows from 3e-7 at p=4 to 2.5e-3--1.1e-2 at p=9.

**Accuracy.** In FP64 the tensorized path matches split3 to at most 4e-13
for p=4--8 and about 1e-11 at p=9, the level at which the FP64 LU solvers
already disagree with each other.

**Status and scope.** Only `fused` and `split3` are selectable today
(`raw_local_assembly="auto"`). The tensorized variant and the split3 MAGMA
stage are open items in `TODO.md`. The tensorized variant emits face BSR
only, like split3, while fused also emits COO and CSR. All numbers come from
one mesh, one problem, and one GPU; requalify them on an FP64-capable GPU
against the baseline document above.

Discontinuous advection must retain both element-side face contributions. The
mathematical contract is documented in
[`../algorithms/advection_reaction/`](../algorithms/advection_reaction/), and
the implementation audit is summarized in
[`raw_cuda.md`](raw_cuda.md).

## Diffusion-Reaction Modes

The fused diffusion path constructs the local mixed operator, condenses it,
applies trace-boundary elimination, and emits reduced COO or CSR values. Source
moments, compact boundary data, reference derivative matrices, and face
reference tensors are prepared outside the hot element kernel and are cached
where the reusable solver permits.

Use the capability table for current support. Historical restrictions and
modal-trace AMGX investigations are retained as research evidence rather than
repeated here.

## Direct-CSR Timing Anchor

A paired July 2026 diffusion run used `p=6`, `mesh_size=0.08`,
`legacy-lagrange`, symmetric volume quadrature, and block size 128. The
matrix had 298,599 reduced trace DOFs and 10,412,451 nonzeros.

| Format | CSR/view | AMGX setup/upload | AMGX solve | AMGX subtotal |
|---|---:|---:|---:|---:|
| Direct CSR | 0.00019 s | 0.53684 s | 0.17208 s | 0.709 s |
| COO then CSR | 0.28348 s | 0.50783 s | 0.17192 s | 0.963 s |

The direct path removed the measured COO conversion cost without changing
iterations or field errors. Treat this as dated evidence, not a portable
performance guarantee.

## Validation Anchors

- `tests/test_diffusion_reaction_assembly_parity.py` compares NumPy, Numba,
  CuPy, raw-CUDA COO, and raw-CUDA CSR assembly over the qualified order range.
- `tests/test_cupy_backend.py` covers raw-CUDA/CuPy advection parity,
  discontinuous coefficients, matrix formats, LU modes, and reconstruction.
- `tests/test_raw_cuda_policy.py` covers launch-policy resolution and failure
  behavior.

Raw logs and historical runner comparisons remain under `run_logs/`.
Curated solver conclusions are indexed from
[`../research/solver_studies/`](../research/solver_studies/).
