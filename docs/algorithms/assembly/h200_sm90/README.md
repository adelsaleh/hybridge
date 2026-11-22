# Assembly on the NVIDIA H200 NVL (sm_90), October 2026

Experiments, timings and conclusions for every GPU assembly path on the H200,
from the Q1 campaign (2026-10-27) and the tuning work of 2026-10-27/28. The
algorithms and switches referred to here are described in
[`../README.md`](../README.md).

## 1. Platform and method

| Item | Value |
|---|---|
| GPU | NVIDIA H200 NVL (compute capability 9.0), 132 SMs, 143.8 GB HBM3e (~4.1 TB/s measured copy), 60 MB L2 |
| Peaks used | FP64 30 TFLOP/s on the vector pipes, 60 TFLOP/s on the tensor cores (DMMA); FP32:FP64 = 2 |
| Per SM | 64K registers, 228 KB shared memory (227 KB opt-in per block), 64 warps |
| Software | CUDA driver 13.1, runtime 13.2 (NVRTC), nvcc 13.1 for offline `ptxas -v`, CuPy 14.2, cuTENSOR 2.8, MAGMA 2.10 (sm_90 build) |
| Host | one H200 node, Slurm batch job |
| Profiling | Nsight Systems works unprivileged (`scripts/gpu/nsys_kernel_summary.py`); performance counters (Nsight Compute) are admin-only, so per-phase data comes from the `clock64` phase build |

Timings are CUDA-event medians of 3-5 repeats after 1-2 warmups, device time
of the assembly kernels unless stated. Meshes:

| Name | Triangles | Use |
|---|---|---|
| `star:0.0095` | 117,439 | AR reference mesh (Q1 / Blackwell baseline) |
| `star:0.005` | 360,379 | AR large |
| `star:0.00199` | 1,995,756 | AR 2M |
| `rect` 224x224 | 100,352 | DR tuning |
| `rect` 128x128 / 256x256 | 32,768 / 131,072 | ADR |

Records: `/scratch/user/hybridge/<study>/<step>/` with copies in
`run_outputs/campaigns/<study>/<step>/` (studies `asm_h200_tuning`,
`ar_batched_lu`, `dr_review`, `cupy_adr_device`; Q1 in
`run_outputs/campaigns/h200_expanded_20261027/`).

## 2. Advection-reaction

### 2.1 Starting point (Q1, raw kernels vs. tensor path)

* The production **fused** kernel with conflict-averaged upwind compiles to
  254-255 registers at p=7-9 on sm_80-90 (p=9 spills 32/88 B; 80-88 registers
  for the same source on sm_100/120). At 128 threads that leaves 8 warps per
  SM. Plain upwind uses 82-92 registers; `ptxas -O1` gives 86 and
  `pointwise_beta=True` 96.
* **split3**: no spills, build/solve 34-42 registers (40 warps/SM at p=9),
  scatter 126 registers; the build is ~64% of its time at p=9.
* **tensor** (cuTENSOR + MAGMA) was already ~2x faster than the raw kernels at
  p=6-9 (p=9: 35.5 ms vs 93-95 ms at 117k), because its build runs on the FP64
  tensor cores (5.7 ms vs 56 ms split3 build).

### 2.2 Tensor path as the sm_90 default (asm-32..34, MAGMA solve)

| p | 117k: tensor | split3 | 360k: tensor | split3 |
|---|---|---|---|---|
| 6 | 7.6 ms | 14.6 ms | 23.1 ms | 44.7 ms |
| 7 | 17.9 | 26.7 | | |
| 8 | 26.8 | 47.8 | | |
| 9 | 38.1 | 93.2 | 116 | 287 |

`auto` was made to pick tensor on sm_90 for p>=6 when its workspace fits 25%
of free memory (workspace 23 GB at 360k, p=9). Parity with split3: matrix
within 7e-13, local response 1.1e-10 at 360k p=9 (explained in 2.5).

### 2.3 Nsight Systems breakdown (asm-35)

FP64 tensor-core kernels confirmed (`sm90_xmma_..._f64_tensor16x8x16`,
`cutlass_sm90 d1684`, `cutlass_80 d884`). The MAGMA batched LU
(`dgetf2_fused`, trsm, laswp, pivinfo) was ~26 ms of each p=9 assembly at
117k, **58-63% of the tensor time**, at ~1.3 TFLOP/s against a ~2.5 ms
memory floor.

### 2.4 split3 volume pre-pass (asm-36..38)

At p=8-9 the split3 build contracts the dense reference tensor per element
with no reuse across elements (L2-bound). Lifting it to an element-batched
GEMM, p=9 build at 117k:

| Volume operator | Build | split3 total 117k | total 360k |
|---|---|---|---|
| inline (default) | 54.3 ms | 94.0 ms | 290.9 ms |
| hand DMMA kernel | 34.1 | 73.8 | 225.4 |
| cuBLAS | 26.3 | 65.9 | 201.1 |

nsys: the hand DMMA GEMM takes 11.3 ms (~7 TFLOP/s) against ~3.5 ms for
cuBLAS; the remaining build kernel (24.4 ms) is latency-bound. The pre-pass
stays opt-in (`HYBRIDGE_RAW_SPLIT3_VOLUME`) since the tensor path is faster
still.

### 2.5 Batched local solve: fused Gauss-Jordan vs. libraries (lu-02/04/10)

Real `A_K`, `[B_K|f_K]` captured from the tensor assembly; ms per batched solve
(useful GFLOP/s = `E(2n^3/3 + 2n^2 nrhs)`):

| p (n, nrhs) | 117k MAGMA | cuBLAS | coop (best) | **fused** (tuned) | vs MAGMA |
|---|---|---|---|---|---|
| 6 (28, 22) | 2.78 | 7.43 | 5.46 | **2.10** | 1.33x |
| 7 (36, 25) | 11.50 | 20.19 | 9.65 | **3.43** | 3.35x |
| 8 (45, 28) | 16.61 | 26.81 | 18.77 | **8.60** | 1.93x |
| 9 (55, 31) | 24.17 | 36.21 | 33.26 | **11.10** (3.2 TFLOP/s) | 2.18x |
| 8, 360k | 50.35 | | 57.32 | **26.43** | 1.91x |
| 9, 360k | 73.92 | | 101.66 | **34.06** | 2.17x |

* MAGMA options: recursive panel 16 is 1.10x faster than getrf; `nopiv` runs
  in 12.6 ms but its forward error is 25x worse, so it is ruled out.
* Pivoting is rare (0.2-0.3% of elements exchange a row) but needed: the
  matrices are not diagonally dominant (median row ratio 0.2-0.4); local
  condition numbers have median 75-220 and reach 2.0e4 (117k) / 2.5e5 (360k).
* Accuracy at 360k, p=9 (residual `||AX-B||/(||A|| ||X||)`, forward error on
  2,048 sampled elements against a refined reference): MAGMA 5.8e-16 /
  1.7e-13, coop 5.5e-16 / 3.1e-13, fused 1.1e-15 / 2.6e-13. All are backward
  stable; their solutions differ by up to 1.8e-10 only on the worst-conditioned
  elements. This also explains the 1.1e-10 tensor-vs-split3 response gap.

### 2.6 End-to-end tensor assembly with the fused solve (lu-06..09, lu-15)

| p | 117k fused / MAGMA / split3 | 360k fused / MAGMA / split3 |
|---|---|---|
| 6 | **5.94** / 6.65 / 14.5 ms | **20.9** / 23.1 / 44.6 ms |
| 7 | **8.72** / 16.82 / 26.6 | **28.5** / 53.2 / 81.7 |
| 8 | **16.71** / 24.69 / 47.7 | **58.1** / 82.3 / 146.5 |
| 9 | **22.43** / 35.50 / 94.3 | **68.7** (2 chunks) / 116.5 / 291.2 |

At 117k, p=9 the stages are prepare 1.0, build 5.7, solve 11.1, condense 2.4,
scatter 2.2 ms. Parity with split3: 1.4e-14 (p=6) to 1.4e-11 (p=9) at 117k,
1.2-1.6e-10 at 360k p=8-9.

### 2.7 Chunking and 2M triangles (lu-13..18)

| Check | Result |
|---|---|
| Forced 12 uneven chunks (10,007 elements) at 117k | parity with split3 unchanged (1.4e-11 at p=9); 23.2 vs 22.4 ms |
| Default chunking at 360k, p=9 (2 x 180k) | 68.7 ms vs 76.3 ms in one pass (condense 15.0 -> 7.3 ms) |
| 2M (1,995,756 triangles), p=9, tensor + fused solve | **379.4 ms**, workspace 17.6 GB (prepare 17, build 96, solve 188, condense 40, scatter 38 ms) |
| 2M, p=9, raw `fused` kernel (on the fly, block 128) | 1,957.7 ms (255 registers, 8 warps/SM) |
| 2M parity tensor vs raw fused | 1.05e-9 matrix, 8.5e-10 RHS |

Without chunking the tensor stage buffers would need 128 GB (155 GB with the
response copy) at 2M. Running two paths in one sweep at 2M needs
`--no-local-response` (each keeps a 27 GB FP64 response otherwise).

### 2.8 Automatic choice after the changes (lu-12, lu-19..21)

| p | auto picks | 117k | 360k |
|---|---|---|---|
| 5 | tensor (was fused) | 6.57 ms (split3 6.94, fused 7.35) | 20.2 ms (21.2, 22.5) |
| 6-9 | tensor | 5.93 / 8.71 / 16.70 / 22.42 ms | |
| 9, 2M | tensor (was split3 under the 25% rule) | 379.2 ms | |

At p=5 the condense stage dominates the tensor time (3.6 ms at 117k, more
than at p=6), so the cuTENSOR plan for that shape has room left.

## 3. Diffusion-reaction (Poisson)

### 3.1 Kernel resources (offline `ptxas`, 537 variants)

`poisson_local` never spills and uses no static shared memory: 56-96
registers for assembly, 64-128 for RHS refresh / reconstruction. Without
`__launch_bounds__`, registers or shared memory cap residency at 16-28 warps
per SM; retaining the coupling at p=9 (118 KB) leaves one block per SM. Legacy
kernels: serial 104-174 registers and up to 44.5 KB per thread (5-21 resident
elements per SM); cooperative 96 KB at p=9 (2 blocks/SM).

### 3.2 Quick vs. thorough tuning (dr-01/02/04, 100,352 triangles, CSR)

| p | quick, old ranking: LU / Cholesky | thorough: LU / Cholesky | quick, FP64-rich priors: LU / Cholesky |
|---|---|---|---|
| 1 | 1.01 / 0.82 ms | 0.29 / 0.28 | 0.29 / 0.25 |
| 2 | 2.25 / 1.83 | 0.56 / 0.72 | 0.56 / 0.72 |
| 4 | 5.58 / 4.75 | 2.86 / 3.18 | 2.86 / 3.18 |
| 6 | 18.2 / 17.3 | 12.7 / 13.2 | 12.6 / 13.2 |
| 7 | 33.7 / 35.7 | 23.5 / 28.1 | 23.4 / 25.2 |
| 8 | 55.0 / 57.5 | 44.5 / 51.2 | 44.5 / 51.2 |
| 9 | 80.2 / 100.6 | 71.8 / 75.4 | 71.4 / 75.3 |

The old quick ranking (four candidates, retained coupling gated on FP32:FP64
>= 16) settled on 128 threads x 8 columns at every p>=2 and was 1.1-4x slower
than the thorough winner; the old thorough records were not reused at 100k
because the cache key includes the log2 mesh bucket and the boundary state.
With the FP64-rich priors the quick tuner reaches the thorough result at every
order. Reconstruction with the Cholesky cache: 1.35 ms at p=9 (LU cache 14.1,
none 30 ms).

### 3.3 Where the remaining time is (estimates)

At p>=4 the kernel runs at ~20% of FP64 peak and <1.5% of HBM bandwidth
(roofline at p=9: FP64 13.7 ms, HBM 0.67 ms, measured 71 ms). Per-column
recomputation of the coupling and flux recovery is ~82% of its FLOPs; the
factorizations take 113 (LU) / 169 (Cholesky) barriers per element; the
triangular solves use one thread per column. Ranked follow-ups (not done):
warp-per-column solves (3-5x on RHS refresh / reconstruction), building the
per-element coupling once (7-11x fewer FLOPs at p=9), element-batched DMMA or
library builds (est. 8-15 ms at p=9, 100k), a Cholesky-only factorization over
the trailing triangle with fewer barriers.

### 3.4 CuPy reference

CuPy Schur-Cholesky CSR is 2.3-4.2x slower than raw: p=9 at 100k 246.5 ms vs
105.8 ms (raw wrapper time; 79.9 ms kernel).

## 4. Advection-diffusion-reaction (tensor kernel)

### 4.1 Starting point (asm-30, Q1)

Variable full diffusion (kind 6) was 4-5.5x slower than constant full: at
131,072 triangles p=7/8/9 took 473 / 946 / 2,796 ms (constant 106 / 259 /
506 ms); 32k p=9 700 ms. Forcing the mass factor into global memory (asm-31)
gave -8% at p=9 and +10-33% at p=7-8.

Resources (nvcc 13.1 offline; NVRTC on the device reports up to 168 registers
in the phase-clock build): p=9 variable compact layout 132 registers, 139.6 KiB
shared -> **1 block / 4 warps per SM** at 128 threads; external mass 45 KiB ->
4 blocks/SM. p=2 and p=4 (single-thread fallback kept at p<=6) use 255
registers and spill. The 110-row warp solves read their register array through
local memory (`warp_row_value` for 3+ rows per lane). Host preparation at 131k
p=9 adds 1.76 s (coefficients 0.91 + diffusion 0.87 s), scaling with the
400-point volume rule.

### 4.2 Phase breakdown (asm-41/42, `HYBRIDGE_RAW_ADR_PHASE_CLOCK`)

p=9 variable-full, 131k, 128 threads, 2.87 s (counter overhead ~2.6%):

| Phase | Share |
|---|---|
| triangular solves with the 110x110 mass factor (Schur build + output columns) | 39% |
| local matrix by quadrature (NQ=400) | 20% |
| mass LU 110x110 | 13% |
| coupling `N-D` recomputation (compact layout) | 11% |
| mass assembly | 6% |
| rest (gradient, 55x55 Schur LU/solves, scatter) | 11% |

p=7 variable-full: mass solves 64%. p=9 constant-full: local matrix 39%,
coupling 23%. With the mass factor in global memory (asm-42) occupancy rises
to 4 blocks/SM and the quadrature phases speed up 2-2.6x, but the mass LU and
solves become L2-latency-bound (74% of the time), netting only 9%.

### 4.3 Launch shape (asm-43/44/47)

| p=9 variable-full, 131k | 128 threads | 256 | 512 |
|---|---|---|---|
| compact layout | 2,794 ms | 1,639 | — |
| non-compact (`N-D` in shared, 187 KB, `maxrregcount=128`) | 2,448 | 1,498 | **1,403** |

Time halves per doubling of threads while one block fits per SM; p=8 variable
948 -> 720 ms at 256 threads; constant kinds are slower at 256 (fewer resident
blocks). All variants are bitwise identical to the 128-thread kernel
(2,048-element parity, p=7-9).

### 4.4 Smaller volume rule (asm-45, not adopted)

`volume_quad_1d` 10/11 at p=8/9 (NQ 100/121 instead of 324/400): p=9
variable-full 2,346 / 1,484 ms at 128 / 256 threads (-9 to -16%),
constant-full -29%, host preparation -0.7 s. This changes the discretization
(exact only for linear coefficients) and needs an accuracy qualification
before it can be considered.

### 4.5 Explicit mass inverse (asm-50..54)

With the wide launch, the remaining p=9 time was 64% mass LU + solves (asm-46).
Replacing them by an in-place Gauss-Jordan inverse and block-wide products:

| 131k variable-full, auto launch | p=7 | p=8 | p=9 |
|---|---|---|---|
| LU + warp solves (asm-49) | 474 ms | 720 ms | 1,405 ms |
| explicit inverse (asm-51) | **238** | **522** | **894** |
| reconstruction LU / inverse | 272 / 196 | 525 / 448 | 970 / 755 |

p=9 phases with the inverse (asm-52, 902 ms): mass solves 697 -> 83 ms; the
Gauss-Jordan inversion itself is now the largest phase (324 ms, 36%), then the
local matrix (17%) and mass assembly (13%). Accuracy: matches the LU kernel to
1e-10 and the independent CuPy reference to 5e-10 (asm-50, all cache
policies); the full ADR GPU suite passes with it on by default (asm-53, 503
tests).

### 4.6 Defaults now vs. before (asm-54 vs. asm-30 best block)

| Mesh | p | variable-full assembly / reconstruction | speed-up | constant-full |
|---|---|---|---|---|
| 32k | 7 | 119 / 68 -> **60 / 49 ms** | 1.98x | 26 ms, unchanged |
| 32k | 8 | 237 / 195 -> **131 / 112** | 1.82x | 65, unchanged |
| 32k | 9 | 700 / 512 -> **224 / 189** | 3.13x | 127, unchanged |
| 131k | 7 | 473 / 272 -> **239 / 197** | 1.98x | 106, unchanged |
| 131k | 8 | 946 / 776 -> **521 / 449** | 1.81x | 259, unchanged |
| 131k | 9 | 2,796 / 2,046 -> **892 / 755** | 3.13x | 506, unchanged |

Next levers at p=9: the inversion (36%; blocked/DMMA Gauss-Jordan), the
400-point quadrature (~30%), and a split build/factor/solve pipeline on
batched GEMMs for the remaining gap to ~0.3 s.

## 5. Vendored CuPy ADR assembler

Q1 measured 0.55-13.2 s per warm assembly at 32,768 triangles (p=1-9), with
the GPU doing only 2-220 ms: the host route runs single-threaded NumPy BLAS
(`OPENBLAS_NUM_THREADS=1`) and ~520 ms of Python loops for face topology and
gather maps at every p. The device route (branch `cupy-adr-device`):

| Triangles | p | host route first / warm | device route first / warm | peak device memory |
|---|---|---|---|---|
| 32,768 | 1 | 613 / 549 ms | 141 / 15.7 ms | 0.11 -> 0.19 GiB |
| 32,768 | 4 | 2,308 / 1,472 | 188 / 63.4 | 1.85 -> 1.73 |
| 32,768 | 7 | 6,696 / 5,809 | 405 / 270 | 9.90 -> 1.98 |
| 32,768 | 9 | 15,000 / 13,134 | 787 / 650 | 22.5 -> 2.17 |
| 199,712 | 4 | 9,895 / 8,552 | 450 / 332 | 11.3 -> 2.16 |
| 199,712 | 7 | 36,371 / 30,049 | 1,655 / 1,530 | 60.3 -> 4.03 |

Parity with the host route <= 1e-12 on the standard cases (1e-10 on a 1e-6
anisotropy case, where the host route already differs from NumPy by 6e-11).
Half of the remaining time is copying results back to the host, which the
worker contract requires. The raw tensor kernel stays 10-30x faster.

## 6. Defaults on sm_90 after this study

| Family | Setting | Value on sm_90 |
|---|---|---|
| AR | `auto` | tensor from p=6 (p=5 on [ar-batched-lu]) when inputs allow, else split3 (p>=8) / fused |
| AR | tensor local solver | fused [ar-batched-lu]; MAGMA on `h200-asm-tuning` |
| AR | tensor chunk | 16 GiB of stage buffers [ar-batched-lu] |
| AR | split3 volume | inline (pre-pass opt-in) |
| DR | quick tuner | FP64-rich assembly priors |
| ADR | block size `auto`, variable kinds | 256 at p=8; 512 + non-compact + 128-register cap at p=9 |
| ADR | kind-6 mass | explicit inverse at p>=7 (`HYBRIDGE_RAW_ADR_MASS_INVERSE=off` for LU) |

## 7. Reproduce

```bash
S=python   # from the repository root, on one H200
# AR sweep (split3 vs tensor, auto check), 117k and 360k
$S scripts/gpu/sweep_advection_blocks.py --boundary tangent \
    --formats bsr --variants conflict --paths split3,tensor --split3-blocks= --split3-auto \
    --auto-check --meshes star:0.0095,star:0.005 --orders 5-9 --output OUT/sweep.jsonl
# AR batched local solve on captured systems
$S scripts/gpu/benchmark_ar_local_lu.py --mesh star:0.005 \
    --orders 8,9 --methods magma,coop,fused --output OUT/solvers.jsonl
# DR tuning (quick and --thorough)
HYBRIDGE_CUDA_TUNING_CACHE_DIR=CACHE $S scripts/gpu/tune_poisson_raw_cuda.py \
    --orders 1-9 --mesh-sizes 224 --trace-bases legendre-modal --formats csr \
    --cache-policies none,schur-lu,schur-cholesky --output OUT/dr.json
# ADR (auto launch), optionally with the phase clock
HYBRIDGE_RAW_ADR_PHASE_CLOCK=1 $S \
    scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_raw_cuda.py --nx 256 \
    --orders 7 8 9 --coefficients variable-full constant-full --bases legacy-lagrange \
    --matrix-formats bsr --block-sizes auto --cache-local-factors none --reconstruction \
    --output OUT/adr.jsonl
# ADR layout/launch variants and parity
$S scripts/gpu/adr_variant_bench.py --nx 256 --orders 9 \
    --layout noncompact --maxrregcount 128 --block-sizes 128 256 512 --output OUT/adr.jsonl
```

## 8. Code changes from this study (2026-10-27/28)

| Area | Content |
|---|---|
| ADR, AR | ADR mass placement switch, tensor AR auto on sm_90, split3 volume pre-pass, LU-mode fix |
| ADR | ADR phase clock, RawModule resource recording, ADR variant driver |
| ADR, DR | ADR wide launch, Poisson FP64-rich priors, explicit kind-6 mass inverse (default) |
| AR | fused batched solve, tensor integration, sm_90 default |
| AR | element chunking, sweep `--no-local-response`, p>=5 / chunk-aware auto policy |
| ADR | device route of the vendored CuPy ADR assembler |
