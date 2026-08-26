# Advection BICGSTAB CSR/BSR findings with CUDA 13 generic SpMV

Date: 2026-08-24

## Runtime and configuration

- GPU: Quadro RTX 6000, 24 GiB, compute capability 7.5.
- AMGX: local `hdg-cuda13-integration` checkout rebuilt with CUDA 13.0.1; PyAMGX
  resolved `/tmp/AMGX-build-cuda13.0.1/libamgxsh.so`.
- CUDA libraries: cuBLAS 13 and cuSPARSE/cuSOLVER 12 from
  `/tmp/cuda-13.0.1`; CuPy 14.2.0 reported runtime 13.2 and driver 13.0.
- Solver: unpreconditioned `BICGSTAB`, `bsr_spmv_backend=cusparse_generic`,
  relative tolerance `1e-11`, maximum 1500 iterations, external row scaling.
- Assembly: raw CUDA, fused local assembly, cooperative LU, zero-flux square
  transport case, `dub_orth` volume basis and legacy-Lagrange trace basis.

## Matched benchmark method

Each order and storage format received one warm-up solve. Measured trials then
alternated CSR/BSR order to reduce drift. The nx=64 case used five trials; the
finer nx=128 and nx=256 cases used three. The matrix coefficients and discrete
problem were the same; only scalar CSR versus native face BSR storage/upload
changed. A speedup above 1 means BSR was faster.

| nx | triangles | trials | p1 (2x2) | p2 (3x3) | p3 (4x4) | p4 (5x5) | p5 (6x6) | p6 (7x7) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 8,192 | 5 | 0.856 | 0.930 | 1.117 | 0.967 | 1.052 | 1.121 |
| 128 | 32,768 | 3 | 1.114 | 0.939 | 1.200 | 1.000 | 1.148 | 1.261 |
| 256 | 131,072 | 3 | 1.162 | 0.970 | 1.326 | 1.005 | 1.218 | 1.313 |

The table above is the median AMGX-solve speedup, CSR time divided by BSR time.
End-to-end wall-time speedups were:

| nx | p1 | p2 | p3 | p4 | p5 | p6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.978 | 0.983 | 1.033 | 0.996 | 1.016 | 1.035 |
| 128 | 1.047 | 0.974 | 1.090 | 1.005 | 1.076 | 1.118 |
| 256 | 1.115 | 0.979 | 1.252 | 1.014 | 1.167 | 1.209 |

At nx=256, median CSR/BSR AMGX solve milliseconds and iteration counts were:

| p | block | CSR ms / iters | BSR ms / iters | solve speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2x2 | 221.02 / 423 | 190.22 / 427 | 1.162 |
| 2 | 3x3 | 353.51 / 418 | 364.33 / 418 | 0.970 |
| 3 | 4x4 | 474.22 / 381 | 357.75 / 384 | 1.326 |
| 4 | 5x5 | 642.15 / 371 | 638.97 / 375 | 1.005 |
| 5 | 6x6 | 885.66 / 385 | 727.12 / 385 | 1.218 |
| 6 | 7x7 | 1123.36 / 379 | 855.45 / 380 | 1.313 |

All solves completed and the median physical relative residuals remained below
`5.4e-11`. Block-vector convergence norms can change the stopping iteration by
a few iterations even when the matrix coefficients are identical, so the p2
case, with exactly matched iteration counts, is the cleanest whole-solve
comparison.

## Nsight dispatch proof and interpretation

A fresh-process p=2 BSR profile recorded 194 `cusparseSpMV` ranges and 194
`cusparse::...bsrmv_tiny_core<...,3,...>` GPU kernels. It recorded
`cusparseCreateBsr` and `cusparseSpMV_bufferSize` ranges and no legacy
`cusparseDbsrmv` or AMGX custom 3x3 SpMV kernel. This proves the patched
BICGSTAB route overrides the historical 3x3 specialization.

For the matched p=2 profiles, the BSR SpMV kernel totaled 4.613 ms over 194
calls, about 23.8 us per call. The CSR SpMV core, fixup, and partition kernels
totaled about 6.532 ms, so the BSR GPU SpMV work itself was about 1.42x faster.
The full BSR solve was nevertheless slower at p=2. The profile shows substantial
block-vector `strided_reduction` work, while both AMGX generic paths recreate
sparse/dense descriptors, query the buffer size, and acquire/release workspace
inside every SpMV. Those non-SpMV costs erase the 3x3 kernel saving.

The fine-mesh results confirm that fixed overhead is increasingly amortized for
2x2, 4x4, 6x6, and 7x7 blocks. The 3x3 case remains a real exception, and 5x5 is
approximately neutral. Before changing AMGX again, compare 3x3 generic BSR with
the old AMGX 3x3 kernel under identical block-vector reduction semantics.

## Legacy test2 on an unstructured square

A second warmed comparison used the legacy `test2` transport problem on Gmsh
unstructured meshes of `[-1,1]^2`. The discrete coefficients were
`beta=(x,-y)`, reaction and source `y^2`, with the legacy oscillatory exact
Dirichlet condition. Assembly used raw CUDA fused local kernels, cooperative
LU, `dub_orth` elements, legacy-Lagrange traces, and boundary elimination.

Plain BICGSTAB used external row scaling. PBICGSTAB used no external scaling
and the active block-size-safe preconditioner
`JACOBI_L1(max_iters=1, relaxation_factor=1,
jacobi_l1_scalar_rows_for_blocks=1)`. AMGX solved to relative tolerance
`1e-10`, with at most 1500 iterations. Each order/format received one warm-up
and three measured solves; table entries are medians and speedup means CSR time
divided by BSR time.

The PBICGSTAB screen was performed first at `ms=0.02`, p=6, pure 7x7 BSR.
Unscaled scalar-row L1 Jacobi converged in 213 iterations with a 119.46 ms AMGX
solve, compared with 246 iterations and 244.12 ms for one unscaled block-Jacobi
step. Two scaled block-Jacobi steps took 225.28 ms despite reducing the median
to 156 iterations, and the unscaled variant failed. `MULTICOLOR_DILU` is not
compiled for block size 7 in this build. Standard aggregation AMG also cannot
construct its `LOW_DEG` hierarchy at block size 7. Thus scalar-row L1 Jacobi is
the best PBICGSTAB preconditioner found that works across p=1..6 pure BSR.

### Direct block DILU follow-up

AMGX `MULTICOLOR_DILU` is a direct block DILU(0)-style preconditioner; it is not
the separate `MULTICOLOR_ILU` factorization. The existing GPU dispatch supports
the HDG face-block dimensions 2x2 through 5x5, corresponding to p=1..4. A
warmed screen compared both `MIN_MAX` and `PARALLEL_GREEDY` coloring with
relaxation factors 0.7, 0.9, and 1.0. Unscaled parallel-greedy coloring at 0.7
was the best variant.

Matched unscaled BSR medians on the 92,552-triangle mesh were:

| p | PBICGSTAB+L1 wall / iters | PBICGSTAB+DILU wall / iters | L1/DILU wall |
| ---: | ---: | ---: | ---: |
| 1 | 270.16 ms / 499 | 255.77 ms / 162 | 1.056 |
| 2 | 442.15 ms / 500 | 367.05 ms / 157 | 1.205 |
| 3 | 460.54 ms / 461 | 424.33 ms / 160 | 1.085 |
| 4 | 697.10 ms / 432 | 793.03 ms / 162 | 0.879 |

The iteration reduction is large, but dense block triangular work grows enough
that p=4 loses overall. The promoted experimental config is therefore limited
to p=1..3 and does not replace plain scaled BICGSTAB as the general default.

For completeness, temporary 6x6 and 7x7 dispatches were compiled and passed a
finite-output random-matrix smoke test. They were then rejected on the actual
p=5/6 advection systems: every unscaled weight/coloring variant became
non-finite, while external scaling kept values finite but failed to converge in
1500 iterations. Those dispatches were removed instead of being advertised as
supported. The retained native smoother test covers finite output for supported
block sizes 1 through 5.

### AMGX solve speedup over mesh size

Plain BICGSTAB:

| mesh size | triangles | p1 | p2 | p3 | p4 | p5 | p6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 248 | 0.773 | 0.708 | 0.791 | 0.778 | 0.771 | 0.774 |
| 0.10 | 944 | 0.838 | 0.763 | 0.912 | 0.779 | 0.725 | 0.779 |
| 0.05 | 3,704 | 0.931 | 0.787 | 0.956 | 0.892 | 0.975 | 1.047 |
| 0.02 | 23,254 | 1.077 | 0.960 | 1.287 | 1.015 | 1.114 | 1.212 |
| 0.01 | 92,552 | 1.132 | 0.961 | 1.305 | 1.031 | 1.188 | 1.304 |

PBICGSTAB with scalar-row L1 Jacobi:

| mesh size | triangles | p1 | p2 | p3 | p4 | p5 | p6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 248 | 0.784 | 0.705 | 0.785 | 0.771 | 0.771 | 0.790 |
| 0.10 | 944 | 0.838 | 0.760 | 0.912 | 0.777 | 0.732 | 0.786 |
| 0.05 | 3,704 | 0.930 | 0.791 | 0.945 | 0.907 | 0.956 | 1.021 |
| 0.02 | 23,254 | 1.097 | 0.967 | 1.209 | 0.981 | 1.090 | 1.174 |
| 0.01 | 92,552 | 1.134 | 0.947 | 1.269 | 1.041 | 1.170 | 1.276 |

BSR loses at genuinely small sizes because launch, descriptor, workspace, and
block-vector overhead dominate. By 23,254 triangles it wins clearly for p=1,
3, 5, and 6. The generic 3x3 BSR path remains slower than CSR at every size;
5x5 is neutral or only slightly faster.

At `ms=0.01`, BSR end-to-end wall speedups for p=1..6 were
`1.090, 0.974, 1.238, 1.030, 1.149, 1.214` for BICGSTAB and
`1.103, 0.965, 1.220, 1.047, 1.151, 1.213` for PBICGSTAB. Fine-mesh BSR
absolute solve results were:

| p | BICGSTAB ms / iters | PBICGSTAB+L1 ms / iters | BIC/PBIC time |
| ---: | ---: | ---: | ---: |
| 1 | 198.69 / 544 | 196.93 / 498 | 1.009 |
| 2 | 313.60 / 465 | 358.31 / 499 | 0.875 |
| 3 | 341.98 / 471 | 359.58 / 461 | 0.951 |
| 4 | 571.52 / 454 | 576.84 / 433 | 0.991 |
| 5 | 672.33 / 462 | 671.09 / 436 | 1.002 |
| 6 | 774.24 / 455 | 797.76 / 445 | 0.971 |

The L1 preconditioner often lowers iteration count, but its per-iteration work
usually consumes the saving. Plain scaled BICGSTAB therefore remains the best
overall default for this problem. PBICGSTAB plus unscaled scalar-row L1 Jacobi
is the preferred compatible PBICGSTAB fallback, not a general speed winner.

Two measured cells initially landed on the wrapper's exact residual-validation
boundary: AMGX reported success and the physical residual met `1e-10`, while
the independently recomputed solver residual differed at roundoff scale. Those
cells were repeated with wrapper `rtol=1.01e-10`; AMGX's own tolerance remained
`1e-10`.

## Raw artifacts

The detailed JSONL/logs and Nsight reports are temporary workspace artifacts:

- `/tmp/codex-adv-csr-bsr-cuda13-patched-n64-r5.{jsonl,log}`
- `/tmp/codex-adv-csr-bsr-cuda13-patched-n128-r3.{jsonl,log}`
- `/tmp/codex-adv-csr-bsr-cuda13-patched-n256-r3.{jsonl,log}`
- `/tmp/codex-amgx-bicgstab-{bsr,csr}-p2-cuda13-patched.nsys-rep`
- `/tmp/codex-test2-match-ms0{20,10,05,02,01}.{jsonl,log}`
- `/tmp/codex-test2-retry-ms{002-p5-bic-csr,001-p2-bic-bsr}.{jsonl,log}`
- `/tmp/codex-test2-pbic-safe-screen-ms002-p6.jsonl`
- `/tmp/codex-test2-dilu-screen-ms002-p1-4.{jsonl,log}`
- `/tmp/codex-test2-dilu-confirm-ms001-p1-4.{jsonl,log}`
- `/tmp/codex-test2-dilu-p5-6-smoke.jsonl`
- `/tmp/codex-test2-dilu-p5-6-{scaled,minmax}-smoke.jsonl`

After the AMGX shared library and native launcher were rebuilt,
`KrylovBsrSpmvBackend`, `DeviceMemoryStats`, `GMRESReliableResidual`,
`BiCGStabResidual`, and `SmootherBlocksizes` all passed in `dDDI` mode.
