# Raw CUDA Fused Cooperative LU Findings - 2026-07-20

## Scope

This note records the fused Raw CUDA advection-reaction update that followed the
2026-07-19 serial/cooperative baseline work. The implementation is in:

- `hdgfem/backends/cupy_adv_rea_raw.py`
- `scripts/run_adv_rea_gpu4_hdg.py`

The focus was to reduce fused local-kernel cost, add an opt-in cooperative LU
stage, document the CUDA synchronization contract, and extend the fused raw path
to higher polynomial orders where shared memory allows it.

## Implementation Summary

The fused advection kernel now caches short per-element coefficient vectors in
shared memory:

- `source_coeffs[K, :]`
- `beta_x[K, :]`
- `beta_y[K, :]`
- optional non-scalar `reaction_coeffs[K, :]`
- transformed beta coefficients:
  - `beta_ref0[k] = beta_x[k] * inv_t00 + beta_y[k] * inv_t10`
  - `beta_ref1[k] = beta_x[k] * inv_t01 + beta_y[k] * inv_t11`

The volume advection loop uses `beta_ref0` and `beta_ref1`, avoiding repeated
global coefficient loads and repeated affine inverse multiplies per matrix
entry. Trace Schur/RHS emission now computes the oriented lift row once per row
task and reuses it for the source-column RHS dot and every trace-column Schur
dot.

The local LU stage has two modes:

- `safe`: default. This is the validated baseline and keeps the original stable
  pivot/search/swap handoff while retaining cooperative trailing updates.
- `coop`: opt-in for `--raw-local-assembly fused`. This mode performs a
  cooperative pivot-column scan, shared-memory pivot reduction, parallel
  multiplier scaling, and parallel trailing Schur update. Row swaps remain on
  thread 0 because fully parallel row-swap variants reproduced illegal-address
  failures in the fused kernel.

The fused raw advection path now validates p <= 8 (`el_dof <= 45`) for the
legacy-lagrange and legendre-modal trace bases. The precomputed raw path keeps
the original p <= 6 guard. The runner rejects `--raw-lu-mode coop` unless the assembly backend is
`raw-cuda` and the local assembly mode is `fused`.

## Correctness Checks

Assembly comparisons against the precomputed raw path for p <= 6 used exact COO
row/column equality and roundoff-level matrix/RHS differences:

| case | rows/cols | max matrix diff | max RHS diff |
|---|---|---:|---:|
| p=3, ms=0.2 | exact | 1.717e-16 | 8.327e-16 |
| p=4, ms=0.2 | exact | 1.422e-16 | 5.551e-16 |
| p=5, ms=0.2 | exact | 1.919e-16 | 6.453e-16 |
| p=6, ms=0.2 | exact | 2.528e-16 | 6.106e-16 |
| p=6, ms=0.02 | exact | 5.052e-17 | 2.307e-16 |

Because the precomputed raw path remains capped at p <= 6, p=7 and p=8 were
compared against the CuPy assembler:

| case | rows/cols | max matrix diff | max RHS diff |
|---|---|---:|---:|
| p=7, ms=0.2 | exact | 2.463e-16 | 1.069e-15 |
| p=8, ms=0.2 | exact | 3.416e-16 | 9.506e-16 |

Representative full solves completed through assembly, AMGX, fused raw
reconstruction, and error evaluation:

| case | trace dofs | nnz | scaled rel residual | L2 error | Linf error |
|---|---:|---:|---:|---:|---:|
| p=3, ms=0.2 | 1,408 | 26,880 | 3.456e-13 | 4.265e+00 | 7.045e+00 |
| p=8, ms=0.2 | 3,168 | 136,080 | 6.451e-13 | 3.885e-02 | 3.265e-01 |
| p=6, ms=0.02 | 242,767 | 8,457,645 | 1.982e-12 | 6.521e-08 | 2.354e-06 |

## Safe vs Cooperative LU Kernel Timing

The following table uses warmed fused raw assembly-only runs at `ms=0.03`, block
size 32. Safe and cooperative outputs matched exactly for rows, columns, matrix
data, and RHS in these checks.

| p | triangles | trace dofs | safe kernel | coop kernel | speedup | faster |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 10,472 | 62,296 | 0.002964 s | 0.002478 s | 1.196x | 16.4% |
| 4 | 10,472 | 77,870 | 0.006965 s | 0.005574 s | 1.249x | 20.0% |
| 5 | 10,472 | 93,444 | 0.013598 s | 0.009422 s | 1.443x | 30.7% |
| 6 | 10,472 | 109,018 | 0.023749 s | 0.016519 s | 1.438x | 30.4% |
| 7 | 10,472 | 124,592 | 0.041272 s | 0.032567 s | 1.267x | 21.1% |
| 8 | 10,472 | 140,166 | 0.083809 s | 0.066202 s | 1.266x | 21.0% |

Spot checks on p=6 and p=8 were consistent:

| case | block | safe kernel | coop kernel | speedup |
|---|---:|---:|---:|---:|
| p=6, ms=0.02 | 32 | 0.046462 s | 0.035344 s | 1.315x |
| p=6, ms=0.02 | 64 | 0.044620 s | 0.035018 s | 1.274x |
| p=8, ms=0.2 | 32 | 0.002559 s | 0.002086 s | 1.227x |
| p=8, ms=0.2 | 64 | 0.003038 s | 0.002384 s | 1.274x |

## Fused Raw Cooperative vs Vectorized CuPy

At `ms=0.03`, the fused raw cooperative path is faster in assembly for p=3..8,
while total runtime improves less because global CSR construction, AMGX, mesh
generation, and error evaluation are shared costs.

| p | CuPy assembly | raw fused coop assembly | assembly speedup | CuPy total | raw total | total speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.3749 s | 0.1346 s | 2.79x | 2.368 s | 2.117 s | 1.12x |
| 4 | 0.3451 s | 0.1386 s | 2.49x | 2.309 s | 2.078 s | 1.11x |
| 5 | 0.3466 s | 0.1475 s | 2.35x | 2.272 s | 2.144 s | 1.06x |
| 6 | 0.3810 s | 0.1688 s | 2.26x | 2.405 s | 2.190 s | 1.10x |
| 7 | 0.4156 s | 0.2231 s | 1.86x | 2.544 s | 2.304 s | 1.10x |
| 8 | 0.4755 s | 0.3596 s | 1.32x | 2.834 s | 2.651 s | 1.07x |

High-order larger comparisons:

| p | mesh size | CuPy assembly | raw fused coop assembly | assembly speedup | CuPy total | raw total | total speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.010 | 1.0206 s | 0.5507 s | 1.85x | 8.494 s | 7.656 s | 1.11x |
| 8 | 0.010 | 1.5333 s | 0.9454 s | 1.62x | 9.839 s | 8.910 s | 1.10x |
| 8 | 0.008 | 2.1301 s | 1.3187 s | 1.62x | 15.076 s | 13.924 s | 1.08x |

For `p=8, ms=0.008`, reconstruction was also faster with fused raw cooperative
assembly data: CuPy reconstruction took 1.601 s, while raw fused reconstruction
took 0.992 s, a 1.61x speedup. AMGX/global solve time was effectively unchanged.

## Full Solve Sweeps

Fused raw cooperative, `ms=0.05`:

| p | triangles | trace dofs | nnz | raw kernel | assembly | AMGX solve | total | L2 error | Linf error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3,704 | 21,904 | 432,960 | 0.03803 s | 0.12876 s | 0.02532 s | 1.76955 s | 2.706e-02 | 2.893e-01 |
| 4 | 3,704 | 27,380 | 676,500 | 0.04607 s | 0.13823 s | 0.02750 s | 1.84971 s | 3.431e-03 | 3.299e-02 |
| 5 | 3,704 | 32,856 | 974,160 | 0.04555 s | 0.14786 s | 0.03091 s | 1.87440 s | 4.572e-04 | 1.128e-02 |
| 6 | 3,704 | 38,332 | 1,325,940 | 0.04457 s | 0.15170 s | 0.03438 s | 1.92846 s | 3.462e-05 | 8.328e-04 |
| 7 | 3,704 | 43,808 | 1,731,840 | 0.05373 s | 0.19526 s | 0.03618 s | 1.92247 s | 4.872e-06 | 2.385e-04 |
| 8 | 3,704 | 49,284 | 2,191,860 | 0.06600 s | 0.28471 s | 0.03726 s | 2.23904 s | 3.250e-07 | 1.616e-05 |

Fused raw cooperative, `ms=0.03`:

| p | triangles | trace dofs | nnz | raw kernel | assembly | AMGX solve | total | L2 error | Linf error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 10,472 | 62,296 | 1,237,344 | 0.03899 s | 0.13462 s | 0.05658 s | 2.11745 s | 3.711e-03 | 5.545e-02 |
| 4 | 10,472 | 77,870 | 1,933,350 | 0.04079 s | 0.13863 s | 0.05832 s | 2.07833 s | 2.856e-04 | 4.225e-03 |
| 5 | 10,472 | 93,444 | 2,784,024 | 0.04665 s | 0.14753 s | 0.06878 s | 2.14414 s | 1.865e-05 | 5.428e-04 |
| 6 | 10,472 | 109,018 | 3,789,366 | 0.05582 s | 0.16883 s | 0.07430 s | 2.19003 s | 1.154e-06 | 3.787e-05 |
| 7 | 10,472 | 124,592 | 4,949,376 | 0.07848 s | 0.22307 s | 0.08085 s | 2.30386 s | 6.731e-08 | 2.189e-06 |
| 8 | 10,472 | 140,166 | 6,264,054 | 0.12491 s | 0.35959 s | 0.08699 s | 2.65139 s | 3.436e-09 | 1.625e-07 |

## Heavy Mesh Limits

The following full solves used the 24 GB Quadro RTX 6000 test GPU.

| p | mesh size | triangles | trace dofs | nnz | raw kernel | assembly | AMGX solve | reconstruction | total | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7 | 0.010 | 92,552 | 1,107,424 | 44,194,560 | 0.29770 s | 0.55072 s | 1.31769 s | 0.25909 s | 7.65577 s | fits |
| 8 | 0.010 | 92,552 | 1,245,852 | 55,933,740 | 0.59927 s | 0.94537 s | 1.56684 s | 0.63486 s | 8.91009 s | fits |
| 8 | 0.008 | 144,686 | 1,948,761 | 87,532,245 | 0.91888 s | 1.31873 s | 3.15415 s | 0.99070 s | 13.92410 s | fits |
| 8 | 0.006 | 258,002 | 3,477,015 | 156,249,243 | 1.63316 s | 2.15901 s | 7.33052 s | 1.77722 s | 25.24189 s | fits |
| 6 | 0.005 | 369,790 | 3,877,195 | 135,545,025 | 0.57153 s | 1.00929 s | 8.21708 s | 0.50168 s | 28.98051 s | fits |
| 7 | 0.005 | 369,790 | about 4.44M | about 177M | 1.11096 s | 1.63712 s | n/a | n/a | n/a | COO-to-CSR OOM |
| 8 | 0.005 | 369,790 | about 4.99M | about 224M | 2.31918 s | 2.98226 s | n/a | n/a | n/a | COO-to-CSR OOM |

The p=7/p=8 `ms=0.005` cases assembled successfully but ran out of memory during
CuPy COO-to-CSR conversion, before AMGX setup. This makes global sparse matrix
construction the immediate scaling bottleneck for high-order heavy cases.


## Modal Trace Compatibility Update

The raw advection kernels originally rejected `legendre-modal` traces because the
column-orientation transform was hard-coded for nodal dof reversal. The CUDA
source generator now emits a trace-basis orientation helper:

- legacy-lagrange: negative local orientation maps global dof `j` to local dof
  `NTR - 1 - j` with sign `+1`;
- legendre-modal: negative local orientation maps global mode `j` to local mode
  `j` with sign `(-1)^j`.

The row-side orientation still comes from
`face_trace_test_element_trial_oriented[loc2oriented_face_coupling]`, matching
the CuPy reference path. The same column transform is used in raw assembly,
boundary-trace RHS elimination, precomputed raw reconstruction, and fused raw
reconstruction.

Modal assembly comparisons against CuPy with fused raw cooperative LU:

| case | rows/cols | max matrix diff | max RHS diff |
|---|---|---:|---:|
| p=3, ms=0.2 | exact | 3.608e-16 | 9.992e-16 |
| p=4, ms=0.2 | exact | 3.886e-16 | 7.355e-16 |
| p=6, ms=0.2 | exact | 7.772e-16 | 2.887e-15 |
| p=8, ms=0.2 | exact | 1.173e-15 | 2.574e-15 |
| p=6, ms=0.03 | exact | 2.619e-16 | 1.069e-15 |

Modal reconstruction comparisons used a deterministic global trace vector and
compared fused raw reconstruction against the CuPy reconstruction path:

| case | max coefficient diff | relative diff |
|---|---:|---:|
| p=3, ms=0.2 | 7.454e-15 | 1.704e-15 |
| p=6, ms=0.2 | 9.992e-15 | 3.961e-15 |
| p=8, ms=0.2 | 3.109e-14 | 7.663e-15 |

The precomputed raw path also matched the CuPy modal reference for p=3 and p=6
at `ms=0.2` with exact rows/columns and max matrix/RHS differences below
7e-16. Representative fused raw cooperative modal full solves:

| p | mesh size | trace dofs | nnz | raw kernel | assembly | AMGX solve | reconstruction | total | L2 error | Linf error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.05 | 21,904 | 432,960 | 0.04143 s | 0.14996 s | 0.02326 s | 0.00135 s | 1.94219 s | 1.972e-02 | 2.977e-01 |
| 6 | 0.05 | 38,332 | 1,325,940 | 0.04327 s | 0.15453 s | 0.03376 s | 0.00767 s | 1.90189 s | 3.027e-05 | 8.430e-04 |
| 8 | 0.05 | 49,284 | 2,191,860 | 0.06634 s | 0.28715 s | 0.03836 s | 0.03486 s | 2.26250 s | 3.132e-07 | 1.614e-05 |
| 8 | 0.03 | 140,166 | 6,264,054 | 0.11919 s | 0.35019 s | 0.09417 s | 0.08924 s | 2.70257 s | 3.346e-09 | 1.614e-07 |

## Reproducible Commands

p8, `ms=0.01`:

```bash
LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib HDGFEM_GPU4_AMGX_MONITOR=0 \
.venv/bin/python scripts/run_adv_rea_gpu4_hdg.py --case test2 -o 8 -ms 0.01 -mt rectangle \
  --basis dub_orth --trace-basis legacy-lagrange --assembly-backend raw-cuda \
  --raw-local-assembly fused --raw-lu-mode coop --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 18 --error-volume-quad-1d 28 -pr 4 -v 2
```

p8, `ms=0.006`:

```bash
LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib HDGFEM_GPU4_AMGX_MONITOR=0 \
.venv/bin/python scripts/run_adv_rea_gpu4_hdg.py --case test2 -o 8 -ms 0.006 -mt rectangle \
  --basis dub_orth --trace-basis legacy-lagrange --assembly-backend raw-cuda \
  --raw-local-assembly fused --raw-lu-mode coop --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 18 --error-volume-quad-1d 28 -pr 3 -v 2
```

p6, `ms=0.005`:

```bash
LD_LIBRARY_PATH=/tmp/AMGX-build:/tmp/AMGX-install/lib HDGFEM_GPU4_AMGX_MONITOR=0 \
.venv/bin/python scripts/run_adv_rea_gpu4_hdg.py --case test2 -o 6 -ms 0.005 -mt rectangle \
  --basis dub_orth --trace-basis legacy-lagrange --assembly-backend raw-cuda \
  --raw-local-assembly fused --raw-lu-mode coop --raw-block-size 32 \
  --trace-ordering none --volume-quad-1d 14 --error-volume-quad-1d 22 -pr 2 -v 2
```

## Next Bottlenecks

The fused local kernel is now faster than the vectorized CuPy path for assembly,
and cooperative LU improves the raw fused kernel by roughly 16-31% in the tested
p=3..8 range. For large p7/p8 runs, the next meaningful performance and memory
targets are:

- lower-memory reduced COO/CSR construction;
- avoiding duplicate sparse sort/sum work before AMGX upload;
- AMGX configuration robustness and solve time on 100M+ nonzero systems;
- optionally caching static map arrays across repeated runs.
