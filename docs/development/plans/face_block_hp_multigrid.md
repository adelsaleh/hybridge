# Face-Block hp-Multigrid For HDG Poisson

Status: active numerical and GPU implementation plan. This document is not a
supported solver contract. The production baseline remains the validated AMGX
classical CSR or fine-BSR/scalar-hierarchy path until the acceptance gates below
are met.

The precise ownership boundary between HDGFEM, direct cuSPARSE, PyAMGX, the
modified AMGX hybrid/pure-BSR paths, and the remaining custom smoother kernels
is maintained in the
[BSR and AMGX dependency map](../../backends/bsr_amgx_dependency_map.md).

## Decision

The candidate is a fixed, symmetric face-block p/h multigrid V-cycle used as an
SPD preconditioner for PCG:

1. assemble the condensed Poisson trace operator directly in face BSR;
2. express every face in a canonical, nested, L2-orthonormal Legendre basis;
3. p-coarsen on the unchanged face graph to the constant mode;
4. apply classical scalar h-AMG only to that p=0 operator;
5. use face-block Chebyshev--block-Jacobi smoothing on p>0 levels;
6. use transpose restriction, reversed post-smoothing, and a fixed coarse
   application so the completed V-cycle is suitable for ordinary PCG.

The working name is `FB-HP-MG-PCG`. The first prototype may use PCGF while
symmetry and positive-definiteness are being measured; PCG is enabled only
after those properties pass explicit tests.

## Existing Evidence And Baseline

The coefficient-exact hybrid AMGX path keeps the fine operator in BSR and uses
a scalar hierarchy. On the 152,909-triangle radius-5 trigonometric-Poisson disk
at p=1..6 it needs 20--23 iterations and is the primary performance baseline.
The dense pure-BSR classical hierarchy needs 15--53 iterations, but is
3.05--4.45 times slower for p=3..6. The missing ingredient is therefore coarse
space quality and work distribution, not another scalar weight on the existing
whole-face interpolation.

The following have already been tested and are not the next experiment:

- identity-lifted `w_ic I_b` interpolation;
- dense block-Jacobi-smoothed interpolation on a fixed D2 support;
- larger support, repeated smoothing, right normalization, and weight damping;
- raw, normalized, and inverse-diagonal block strength metrics;
- scalar-guided whole-face promotion and block Extended+i interpolation.

The p-coarsening hypothesis has now passed its first production-size numerical
checkpoint. On the 150,209-triangle radius-5 disk at p=6, the dedicated scalar
p=0 cycle described below makes the complete modal V-cycle self-adjoint to
about 3e-18 and positive on the sampled action. The remaining work is the full
Phase-2 parameter ablation, persistent-buffer/launch optimization,
repeated-RHS timing, and qualification through p=9.

### Production-size Phase-2 checkpoint (2026-08-19)

The system had 224,862 free faces, 1,574,034 degree-6 trace unknowns, and
1,122,504 face blocks. Generic cuSPARSE BSR SpMV took 0.962 ms versus 1.019 ms
for the row-owned raw kernel, with relative parity 2.19e-16. Results below use
order-3 face-block Chebyshev smoothing, one pre- and one reversed post-sweep,
and the common true-residual tolerance 1e-8.

#### Retained GPU kernel roles

All four implementations remain in the prototype. They form a validation ladder
from a readable array formulation to the low-level production candidate.

- **Transparent CuPy reference smoother (`cupy`).** This intentionally
  materializes the residual, calls the selected BSR `matvec`, applies every
  full dense inverse diagonal block with `cupy.matmul`, and performs a
  separate vector update. With `spmv_backend=auto`, the matrix action is
  generic cuSPARSE BSR where supported. This is the simple gateway and numerical
  oracle for future dense-BSR work and is not scheduled for removal.
- **Generic cuSPARSE BSR SpMV (`cusparse-generic-bsr`).** This remains the
  default standalone action for PCG `A*p`, true-residual recomputation, and
  the CuPy reference smoother. Descriptors, preprocessing state, and workspace
  are matrix-owned and reused.
- **Row-owned raw-CUDA BSR SpMV (`raw-cuda`).** This is the race-free fallback
  and independent benchmark/parity control. It retains full dense face blocks
  and remains available when generic cuSPARSE rejects a block size or runtime.
- **Warp-owned fused dense-BSR smoother (`fused-raw-cuda`).** One warp owns
  one face row, loads each neighboring trace block once, broadcasts its entries
  with warp shuffles, and fuses dense BSR SpMV, residual formation, the complete
  dense diagonal-block inverse, and the Chebyshev update without global
  residual/update temporaries.

| p schedule | smoother | outer | iterations | V-cycle | hot solve | true relative residual |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 6 -> 3 -> 1 -> 0 | CuPy reference | PCG | 20 | 45.52 ms | 0.958 s | 4.48e-9 |
| 6 -> 0 | CuPy reference | PCG | 22 | 26.19 ms | 0.625 s | 4.27e-9 |
| 6 -> 0 | fused dense BSR | PCG | 22 | 10.67 ms | 0.280 s | 4.27e-9 |
| 6 -> 3 -> 1 -> 0 | CuPy reference | PCGF | 20 | 45.86 ms | 0.959 s | 4.48e-9 |

The two-run direct-schedule medians confirm the earlier single-run result:
fusion reduces the p=6 V-cycle by 59.3% and the hot solve by 55.3%, with unchanged
iterations and true residual. Generic cuSPARSE BSR remains the default for
standalone operator applications such as PCG `A*p`; the fused kernel is the
smoother specialization.

#### Direct p-to-zero degree/mesh sweep

The calibrated radius-5 unstructured disks contain 99,896, 124,831, and
150,209 triangles. Every mesh/degree/backend combination was run twice: 72
successful solves in total. Each reported V-cycle is itself the median of 10
warmed CUDA-event samples. Both paths use direct `p -> 0` coarsening,
order-3 Chebyshev, symmetric `1+1` smoothing, the dedicated fixed scalar-p=0
AMGX cycle, FP64 PCG, and true-residual tolerance 1e-8. CuPy and fused pairs
have identical iteration counts; all residuals pass and symmetry defects remain
at roundoff scale.

The displayed solve time is the hot outer Krylov phase only. For the custom
path it begins immediately before the first preconditioner application and ends
after final FP64 true-residual synchronization. It excludes raw-CUDA assembly,
the modal transform, p-level construction, spectral estimation, diagonal-block
inversion, the scalar-AMGX hierarchy setup, and the separate diagnostic V-cycle
samples. The historical `amgx_solve_seconds` values likewise exclude
`amgx_setup_seconds`, so the comparison below is hot-solve versus hot-solve,
not setup-inclusive or end-to-end time.

| triangles | p | it | CuPy V-cycle (ms) | fused V-cycle (ms) | V speedup | CuPy hot PCG (s) | fused hot PCG (s) | hot speedup | residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99,896 | 1 | 13 | 8.287 | 6.043 | 1.37x | 0.1193 | 0.0900 | 1.33x | 7.45e-09 |
| 99,896 | 2 | 15 | 11.606 | 6.687 | 1.74x | 0.1880 | 0.1142 | 1.65x | 4.75e-09 |
| 99,896 | 3 | 16 | 11.811 | 7.133 | 1.66x | 0.2049 | 0.1292 | 1.59x | 6.38e-09 |
| 99,896 | 4 | 17 | 15.994 | 7.925 | 2.02x | 0.2930 | 0.1558 | 1.88x | 6.74e-09 |
| 99,896 | 5 | 17 | 16.707 | 8.675 | 1.93x | 0.3072 | 0.1703 | 1.80x | 6.78e-09 |
| 99,896 | 6 | 18 | 19.974 | 9.441 | 2.12x | 0.3869 | 0.1977 | 1.96x | 5.87e-09 |
| 124,831 | 1 | 14 | 10.666 | 7.838 | 1.36x | 0.1609 | 0.1217 | 1.32x | 2.65e-09 |
| 124,831 | 2 | 15 | 14.773 | 8.493 | 1.74x | 0.2373 | 0.1436 | 1.65x | 5.74e-09 |
| 124,831 | 3 | 16 | 15.011 | 9.073 | 1.65x | 0.2563 | 0.1617 | 1.59x | 7.44e-09 |
| 124,831 | 4 | 17 | 20.302 | 10.120 | 2.01x | 0.3715 | 0.1964 | 1.89x | 5.86e-09 |
| 124,831 | 5 | 17 | 21.120 | 10.964 | 1.93x | 0.3867 | 0.2151 | 1.80x | 7.76e-09 |
| 124,831 | 6 | 18 | 25.175 | 12.017 | 2.09x | 0.4866 | 0.2510 | 1.94x | 6.38e-09 |
| 150,209 | 1 | 18 | 10.377 | 6.403 | 1.62x | 0.1808 | 0.1245 | 1.45x | 5.11e-09 |
| 150,209 | 2 | 19 | 15.679 | 6.971 | 2.25x | 0.2832 | 0.1498 | 1.89x | 7.89e-09 |
| 150,209 | 3 | 21 | 14.604 | 7.544 | 1.94x | 0.3198 | 0.1779 | 1.80x | 6.07e-09 |
| 150,209 | 4 | 21 | 20.422 | 8.185 | 2.50x | 0.4855 | 0.2058 | 2.36x | 8.20e-09 |
| 150,209 | 5 | 22 | 21.282 | 9.284 | 2.29x | 0.5087 | 0.2456 | 2.07x | 6.91e-09 |
| 150,209 | 6 | 22 | 26.188 | 10.671 | 2.45x | 0.6250 | 0.2797 | 2.23x | 4.27e-09 |

Fusion wins against the transparent CuPy reference in all 18 cases. V-cycle
speedups are 1.36--2.50x and hot-solve speedups are 1.32--2.36x. The gain
generally grows with block size and mesh size, consistent with eliminating
intermediate global vectors and reusing each dense neighboring face vector.

#### Comparison with the previous CSR and hybrid kernels

The table below joins the new hot-solve medians with measured-repeat medians
from `docs/research/solver_studies/classical_amg_bsr_sweep_samples_2026_08.csv`.
It also distinguishes BSR block nonzeros, `nnzb`, from scalar-expanded
nonzeros: `nnz = nnzb * (p+1)^2`. Trace DOFs count only free trace faces after
Dirichlet elimination. Historical paths use PCGF and their recorded
classical-AMG policy; the custom path uses the symmetric PCG/V-cycle above.
These are same-matrix, same-hardware, same-tolerance historical hot-solve
comparisons, not yet interleaved identical-runner timings.

| triangles | p | trace DOFs | BSR nnzb | scalar nnz | CuPy hot PCG (s) | fused hot PCG (s) | historical CSR hot PCGF (s) | historical hybrid hot PCGF (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 99,896 | 1 | 298,952 | 745,908 | 2,983,632 | 0.1193 | 0.0900 | 0.0616 | 0.0576 |
| 99,896 | 2 | 448,428 | 745,908 | 6,713,172 | 0.1880 | 0.1142 | 0.0877 | 0.0550 |
| 99,896 | 3 | 597,904 | 745,908 | 11,934,528 | 0.2049 | 0.1292 | 0.1097 | 0.0831 |
| 99,896 | 4 | 747,380 | 745,908 | 18,647,700 | 0.2930 | 0.1558 | 0.1474 | 0.1466 |
| 99,896 | 5 | 896,856 | 745,908 | 26,852,688 | 0.3072 | 0.1703 | 0.1871 | 0.1521 |
| 99,896 | 6 | 1,046,332 | 745,908 | 36,549,492 | 0.3869 | 0.1977 | 0.2544 | 0.1843 |
| 124,831 | 1 | 373,670 | 932,529 | 3,730,116 | 0.1609 | 0.1217 | 0.0772 | 0.0451 |
| 124,831 | 2 | 560,505 | 932,529 | 8,392,761 | 0.2373 | 0.1436 | 0.1113 | 0.0688 |
| 124,831 | 3 | 747,340 | 932,529 | 14,920,464 | 0.2563 | 0.1617 | 0.1350 | 0.1011 |
| 124,831 | 4 | 934,175 | 932,529 | 23,313,225 | 0.3715 | 0.1964 | 0.1823 | 0.1821 |
| 124,831 | 5 | 1,121,010 | 932,529 | 33,571,044 | 0.3867 | 0.2151 | 0.2344 | 0.1896 |
| 124,831 | 6 | 1,307,845 | 932,529 | 45,693,921 | 0.4866 | 0.2510 | 0.3173 | 0.2290 |
| 150,209 | 1 | 449,724 | 1,122,504 | 4,490,016 | 0.1808 | 0.1245 | 0.0893 | 0.0542 |
| 150,209 | 2 | 674,586 | 1,122,504 | 10,102,536 | 0.2832 | 0.1498 | 0.1414 | 0.0815 |
| 150,209 | 3 | 899,448 | 1,122,504 | 17,960,064 | 0.3198 | 0.1779 | 0.1608 | 0.1216 |
| 150,209 | 4 | 1,124,310 | 1,122,504 | 28,062,600 | 0.4855 | 0.2058 | 0.2183 | 0.2179 |
| 150,209 | 5 | 1,349,172 | 1,122,504 | 40,410,144 | 0.5087 | 0.2456 | 0.2815 | 0.2274 |
| 150,209 | 6 | 1,574,034 | 1,122,504 | 55,002,696 | 0.6250 | 0.2797 | 0.3806 | 0.2739 |

#### Solver setup plus hot solve (assembly excluded)

For completeness, the following adds the one-time solver/preconditioner setup
to the hot Krylov time. Custom setup is `prototype_setup_seconds`: normalized
p-level extraction, dense face-diagonal inversion, spectral estimation, and
the scalar-p=0 AMGX hierarchy setup. Historical setup is
`amgx_setup_seconds`. PDE assembly and reconstruction remain excluded.
Every hot PCG/PCGF value already contains all preconditioner applications made
during Krylov iteration. Runtime/NVRTC compilation and the separately timed
diagnostic V-cycle samples are not charged to this solver total.

| triangles | p | CuPy setup + hot PCG (s) | fused setup + hot PCG (s) | historical CSR setup + hot PCGF (s) | historical hybrid setup + hot PCGF (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 99,896 | 1 | 1.3172 | 1.2902 | 0.2244 | 0.2224 |
| 99,896 | 2 | 1.3859 | 1.3156 | 0.4334 | 0.0914 |
| 99,896 | 3 | 1.4044 | 1.3494 | 0.1806 | 0.1282 |
| 99,896 | 4 | 1.4650 | 1.3352 | 0.2289 | 0.2109 |
| 99,896 | 5 | 1.4905 | 1.3398 | 0.4389 | 0.2430 |
| 99,896 | 6 | 1.5574 | 1.3779 | 0.5042 | 0.3004 |
| 124,831 | 1 | 2.2216 | 2.1876 | 0.3524 | 0.0741 |
| 124,831 | 2 | 2.3134 | 2.2359 | 0.7356 | 0.1171 |
| 124,831 | 3 | 2.3320 | 2.2337 | 0.2406 | 0.1536 |
| 124,831 | 4 | 2.4030 | 2.2256 | 0.3019 | 0.2623 |
| 124,831 | 5 | 2.4294 | 2.2591 | 0.6634 | 0.3021 |
| 124,831 | 6 | 2.5451 | 2.3071 | 0.7291 | 0.3764 |
| 150,209 | 1 | 0.4748 | 0.4179 | 0.5448 | 0.0883 |
| 150,209 | 2 | 0.5910 | 0.4502 | 1.1853 | 0.1393 |
| 150,209 | 3 | 0.6225 | 0.4702 | 0.3106 | 0.1839 |
| 150,209 | 4 | 0.7724 | 0.4980 | 0.3870 | 0.3118 |
| 150,209 | 5 | 0.7896 | 0.5363 | 0.9822 | 0.3631 |
| 150,209 | 6 | 0.9292 | 0.5954 | 1.0425 | 0.4512 |

These setup-inclusive measurements change the conclusion for one-shot solves:
the established hybrid path wins all 18 cases. The custom setup also shows
strong nonmonotonic mesh dependence in these samples, so setup optimization and
an interleaved cold/warm benchmark are required before drawing scaling claims.
The hot table remains the relevant amortized comparison when a fixed Poisson
operator and hierarchy are reused for many right-hand sides.

#### Chebyshev screen and zero-start fast path

A focused 150,209-triangle p=4--6 screen selected direct `p -> 0`, order-2
Chebyshev, symmetric `1+1` smoothing, and ordinary PCG. Relative to order 3,
order 2 needs one or two extra iterations but removes two dense-BSR smoother
stages per V-cycle and wins in hot time. Diagnostic `0+3` smoothing has
symmetry defects between 2e-6 and 2e-5, requires PCGF, and is slower. The
halving schedules reduce the iteration count to 19--20 but nearly double the
V-cycle cost. Block-L1 was not pursued because full face-block Jacobi remains
SPD and satisfies the iteration gate.

The first Phase-3 optimization exploits the provably zero correction at the
start of every V-cycle. Its first pre-smoothing stage now evaluates
`x = omega * D_face^{-1} rhs` with a dedicated warp kernel, without reading the
BSR operator or forming `A * 0`; it also avoids zero-filling that initial output
buffer. Two measurements per degree, with one p=5 tie-breaker for host jitter,
give:

| p | iterations | Cheb-2 before (s) | zero-start V-cycle (ms) | zero-start hot PCG (s) | historical hybrid (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 23 | 0.1997 | 5.407 | 0.1582 | 0.2179 |
| 5 | 24 | 0.2071 | 5.965 | 0.1847 | 0.2274 |
| 6 | 23 | 0.2257 | 6.655 | 0.2007 | 0.2739 |

True relative residuals are unchanged at 4.42e-9, 2.31e-9, and 5.56e-9. The
V-cycle improves by 23% at p=4 and about 14% at p=5,6; hot solve improves by
21%, 11%, and 11% against the pre-optimization Cheb-2 samples. Against the
historical hybrid hot medians, the optimized path is 27%, 19%, and 27% faster.
These are promising but not yet the required identical-runner interleaved
comparison.

Persistent per-level correction/scratch/residual/coarse-RHS storage then
removes solve-time V-cycle vector allocation, performs `rhs-A*x` in place,
restricts directly into the retained low-mode buffer, and adds the coarse
correction directly into the low Legendre coefficients without materializing a
full prolongation vector. The returned correction is borrowed workspace storage;
the prototype is deliberately serial and rejects reentrant application.

| triangles | p | trace DOFs | BSR nnzb | scalar nnz | workspace MiB | zero-start hot PCG (s) | persistent hot PCG (s) | gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150,209 | 4 | 1,124,310 | 1,122,504 | 28,062,600 | 27.449 | 0.1582 | 0.1562 | 1.3% |
| 150,209 | 5 | 1,349,172 | 1,122,504 | 40,410,144 | 32.596 | 0.1847 | 0.1828 | 1.0% |
| 150,209 | 6 | 1,574,034 | 1,122,504 | 55,002,696 | 37.742 | 0.2007 | 0.1977 | 1.5% |

The corresponding V-cycle medians are 5.311, 5.880, and 6.526 ms, improvements
of 1.8%, 1.4%, and 1.9%. Iterations and true residuals remain exactly unchanged.
The modest gain is consistent with CuPy's caching allocator already making raw
allocation inexpensive; the persistent storage is still required groundwork
for transfer fusion and CUDA-graph capture.

A subsequent directly restricted residual experiment computed only the retained
modal rows of `rhs-A*x`. The first one-face-per-warp version was slower. A
power-of-two subwarp revision processed four p=4--6 faces per warp, but still did
not beat the full generic-cuSPARSE BSR residual consistently:

| triangles | p | trace DOFs | BSR nnzb | scalar nnz | cuSPARSE full V-cycle (ms) | custom restricted V-cycle (ms) | cuSPARSE hot PCG (s) | custom hot PCG (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150,209 | 4 | 1,124,310 | 1,122,504 | 28,062,600 | 5.311 | 5.534 | 0.1562 | 0.1617 |
| 150,209 | 5 | 1,349,172 | 1,122,504 | 40,410,144 | 5.880 | 5.688 | 0.1828 | 0.2019 |
| 150,209 | 6 | 1,574,034 | 1,122,504 | 55,002,696 | 6.526 | 6.430 | 0.1977 | 0.1990 |

The p=5 hot custom sample contains visible host/synchronization jitter, but even
the CUDA-event V-cycle results show only small mixed changes (-4.2%, +3.3%,
+1.5%). Reducing coefficient rows did not overcome the custom kernel's lower
utilization relative to cuSPARSE. The custom kernel and configuration surface
were therefore removed; all ordinary and residual BSR SpMV now rely on the
cached generic-cuSPARSE implementation. The restored-default confirmation gives
0.1555, 0.1818, and 0.1976 s with unchanged iterations and residuals.

The degree-dependent conclusion is now:

- At p=1--3, keep the established hybrid path until the same optimization is
  measured there; fixed V-cycle and coarse-wrapper costs dominate small blocks.
- At p=4--6, direct-to-zero Cheb-2 block-Jacobi is the selected custom policy.
- The directly restricted residual was tested and rejected; generic cuSPARSE
  remains the only BSR SpMV implementation. Keep the fixed-five-slot structure
  as assembly/topology metadata, not as another SpMV kernel.
- The next hot-path targets are persistent cuSPARSE descriptor/preprocess reuse,
  fused PCG vector updates and reductions, CUDA-graph capture where the solver
  stack permits it, and an identical-runner interleaved comparison.
- Production promotion still requires setup reuse/optimization and an
  end-to-end win under the common solver contract.

The dedicated p=0 AMGX cycle is scalar classical PMIS/D2 with one
`JACOBI_L1` pre/post sweep, no aggressive level, no correction scaling, and
fixed one-cycle work. It must not inherit the full-order nodal Chebyshev preset.
The inherited preset produced a 9.61e-5 symmetry defect; AMGX-matching PCGF
still converged, but required 30 iterations and 1.477 s. This confirms the known
result that the nodal-tuned PCGF/AMG configuration should not be judged or used
as a full-order modal Legendre baseline. The fair incumbent remains full AMGX
on nodal trace coordinates.

## Discrete Representation

For degree p, one face has b=p+1 unknowns and the trace operator is

```text
A_p = [A_fg],       A_fg in R^(b x b),       at most five blocks per row
```

for a two-dimensional manifold triangular mesh. Full row storage is retained
for race-free row-owned SpMV.

The assembly basis is the current Legendre basis `P_j`. The normalized solver
basis is

```text
psi_j = sqrt((2*j + 1)/2) * P_j.
```

With `S=diag(sqrt((2*j+1)/2))`, transform once as

```text
A_modal = S A S,       g_modal = S g,       a_assembly = S a_modal.
```

Face reversal acts diagonally by `(-1)^j`; the existing modal raw-CUDA assembly
already implements this orientation convention. For a coarse degree pc, nested
injection retains modes 0..pc. Hence the Galerkin block is the leading
`(pc+1) x (pc+1)` principal subblock. This is a Galerkin trace level, not a
freshly condensed lower-degree HDG discretization.

Default degree schedule:

```text
p -> floor(p/2) -> ... -> 1 -> 0
```

The prototype must also retain `p -> 0` as an ablation.

## V-Cycle

On every p>0 level, use the SPD face diagonal `M=blockdiag(A_ff)` and a fixed
Chebyshev polynomial in `M^-1 A`. Estimate the upper spectral bound once per
level, inflate it for safety, and use a fixed lower fraction. Start with orders
2 and 3. Apply one pre-smoothing polynomial and the reversed adjoint sequence
after coarse correction. Restriction truncates high modal coefficients;
prolongation injects low coefficients.

At p=0, apply one reusable classical AMGX V-cycle on the scalar face graph. The
initial numerical prototype may synchronize and allocate in its Python wrapper;
those costs must be reported separately and must not be interpreted as the
production hot-solve time.

The outer convergence contract remains the repository contract:

```text
||b - A x|| <= max(atol, rtol * ||b||).
```

Also report `||r_k||/||r_0||`, but do not substitute it for the contract above.
Recompute the true FP64 residual periodically and before accepting a solve.

## GPU Operator Policy

Use direct face BSR throughout p-level work. For SpMV:

1. prefer CUDA 13 generic BSR `cusparseSpMV` when the runtime supports it;
2. cache matrix/dense-vector descriptors, preprocessing state, and workspace
   for the lifetime of the matrix structure;
3. retain a row-owned specialized raw-CUDA kernel as a correctness fallback and
   benchmark control;
4. specialize block sizes 1..10, prioritizing b=2..7 now and b=8..10 for
   Poisson p=7..9;
5. invalidate cached state on structure, pointer, value-type, block order,
   device, or incompatible-stream changes.

The installed CuPy wheel currently exposes generic CSR but not a public BSR
matrix class or `cusparseCreateBsr`, so the HDG prototype uses a narrow local C
binding to the loaded cuSPARSE library. It must fall back cleanly when the
symbol or runtime support is absent; changing or rebuilding CuPy is not required
for this phase.

## Phased Roadmap

### Phase 0 — freeze baselines

- Record scalar CSR, coefficient-exact hybrid fine-BSR/scalar-hierarchy, and
  dense pure-BSR results with CUDA-event timings and independent residuals.
- Treat the hybrid path as the primary target and scalar CSR as the portability
  reference.

### Phase 1 — validate modal face BSR

- Verify direct BSR/CSR matvec parity, bilinear symmetry, positive Rayleigh
  quotients, diagonal location, five-block topology, and orientation changes.
- Verify normalized-basis round trips and transformed operator equivalence.

### Phase 2 — inexpensive numerical prototype

- Build p-levels by principal modal-block extraction on CuPy arrays.
- Compare halving and direct-to-zero schedules.
- Compare order-2/order-3 Chebyshev, `1+1` and diagnostic `0+3` smoothing, and
  block Jacobi versus block-L1 only if Jacobi fails.
- Reuse one dedicated scalar AMGX hierarchy at p=0; do not inherit the
  full-order nodal smoother/coarsening preset.
- Measure V-cycle symmetry and positive curvature before selecting PCG; retain
  the AMGX Polak--Ribiere PCGF recurrence as the diagnostic fallback.
- Compare iteration counts with the fine-BSR/scalar-AMG baseline before any
  production integration.

### Phase 3 — optimize p-level primitives

- Cache generic-cuSPARSE BSR descriptors/preprocessing/workspace.
- Benchmark against specialized row-owned kernels for b=1..10.
- [x] Fuse face SpMV, residual, dense diagonal-block action, and Chebyshev
  update after the unfused reference passes parity. GPU parity covers block
  sizes 2, 5, 7, and 10; the production p=6 result is recorded above.
- Remove solve-time allocations and host synchronization.

### Phase 4 — establish the SPD contract

- Enforce `R=P^T`, reversed adjoint post-smoothing, fixed spectral intervals,
  fixed coarse work, and an SPD terminal solve.
- Add numerical preconditioner-adjointness, positive-curvature, and PCG-vs-
  trusted-solve tests.

### Phase 5 — production FB-HP-MG with AMGX at p=0

- Add a backend selected independently of the existing AMGX CSR/BSR paths.
- Reuse p-level matrices, p=0 AMG hierarchy, coarse factors, and warm trace
  guesses over repeated Poisson right-hand sides.
- Keep scalar CSR/hybrid dispatch for low degree when measurements favor it.

### Phase 6 — profile and fuse

- Capture the fixed V-cycle in CUDA graphs, fuse compatible PCG updates and
  reductions, and publish per-level timing/operator-complexity data.
- Implement a custom scalar h-hierarchy only if p=0 AMGX is a measured hot-path
  bottleneck; it is not a prerequisite for validating p-coarsening.

### Phase 7 — qualification and dispatch

- Sweep the radius-5 unstructured trigonometric-Poisson disk at production
  sizes and p=1..9.
- Select the degree-dependent dispatcher from end-to-end warm-solve evidence,
  not from SpMV alone.

## Acceptance Gates

Correctness gates:

- BSR/CSR matvec relative difference <= 5e-13 in FP64;
- bilinear symmetry defect <= 1e-13 on representative matrices;
- modal transform and transfer adjointness near FP64 rounding error;
- every Galerkin level has positive sampled Rayleigh quotients;
- final true residual satisfies the common solver contract;
- reconstructed HDG fields agree with the trusted trace solution.

Performance gates:

- initial prototype iterations no more than 1.25 times the hybrid baseline;
- p>=4 fine SpMV target at least 1.4 times scalar CSR after setup is amortized;
- projected, then measured, warm solve at least 10% faster than scalar CSR;
- production promotion requires beating the hybrid path on the same matrix,
  right-hand side, initial guess, tolerance, precision, and GPU.

No method is promoted because of a faster standalone SpMV while its complete
solve is slower or less accurate.

## Long-Term Degree Extension

The BSR route is the priority for raw-CUDA Poisson p=7,8,9. Extend local
assembly limits and shared-memory/cooperative-solve policies for element degrees
36, 45, and 55 only after recording occupancy and capacity. Direct BSR block
sizes 8, 9, and 10, modal orientation, reconstruction, cached-RHS assembly, and
BSR-vs-host/CSR parity are required. COO/expanded CSR support at those degrees
is a secondary validation/debug path, not the production optimization target.
