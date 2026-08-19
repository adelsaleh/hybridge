# Classical AMG With A BSR Fine Operator: August 2026

This study validates the hybrid classical-AMG algorithm described in
[`../../backends/amgx_classical_bsr.md`](../../backends/amgx_classical_bsr.md).
It is dated performance evidence, not a backend support or default-policy
claim.

## Environment And Case

- GPU: NVIDIA Quadro RTX 6000, compute capability 7.5;
- driver: 580.173.02;
- CUDA Toolkit/runtime used to build AMGX: 13.0;
- AMGX: 2.5.0 with the local classical-BSR compatibility changes;
- Python: 3.12.3;
- CuPy: 14.1.1;
- domain: unstructured disk of radius 5;
- PDE: trigonometric Poisson manufactured solution;
- diffusion stabilization: global length scale;
- AMGX configuration:
  `configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json`.

Both paths used raw-CUDA HDG assembly and the same PCGF/classical-AMG
configuration: PMIS, D2 interpolation, Chebyshev acceleration, a V-cycle, and
zero/three pre/post smoothing sweeps. Only the fine-operator storage path was
changed between CSR and BSR.

The comparison driver supports repeated, order-alternating measurements:

```bash
python -m scripts.diffusion_reaction.compare_cuda_bsr_csr \
  --domain disk \
  --radius 5 \
  --mesh-size 0.0348 \
  --order 2 \
  --only both \
  --bsr-amgx-config \
    configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json \
  --repeats 3 \
  --format-order alternate \
  --csv /tmp/classical-amg-bsr-p2.csv \
  --verbosity 0
```

## Initial Spot Check

The initial 152,909-triangle, degree-2 paired run used a face block size of 3
and 686,724 reduced trace unknowns. It found 19 CSR iterations and 20 BSR
iterations, independently checked physical relative residuals of `8.822e-09`
and `4.666e-09`, and an AMGX solve time of 0.150 s and 0.116 s respectively.
The BSR/CSR primal-coefficient relative difference was `8.733e-09`. Its
CSR-first setup and total timings included warmup bias, so the sweep below
supersedes them for timing conclusions.

## Repeated Sweep Protocol

Three radius-5 meshes were calibrated to 99,896, 124,831, and 150,209
triangles. Degrees 1 through 6 correspond to face block sizes 2 through 7.
Every mesh/degree pair was run three times for each format, giving 108
successful solves. Repeat 1 was retained as warmup and parity evidence.
Repeats 2 and 3 reversed the format order; their median is the timing reported
below. Because there are two measured samples, this median is their midpoint.

The complete per-run data, including assembly, format conversion, setup,
solve, reconstruction, residual, and parity fields, are in
[`classical_amg_bsr_sweep_samples_2026_08.csv`](classical_amg_bsr_sweep_samples_2026_08.csv).

All independently checked physical relative residuals were at most
`9.843e-09`. The largest relative primal-coefficient difference between the
independently stopped CSR and BSR solves was `1.467e-08`, consistent with the
approximately `1e-8` stopping tolerance. Iteration counts were deterministic
within each mesh/degree/format case.

### Degree Summary

Ratios greater than one favor BSR. Each entry is the median across the three
meshes; the range follows in parentheses. `Pattern` is the BSR/CSR compressed
index-storage ratio.

| Degree | Block | Setup CSR/BSR | Solve CSR/BSR | Total CSR/BSR | Pattern |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 9.49 (0.99–13.36) | 1.65 (1.07–1.71) | 2.51 (1.00–3.03) | 27.28% |
| 2 | 3 | 12.91 (9.49–18.06) | 1.62 (1.59–1.74) | 3.19 (2.38–4.25) | 12.50% |
| 3 | 4 | 2.01 (1.57–2.40) | 1.32 (1.32–1.34) | 1.21 (1.14–1.26) | 7.14% |
| 4 | 5 | 1.49 (1.27–1.80) | 1.00 (1.00–1.01) | 1.06 (1.04–1.10) | 4.62% |
| 5 | 6 | 3.81 (2.77–5.16) | 1.24 (1.23–1.24) | 1.40 (1.27–1.58) | 3.23% |
| 6 | 7 | 2.79 (2.15–3.73) | 1.39 (1.38–1.39) | 1.25 (1.19–1.35) | 2.38% |

### Per-Case Warm Timings

`It C/B` gives CSR and BSR iteration counts. Setup, solve, and total times are
seconds. `Solve x` and `Total x` are CSR/BSR ratios.

| Triangles | p | Trace DOFs | It C/B | Setup C/B | Solve C/B | Solve x | Total C/B | Total x |
|---:|---:|---:|:---:|:---:|:---:|---:|:---:|---:|
| 99,896 | 1 | 298,952 | 21/21 | 0.163/0.165 | 0.0616/0.0576 | 1.07 | 0.328/0.329 | 1.00 |
| 99,896 | 2 | 448,428 | 19/20 | 0.346/0.036 | 0.0877/0.0550 | 1.59 | 0.577/0.242 | 2.38 |
| 99,896 | 3 | 597,904 | 21/22 | 0.071/0.045 | 0.1097/0.0831 | 1.32 | 0.444/0.389 | 1.14 |
| 99,896 | 4 | 747,380 | 21/22 | 0.082/0.064 | 0.1474/0.1466 | 1.01 | 0.576/0.553 | 1.04 |
| 99,896 | 5 | 896,856 | 19/20 | 0.252/0.091 | 0.1871/0.1521 | 1.23 | 0.967/0.759 | 1.27 |
| 99,896 | 6 | 1,046,332 | 21/22 | 0.250/0.116 | 0.2544/0.1843 | 1.38 | 1.352/1.135 | 1.19 |
| 124,831 | 1 | 373,670 | 21/21 | 0.275/0.029 | 0.0772/0.0451 | 1.71 | 0.471/0.188 | 2.51 |
| 124,831 | 2 | 560,505 | 19/20 | 0.624/0.048 | 0.1113/0.0688 | 1.62 | 0.903/0.284 | 3.19 |
| 124,831 | 3 | 747,340 | 21/22 | 0.106/0.053 | 0.1350/0.1011 | 1.34 | 0.540/0.448 | 1.21 |
| 124,831 | 4 | 934,175 | 21/22 | 0.120/0.080 | 0.1823/0.1821 | 1.00 | 0.713/0.674 | 1.06 |
| 124,831 | 5 | 1,121,010 | 19/20 | 0.429/0.113 | 0.2344/0.1896 | 1.24 | 1.305/0.933 | 1.40 |
| 124,831 | 6 | 1,307,845 | 21/22 | 0.412/0.147 | 0.3173/0.2290 | 1.39 | 1.759/1.413 | 1.25 |
| 150,209 | 1 | 449,724 | 20/21 | 0.456/0.034 | 0.0893/0.0542 | 1.65 | 0.671/0.221 | 3.03 |
| 150,209 | 2 | 674,586 | 20/20 | 1.044/0.058 | 0.1414/0.0815 | 1.73 | 1.380/0.325 | 4.25 |
| 150,209 | 3 | 899,448 | 21/22 | 0.150/0.062 | 0.1608/0.1216 | 1.32 | 0.644/0.512 | 1.26 |
| 150,209 | 4 | 1,124,310 | 21/22 | 0.169/0.094 | 0.2183/0.2179 | 1.00 | 0.886/0.803 | 1.10 |
| 150,209 | 5 | 1,349,172 | 19/20 | 0.701/0.136 | 0.2815/0.2274 | 1.24 | 1.731/1.097 | 1.58 |
| 150,209 | 6 | 1,574,034 | 21/22 | 0.662/0.177 | 0.3806/0.2739 | 1.39 | 2.354/1.745 | 1.35 |

## Findings

- BSR preserved practical convergence parity. It usually required one more
  PCGF iteration, with several equal-iteration cases.
- The degree-2 case was the strongest and most consistent overall result:
  BSR solved 1.59–1.74 times faster and reduced measured total time by
  2.38–4.25 times.
- Degree 3 gained about 1.32 times in solve time, degree 5 about 1.23 times,
  and degree 6 about 1.38 times. Degree 4 was effectively neutral in solve
  time despite its smaller index storage.
- Degree 1 was neutral on the smallest mesh but gained 1.65–1.71 times in
  solve time on the two larger meshes. Its setup ratios are especially
  allocation-sensitive and should not be generalized from this range.
- Compressed index storage fell from 27.28% of CSR at block size 2 to 2.38%
  at block size 7. Less index traffic did not by itself guarantee faster
  SpMV or solve, as the block-size-5 result demonstrates.
- Measured-repeat stability was good: the largest max/min ratios were 1.011
  for BSR setup, 1.008 for CSR setup, 1.028 for BSR solve, and 1.004 for CSR
  solve. Setup speedups still vary strongly with degree and allocation path,
  so solve time is the safer performance conclusion.

## Conclusion

Across 108 successful solves, the BSR fine operator reproduced the classical
scalar hierarchy's practical convergence while retaining BSR for fine SpMV
and scalar-row Jacobi-L1. It materially improved solve time for block sizes 2,
3, 4, 6, and 7 in this test range, but was neutral at block size 5. It does
not establish a pure-BSR hierarchy: the transfer operators and coarse systems
remain scalar CSR during every V-cycle. The next implementation step remains
the block-graph D2 construction with scalar block weights lifted as
`P_ic = w_ic I_b`.


## Pure-BSR Identity-Lift Follow-Up: 2026-08-19

This follow-up evaluates the opt-in `block_graph_identity` hierarchy described
in the backend document. The initial large sweep compared the production CSR
reference (`PCGF` with coefficient-exact scalar classical AMG) with pure-BSR
`FGMRES` using block-graph D2 and `P_ic=w_ic I_b`. That conservative solver
choice was based on a pre-fix p=6 PCGF breakdown. After correcting the
multilevel vector overrun, a second large sweep established that pure-BSR PCGF
is stable for every tested degree.

### Memory-Safety Regression

A three-level p=6 reproducer exposed an out-of-bounds correction update. The
pure-BSR prolongation passed `P.num_rows*b` to AMGX `axpby`, whose size unit is
already a vector block, so `axpby` multiplied by `b` a second time. For p=6
this produced a sevenfold overrun and eventually corrupted pooled transfer
storage. The fix passes `P.num_rows`, adds an explicit vector-extent check, and
launches the new block-graph kernels on AMGX's configured stream.

After the fix, the 6,369-triangle p=6 reproducer completed at relative residual
`6.833e-9`. CUDA Initcheck and CUDA Memcheck both reported `ERROR SUMMARY: 0
errors`. Tightening the paired CSR/BSR solves from `rtol=1e-8` to `1e-10`
reduced their primal-coefficient relative difference from `1.122e-6` to
`1.187e-8`, confirming convergence toward the same algebraic solution.

### Sweep Protocol

The calibrated radius-5 meshes contain 99,896, 124,831, and 150,209 triangles.
Degrees 1 through 6 give BSR block sizes 2 through 7. Each mesh/degree pair was
run twice per format with alternating format order, giving 72 successful
solves. The tables report the median of the two repetitions. The complete
records are in
[`classical_amg_pure_bsr_sweep_samples_2026_08.csv`](classical_amg_pure_bsr_sweep_samples_2026_08.csv).

All independently checked relative residuals were at most `9.911e-9`. The
largest coefficient relative difference between independently stopped CSR and
BSR runs was `2.577e-5`. That coefficient difference is affected by operator
conditioning and the different Krylov methods; the tighter small-case result
above is the direct parity check.

Ratios greater than one favor BSR. The `Pattern` column is BSR/CSR compressed
index storage.

| Degree | Block | Setup CSR/BSR | Solve CSR/BSR | Total CSR/BSR | Pattern | BSR iterations |
|---:|---:|:---:|:---:|:---:|---:|:---:|
| 1 | 2 | 7.083 (0.349–8.811) | 0.040 (0.040–0.054) | 0.442 (0.351–0.549) | 27.28% | 375–560 |
| 2 | 3 | 7.135 (6.142–7.848) | 0.037 (0.035–0.042) | 0.430 (0.382–0.433) | 12.50% | 488–596 |
| 3 | 4 | 0.652 (0.527–0.789) | 0.029 (0.024–0.030) | 0.184 (0.144–0.190) | 7.14% | 619–754 |
| 4 | 5 | 0.368 (0.310–0.484) | 0.018 (0.017–0.022) | 0.108 (0.096–0.138) | 4.62% | 690–900 |
| 5 | 6 | 0.624 (0.614–0.712) | 0.018 (0.017–0.024) | 0.120 (0.120–0.160) | 3.23% | 701–956 |
| 6 | 7 | 0.479 (0.404–7.456) | 0.022 (0.019–0.024) | 0.153 (0.126–0.157) | 2.38% | 777–959 |

### Per-Case Timings

Times are seconds. `It C/B` gives CSR PCGF and pure-BSR FGMRES iterations.

| Triangles | p | Trace DOFs | It C/B | Setup C/B | Solve C/B | Total C/B | Coeff. diff. |
|---:|---:|---:|:---:|:---:|:---:|:---:|---:|
| 99,896 | 1 | 298,952 | 21/375 | 0.213/0.610 | 0.070/1.730 | 0.866/2.465 | 1.210e-05 |
| 99,896 | 2 | 448,428 | 19/488 | 0.391/0.064 | 0.095/2.263 | 1.070/2.489 | 1.807e-05 |
| 99,896 | 3 | 597,904 | 21/630 | 0.104/0.132 | 0.121/4.118 | 0.860/4.517 | 2.098e-05 |
| 99,896 | 4 | 747,380 | 21/690 | 0.105/0.217 | 0.155/6.958 | 1.045/7.555 | 7.299e-06 |
| 99,896 | 5 | 896,856 | 19/701 | 0.278/0.390 | 0.196/8.312 | 1.478/9.265 | 1.037e-05 |
| 99,896 | 6 | 1,046,332 | 21/777 | 0.272/0.568 | 0.263/10.916 | 1.886/12.337 | 1.360e-05 |
| 124,831 | 1 | 373,670 | 21/447 | 0.321/0.045 | 0.086/1.579 | 0.961/1.751 | 1.083e-05 |
| 124,831 | 2 | 560,505 | 19/596 | 0.683/0.096 | 0.119/3.383 | 1.398/3.660 | 1.649e-05 |
| 124,831 | 3 | 747,340 | 21/619 | 0.144/0.221 | 0.149/5.004 | 1.021/5.537 | 2.459e-05 |
| 124,831 | 4 | 934,175 | 21/824 | 0.143/0.389 | 0.190/10.408 | 1.219/11.244 | 9.651e-06 |
| 124,831 | 5 | 1,121,010 | 19/927 | 0.455/0.741 | 0.242/13.661 | 1.813/15.077 | 1.652e-05 |
| 124,831 | 6 | 1,307,845 | 21/953 | 0.439/1.088 | 0.326/16.769 | 2.381/18.920 | 1.304e-05 |
| 150,209 | 1 | 449,724 | 20/560 | 0.515/0.058 | 0.097/2.443 | 1.166/2.640 | 1.539e-05 |
| 150,209 | 2 | 674,586 | 20/594 | 1.105/0.141 | 0.149/4.069 | 1.907/4.405 | 1.896e-05 |
| 150,209 | 3 | 899,448 | 21/754 | 0.191/0.363 | 0.172/7.279 | 1.154/7.986 | 2.577e-05 |
| 150,209 | 4 | 1,124,310 | 21/900 | 0.191/0.617 | 0.226/13.614 | 1.419/14.753 | 7.928e-06 |
| 150,209 | 5 | 1,349,172 | 19/956 | 0.731/1.171 | 0.289/16.841 | 2.256/18.771 | 1.535e-05 |
| 150,209 | 6 | 1,574,034 | 21/959 | 0.690/0.093 | 0.389/17.570 | 2.979/18.979 | 1.515e-05 |

### Pure-BSR Findings

- The repaired implementation is memory-safe across all tested block sizes and
  multilevel hierarchies.
- Identity-lifted interpolation is not a competitive production
  preconditioner. It required 375–959 FGMRES iterations, compared with 19–21
  PCGF iterations for coefficient-exact scalar classical AMG.
- Pure BSR reduced compressed index storage from 27.28% of CSR at block size 2
  to 2.38% at block size 7, but its solves were 18–59 times slower. Structural
  compression alone cannot compensate for a weak coarse correction.
- Setup was sometimes much faster for block sizes 2 and 3 and highly variable
  with the hierarchy selected by the Frobenius block graph. Solve time
  dominates, so these setup reductions do not change the conclusion.
- `block_graph_identity` should remain experimental. Its dense-interpolation
  successor is implemented and validated below; it repairs most of the coarse
  approximation deficit but is not yet competitive with the hybrid path for
  block sizes four through seven.


### Pure-BSR PCGF Follow-Up

After the correction-size fix, PCGF was run once on all 18 large
mesh/degree cases using
`diff_rea_gpu4_hdg_pcgf_cheb_l1_block_graph_identity_bsr.json`. All cases
converged with independently checked relative residual at most `9.869e-9`.
The samples are in
[`classical_amg_pure_bsr_pcgf_samples_2026_08.csv`](classical_amg_pure_bsr_pcgf_samples_2026_08.csv).

`FGMRES/PCGF speed` is the previous two-run median FGMRES solve time divided by
the exploratory PCGF solve time. `PCGF/CSR slowdown` compares PCGF pure BSR
with the previous scalar-CSR solve median. PCGF is consistently better than
FGMRES, but the identity-lifted hierarchy remains far slower than the
coefficient-exact scalar hierarchy.

| p | PCGF iterations | FGMRES/PCGF speed | PCGF/CSR slowdown |
|---:|:---:|:---:|:---:|
| 1 | 312–355 | 1.57 (1.34–1.99) | 12.6 (11.8–18.6) |
| 2 | 395–475 | 1.62 (1.56–1.79) | 15.9 (15.2–16.8) |
| 3 | 423–504 | 1.87 (1.65–1.88) | 20.3 (18.1–22.7) |
| 4 | 473–562 | 1.85 (1.75–1.89) | 29.7 (25.7–31.9) |
| 5 | 493–654 | 1.72 (1.70–1.85) | 30.4 (24.9–33.9) |
| 6 | 500–590 | 1.94 (1.84–2.04) | 23.3 (22.5–25.2) |

On the paired 150,209-triangle p=6 case, scalar CSR PCGF took 21 iterations and
`0.392 s`; pure-BSR PCGF took 590 iterations and `9.034 s`. Their independently
checked relative residuals were `9.843e-9` and `4.302e-9`, and their primal
coefficient relative difference was `3.962e-7`. Compared with the hybrid
BSR/scalar-hierarchy measurements, pure-BSR PCGF was 22.3--43.1 times slower
across individual cases; the largest p=6 case was 33.1 times slower (`9.062 s`
versus `0.274 s`). The earlier assertion that the identity-lifted hierarchy was
not CG-compatible was therefore an artifact of the out-of-bounds correction
update. The remaining defect is coarse-space quality.

### Dense Block Interpolation: First Validation

The new `block_graph_dense` mode starts from the same D2 support and applies

\[
\widetilde P=P_0-\omega D^{-1}AP_0,
\qquad (P_0)_{ic}=w_{ic}I_b.
\]

It then corrects each block row so that `sum_c P_ic = I_b`, constructs the
exact conjugate transpose, and evaluates the dense BSR Galerkin product
`P^* A P`. The identity-lifted mode is unchanged, so the two modes isolate
the effect of block-valued interpolation on an identical coarse block graph.

The first multilevel runs exposed one support restriction: the existing
aggressive D2 pass can leave fine block rows with no interpolation entry. Such
a row cannot satisfy `sum_c P_ic = I_b`, and relaxing the numerical constraint
tolerance merely hid the structural error. The production dense configuration
therefore uses `aggressive_levels=0`; AMGX rejects the unsupported combination
explicitly until support completion is implemented.

The first radius-5 trigonometric-Poisson results used 4,658 triangles. At the
ordinary exploratory tolerance, p=2 dense interpolation reduced PCGF from 91
identity-hierarchy iterations to 14 (0.072 s versus 0.157 s solve), and p=6
reduced it from 117 to 22 (0.106 s versus 0.306 s). These identity comparisons
include its prior aggressive first level and are therefore measures of the
complete configurations, not interpolation-only timings.

For the stricter p=6 parity run at relative tolerance `1e-10`, scalar CSR used
25 iterations, 0.131 s setup, and 0.065 s solve. Dense pure BSR used 27
iterations, 0.077 s setup, and 0.098 s solve. The reconstructed primal
coefficient vectors differed by `1.476e-10`; BSR compressed-pattern storage was
2.38% of CSR's. This is the key numerical result: block interpolation repairs
the coarse-space quality deficit of `w_ic I_b`, but the pure-BSR V-cycle is not
yet faster than the highly optimized scalar CSR cycle.

### Dense Pure-BSR Three-Way Large-Mesh Sweep

The full comparison used the unstructured radius-5 trigonometric-Poisson disk
with 152,909 triangles, relative tolerance `1e-8`, and two solves per path in
reversed order. The three paths were scalar CSR with scalar classical AMG,
face BSR with the coefficient-exact scalar hierarchy (`scalar_expand`, called
hybrid below), and face BSR with the dense pure-BSR hierarchy. The complete
samples are in
[`classical_amg_dense_bsr_three_way_samples_2026_08.csv`](classical_amg_dense_bsr_three_way_samples_2026_08.csv).

Times below are medians of the two runs. Triples are
`CSR / hybrid / dense pure BSR`.

| p | trace DOFs | iterations | AMGX setup (s) | AMGX solve (s) | pure/hybrid solve | BSR/CSR pattern |
|---:|---:|:---:|:---:|:---:|---:|---:|
| 1 | 457,816 | 21 / 21 / 15 | 0.519 / 0.035 / 0.440 | 0.102 / 0.069 / 0.086 | 1.25 | 27.28% |
| 2 | 686,724 | 19 / 20 / 20 | 1.086 / 0.060 / 1.126 | 0.137 / 0.083 / 0.175 | 2.11 | 12.50% |
| 3 | 915,632 | 21 / 22 / 39 | 0.153 / 0.064 / 0.250 | 0.163 / 0.121 / 0.369 | 3.05 | 7.14% |
| 4 | 1,144,540 | 21 / 23 / 43 | 0.176 / 0.094 / 0.347 | 0.222 / 0.229 / 0.710 | 3.11 | 4.62% |
| 5 | 1,373,448 | 19 / 20 / 49 | 0.728 / 0.139 / 0.466 | 0.284 / 0.228 / 0.929 | 4.07 | 3.23% |
| 6 | 1,602,356 | 21 / 22 / 53 | 0.702 / 0.178 / 0.640 | 0.387 / 0.277 / 1.231 | 4.45 | 2.38% |

All 36 solves converged. The maximum independently reconstructed physical
relative residual was `9.093e-9`. Relative to CSR, the maximum primal
coefficient difference was `9.220e-9` for hybrid and `2.237e-8` for dense pure
BSR, consistent with the solve tolerance and differing preconditioned stopping
histories.

Dense interpolation is a major numerical repair over identity lifting: on the
nearby earlier large p=6 run it reduced pure-BSR PCGF from 590 to 53 iterations.
It is competitive at small blocks, beating scalar CSR solve time at p=1 and
matching CSR's iteration count at p=2. Its quality degrades from p=3 onward,
and the dense-transfer V-cycle costs more per iteration. The coefficient-exact
hybrid is therefore the production BSR candidate: it is fastest at every degree
except p=4, where scalar CSR is 3% faster, and it preserves BSR's compressed
fine-level pattern.

Remaining validation is native coefficient-level constant-mode/transpose/RAP
coverage plus CUDA Initcheck and Memcheck. The next performance work should
profile dense prolongation, restriction, coarse BSR SpMV, smoothing, and the
reference Galerkin setup separately. A richer block interpolation or additional
smoothing is needed before pure BSR can match the hybrid for p >= 3.

### Dense-Interpolation Parameter Isolation and Next Candidate

A follow-up on the same 152,909-triangle p=6 case ruled out two cheap scalar
explanations. Raising `interp_max_elements` from 4 to 8, 16, or unlimited moved
the PCGF count only from 53 to 54. With support fixed at 4, sweeping the single
Jacobi interpolation weight through `0.25, 0.4, 0.55, 2/3, 0.8, 1.0, 1.2`
kept the count between 52 and 54. The hybrid gap is therefore not caused by the
D2 truncation or a poorly tuned single damping constant.

The block strength threshold has a larger but discontinuous effect. At p=6,
raising it from `0.25` to `0.47` reduces 53 iterations/1.227 s solve to
41/1.038 s; `0.48` selects a different hierarchy and jumps to 68 iterations.
At p=2, `0.47` changes 20 to 24 iterations but reduces setup/solve from
1.222/0.217 s to 0.295/0.181 s. Post-rebuild tests must therefore include both
thresholds and must not attribute their difference to interpolation smoothing.

The follow-up factorial screen used p=2 and p=6, both strength thresholds,
one through four true dense-block smoothing steps, and additive versus right
block-row normalization. The 32 records are in
[`classical_amg_dense_bsr_interpolation_screen_2026_08.csv`](classical_amg_dense_bsr_interpolation_screen_2026_08.csv).
All successful solves have independently checked relative residual below
`8e-9`.

| p | threshold | constraint | steps 1 / 2 / 3 / 4 iterations | conclusion |
|---:|---:|:---|:---:|:---|
| 2 | 0.25 | additive | 20 / 20 / 20 / 19 | negligible gain |
| 2 | 0.47 | additive | 24 / 24 / 24 / 25 | no gain |
| 6 | 0.25 | additive | 53 / 54 / 53 / 52 | negligible gain |
| 6 | 0.47 | additive | 41 / 69 / 66 / 68 | additional sweeps regress |
| 2 | 0.25 or 0.47 | right normalize | 49 or 48 / fail / fail / fail | rejected |
| 6 | 0.25 or 0.47 | right normalize | 146 or 122 / fail / fail / fail | rejected |

The multi-sweep hypothesis is therefore rejected. Right normalization weakens
the one-sweep hierarchy and, on later levels with two or more sweeps, either
makes a coarse diagonal block singular or loses the constant-mode constraint.
The original one-step additive implementation remains the compatibility
baseline.

The next screen targeted coarse-face selection with the diagonal-normalized
block coupling

```text
s_ij = ||A_ij||_F / sqrt(||A_ii||_F ||A_jj||_F).
```

The 21 raw and normalized samples are in
[`classical_amg_block_strength_metric_screen_2026_08.csv`](classical_amg_block_strength_metric_screen_2026_08.csv).
At p=6, normalized strength moves smoothly from 59 iterations at threshold
`0.10` to its best value of 41 at `0.44`, then crosses a hierarchy transition
to 65 at `0.46`. Raw Frobenius also reaches 41 iterations, at threshold `0.47`.
At p=2, normalized `0.44` uses 24 iterations. Scalar diagonal normalization
therefore shifts the transition but does not improve the best coarse space.

The next opt-in metric retains the full internal polynomial-mode action:

```text
s_ij = sqrt(||A_ii^-1 A_ij||_F * ||A_jj^-1 A_ji||_F).
```

It is symmetric and dimensionless, still supplies one scalar graph vertex per
face to AHAT/PMIS/D2, and retains dense BSR transfers, `R=P^*`, BSR Galerkin
products, and BSR V-cycle operations. Its extra dense-block inversions and
products occur only during hierarchy setup. Raw Frobenius remains the
compatibility default.

The 15 inverse-scaled samples are in
[`classical_amg_inverse_scaled_strength_screen_2026_08.csv`](classical_amg_inverse_scaled_strength_screen_2026_08.csv).
The metric reaches 42 p=6 iterations at thresholds `0.42` and `0.44`, versus
41 for the cheaper raw and diagonal-normalized reductions. Its setup grows to
about 2.6 s near that optimum. At p=2 it gives 20 iterations at threshold
`0.25` and 26 at `0.42`. This rejects scalar edge weighting as the primary
remaining cause: three increasingly mode-aware reductions all select
coarse-face spaces of essentially the same quality.

The next candidate changes the coarse-face decision itself. During setup it
expands the BSR operator coefficient-exactly, runs the established scalar
strength and PMIS selection, and promotes a whole face whenever any scalar mode
on that face is coarse. The temporary scalar operator and labels are then
released. D2 interpolation is built on the block graph, and dense `P`, exact
`R=P^*`, Galerkin operators, smoothers, and all V-cycle work remain BSR. This
uses the hybrid hierarchy's mode-aware evidence without retaining scalar CSR
on any solve level.


### Scalar-Guided Face Promotion Result and Block Extended+i Candidate

The coefficient-exact `scalar_guided_any` experiment is rejected. On the
4,658-triangle p=2 smoke case it reduced PCGF from 14 to 12 iterations, but on
the production-size 152,909-triangle p=2 case it increased the count from 20
to 28 and solve time from 0.218 s to 0.811 s. At p=6, setup could not allocate
the temporary coefficient-exact scalar expansion: the 24 GiB device was
already using 23.243 GiB and AMGX reported 21.763/22.197 GiB live/reserved.
The failure occurred while allocating the temporary scalar values, before a
hierarchy was formed. Promoting a face when any independently selected scalar
mode is coarse is both numerically too aggressive at scale and incompatible
with the available setup-memory headroom.

The next opt-in interpolation remains block-native and leaves the established
block-graph PMIS/D2 support unchanged. For a fine face `i` and a coarse face
`c` in that support, it forms the approximate block-elimination weight

```text
B_ic = A_ic - sum_(j strong fine neighbor of i) A_ij A_jj^-1 A_jc,
P_ic^(0) = -A_ii^-1 B_ic.
```

An additive block-row correction then enforces `sum_c P_ic = I_b`. Coarse rows
remain exact block injection, restriction is the exact conjugate transpose,
and Galerkin operators and V-cycle operations remain BSR. This is a
matrix-valued Extended+i analogue rather than another scalar edge reduction;
it is selected with
`block_graph_dense_interpolation_mode=extended_i`. The existing one-step
projected-Jacobi construction remains the default and comparison baseline.


The eight validation records are in
[`classical_amg_extended_i_interpolation_screen_2026_08.csv`](classical_amg_extended_i_interpolation_screen_2026_08.csv).
Both interpolation paths use the same pure-BSR PMIS/D2 hierarchy, exact
`R=P^*`, Galerkin implementation, smoother, and PCGF tolerance.

| mesh | p | threshold | Jacobi iterations / solve (s) | Extended+i iterations / solve (s) |
|:---|---:|---:|:---:|:---:|
| 4,658 triangles | 2 | 0.25 | 14 / 0.073 | 14 / 0.072 |
| 4,658 triangles | 6 | 0.25 | 22 / previous baseline | 21 / 0.098 |
| 152,909 triangles | 2 | 0.25 | 20 / 0.218 | 21 / 0.226 |
| 152,909 triangles | 6 | 0.25 | 53 / 1.283 | 55 / 1.329 |
| 152,909 triangles | 6 | 0.47 | 41 / 1.091 | 43 / 1.124 |

All newly executed solves have independently checked relative residual below
`5.5e-9`. Extended+i is correct, memory-safe, and gives a one-iteration gain
on the small p=6 case, but it is consistently one to two iterations slower on
the production mesh at both tested strength thresholds. It is therefore
rejected as the next production interpolation and retained only as an opt-in
diagnostic. The scalar edge metric, support size, damping, repeated Jacobi
smoothing, normalization, scalar-guided face promotion, and this local
block-elimination formula have now all failed to close the p>=3 hybrid gap.
