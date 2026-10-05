# Large Poisson solver comparison — 2026-09-22

All four solvers passed on one captured **315,425-triangle, p=6** isotropic
Poisson system: **3,307,381 trace unknowns**, 7-by-7 face blocks, and
115,630,053 stored scalar matrix entries. The domain is the radius-1 disk,
with diffusion 1 and stabilization tau=1. The physical RHS is the first
saved vortex-gas Poisson RHS, not a manufactured-solution accuracy test.
No new assembly, time integration, build, or kernel compilation was run.

The existing diffusion-reaction comparison script now provides the
`poisson_300k_p6` replay preset; no new runner script was created.

## Confirmed timings

Hardware: Intel Xeon w7-3455 (24 physical cores, SMT disabled) and NVIDIA
RTX PRO 5000 Blackwell. GPU runtime reported CUDA 13.0.2. All times below
are **seconds**, from independent confirmation rather than tuning pilots.

| Solver | Fresh setup + first solve, median | Reused setup solve, mean | Iterations | Worst physical relative residual |
|---|---:|---:|---:|---:|
| PyPardiso LU, 16 threads | 14.3236 | 3.21866 | direct | 4.75e-12 |
| AMGX hybrid BSR/CSR, PCGF, Chebyshev-L1 0/2 sweeps | 0.568394 | 0.293020 | 21 | 9.34e-12 |
| pMG-AMG, standard policy, PCGF | 0.609152 | 0.340672 | 41 | 8.24e-12 |
| Element ASM+PP(24), GMRES(75), CGS2 | 28.5134 | 25.6145 | 375 | 4.41e-12 |

AMGX is approximately **25.2x faster fresh and 11.0x faster reused** than
the measured PyPardiso LU path. pMG-AMG also wins both comparisons; this
ASM+PP configuration does not. These conclusions are specific to this
matrix and the tested configurations, not all Poisson or ADR problems.

Fresh confirmation ranges were 14.3180–14.3899 s (LU),
0.563135–0.570808 s (AMGX), 0.605399–0.615842 s (pMG-AMG), and
28.2347–28.5690 s (ASM+PP). Reused ranges were 3.17989–3.24380 s,
0.291559–0.295443 s, 0.338746–0.342864 s, and 25.1851–25.9889 s,
respectively.

## Protocol and interpretation

- Every worker verifies the original BSR values, indices and row-pointer
  hashes; all four use the same RHS. All iterative solves start from zero,
  including reused-setup solves. Each setup solves the same RHS twice.
- The target is an independently checked original nodal-coordinate
  relative residual of 1e-10. Iterative internal targets are stricter.
  pMG's Legendre congruence uses a norm bound to enforce the same original
  residual target. Modal trace differences against the archived reference
  were below 1.16e-10; this is not a PDE discretization-error measurement.
- Seven pilots each discarded one setup/two-solve warmup, then measured
  two independent setups. LU screened 8, 16 and 24 threads. AMGX screened
  the historically leading hybrid 0/2 and 0/3 post-smoothing configurations.
  The selected LU count was 16 for both fresh and reused objectives;
  AMGX selected 0/2 for both. This is best among tested configurations,
  not a global optimum. ASM polynomial degree and pMG policy were fixed.
- Each selected configuration was confirmed in a separate process with
  one discarded setup/two-solve warmup and three measured setups. Every
  pilot and confirmation passed (11 jobs total). No solver jobs overlap.
- Fresh time includes input representation conversion/upload, setup, and
  first solve. Reused time excludes setup. Runtime initialization, file
  loading, discarded warmups, independent host validation, reconstruction
  and cleanup are excluded. GPU RHS/solutions remain device-resident during
  solve timing; LU inputs/outputs are host-resident.
- AMGX uses the generic cuSPARSE BSR fine operator and scalar-expanded
  coarse hierarchy. pMG uses the production symmetric PCGF implementation,
  not the nonsymmetric ADR symmetric-part/GMRES adapter. ASM reuses the
  package's exact element-patch construction, cuBLAS batched inverses,
  fused patch application, harmonic-Ritz polynomial, and GMRES with a
  generic cuSPARSE BSR operator. No numerical kernels were added.
- **The initial direct baseline is real nonsymmetric LU (mtype=11), matching
  the ADR campaign. The SPD follow-up is recorded below.** Reused time includes the
  current PyPardiso wrapper's matrix-hash/factor-reuse checks; it is not
  a kernel-only triangular-substitution time. Both checks and factor
  reuse remain enabled, with phase 33 verified after each solve. These
  measurements do not establish the maximum possible direct-solver speed.

## LU memory

Confirmed LU peak process RSS/HWM was **16.77 GiB**. MKL reported an
estimated in-core solver peak of **14.05 GiB**, with **1,081,215,135 factor
nonzeros**, a **9.35x fill ratio**, and **zero perturbed pivots**.
The RSS includes the BSR/CSR matrices, factors, interpreter and validation
temporaries; it is not factor storage alone. MKL estimates and process RSS
are distinct measurements and must not be added together.

The monitor enforced a best-effort 72 GiB RSS limit, 16 GiB available-memory
reserve and 1800 s per-job timeout. All jobs remained within these limits.
Reported CPU RSS for GPU workers is not GPU VRAM; per-job JSON additionally
records CuPy pool reservations and AMGX native allocation counters, which
are not a synchronized whole-process GPU high-water mark.

## Reproduce

From the repository root, use a **new empty output directory**:

```bash
HYBRIDGE_CUDA13_ROOT=/path/to/cuda-13 \
HYBRIDGE_AMGX_BUILD_ROOT=../AMGX-build-cuda13 \
HYBRIDGE_AMGX_INSTALL_ROOT=../AMGX-install-cuda13 \
PYTHONDONTWRITEBYTECODE=1 NUMBA_DISABLE_JIT=1 OPENBLAS_NUM_THREADS=1 \
scripts/gpu/run_cuda13.sh .venv/bin/python -u -B \
  -m scripts.diffusion_reaction.compare_cuda_bsr_csr \
  --preset poisson_300k_p6 \
  --output run_outputs/solver_studies/poisson_300k_p6_repeat \
  --rtol 1e-10 --maxiter 2000 --pp-degree 24 \
  --repeats 3 --warmup 1 --threads 8 16 24 \
  --timeout 1800 --max-rss-gib 72 --reserve-gib 16 --execute
```

Omit `--execute` for a read-only plan. The preset requires the original
capture in `artifacts/full_bsr_convergence_20260914/p6_300k_controls`
and its cached mesh. A missing kernel cache stops execution rather than
compiling. The original BSR-vs-CSR assembly benchmark mode is unchanged.

Sources: full result summary (`run_outputs/solver_studies/poisson_300k_p6_20260922/summary.json`, local, untracked),
arguments (`run_outputs/solver_studies/poisson_300k_p6_20260922/arguments.json`, local, untracked),
and each `pilot_*/` / `confirmation_*/` directory's `result.json` and
`worker.log`. The original matrix-values SHA256 is
`da78acca27b1aafa729f6a88cc6b635159c4d7229a53021ea49227939b28956d`.

Validation: 15 runner/preset tests passed with Numba JIT disabled, in
addition to the completed large-system residual and reference checks.

## Follow-up: SPD Cholesky and pMG tuning

The follow-up uses the **same captured matrix/RHS**, zero starts, physical
relative-residual target `1e-10`, and independent pilot/confirmation protocol.
The separate short square-stress campaign was allowed to finish before these
measurements. No compilation, new assembly or time integration was performed.
The production solver defaults have **not** been changed.

### Lighter pMG beats heavier cycles on this matrix

The initial eight-policy screen measured the following pilot times (seconds):

| pMG policy | Fresh | Reused | Iterations |
|---|---:|---:|---:|
| Standard: p6 -> p0, Chebyshev 2, balanced 1/1 | 0.6691 | 0.3842 | 41 |
| Same, Chebyshev 1 | 0.5842 | 0.2936 | 49 |
| Same, Chebyshev 3 | 0.7900 | 0.5239 | 41 |
| Same, Chebyshev 4 | 0.9728 | 0.6808 | 41 |
| Standard, coarse L1-Jacobi 2/2 | 0.6161 | 0.3331 | 34 |
| Standard, coarse W-cycle | 0.7510 | 0.4419 | 30 |
| p6 -> p3 -> p1 -> p0, Chebyshev 2, 1/1, coarse 1/1 | 1.0999 | 0.7366 | 44 |
| Robust: halving, Chebyshev 4, 2/2, coarse 2/2 | 2.6397 | 2.0742 | 34 |

Every pMG candidate passed its symmetry/curvature gates and physical residual
checks. Fewer outer iterations did not imply a faster solve: more work per
cycle outweighed the reduction. A ninth, targeted trial combining Chebyshev 1
with coarse 2/2 smoothing did not improve the selected pilot objectives.

In the final independent GPU confirmation:

| Solver | Fresh median (s) | Reused mean (s) | Iterations | Worst physical relative residual |
|---|---:|---:|---:|---:|
| AMGX hybrid 0/2 control | 0.644980 | 0.342733 | 21 | 9.34e-12 |
| pMG-AMG, standard + Chebyshev 1 | 0.585596 | 0.312730 | 49 | 8.22e-12 |

The tuned pMG is **9.2% lower fresh time and 8.8% lower reused time** than
the contemporaneous AMGX control (about 1.10x speedup). Do not mix the earlier
AMGX run's 0.2930 s with these new timings: repeated GPU timings vary between
sessions. The first tuning campaign's independent confirmation also favored
Chebyshev 1. This is the best tested policy for this matrix, not a universal
replacement for the production default or a globally optimized AMG comparison.

The selected setting can be reused via
`FaceBlockHpMgPcgSolver(..., preconditioner_policy="standard",
preconditioner_tuning={"chebyshev_order": 1})`. It retains direct p6 -> p0,
balanced fine 1/1, scalar-AMGX V-cycle with coarse 1/1 L1-Jacobi, strength
0.40, dense-LU cutoffs 128/256, fused fine smoothing and outer PCGF.

### Cholesky accuracy caveat

PyPardiso supports `mtype=2`, which selects SPD Cholesky without pivoting.
Its upper-triangle storage represents a symmetric interpretation of the
approximately symmetric captured matrix. Both the default and two-step
native-refinement trials factored successfully at 8, 16 and 24 threads, with
zero perturbed pivots, but failed our original-matrix residual target at
approximately **1.269e-10**. They are retained as failed accuracy checks,
not counted as successful timed solutions. The native refinement parameter
is PyPardiso's one-based `iparm(8)` (C `iparm[7]`); actual executed steps
are recorded after solves. See the
[Intel parameter reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/pardiso-iparm-parameter.html).

A diagnostic of the uncorrected solution measured approximately 6.6e-12
against the symmetric upper-triangle interpretation, confirming that MKL's
internal refinement does not enforce the original full-matrix residual.
One correction solve with the retained Cholesky factors and residual of the
original matrix reduced that residual to approximately 5.1e-12. This path is
labelled **Cholesky + original-system correction**, not plain Cholesky.
The original matrix is not symmetrized or replaced for validation. All outer
residual evaluations and correction solves are included in timed work.

The corrected CPU screen completed successfully (three pilots and two
independent confirmations). Pilots selected 16 threads for fresh solves and
8 for reused factors; the confirmed reused difference is small (0.4%).

| Direct path | Fresh median (s) | Reused mean (s) | Peak process RAM (GiB) | MKL estimated peak (GiB) |
|---|---:|---:|---:|---:|
| Earlier LU, 16 threads | 14.3236 | 3.21866 | 16.77 | 14.05 |
| Cholesky + original-system correction, 16 threads | 14.3283 | 4.19470 | 10.11 | 7.64 |
| Cholesky + original-system correction, 8 threads | 15.1801 | 4.17747 | 9.60 | 7.19 |

All measured corrected solves used **one** original-system correction and
passed with relative residual at most **5.11e-12**. At 16 threads the median
preparation cost was 4.277 s (including symmetry checking/upper conversion),
and median factorization cost was 5.878 s. The fresh range was 14.2694–14.3739 s;
reused range 4.17123–4.22967 s. At 8 threads those ranges were
15.1583–15.1817 s and 4.14480–4.21855 s.

Cholesky stored **574,617,698 factor nonzeros** versus LU's 1,081,215,135,
with zero perturbed pivots. Its input triangle contains 59,468,717 entries;
the reported 9.66x fill is relative to that triangle, not the full CSR matrix.
The 16-thread process RAM peak fell about **40%** from the previous LU run.
However, the original-system correction and retained PyPardiso reuse-check
overhead remove an end-to-end time advantage here: fresh performance is
essentially unchanged, and the reused path is slower than the previous LU
measurement. These remain wrapper-level results, not the limit of native
triangular-substitution performance. Removing matrix-hash overhead has not
been tested and is not included in these claims.

Corrected CPU records:
`run_outputs/solver_studies/poisson_300k_p6_spd_original_correction_20260922/summary.json`.
The same matrix/RHS hashes, 72 GiB RSS guard and 16 GiB reserve were retained.

Follow-up GPU records:
`run_outputs/solver_studies/poisson_300k_p6_spd_pmg_20260922/summary.json`
and `run_outputs/solver_studies/poisson_300k_p6_spd_refined_pmg_20260922/summary.json`.
These campaigns complete with nonzero exit codes because the uncorrected SPD
pilots failed; their GPU confirmations passed.

To reproduce the expanded screen using the corrected SPD path, use the same
environment prefix as above with a new output directory:

```bash
scripts/gpu/run_cuda13.sh .venv/bin/python -u -B \
  -m scripts.diffusion_reaction.compare_cuda_bsr_csr \
  --preset poisson_300k_p6 --replay-suite spd-pmg \
  --spd-refinement 2 --spd-original-refinement 2 \
  --output run_outputs/solver_studies/poisson_300k_p6_spd_pmg_repeat \
  --rtol 1e-10 --maxiter 2000 --repeats 3 --warmup 1 \
  --threads 8 16 24 --timeout 1800 --max-rss-gib 72 --reserve-gib 16 --execute
```

`--replay-suite spd` runs only the CPU Cholesky screen;
`--replay-suite spd-pmg-refined` tests the two light pMG variants with an AMGX
control and Cholesky. The refinement flags remain explicit in each case.

Final validation: **35 host-only regression tests passed**, with Numba JIT
disabled. Cached-kernel-only guards remained enabled for every GPU worker;
all nine tested pMG variants completed without new compilation.
