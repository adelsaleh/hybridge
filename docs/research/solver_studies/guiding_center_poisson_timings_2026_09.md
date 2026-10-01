Poisson HDG timing investigation — 2026-09-13

The main optimization target is the repeated global solve, especially its scalar
AMGX coarse correction. In the saved vortex-gas runs, global solution consumes
89–92% of Poisson wall time, cached RHS assembly 5–8%, and reconstruction about
2%. Matrix, local factors, and global hierarchy are already reused. There is
also an opportunity to remove unnecessary Poisson calls during H1/H2 startup.

This investigation reads existing logs and current source. No build, CUDA
compilation, simulation, or time integration was run. Solver code and defaults
were not changed. Proposed speedups remain unmeasured.

The reproducible evidence is in
[summary.json](../../../artifacts/poisson_timing_investigation_20260913/summary.json)
and [summary.csv](../../../artifacts/poisson_timing_investigation_20260913/summary.csv).
JSON includes input paths, byte counts, SHA-256 hashes, distributions,
startup/retry records, tau groups, and console statistics. Inspected source
hashes are in
[source_provenance.json](../../../artifacts/poisson_timing_investigation_20260913/source_provenance.json).
The worktree was already extensively modified; current source hashes do not
establish the source version that produced older logs.

**Measured costs.** These are medians in milliseconds per accepted step,
summed over every Poisson stage. All five vortex-gas logs identify p=6 and
113,894 triangles. Steps 0–2 are excluded to separate setup and the two BDF3
startup steps. Tau-recovery steps are excluded here and retained separately in
JSON. Assembly is RHS-only on these ordinary steps. Each column has its own
median, so columns need not add exactly to the wall median.

| Scheme | Actual dt | Ordinary steps | Poisson calls/step | RHS assembly | Global solve | Reconstruction | Poisson wall | Iterations/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SI Euler | 0.01 | 998 | 1 | 12.14 | 166.71 | 2.80 | 184.28 | 23 |
| SI BDF2 | 0.05 | 998 | 1 | 8.92 | 152.07 | 2.53 | 164.90 | 25 |
| H1-BDF3 | 0.01 | 89 | 2 | 17.80 | 256.91 | 5.05 | 282.66 | 42 |
| H2-BDF3 | 0.005 | 98 | 2 | 17.80 | 207.52 | 5.04 | 232.98 | 34 |
| IMEX-ARK3 | 0.05 | 996 | 4 | 35.70 | 562.38 | 10.13 | 613.72 | 92 |

The input filenames are `run_outputs/guiding_center/euler_vortex_gas_`
followed by the scheme with underscores and its `p6*timings.jsonl` suffix.
Use recorded times, not filename dt/T tokens: several runs override their
response files. Different dt values, physical intervals, and sessions mean
this is not a matched comparison at equal accuracy. Recorded final times are
10, 50, 0.91, 0.5, and 50 respectively. Successful recorded steps do not prove
that a requested longer run completed. Four console logs explicitly identify
an RTX PRO 5000 Blackwell; the SI Euler log lacks that device identification.
Its higher time per iteration/RHS kernel must not be attributed to Euler itself.

Poisson accounts for 55.1%, 54.9%, 66.5%, 49.9%, and 64.6% of the respective
linear-step wall sums. These denominators exclude subsequent plotting and
accepted-state diagnostics. Local optimizations have substantially smaller
potential overall savings than global optimizations.

Other guiding-center evidence supports the same priority:

| IMEX-ARK3 case | Triangles | Actual dt | Ordinary steps | RHS ms/step | Global ms/step | Reconstruction ms/step | Global share of Poisson wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smooth diocotron m9 | 72,961 | 0.5 | 1,398 | 23.08 | 668.28 | 6.75 | 94.9% |
| Gaussian diocotron m64 | 72,961 | 0.5 | 798 | 23.12 | 632.75 | 6.75 | 94.3% |
| Gaussian diocotron m128 | 113,894 | 0.025 | 798 | 44.09 | 301.56 | 10.13 | 83.7% |

All three use four Poisson evaluations per ordinary step and tau=128,000.
Median iteration totals are 166, 158, and 49. Density histories, dt, and warm
starts affect those counts. Local assembly matters relatively more for m128,
but its global solve still dominates. These are timing observations, not
accuracy qualifications.

**Stage counts and existing reuse.** The call paths are in
[run_guiding_center_cases.py](../../../scripts/guiding_center/run_guiding_center_cases.py),
[hybrid_bdf3.py](../../../scripts/guiding_center/time_schemes/hybrid_bdf3.py), and
[imex_ark3.py](../../../scripts/guiding_center/time_schemes/imex_ark3.py).

- SI Euler/BDF2 reuse the accepted field for transport, then solve Poisson for
  the new density. Their guess already uses up to quadratic extrapolation
  through `_fixed_operator_trace_predictor`.
- H1 solves for the AB3 density predictor and accepted BDF3 density. H2 solves
  for the BDF3 predictor and accepted corrected density. Both use nearest-time
  stage trace guesses and two Poisson solves after startup.
- Default BDF3 startup uses SI-Euler paths of 1, 2, and 3 substeps, then an
  extrapolated endpoint: 1+2+3+1=7 Poisson calls per startup step. H1 startup
  Poisson walls are 1.012/1.031 s; H2 gives 0.968/0.955 s. Optional SSPRK3
  startup has three calls per step and its own stability considerations.
- ARK reuses the previous accepted field for stage 1, solves for stages 2–4,
  then for the accepted weighted density. Stage 4 and the endpoint have the
  same time but generally different densities: the explicit final tableau row
  differs from the weights. Blind reuse would change the scheme.

The device solver caches its matrix and scalar Schur-Cholesky data. Native
PCG keeps the p=6→0 hierarchy, fine cuSPARSE BSR descriptor/preprocessing,
Chebyshev-2 smoother, Krylov workspaces, and p=0 AMGX hierarchy. Compact
reconstruction reuses both the source solution computed during RHS assembly
and a precomputed scalar trace response. These are existing optimizations.

Ordinary endpoint records show operator reuse, compact reconstruction, native
hierarchy reuse, and no native fallback. ARK console evidence shows three
hierarchy creations: initialization and tau increases at steps 215 and 546.
These recovery steps perform 7 and 10 Poisson evaluations, with walls 2.004 and
2.387 s. An endpoint reuse flag cannot reveal an earlier-stage rebuild.
Ordinary global step medians at tau=1,000/2,000/4,000 are
562.75/557.81/562.27 ms; different time intervals prevent a causal tau comparison.

**Global optimization priorities.** Paired coarse-AMGX/native-PCG console
records give this attribution. The sample includes every matched reused solve
with positive iterations, including startup/retries, separately from the
ordinary-step sample. Percentages divide summed coarse wall by summed native
solve wall. Calls from an incomplete final step may also appear in this
console sample. Printed times have five-decimal precision.

| Scheme | Matched reused solves | Median coarse ms/application | Coarse fraction of native solve |
|---|---:|---:|---:|
| SI Euler | 1,000 | 5.508 | 76.0% |
| SI BDF2 | 1,000 | 4.438 | 73.0% |
| H1-BDF3 | 193 | 4.432 | 72.7% |
| H2-BDF3 | 210 | 4.429 | 72.4% |
| IMEX-ARK3 | 4,007 | 4.444 | 72.7% |

`AmgxScalarVcycle.__call__` in
[face_hp_multigrid.py](../../../hdgfem/linalg/multigrid/face_hp.py) measures
all of `PyAMGXCsrDeviceSolver.solve`. Thus 73% is **AMGX plus adapter cost**,
not a measured kernel-only share. In
[advection_cuda.py](../../../hdgfem/transport/cuda.py), the adapter
allocates/zeros output, uploads RHS and zero guess through raw pointers,
executes a cycle, downloads the result, synchronizes, then queries status,
iterations, and residual history. `_amgx_config_for_solve` forces monitoring
and history even for this fixed-work preconditioner. These pointers refer to
device arrays; this is not evidence of full-vector GPU→CPU→GPU transfers.

1. Extend the shared AMGX adapter with a fixed-cycle preconditioner application:
   persistent output storage and optional residual-history collection while
   preserving general-solver diagnostics. First measure vector operations,
   actual cycle execution, synchronization, and metadata queries separately.
   Keep error propagation, zero-start one-cycle semantics, stream ordering,
   and outer true-residual validation. The fraction of the 4.4 ms/application
   avoidable in the adapter is currently unknown.
2. Reduce iterations using stage-aware guesses or a small recycled correction
   space across the fixed operator. H1/H2/ARK use nearest traces, without
   SI Euler/BDF2's accepted-history extrapolator. Compare initial true residuals
   and include guess construction cost. Reset recycled operator products after
   tau changes. Different densities at the same time need density/correction
   information, not duplicate-time polynomial interpolation. Keep tolerances fixed.
3. Fuse PCG vector updates and weighted residual reductions after understanding
   the coarse path. The loop already groups norm, curvature, and rho into one
   host read per iteration; the coarse adapter adds synchronization. Preserve
   periodic/exit true residuals, curvature checks, and restart on false
   convergence. Fine BSR descriptors/preprocessing are already persistent.

Changing p=0 AMG or capturing compatible GPU work in a CUDA graph is a larger
experiment. The existing
[multigrid study](../../development/plans/face_block_hp_multigrid.md) screened
Chebyshev order/schedules and rejected a custom restricted-residual SpMV path.
It also records failed symmetry with the full-order nodal AMG configuration.
These should not be presented as untested wins. Current evidence does not
justify replacing the solver or loosening its residual contract.

Illustration only: halving the roughly 73% coarse portion predicts about 1.5×
Poisson throughput and 1.27× linear-step throughput for vortex-gas ARK, with
all other costs and iterations fixed. This is an Amdahl estimate, not an
achieved or promised speedup.

**RHS assembly and reconstruction.** Typical vortex-gas BDF2/H1/H2/ARK
RHS assembly is 8.9 ms/call: approximately 0.40 ms for moments and 8.48–8.50 ms
for the fused phase. That phase includes source-array compaction and RHS
zeroing as well as kernel execution. In
[diffusion_cupy.py](../../../hdgfem/mixed/cupy.py),
`compact_diffusion_rhs` uses one warp per element, two sequential 28-row
triangular solves at p=6, then source-flux recovery and atomic scatter.
Dependent warp reductions/synchronizations are an optimization hypothesis;
these logs do not isolate their cost from flux/scatter.

Let M be reference scalar mass, J the element Jacobian, S the local scalar
Schur matrix, B its condensed trace coupling, and R=S⁻¹B the cached response:

```
f        = J M rho
u_source = S⁻¹ f
u        = u_source + R lambda
local trace RHS contribution = Bᵀ S⁻¹ f = Rᵀ f
```

The last identity uses the coupling-adjoint/symmetry assumptions checked by
the Cholesky cache. Signs follow the current positive source-scatter convention.
Nonzero traces also need boundary contributions; the current compact cached
path requires zero Dirichlet data.

- Construct W=S⁻¹(JM) once with the existing Cholesky helper, then apply
  `u_source=W rho`. This can fuse moments and replace the triangular dependency
  chain with a matrix-vector product, keeping existing flux/scatter and
  reconstruction. Here FP64 W occupies about 681 MiB, the same size as the
  scalar factor; retaining both adds that memory. Conditioning, precision,
  mesh quality, and tau invalidation require parity checks.
- `Rᵀ f` avoids local solve/source-flux recovery during trace RHS assembly.
  Reconstruction still needs `u_source`: merely deferring its solve is not a
  total Poisson optimization. Compare RHS plus reconstruction together.
- Extend `source_moments_cupy` with scalar-only output. It currently allocates
  `(K,3*NEL)` with two zero blocks and copies the first block to contiguous
  `(K,NEL)` storage. Here those sizes are about 73 and 24 MiB. Reuse the
  shared projection helper. At approximately 0.40 ms, this is secondary.
- Reconstruction costs about 2.53 ms/call and already reuses source/trace
  responses. Response layout `(K,NEL,3*NTR)` gives strided loads as warp lanes
  index NEL; benchmark a transposed layout. The kernel writes u twice and the
  wrapper extracts contiguous qx/qy from mixed storage. Direct component
  output could reduce copying, materializing mixed unknowns only for consumers
  needing them. Retained stage fields must keep owning storage.

Nine synthetic small local-block cases for p=1,4,6 and tau=1,1,000,128,000
verified source-map, adjoint-RHS, and reconstruction identities with relative
error/residual below 1e-9. See
[check_local_maps.py](../../../artifacts/poisson_timing_investigation_20260913/check_local_maps.py)
and [results](../../../artifacts/poisson_timing_investigation_20260913/check_local_maps.json).
These validate algebra, not actual mesh conditioning, CUDA execution, FP32,
or speed. Production changes still need those checks.

**Startup call reduction.** `HybridBDF3Stepper.advance` solves Poisson after
a branch's last transport substep and builds a `branch_drift` that no later
substep consumes. Its trace serves only as a later warm-start candidate.
Skipping three branch-final evaluations retains 0+1+2 intermediate solves and
the extrapolated endpoint: **7→4 per startup step**, six fewer calls over two
startup steps. Branch transport and extrapolation stay unchanged in exact
arithmetic. Warm starts/finite-tolerance results can change and need comparison.
Preserve time-dependent boundaries, endpoint postprocessing, and counts.
This affects startup, not steady-state H1/H2 cost.

**Timing issues to correct before finer profiling.**

- `poisson_time_solve`, `poisson_solver_iterations`, and `poisson_detail_*`
  describe the endpoint. Use `poisson_step_time_*`, `poisson_step_iterations`,
  and stage wall lists for whole-step analysis. H2 endpoint solve median is
  67.59 ms versus 207.52 ms for all stages; ARK gives 122.63 versus 562.38 ms.
- The compact wrapper records `cached_rhs.total`, but the solver computes
  `solver.headline.unaccounted` using `timings.get('total', 0)`. It labels the
  entire cached RHS wall as unaccounted. This is a key mismatch, not another
  hidden 9 ms operation. Main assembly wall remains usable.
- `cached_rhs.local_solve` aliases `cached_rhs.fused_solve_flux_scatter`;
  do not sum both. `raw.reconstruction.local_factors.reused=0` describes raw
  LU factors; the CuPy flag confirms Cholesky reuse.
- Native `timings.solve` is setup plus internal PCG elapsed, excluding some
  input/output transforms and logging. Use Poisson wall as the outer check.
  Coarse elapsed/count are printed but absent from native-result JSONL metrics.
  Add structured per-stage coarse and reconstruction-kernel metrics through
  existing result/diagnostic helpers, retaining all attempts.

Follow-up measurements should use a fixed matrix and multiple saved RHSs,
identical guesses/tolerances/precision, and one implementation change at a time.
Separate setup from repeated applications. Compare moments, reduced RHS, true
trace residual, potential, and flux, preserving ownership/tau invalidation.
Runtime gains remain to be measured. This investigation needs no build;
future solver changes may require a user-run build and authorized diagnostic
benchmark.

Reproduce the offline analysis from the repository root:

```bash
python3 scripts/guiding_center/diagnostics/analyze_poisson_timings.py \
  run_outputs/guiding_center/euler_vortex_gas_{si_euler,si_bdf2,h1_bdf3,h2_bdf3,imex_ark3}_p6*_timings.jsonl \
  run_outputs/guiding_center/diocotron_{smooth_m9,gaussian_m64,gaussian_m128}_ark3_p6*_timings.jsonl \
  --output-dir artifacts/poisson_timing_investigation_20260913

OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  artifacts/poisson_timing_investigation_20260913/check_local_maps.py
```
