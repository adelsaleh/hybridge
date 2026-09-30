# High-resolution equilibrium optimizer performance audit (2026-08-20)

## Executive result

The 933.405 s P4 production run was dominated by 80 reduced-optimization
passes, not by initialization or the final equilibrium solve. The primary
optimization is algorithmic: the stopping test is incompatible with the
certified-subband outcome, so the optimizer cannot terminate naturally and
runs to `max_opt_it`. The primary solver improvement is GMRES with Hypre
BoomerAMG, which reproduced the difficult first six MUMPS iterations in 43.84 s
instead of 174.57 s (3.98x faster inside the reduced loop).

## Production-run accounting

- Total run: 933.405 s.
- Reduced optimization: 895.221 s (95.9%).
- Homotopy tangent solves: 8.355 s.
- Homotopy Newton solves: 16.931 s.
- Final exact Newton: 3.918 s.
- Remaining setup and output: about 9 s.
- Outer rows: 80, with 73 accepted and 7 rejected.
- Trial Newton corrections: 90 total.

For the last 60 iterations, each pass was almost invariant at about 9.67 s:
about 4.0 s for the sensitivity factorization/solves, 4.0 s for one trial
Newton factorization/solve, and 1.65-1.70 s for diagnostics and gradient work.

## Where useful progress stopped

| outer iteration | cumulative optimizer time | leakage rel. | missing rel. | certified area fraction | rho mismatch |
|---:|---:|---:|---:|---:|---:|
| 1 | 80.87 s | 0.19665 | 0.20906 | 0.53036 | 0.37363 |
| 10 | 230.06 s | 0.18504 | 0.21626 | 0.52019 | 0.37435 |
| 20 | 325.57 s | 0.18453 | 0.21772 | 0.51918 | 0.37629 |
| 40 | 518.16 s | 0.18359 | 0.21852 | 0.51821 | 0.37667 |
| 79 | 895.22 s | 0.18180 | 0.22265 | 0.51507 | 0.38171 |

Iterations 11-79 cost about 665 s. They reduced relative leakage by only
0.00323 (1.75% relative), while missing area, certified area, and density
mismatch became worse. The trust radius remained at 1.277e-5 and the method
made tiny boundary steps.

The normal stopping test requires leakage <= 0.02, missing <= 0.05, and a
gradient tolerance of 1e-8. This branch cannot meet those geometric targets.
Moreover, the logged `projectedGradNorm` in the reduced loop is the norm of the
selected raw reduced gradient, not a constrained/KKT projected gradient; it
need not vanish at a constrained solution. The final code nevertheless accepts
a certified subband. The reduced-loop stopping rule and final success rule are
therefore inconsistent.

Recommended termination:

1. Add certified-subband convergence: certified leakage zero (within
   tolerance), certified area fraction above the requested minimum, converged
   PDE residual, and stable thresholds/metrics over a short window.
2. Add stagnation/Pareto stopping based on relative objective improvement and
   threshold displacement over 3-5 accepted steps.
3. Compute a true projected/KKT stationarity measure for the active threshold,
   width, and geometric constraints.
4. Keep the best Pareto/certified state rather than automatically returning
   the last accepted state.

Stopping around iteration 10 would have reduced this production run from
933 s to roughly 270 s with MUMPS while producing better certified-area and
density-match diagnostics.

## Measured kernel timings

An instrumented P4/MUMPS pass at 690,677 dofs measured:

- sensitivity Jacobian assembly: 0.120 s;
- two sensitivity RHS assemblies: 0.058 s;
- one MUMPS factorization plus two sensitivity solves: 3.812 s;
- functional-gradient assembly: 1.146 s;
- repeated band and legacy diagnostic block: 0.38-0.69 s;
- three rejected-trial Newton factorizations/solves: 11.783 s;
- all other work around those three Newton corrections: 0.164 s.

MUMPS numerical factorization is therefore the dominant kernel. Matrix
assembly is already cheap, and the two sensitivities already share a single
factorization.

## Solver experiments

The matching first six reduced rows gave:

| solver | reduced-loop time | result |
|---|---:|---|
| MUMPS | 174.574 s | robust reference |
| GMRES + Hypre BoomerAMG | 43.836 s | same accept/reject path and metrics |
| CG + Hypre BoomerAMG | -- | failed in outer trial 1 with `DIVERGED_INDEFINITE_MAT` |

GMRES/BoomerAMG was 3.98x faster over the difficult prefix. One sensitivity
block fell from 4.01 s to 1.22 s, and one three-correction trial fell from about
11.8 s to 2.11 s. The full six-pass benchmark, including homotopy, setup, final
projection, and diagnostics, finished in 66.98 s.

Recommended solver split:

- CG/AMG for known SPD stiffness, torsion, Poisson, and H-minus-one systems.
- GMRES/BoomerAMG for reduced sensitivities and trial Newton Jacobians.
- MUMPS fallback when GMRES diverges or exceeds an iteration threshold, and
  optionally for the final certification solve.

The accepted equilibrium Jacobian may be coercive, but an off-branch trial
Jacobian can become indefinite. This explains why CG is unsafe as a global
preset and why prior CG/GAMG and GMRES/GAMG attempts did not establish that
all AMG approaches fail.

## Source-level avoidable work

1. An accepted trial computes density and the full legacy diagnostics at the
   end of an iteration. The unchanged state is interpolated and diagnosed
   again at the beginning of the next iteration. Cache the accepted Newton
   result, band metrics, density, and diagnostics.
2. Full `compute_metrics` performs min/max reductions, residual assembly, and
   roughly a dozen scalar integrations. Most are logging-only. Evaluate them
   every N iterations and at finalization; keep only acceptance-critical band
   and branch metrics in the hot path.
3. Standard outer Newton and sensitivity helpers recreate PETSc matrices,
   KSPs, and preconditioners each call. Introduce run-scoped workspaces, reuse
   fixed sparsity, and reuse the Hypre hierarchy for nearby Jacobians until
   iteration counts deteriorate.
4. Use nonzero initial guesses for successive sensitivity solves, whose state,
   matrix, and RHS change only slightly late in the run.
5. Use phase-specific inexact Krylov tolerances. A fixed KSP rtol of 1e-10 is
   unnecessary when the current nonlinear target is about 1e-7. Tighten only
   as Newton and final certification require it.

Caching/throttling diagnostics should save roughly 0.5-1.0 s per accepted
iteration. KSP/AMG hierarchy reuse may provide an additional material gain,
but must be benchmarked with rebuild-on-degradation logic.

## Initialization

The separate fine H-minus-one scan used 1,650 frozen evaluations. Frozen scan
work cost about 107.7 s and its eight-stage homotopy cost 27.2 s. Refinement
passes 3 and 4 cost 37.1 s together but changed the thresholds only by about
3e-5 after pass 2. Do threshold search on a coarser mesh/order, stop refinement
on seed improvement, and transfer the two scalar thresholds to the fine mesh.

More generally, optimize thresholds on a coarse mesh and use the fine P4 mesh
only for a few corrections and final certification. Since the control space is
only two-dimensional, this is likely the highest-payoff workflow change after
fixing termination.

## Numerical-library build

The active MUMPS 5.8.2 shared library resolves `libblas.so.3` to conda's netlib
BLAS package (`libblas 3.9.0 ... netlib`), although stale OpenBLAS packages are
also installed. Benchmark a clean cloned DOLFINx environment selecting a
current OpenBLAS or MKL BLAS variant. Do not mutate the working environment
until numerical agreement and MUMPS timing are verified.

## Proposed implementation/benchmark order

1. Add correct certified/stagnation termination and retain the best certified
   state. This removes most of the 80 passes.
2. Add separate solver controls and an automatic GMRES/BoomerAMG-to-MUMPS
   fallback. Validate at least 15-20 outer iterations and the final residual.
3. Cache accepted-state work and throttle logging-only diagnostics.
4. Reuse KSP/matrix/AMG workspaces and add adaptive linear tolerances.
5. Add coarse-to-fine threshold/state transfer.
6. Benchmark MUMPS ordering/BLAS variants only after the algorithm and Hypre
   path are in place; MUMPS then becomes a fallback rather than the hot path.

Expected combined result: under two minutes for a production-quality solve is
plausible from the measured 4x Hypre loop speedup plus termination near 10
iterations. This is an estimate and should be validated with a complete final
certification and guiding-center stationarity check.

## Implemented and validated result

The optimizer now defaults to a phase-specific `optimized` solver preset:

- CG/BoomerAMG for fixed stiffness systems;
- GMRES/BoomerAMG for homotopy, Newton, and sensitivity systems;
- MUMPS for the final exact certification;
- automatic MUMPS retry after an iterative linear-solver failure.

The all-MUMPS path remains available through `--solver-preset legacy` or the
global `--linear-solver mumps` override. Each phase also has its own solver
override.

Certified-subband stagnation now terminates after a configurable number of
accepted steps without meaningful score improvement, and the best certified
state is restored before final exact projection. A reduced gradient is also
cached across consecutive trust-region rejections because the state and
thresholds are unchanged; the cache is invalidated on acceptance.

The final P4 validation used 86,112 triangles, 690,677 dofs, 20 MPI ranks, the
same production mesh and seed, and the same tight Newton tolerances as the
933.405 s reference. Results:

| run | outer rows | reduced-loop time | sensitivity time | total time |
|---|---:|---:|---:|---:|
| MUMPS reference | 81 | 895.677 s | 322.008 s | 933.405 s |
| optimized, before rejection cache | 16 | 80.002 s | 19.025 s | 106.271 s |
| optimized, with rejection cache | 16 | 70.823 s | 10.720 s | 97.023 s |

The final implementation is 9.62x faster end to end. It stopped at iteration
15 after nine accepted steps, restored the best certified state from iteration
7, and finished with residual `1.009e-14`, certified area fraction `0.522788`,
and zero certified leakage. Seven sensitivity blocks were eliminated by the
rejection cache. The equilibrium checkpoint is in
`projects/diocotron/runs/scratch/torsion_reduced_optimization_homotopy/optimized_h002_p4_cached_certstop_validation_20260820/out/equilibrium.npz`.
