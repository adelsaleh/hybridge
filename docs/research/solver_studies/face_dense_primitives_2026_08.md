# Face-dense primitive and polynomial benchmark (2026-08-13)

## Scope

This study isolates the GPU costs behind the face-dense GMRES result in
`amgx_vs_face_dense_2026_08.md`.  It uses the classical trigonometric Poisson
problem on the radius-5 disk with 32,449 triangles, orders p=1,...,6,
`dub_orth`, the branch's legacy Lagrange face trace, 2p volume/edge
quadrature, `tau=1`, and boundary elimination.  The device was a Quadro RTX
6000 and the requested relative residual tolerance was 1e-8.

The face implementation is commit `ea5ad26281f9e988194d9352399ccc4354a6633e`.
Primitive medians use CUDA events around preallocated operations.  The main
grid uses 3 warm-ups and 12 samples; p=1 primitives were repeated with 100
warm-ups and 100 samples to remove a visible initial GPU clock-ramp artifact.
Variant outputs agreed to relative differences between 1e-16 and 6.4e-15.

That implementation and its focused CPU/GPU tests are now selectively ported
into the current tree. Reproduction should use
`scripts.diffusion_reaction.benchmark_face_dense_primitives`; the values in
this report remain historical measurements from the commit stated above.

## Primitive timings

All times are milliseconds per application.

| p | dofs | operator matmul | operator raw | operator fused | fastest | ASM matmul | ASM raw | ASM fused | fastest |
|---:|---:|---:|---:|---:|:---|---:|---:|---:|:---|
| 1 | 96,928 | 0.230 | 0.058 | 0.053 | fused | 0.272 | 0.062 | 0.044 | fused |
| 2 | 145,392 | 0.297 | 0.140 | 0.141 | raw | 0.606 | 0.085 | 0.071 | fused |
| 3 | 193,856 | 0.446 | 0.154 | 0.130 | fused | 0.712 | 0.126 | 0.141 | raw |
| 4 | 242,320 | 0.602 | 0.250 | 0.266 | raw | 0.939 | 0.276 | 0.365 | raw |
| 5 | 290,784 | 0.606 | 0.344 | 0.422 | raw | 1.166 | 0.313 | 0.471 | raw |
| 6 | 339,248 | 0.804 | 0.494 | 0.652 | raw | 1.404 | 0.413 | 0.635 | raw |

The cuBLAS-backed `matmul` path never wins.  The custom raw face operator is
best from p=4 upward; fusion is helpful at p=1 and p=3 but essentially tied
at p=2.  Fused ASM wins at p=1 and p=2, while the separate raw restriction,
local solve, and prolongation path wins at p=3,...,6.  These crossover choices
should be profiled/autotuned rather than selected globally.

## Polynomial degree tradeoff

The solver uses the fastest primitive variants above for each p.  `Best face`
is the fastest converged hot solve among polynomial degrees 2, 4, 8, 12, and
18.  The AMGX column is the exact iterate time from the matched 32,449-element
dense-128 PCGF/AMG campaign; it excludes AMGX setup, just as the face column
excludes polynomial setup.

| p | best degree | face iterations | best face solve | degree-18 solve | AMGX iterations | AMGX iterate | face/AMGX |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 41 | 107.3 ms | 107.3 ms | 21 | 38.2 ms | 2.81x |
| 2 | 18 | 44 | 201.1 ms | 201.1 ms | 20 | 45.4 ms | 4.43x |
| 3 | 18 | 47 | 267.7 ms | 267.7 ms | 21 | 56.7 ms | 4.73x |
| 4 | 8 | 96 | 565.6 ms | 585.6 ms | 22 | 69.6 ms | 8.13x |
| 5 | 8 | 100 | 733.2 ms | 736.7 ms | 19 | 78.8 ms | 9.31x |
| 6 | 8 | 100 | 1,025.8 ms | 1,087.3 ms | 21 | 93.8 ms | 10.93x |

Degree 18 minimizes the face iteration count, but degree 8 minimizes hot solve
time for p=4,...,6.  At p=2, degrees 12 and 18 are within about 1%.  Degree 2
does not meet the requested tolerance within 500 iterations for p=2,...,6;
degree 4 converges but is never time-optimal.  The compact CSV contains every
degree, status, residual, setup time, application time, and solve time.

## GMRES operation attribution

The recommended degree-18 solve was rerun with CUDA-event instrumentation.
Event insertion perturbs the solve, so these numbers are attribution rather
than primary time-to-solution measurements.

| p | iterations | polynomial share | outer matvec share | projection + correction | norm share | coefficient D2H event |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | 81.0% | 4.3% | 5.9% | 4.5% | 1.220 ms |
| 2 | 44 | 88.0% | 4.6% | 2.6% | 2.6% | 0.792 ms |
| 3 | 47 | 89.9% | 3.4% | 2.7% | 2.1% | 0.859 ms |
| 4 | 51 | 93.3% | 3.0% | 1.8% | 1.1% | 0.907 ms |
| 5 | 51 | 93.5% | 3.2% | 1.7% | 0.8% | 1.472 ms |
| 6 | 53 | 94.5% | 3.0% | 1.4% | 0.5% | 0.890 ms |

The p=6 profile spends 1,019.6 ms of 1,078.5 ms of recorded GPU-operation
time inside polynomial applications.  One degree-18 application performs 18
face-operator applications and 19 ASM applications.  GMRES orthogonalization
itself is small: projection plus correction totals 15.6 ms at p=6.

The blocking coefficient transfer occurs once per CGS Arnoldi step.  Its
reported host wall time can be large because it is also the point that waits
for preceding polynomial kernels.  It must not be interpreted as copy cost:
at p=6 the transfer's CUDA-event time is 0.890 ms in total, although its
synchronizing host wall time is 924.6 ms.

## Conclusions

The face solver is not limited by dense-vs-sparse library choice or by GMRES
orthogonalization.  It is limited by applying a high-degree polynomial whose
stages each traverse the duplicated face operator and element ASM data.  The
custom kernels are already substantially faster than the cuBLAS batched
`matmul` variants, but 18 operator traversals plus 19 ASM traversals per outer
iteration outweigh one AMGX PCG/AMG iteration.

The next experiments should therefore focus on reducing polynomial work:

1. select operator, ASM implementation, and polynomial degree per p;
2. test whether better spectral intervals/root orderings make degree 8 robust
   with fewer outer iterations;
3. consider a flexible or restarted scheme that can vary polynomial degree;
4. integrate current master raw-CUDA local assembly/reconstruction before
   comparing complete HDG time again.

Raw JSON and the temporary benchmark harness remain under
`/tmp/codex-diffusion-microbench/` and `/tmp/hdg-face-microbench/`.
