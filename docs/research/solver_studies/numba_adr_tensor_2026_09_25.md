# Numba ADR tensor specialization — 2026-09-25

The fused host ADR kernels support variable scalar and elliptic tensor
coefficients, selecting exact constant/isotropic/diagonal/symmetric/general
paths. See the [implementation guide](../../backends/numba_adr.md).

## Numerical qualification

Validation passed 261 focused tensor/stabilization/capability/test-matrix tests
and 91 existing ADR/diffusion solver and Schur-cache regressions. The latter
reported one pre-existing complex-to-real cast warning in a diffusion test.

The tensor suite checks matrix/RHS, trace and local primal/flux parity with
NumPy on sheared meshes at degrees 1 and 3 and both trace bases. Relative
true condensed residuals must be below 1e-10. It also checks mixed element
paths, cross-space fields, both flux postprocessors, mixed host stages,
invalid tensors and separate incidences of coefficient jumps.

Manufactured scalar, diagonal, symmetric and nonsymmetric elliptic cases use
u=sin(pi*x)sin(pi*y), analytic forcing, nonzero constant advection and reaction,
and the default normal-diffusivity global-length stabilization. Degrees 1 and 2
use 2, 4 and 8 subdivisions per direction. Every final raw primal and diffusive
flux L2 rate exceeds p+0.65. This qualifies raw solutions; tensor primal
postprocessing is not implemented.

## Prepared-kernel benchmark

Command (from repository root):

```sh
NUMBA_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.advection_diffusion_reaction.benchmarks.benchmark_tensor_numba --output /tmp/adr_tensor_numba.jsonl
```

[All 42 records](numba_adr_tensor_2026_09_25.jsonl) cover seven coefficient
structures, degrees 2/4/6, 512 triangles and 1/16 Numba OpenMP threads, with
orthonormal volume and Legendre-modal trace bases. Each timed function is
warmed twice; three batched samples record median/min/max and CPU/wall ratio.
Assembly and reconstruction are compared against the same diffusion forced
through coupled LU. Matrix/RHS/recovery parity is checked in every case.
Coefficient lowering, JIT compilation, sparse solution and postprocessing are
excluded. No AMGX build or solve is involved.

Degree 6, 16 threads (general-LU time / specialized time):

| Structure | Assembly | Reconstruction |
| --- | --- | --- |
| constant_isotropic | 1.78x | 1.74x |
| constant_diagonal | 1.70x | 1.96x |
| constant_full | 1.70x | 1.93x |
| variable_isotropic | 1.37x | 1.43x |
| variable_diagonal | 1.31x | 1.47x |
| variable_symmetric | 1.02x | 1.11x |
| variable_full | 0.95x | 0.96x |

The measured 16-thread CPU/wall ratios span 12.2–16.2,
confirming actual parallel CPU use. Constant and scalar/diagonal specializations
show the clearest savings. Symmetric Cholesky gains are modest here; general
LU uses the same path in both runs, so its deviation from 1.0 reflects timing
variation. These short warmed local-kernel timings are not end-to-end solver
speedups or a guarantee that every specialization wins at every order.

Raw CUDA variable/tensor ADR remains a separate TODO.
