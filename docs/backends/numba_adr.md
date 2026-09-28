# Numba ADR diffusion

Fused host Numba assembly and local reconstruction support variable scalar and
elliptic tensor diffusion in `q = -K grad(u)`. Coefficients are sampled through
the shared NumPy/DG preparation helpers before entering compiled kernels.
Use `assembly_backend='numba'` and `hdg_postprocess='none'` for the raw solution.
The result's `diffusion_structure` reports element counts for each selected path.

## Inputs and validation

Scalar constants, scalar callables `k(x, y)`, and scalar DG fields represent
isotropic diffusion. Tensors accept `(K00, K01, K11)` for symmetric inputs,
`(K00, K01, K10, K11)` for general inputs, or a nested 2-by-2 representation.
Components can be constants, callables, or DG fields on the same mesh, including
fields in a different polynomial space. Callables are evaluated directly;
passing a projected DG field instead uses that discrete field.

Every sampled tensor must be finite and have a positive definite symmetric
part. Nonsymmetric elliptic tensors are supported. Validation applies at volume
quadrature points and, when normal-diffusivity stabilization is requested, face
quadrature points; it does not certify positivity between sample points.

## Structural paths

Classification uses exact equality of the discrete samples, independently on
each element. Small nonzero couplings and projection roundoff are never dropped.

| Tensor structure | Flux mass inverse action |
| --- | --- |
| Constant isotropic | Existing scalar Schur kernel |
| Constant diagonal or full | Tensor times reference scalar mass inverse; no weighted factorization |
| Variable isotropic | One scalar Cholesky factor reused for both components |
| Variable diagonal | Two scalar Cholesky factors |
| Variable symmetric | Coupled Cholesky factor |
| Variable nonsymmetric | Coupled pivoted LU factor |

For variable diffusion, the weighted flux mass is the quadrature discretization
of `K^-1`, rather than the inverse of a mass weighted by `K`. Its factors act on
local derivative and trace columns to form the primal Schur complement. The
primal block uses pivoted LU because advection makes it nonsymmetric. Assembly
and reconstruction share the same element algebra and prepared coefficients.
Global constants use a compact four-component descriptor without expanded
volume coefficient tables. All variable factors are element-local scratch.

## Stabilization and postprocessing

The default `GlobalLengthDiffusion` uses
`tau_diff[K,f] = gamma * max_q(n^T K(x_q) n) / ell` on each element-face
incidence, with the existing trace quadrature and physical length policy.
Opposite sides of an interior face remain independent, including coefficient
jumps. This is a sampled maximum, not a certified continuous supremum.
The explicit inverse-h policy uses the same normal diffusivity multiplied by
its existing degree/face-length scale. Explicit scalar or incidence stabilization
tables remain available. Volume-only coefficient tables need explicit
stabilization because they cannot supply face values.

Tensor/variable-diffusion primal postprocessing remains unsupported; select
`hdg_postprocess='none'` or `'flux'`. Requesting `'primal'` or `'both'` fails
before preparation. Raw CUDA ADR still requires constant isotropic diffusion.

## Verification

`tests/test_adr_tensor_numba.py` compares condensed matrices, right-hand sides,
traces, local primal/flux coefficients and true residuals with NumPy for all
seven paths, both trace bases, and degrees 1 and 3 on a sheared mesh. It also
checks mixed element dispatch, cross-space coefficients, invalid tensors,
incidence jumps, and manufactured scalar/diagonal/symmetric/nonsymmetric
convergence at degrees 1 and 2 over three refinements.

`scripts/advection_diffusion_reaction/benchmarks/benchmark_tensor_numba.py` compares the
specializations with a forced general coupled-LU path for the same coefficient
samples, checking parity for every recorded case. Timings exclude JIT warmup, coefficient
preparation and global solves; recorded CPU/wall ratios supplement the requested
Numba thread count.

The [2026-09-25 qualification report](../research/solver_studies/numba_adr_tensor_2026_09_25.md)
contains the benchmark command, scope and all 42 raw records.
