# Raw-CUDA Launch Policy

Public solver and runner defaults use `raw_block_size="auto"`. The value is
resolved before assembly, reconstruction, or cache-key construction; raw-CUDA
kernels continue to receive an integer block size.

## Initial Recommendations

| Equation | Polynomial order | Block size |
| --- | ---: | ---: |
| Advection-reaction | `p <= 6` | 32 |
| Advection-reaction | `p = 7, 8` | 64 |
| Diffusion/Poisson | `p <= 2` | 32 |
| Diffusion/Poisson | `p = 3, 4` | 64 |
| Diffusion/Poisson | `p = 5, 6` | 128 |

The advection rule selects the smallest warp multiple that covers the scalar
element rows. The diffusion rule is a conservative degree-tier heuristic for
its larger Schur-complement construction and shared-memory working set. It is
not runtime autotuning and is not yet a claim of optimality on every GPU.

Explicit `1`, `32`, `64`, and `128` overrides remain supported. Use them for
benchmark reproduction and launch-policy sweeps. The serial value `1` is a
correctness/debug baseline, not a production recommendation.

Orders outside the qualified recommendation range fail clearly instead of
silently choosing a launch shape. Kernel-specific order checks still apply,
including the distinction between precomputed and fused advection kernels.

The open qualification task in `TODO.md` requires warmed sweeps over supported
orders, equation families, trace bases, mesh scales, and repeated assembly/RHS
and reconstruction phases before these recommendations are promoted as tuned
production defaults.
