# Raw-CUDA Backend Notes

This document records raw-CUDA implementation ownership and launch policy.
Current support is defined only by
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).

## Kernel Ownership

The fused raw-CUDA element kernels perform the expensive local work:

- construct equation-specific local operators from prepared coefficient data;
- factor and solve the local condensed systems;
- eliminate known trace columns;
- emit reduced COO entries or direct CSR values;
- accumulate the reduced right-hand side on device;
- reconstruct local fields with the matching equation kernel.

Reference tensors, projected/source moments, compact boundary data, topology,
and reduction maps are prepared before the element launch. Reusable solver
objects cache matrix- and graph-dependent data when their invalidation contract
allows it.

## Launch Policy

Public solvers and runners use `raw_block_size="auto"`. Resolution occurs
before kernel selection and cache-key construction; kernels receive an integer
block size.

| Equation | Polynomial order | Recommended block size |
|---|---:|---:|
| Advection-reaction | `p <= 6` | 32 |
| Advection-reaction | `p = 7, 8` | 64 |
| Diffusion/Poisson | `p <= 2` | 32 |
| Diffusion/Poisson | `p = 3, 4` | 64 |
| Diffusion/Poisson | `p = 5, 6` | 128 |

These are policy defaults covered by launch-resolution and parity tests, not
runtime autotuning or a completed performance sweep. Explicit `1`, `32`,
`64`, and `128` overrides remain available for benchmark reproduction.
Block size 1 is a serial correctness baseline, not a performance
recommendation. Orders outside the policy fail explicitly rather than silently
choosing a launch shape.

## Discontinuous Advection Audit

CuPy assembly stores `beta . n` per element side. Raw-CUDA precomputed,
fused COO, and fused CSR paths preserve those side values through local
coupling, stabilization, and trace-mass construction. Interior sides are
emitted independently and summed into the global trace row; no face averaging
is introduced.

Fused reconstruction is element-local and uses the element's own projected
advection coefficient. It does not reconstruct an averaged interior-face
coefficient.

Validation anchors include:

- `tests/test_advection_reaction_numba.py` for side-weighted trace mass;
- `tests/test_cupy_backend.py` for CuPy/raw-CUDA matrix and RHS parity,
  modal traces, and COO/CSR equivalence;
- `tests/test_advection_reaction_conservation.py` for global conservation and
  no-through-flow boundaries.

## Diffusion Setup Ownership

The raw diffusion kernel owns local Schur construction, local solves, boundary
elimination, reduced emission, and RHS accumulation. It does not materialize
the pure-CuPy path's large `local_lhs`, `solved_el_bd`, `solved_src`,
`trace_blocks`, or `faces` intermediates.

Prepared outside the hot kernel:

| Data | Preparation |
|---|---|
| Source moments | CuPy quadrature/projection or a reusable prepared table |
| Boundary trace | Compact device interpolation/projection and edge mapping |
| Reference derivatives and face tensors | Small order-dependent device tables |
| COO reduction maps | Host-origin maps copied to device |
| CSR pattern and block positions | Raw-CUDA pattern kernels and device prefix operations |
| Mesh/reference mirrors | One-time host-to-device construction for the CuPy space |

The remaining optimization direction is to cache all order-dependent reference
tables, keep compact boundary storage, generalize table-driven coefficients,
and eliminate COO-only host map construction where measurements justify it.

## Failure And Validation Policy

Raw-CUDA requests are preflighted before assembly. Unsupported combinations
must raise a capability error or take an explicitly documented compatibility
path; they must not silently change coefficient semantics, trace basis, matrix
format, or convergence requirements.

The required parity and smoke lanes are documented in
[`../development/alpha_test_matrix.md`](../development/alpha_test_matrix.md).
