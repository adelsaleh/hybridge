# CUDA Execution Paths

This document explains the CUDA-oriented assembly and sparse-solve pipeline.
The supported equation, coefficient, basis, order, and reconstruction
combinations are defined by
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).
This page describes architecture and runner usage; it is not a second support
matrix.

## Environment

CUDA and AMGX are optional dependencies. When AMGX is not visible through the
system loader, expose the directory containing its shared library:

```bash
LD_LIBRARY_PATH=/path/to/amgx/lib:$LD_LIBRARY_PATH \
  .venv/bin/python scripts/gpu/run_advection_reaction_cuda.py --help
```

Package imports remain lazy: importing `hdgfem` does not require CuPy, CUDA,
or PyAMGX.

## Standalone Runners

- `scripts/gpu/run_advection_reaction_cuda.py` exercises CuPy and raw-CUDA
  advection-reaction assembly, Cupyx/PyAMGX solves, and reconstruction.
- `scripts/gpu/run_diffusion_reaction_cuda.py` exercises the corresponding
  diffusion-reaction paths.
- `scripts/gpu/sweep_cuda_hdg.py` runs controlled basis, trace, quadrature,
  and matrix-format comparisons.

The standalone runners are diagnostic and benchmarking entry points. Normal
application workflows should use the public solver classes or maintained case
drivers.

## Assembly And Solve Pipeline

| Path | Assembly storage | Solve handoff | Role |
|---|---|---|---|
| CuPy | Vectorized device arrays followed by CuPy COO/CSR construction | Direct device Cupyx for compatible preconditioners; explicit host staging for host solvers/preconditioners | Device reference and parity path |
| Raw-CUDA COO | Reduced COO triplets and RHS emitted by raw kernels | CuPy COO-to-CSR, then Cupyx or PyAMGX | Inspectable correctness path |
| Raw-CUDA CSR | Values emitted directly into a known reduced CSR pattern | Direct device CSR view | Preferred large-run path |
| Direct CSR-to-AMGX | Existing device `indptr`, `indices`, and values | PyAMGX device upload/view | Avoids host staging and COO conversion |

Raw-CUDA CSR pattern construction and block-position maps remain on device.
The direct path avoids both global COO-to-CSR reconstruction and large
per-element condensed tensor materialization.

The experimental direct-BSR path and its classical-AMG hierarchy lifecycle are
documented in [`amgx_classical_bsr.md`](amgx_classical_bsr.md). Its retained
fine operator is BSR, while the currently validated transfer and coarse
operators remain scalar CSR.

## Advection-Reaction Modes

The fused cooperative path builds local matrices from projected coefficients,
performs local factorization and solves, eliminates known boundary columns, and
emits the reduced operator. The `precomputed` local-assembly path and serial
`safe` LU mode remain compatibility and debugging references while the
cooperative path is qualified.

Discontinuous advection must retain both element-side face contributions. The
mathematical contract is documented in
[`../algorithms/advection_reaction/`](../algorithms/advection_reaction/), and
the implementation audit is summarized in
[`raw_cuda.md`](raw_cuda.md).

## Diffusion-Reaction Modes

The fused diffusion path constructs the local mixed operator, condenses it,
applies trace-boundary elimination, and emits reduced COO or CSR values. Source
moments, compact boundary data, reference derivative matrices, and face
reference tensors are prepared outside the hot element kernel and are cached
where the reusable solver permits.

Use the capability table for current support. Historical restrictions and
modal-trace AMGX investigations are retained as research evidence rather than
repeated here.

## Direct-CSR Timing Anchor

A paired July 2026 diffusion run used `p=6`, `mesh_size=0.08`,
`legacy-lagrange`, symmetric volume quadrature, and block size 128. The
matrix had 298,599 reduced trace DOFs and 10,412,451 nonzeros.

| Format | CSR/view | AMGX setup/upload | AMGX solve | AMGX subtotal |
|---|---:|---:|---:|---:|
| Direct CSR | 0.00019 s | 0.53684 s | 0.17208 s | 0.709 s |
| COO then CSR | 0.28348 s | 0.50783 s | 0.17192 s | 0.963 s |

The direct path removed the measured COO conversion cost without changing
iterations or field errors. Treat this as dated evidence, not a portable
performance guarantee.

## Validation Anchors

- `tests/test_diffusion_reaction_assembly_parity.py` compares NumPy, Numba,
  CuPy, raw-CUDA COO, and raw-CUDA CSR assembly over the qualified order range.
- `tests/test_cupy_backend.py` covers raw-CUDA/CuPy advection parity,
  discontinuous coefficients, matrix formats, LU modes, and reconstruction.
- `tests/test_raw_cuda_policy.py` covers launch-policy resolution and failure
  behavior.

Raw logs and historical runner comparisons remain under `run_logs/`.
Curated solver conclusions are indexed from
[`../research/solver_studies/`](../research/solver_studies/).
