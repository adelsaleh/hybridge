# CUDA Execution Paths

This document explains the CUDA-oriented assembly and sparse-solve pipeline.
The supported equation, coefficient, basis, order, and reconstruction
combinations are defined by
[`../reference/backend_capabilities.md`](../reference/backend_capabilities.md).
This page describes architecture and runner usage; it is not a second support
matrix.

## Environment

CUDA and AMGX are optional dependencies. On the qualified CUDA-13 workspace,
use the checked launcher so compiler, AMGX, runtime, and math-library paths are
set together:

```bash
scripts/gpu/run_cuda13.sh .venv/bin/python \
  scripts/gpu/run_advection_reaction_cuda.py --help
```

The canonical `/tmp` aliases and advanced root overrides are documented in
[`../getting_started/forked_amgx_stack.md`](../getting_started/forked_amgx_stack.md).

Package imports remain lazy: importing `hdgfem` does not require CuPy, CUDA,
or PyAMGX.

## Standalone Runners

- `scripts/gpu/run_advection_reaction_cuda.py` exercises CuPy and raw-CUDA
  advection-reaction assembly, Cupyx/PyAMGX solves, and reconstruction.
- `scripts/gpu/run_diffusion_reaction_cuda.py` exercises the corresponding
  diffusion-reaction paths.
- `scripts/gpu/sweep_cuda_hdg.py` runs controlled basis, trace, quadrature,
  and matrix-format comparisons.
- `scripts/gpu/benchmark_advection_tsle_bsr.py` compares fused and three-stage
  face-BSR assembly without including AMGX solve time.

The standalone runners are diagnostic and benchmarking entry points. Normal
application workflows should use the public solver classes or maintained case
drivers.

## Assembly And Solve Pipeline

| Path | Assembly storage | Solve handoff | Role |
|---|---|---|---|
| CuPy | Vectorized device arrays followed by CuPy COO/CSR construction | Direct device Cupyx for compatible preconditioners; explicit host staging for host solvers/preconditioners | Device reference and parity path |
| Raw-CUDA COO | Reduced COO triplets and RHS emitted by raw kernels | CuPy COO-to-CSR, then Cupyx or PyAMGX | Inspectable correctness path |
| Raw-CUDA CSR | Values emitted directly into a known reduced CSR pattern | Direct device CSR view | Preferred large-run path |
| Raw-CUDA face BSR | Dense trace-face interaction blocks emitted directly into a block pattern | Native FB-HP-MG-PCG or direct PyAMGX BSR upload | Default eligible device format; CSR/COO remain explicit comparison/diagnostic paths |
| Direct CSR-to-AMGX | Existing device `indptr`, `indices`, and values | PyAMGX device upload/view | Avoids host staging and COO conversion |

Raw-CUDA CSR/BSR pattern construction and block-position maps remain on device.
The direct paths avoid both global COO-to-CSR reconstruction and large
per-element condensed tensor materialization.

The hybrid direct-BSR path and its classical-AMG hierarchy lifecycle are
documented in [`amgx_classical_bsr.md`](amgx_classical_bsr.md). Its retained
fine operator is BSR, while transfer and coarse operators remain scalar CSR.
For supported p=4--6 Legendre-modal Poisson systems, the default repeated-solve
path is instead HDGFEM's native [`FB-HP-MG-PCG`](face_hp_mg_pcg.md).

## Advection-Reaction Modes

The fused cooperative path builds local matrices from projected coefficients,
performs local factorization and solves, eliminates known boundary columns, and
emits the reduced operator. The `precomputed` local-assembly path and serial
`safe` LU mode remain compatibility and debugging references while the
cooperative path is qualified.

The explicit `raw_local_assembly="split3"` selector chooses TSLE-BSR, which
separates build, cooperative LU/solve, and Schur/BSR scatter so each stage can
autotune its launch shape. It preserves the same HDG algebra, requires BSR and
cooperative pivoted LU, and retains persistent intermediate workspaces. The
measured `p <= 6` gate keeps fused as the production default; p=7 is the
current TSLE promotion candidate, while p=8--9 is a spill-free but
memory-intensive experimental scope. See [`raw_cuda.md`](raw_cuda.md) for
timings and the
full support/reuse contract.

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
