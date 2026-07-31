# Production GPU GMRES and convergence safeguards

This stage freezes the incomplete element-local GPU assembly work and focuses
on turning the validated face-dense solver into a robust reusable component.
The low-level CUDA GMRES remains available, while
`CuPyProductionGMRESSolver` enables conservative safeguards by default.

## Safeguards

The solver recomputes the unpreconditioned true residual

\[
r_k=b-Ax_k
\]

at the initial guess and after every restart-cycle update.  The recomputed
residual is used to seed the next Arnoldi cycle, so each restart boundary is
also an exact residual-replacement point.  Convergence is accepted only from
this true residual, never from the preconditioned Givens estimate alone.

Explicit terminal states are now available:

- `converged`;
- `max_iterations`;
- `breakdown`;
- `stagnated`;
- `diverged`;
- `non_finite`.

The result includes a human-readable `termination_reason` and one
`CuPyGMRESCycleRecord` per completed restart cycle.  Each cycle record contains
its iteration interval, orthogonalization mode, start/end true residuals,
reduction factor, stagnation count, and whether a CGS-to-CGS2 switch was
scheduled.

## CGS-to-CGS2 fallback

One-pass CGS is substantially faster on the T600.  For robustness, the
production interface can measure the small Arnoldi Gram matrix once per
restart cycle while CGS is active.  If

\[
\max_{i\ne j}|v_i^T v_j|
\]

exceeds the configured threshold, subsequent cycles use CGS2.  The automatic
threshold is `1e-8` in float64 and `1e-3` in float32.  The low-level solver
keeps this monitor disabled unless explicitly requested, preserving previous
benchmark behavior.

## Stagnation and divergence

After every completed cycle, a stagnation counter is incremented when the true
residual has failed to decrease by at least the configured relative amount.
The production default terminates after eight consecutive stagnant cycles.
The divergence guard terminates when the true residual exceeds a configurable
multiple of the initial residual; the default factor is `1e6`.

## Usage

```python
from hdgfem.backends.cupy_solver import (
    CuPyProductionGMRESOptions,
    CuPyProductionGMRESSolver,
)

options = CuPyProductionGMRESOptions(
    restart=75,
    max_iterations=2000,
    rtol=1.0e-8,
    orthogonalization="cgs",
    cgs2_fallback_threshold="auto",
    stagnation_cycles=8,
)

solver = CuPyProductionGMRESSolver(
    operator,
    preconditioner=asm_polynomial,
    options=options,
)
result = solver.solve(rhs)
```

Set `raise_on_failure=True` in the options or on an individual solve to raise
`CuPyGMRESFailure` instead of returning a nonconverged result.

## T600 validation

```bash
PYTHONPATH=. pytest -q \
    tests/test_cupy_solver.py \
    tests/test_cupy_gmres.py
```

Representative production solve:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_production_solver.py \
    --mesh 64 \
    --order 1 \
    --operator raw \
    --preconditioner asm_poly \
    --asm-application fused \
    --polynomial-degree 18 \
    --restart 75 \
    --rtol 1e-8
```

For the `p=4` case, use restart 100 based on the T600 restart study.

## Scope boundary

This stage does not finish quadrature-to-Schur GPU assembly.  The supported
production path remains CPU local condensation followed by exact GPU global
face-dense assembly and a fully GPU-resident iterative solve.  Full local GPU
assembly remains isolated as future work and is not required by this solver.
