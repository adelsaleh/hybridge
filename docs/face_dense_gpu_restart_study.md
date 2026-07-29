# GPU GMRES restart and orthogonality study

The T600 measurements show that batched CGS and CGS2 remove the dominant
DOT/AXPY synchronization overhead.  The remaining numerical question is how
large the restart space must be for the one-element ASM preconditioner, and
whether one-pass CGS preserves a sufficiently orthogonal Arnoldi basis on more
difficult systems.

## Optional orthogonality monitor

`restarted_gmres_cupy` now accepts:

```python
result = restarted_gmres_cupy(
    operator,
    rhs,
    restart=50,
    orthogonalization="cgs2",
    monitor_orthogonality=True,
)
```

After each restart cycle it forms the small Gram matrix

\[
G = V V^T
\]

and records:

\[
\|G-I\|_F,
\qquad
\max_{i\ne j}|G_{ij}|,
\qquad
\max_i |G_{ii}-1|.
\]

The monitor is disabled by default.  It allocates a small Gram matrix and
copies it to the CPU, so it must not be enabled for authoritative
performance timing.

## Restart study

Run, for example:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_restart.py \
    --mesh 64 \
    --order 1 \
    --restarts 10 20 30 50 75 100 \
    --orthogonalizations cgs cgs2 \
    --preconditioner asm \
    --max-iterations 2000 \
    --rtol 1e-8 \
    --output-prefix results/t600_restart_64_p1
```

The script performs two solves per configuration:

1. an unmonitored synchronized solve for time-to-solution;
2. a separate diagnostic solve with orthogonality monitoring.

It reports convergence, iteration count, restart cycles, basis memory,
orthogonality defects, and the difference from the best converged solution.

For a second difficult case:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_restart.py \
    --mesh 64 \
    --order 4 \
    --restarts 30 50 75 100 150 \
    --orthogonalizations cgs cgs2 \
    --preconditioner asm \
    --max-iterations 2000 \
    --rtol 1e-8 \
    --output-prefix results/t600_restart_64_p4
```

## Interpretation

- If CGS and CGS2 have comparable defects and identical convergence, CGS is
  the faster production option for that problem family.
- If CGS loses orthogonality while CGS2 remains stable, use CGS2 or add
  selective reorthogonalization.
- If increasing the restart dimension sharply reduces iteration count and
  total time, the previous stagnation was primarily a restart effect.
- If even large restart dimensions do not converge economically, a stronger
  preconditioner is required.  The next candidate is the polynomial
  preconditioner described in the HDG GPU paper.
