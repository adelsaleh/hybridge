"""Warm-started solve series on a reused operator/preconditioner (time-stepping benchmark).

Each series perturbs the right-hand side by delta*||b|| per step along a fixed seeded random
direction and starts every solve from the previous step's solution, so iteration counts show
what a time stepper saves by warm starting. Only the solve is timed; RHS/guess uploads are
not. Each warm solve is checked by its own true residual (the manufactured-solution checks do
not apply to perturbed right-hand sides), and a non-converged warm solve is recorded, never a
job failure.
"""
import numpy as np


def direction(size, seed=261027):
    vector = np.random.default_rng(seed).standard_normal(size)
    return vector/np.linalg.norm(vector)


def steps(rhs, deltas, count, seed=261027):
    """Yield (delta, step, rhs_k) for every series, each starting again from the base RHS."""
    unit = direction(rhs.size, seed)*float(np.linalg.norm(rhs))
    for delta in deltas:
        current = np.array(rhs, dtype=np.float64, copy=True)
        for step in range(1, count+1):
            current = current+delta*unit
            yield float(delta), step, current


def relative_residual(matvec, x, rhs):
    norm = float(np.linalg.norm(rhs))
    residual = float(np.linalg.norm(matvec(x).reshape(-1)-rhs))
    return residual/norm if norm else residual


def summarize(warm):
    """Per-delta medians over measured warm solves, for compact reports."""
    out = {}
    for delta in sorted({w['delta'] for w in warm}):
        rows = [w for w in warm if w['delta'] == delta]
        out[f'{delta:g}'] = dict(solves=len(rows), passed=sum(w['passed'] for w in rows),
                                 median_iterations=float(np.median([w['iterations'] for w in rows])),
                                 median_solve_ms=float(np.median([w['solve_ms'] for w in rows])))
    return out
