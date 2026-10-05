"""Static study of KKT positivity projections for DG densities; no time integration.

Two bounded checks support ``docs/development/plans/positivity_kkt_bdf2.md``:

``states``
    Apply the element KKT projection to density checkpoints already recorded by
    ``scripts/reports/record_gpu_showcase.py`` (``*.restart.npz``) and report
    how many elements violate positivity, the correction size, the mass returned
    and the element-solver cost, against Zhang-Shu scaling.
``order``
    L2-project a smooth nonnegative function that touches zero onto DG(p) on
    refined meshes and compare the L2 errors of the plain projection, the
    point-constrained KKT projection and Bernstein-coefficient constraints.

The element projection solves, per element K with orthogonal (``dub_orth``)
modes and diagonal mass weights D,

    min 1/2 (x - y)^T D (x - y)  s.t.  V x >= 0 at the points S  [and x_0 = y_0],

as a least-distance problem by the Lawson-Hanson NNLS active-set method: a
Newton step on the KKT system for each active set, with the step-length
safeguard that keeps it finite on the degenerate active sets of near-zero
elements. Elements whose mean is negligible take the constant mean; Zhang-Shu
scaling is the feasible fallback.
"""
from __future__ import annotations

from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path
import time

import numpy as np
from scipy.optimize import nnls

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.core.basis import evaluate_bernstein_basis


ROOT = Path(__file__).resolve().parents[3]


def lattice(n):
    """Equispaced order-``n`` lattice of the reference triangle ``(-1, 1)``."""
    return np.array([(-1 + 2*i/n, -1 + 2*j/n) for i in range(n+1) for j in range(n+1-i)])


def least_distance(W, h):
    """Return argmin ||z|| subject to W z >= h by the NNLS reduction of the LDP."""
    k = W.shape[1]
    E = np.vstack([W.T, h[None, :]])
    target = np.zeros(k + 1)
    target[-1] = 1.0
    u, _ = nnls(E, target, maxiter=50 * W.shape[0])
    residual = E @ u - target
    if np.linalg.norm(residual) < 1e-14:
        raise ValueError("infeasible least-distance problem")
    return -residual[:k] / residual[k]


def _project_rows(args):
    """Project coefficient rows onto {V x >= 0}, optionally keeping mode 0 (the mean)."""
    rows, V, D, preserve_mean = args
    free = np.arange(1, V.shape[1]) if preserve_mean else np.arange(V.shape[1])
    root = np.sqrt(D[free])
    W = V[:, free] / root[None, :]
    out = np.zeros((len(rows), len(free)))
    for i, y in enumerate(rows):
        b = y @ V.T
        scale = np.abs(b).max()
        if preserve_mean:
            mean = y[0]
            theta = mean / (mean - b.min()) if b.min() < 0 else 1.0
            fallback = -(1 - theta) * y[free] * root
            if mean <= 1e-8 * scale:
                out[i] = -y[free] * root
                continue
        else:
            fallback = -y[free] * root
            fallback[0] += max(y[0], 0.0) * root[0]
        try:
            z = least_distance(W, -b)
            if (not np.all(np.isfinite(z)) or (z @ W.T + b).min() < -1e-12 * scale
                    or z @ z > fallback @ fallback):
                z = fallback
        except ValueError:
            z = fallback
        out[i] = z
    return out


def project(c, V, D, pool, *, preserve_mean=True, chunk=400):
    """Element KKT projection of all flagged rows of ``c``; returns (x, flagged count)."""
    free = np.arange(1, c.shape[1]) if preserve_mean else np.arange(c.shape[1])
    flagged = np.flatnonzero((c @ V.T).min(axis=1) < 0)
    chunks = [flagged[s:s + chunk] for s in range(0, len(flagged), chunk)]
    x = c.copy()
    for rows, z in zip(chunks, pool.map(_project_rows, [(c[r], V, D, preserve_mean) for r in chunks])):
        x[rows[:, None], free[None, :]] += z / np.sqrt(D[free])[None, :]
    return x, len(flagged)


def reference_tables(order):
    """Mode weights, volume-quadrature, p-lattice and (2p+2)-lattice evaluation tables."""
    space = DGSpace(rectangle_mesh(1, 1), order, basis_type="dub_orth")
    D = np.diag(space.mass()[0]) / space.mesh.aff_jacs[0]
    return (D, space.quad_data.bas_of_quads.T, space.reference.basis_at(lattice(order)),
            space.reference.basis_at(lattice(2 * order + 2)))


def study_states(paths, workers):
    """Report violations, projection size, mass return and cost for recorded checkpoints."""
    D, VQ, VL, VS = reference_tables(6)
    S = np.vstack([VQ, VL])
    with Pool(workers) as pool:
        for path in paths:
            data = np.load(path, allow_pickle=False)
            c = np.ascontiguousarray(data["rho"])
            p = data["node_coords"][data["triangles"]]
            area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                                - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
            mass = area @ c[:, 0]
            values = c @ S.T
            flagged, nonneg = values.min(axis=1) < 0, c[:, 0] >= 0
            start = time.perf_counter()
            x = c.copy()
            local = np.flatnonzero(flagged & nonneg)
            negative = np.flatnonzero(flagged & ~nonneg)
            x[local], _ = project(c[local], S, D, pool, preserve_mean=True)
            x[negative], _ = project(c[negative], S, D, pool, preserve_mean=False)
            excess = area @ x[:, 0] - mass
            x *= 1 - excess / (area @ x[:, 0])          # multiplicative mass return keeps signs
            elapsed = time.perf_counter() - start
            weight = (area / 2)[:, None] * D
            size = lambda d: np.sqrt((weight * d * d).sum() / (weight * c * c).sum())
            scaled = c.copy()
            low = values.min(axis=1)
            theta = np.where(flagged & nonneg, c[:, 0] / np.maximum(c[:, 0] - low, 1e-300), 1.0)
            scaled[:, 1:] *= theta[:, None]
            scaled[~nonneg] = 0.0
            scaled *= 1 - (area @ scaled[:, 0] - mass) / (area @ scaled[:, 0])
            print(f"{Path(path).name}: t={int(data['step'])*float(data['dt']):.3g} dt={float(data['dt']):g} "
                  f"flagged {flagged.sum()} (negative mean {negative.size}) | min at S {values.min():.2e} -> "
                  f"{(x @ S.T).min():.1e} | ||dρ||/||ρ|| KKT {size(x - c):.2e} Zhang-Shu {size(scaled - c):.2e} | "
                  f"mass returned {excess / mass:.1e}, mass error {abs(area @ x[:, 0] - mass) / mass:.0e} | "
                  f"{elapsed:.2f} s on {workers} workers", flush=True)


def study_order(workers):
    """Compare L2 errors of plain, point-KKT and Bernstein-constrained projections."""
    def f(x, y):
        """Nonnegative, non-polynomial test density touching zero along curves."""
        return np.sin(3.0 * (x * x + y * y) + x) ** 2

    with Pool(workers) as pool:
        for order, levels in ((3, (4, 8, 16, 32, 64)), (6, (4, 8, 16, 32))):
            print(f"p={order}: nx | L2 projection | point KKT | Bernstein >= 0 (observed rates)")
            previous = None
            for nx in levels:
                space = DGSpace(rectangle_mesh(nx, nx), order, basis_type="dub_orth",
                                volume_quad_1d=2 * order + 6)
                c = space.project_callable(f).coeffs
                D = np.diag(space.mass()[0]) / space.mesh.aff_jacs[0]
                VL = space.reference.basis_at(lattice(order))
                S = np.vstack([space.quad_data.bas_of_quads.T, VL])
                bernstein = np.linalg.solve(evaluate_bernstein_basis(order, lattice(order)), VL)
                points, weights = space.mapped_quads(), space.quad_data.Krf_w
                jac = space.mesh.aff_jacs[:, None]

                def error(x):
                    """Quadrature L2 error against the exact density."""
                    diff = x @ space.quad_data.bas_of_quads - f(points[..., 0], points[..., 1])
                    return np.sqrt((jac * weights * diff * diff).sum())

                errors = (error(c), error(project(c, S, D, pool)[0]), error(project(c, bernstein, D, pool)[0]))
                rates = ("" if previous is None else
                         " (" + ", ".join(f"{np.log2(a / b):.1f}" for a, b in zip(previous, errors)) + ")")
                print(f"  {nx:3d} | " + " | ".join(f"{e:.2e}" for e in errors) + rates, flush=True)
                previous = errors


def main():
    """Command-line entry point."""
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    states = sub.add_parser("states", help="project recorded *.restart.npz density checkpoints")
    states.add_argument("checkpoints", nargs="+", type=Path)
    order = sub.add_parser("order", help="static spatial-order check of the projections")
    for command in (states, order):
        command.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.command == "states":
        study_states(args.checkpoints, args.workers)
    else:
        study_order(args.workers)


if __name__ == "__main__":
    main()
