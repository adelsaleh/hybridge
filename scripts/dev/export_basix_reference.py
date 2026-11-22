"""Write reference data computed by basix for the tests of :mod:`hybridge.core.simplex`.

Run this script with any Python environment that has basix 0.9 or newer; it does
not import hybridge. The single argument is the output ``.npz`` file, for example

    python -I scripts/dev/export_basix_reference.py tests/data/simplex_basix_reference.npz

Coordinates. basix uses the reference simplices with vertices ``0`` and the unit
vectors, ``X in [0, 1]^d``. The file stores every point exactly as basix returns
it. The reference simplices of hybridge have vertices ``-1`` and the unit vectors
shifted by ``-1``; the affine map ``x = 2 X - 1`` sends the basix vertex ``k`` to
the hybridge vertex ``k`` and multiplies volumes by ``2^d``.

Arrays stored (``<cell>`` is ``triangle`` or ``tetrahedron``):

- ``<cell>_gll_isaac_p<p>`` and ``<cell>_equispaced_p<p>``, ``p = 1..8``: the
  Lagrange points of degree ``p`` (``basix.create_lattice`` with the points on the
  boundary included), shape ``(n_p, d)`` with ``n_p = C(p + d, d)``, in the lattice
  order of basix. ``gll_isaac`` is the recursive construction of Isaac (SIAM J. Sci.
  Comput. 42, 2020) from the Gauss-Lobatto-Legendre points of the interval.
- ``<cell>_legendre_points``, shape ``(m, d)``: points drawn from the uniform
  distribution on the basix simplex with a fixed seed, and
  ``<cell>_legendre_p<p>``, ``p = 0..6``, shape ``(C(p + d, d), m)``: the
  orthonormal (Legendre) polynomials of basix, of total degree at most ``p`` and
  orthonormal in ``L2`` of the basix simplex, evaluated at these points.
- ``<cell>_xiao_gimbutas_k<k>_points`` (shape ``(q, d)``) and
  ``<cell>_xiao_gimbutas_k<k>_weights`` (shape ``(q,)``), ``k = 1..15``: the
  Xiao-Gimbutas rule exact for polynomials of degree ``k`` on the basix simplex; the
  weights add up to the basix volume ``1/d!``.
- ``basix_version``: the version string of the basix that wrote the file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LATTICE_DEGREES = range(1, 9)
LEGENDRE_DEGREES = range(0, 7)
QUADRATURE_DEGREES = range(1, 16)
LEGENDRE_POINT_COUNT = 24
SEED = 20261122


def collect(basix) -> dict[str, np.ndarray]:
    """Return the arrays described in the module docstring, computed with the module ``basix``."""
    cells = {"triangle": (basix.CellType.triangle, 2), "tetrahedron": (basix.CellType.tetrahedron, 3)}
    rng = np.random.default_rng(SEED)
    arrays: dict[str, np.ndarray] = {"basix_version": np.array(basix.__version__)}
    for name, (cell, dim) in cells.items():
        for p in LATTICE_DEGREES:
            arrays[f"{name}_gll_isaac_p{p}"] = np.asarray(
                basix.create_lattice(cell, p, basix.LatticeType.gll, True, basix.LatticeSimplexMethod.isaac),
                dtype=np.float64,
            )
            arrays[f"{name}_equispaced_p{p}"] = np.asarray(
                basix.create_lattice(cell, p, basix.LatticeType.equispaced, True), dtype=np.float64
            )
        # Uniform points of the simplex {X >= 0, sum X <= 1}: the last d of d + 1 Dirichlet(1) coordinates.
        points = rng.dirichlet(np.ones(dim + 1), size=LEGENDRE_POINT_COUNT)[:, 1:]
        points = np.ascontiguousarray(points, dtype=np.float64)
        arrays[f"{name}_legendre_points"] = points
        for p in LEGENDRE_DEGREES:
            arrays[f"{name}_legendre_p{p}"] = np.asarray(
                basix.tabulate_polynomials(basix.PolynomialType.legendre, cell, p, points), dtype=np.float64
            )
        for k in QUADRATURE_DEGREES:
            quadrature_points, weights = basix.make_quadrature(
                cell, k, rule=basix.QuadratureType.xiao_gimbutas
            )
            arrays[f"{name}_xiao_gimbutas_k{k}_points"] = np.asarray(quadrature_points, dtype=np.float64)
            arrays[f"{name}_xiao_gimbutas_k{k}_weights"] = np.asarray(weights, dtype=np.float64)
    return arrays


def main(argv: list[str] | None = None) -> int:
    """Parse the output path, compute the arrays with basix and write them to one compressed ``.npz`` file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="path of the .npz file to write")
    args = parser.parse_args(argv)
    import basix

    major, minor = (int(part) for part in basix.__version__.split(".")[:2])
    if (major, minor) < (0, 9):
        raise SystemExit(f"basix 0.9 or newer is required; found {basix.__version__}")
    arrays = collect(basix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"wrote {len(arrays)} arrays to {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
