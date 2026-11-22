"""Polynomial bases on reference simplices of dimension 1, 2 and 3.

The one-dimensional bases used for triangle edges (Lagrange polynomials on given
nodes, Legendre polynomials and Bernstein polynomials) were moved here unchanged
from :mod:`hybridge.core.quadrature` and :mod:`hybridge.core.space`, which still
expose them under their original names.
"""

from __future__ import annotations

import numpy as np

from hybridge.runtime.precision import REAL_DTYPE


def _lagrange_basis(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate 1D nodal Lagrange basis functions at ``points``."""
    nodes = np.asarray(nodes, dtype=REAL_DTYPE)
    points = np.asarray(points, dtype=REAL_DTYPE)
    values = np.ones((nodes.size, points.size), dtype=REAL_DTYPE)
    for i in range(nodes.size):
        for j in range(nodes.size):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return np.ascontiguousarray(values)


def _bernstein_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Tabulate the Bernstein trace basis at one-dimensional edge points."""
    from math import factorial

    r = 0.5 * (np.asarray(points, dtype=REAL_DTYPE) + 1.0)
    values = np.empty((order + 1, r.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        coeff = factorial(order) / (factorial(j) * factorial(order - j))
        values[j] = coeff * (1.0 - r) ** (order - j) * r ** j
    return np.ascontiguousarray(values)


def _legendre_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Tabulate the Legendre trace basis at one-dimensional edge points."""
    points = np.asarray(points, dtype=REAL_DTYPE)
    values = np.empty((order + 1, points.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        values[j] = np.polynomial.legendre.Legendre.basis(j)(points)
    return np.ascontiguousarray(values)


__all__: list[str] = []
