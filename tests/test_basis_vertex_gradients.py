"""Gradients of the triangle bases of ``hybridge.core.basis`` at the vertices of the reference triangle."""

from __future__ import annotations

import numpy as np
import pytest

from hybridge.core import basis as legacy


@pytest.mark.parametrize("name, evaluate", [("dub_orth", legacy.evaluate_dubiner_gradients),
                                            ("hier_C0", legacy.evaluate_hierarchical_c0_gradients)])
def test_legacy_vertex_gradients_equal_interior_limit(name, evaluate):
    """The triangle kernels of ``hybridge.core.basis`` give, at each vertex, the limit of the gradient from inside.

    Before the fix, at the vertex (-1, 1) the Dubiner kernel returned
    d/dxi = r + 1 instead of C(r + 3, 3) for p = 1 and d/deta = r (r + 2)/2 instead
    of r (r + 1)(r + 2)/4 for p = 0, and the modified kernel set the derivative of
    the collapsed coordinate to 0, so that the edge modes through that vertex lost
    their xi-derivative. The limit is extrapolated linearly from interior points at
    distances h and 2h along the segment to the centroid. The test uses only
    ``hybridge.core.basis``.
    """
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    centroid = np.array([-1.0 / 3.0, -1.0 / 3.0])
    direction = (centroid - vertices) / np.linalg.norm(centroid - vertices, axis=1, keepdims=True)
    h = 1e-5
    for order in range(1, 9):
        at_vertices = evaluate(order, vertices)
        limit = 2.0 * evaluate(order, vertices + h * direction) - evaluate(order, vertices + 2 * h * direction)
        scale = max(1.0, np.abs(at_vertices).max())
        assert np.abs(at_vertices - limit).max() <= 1e-6 * scale, (name, order)
