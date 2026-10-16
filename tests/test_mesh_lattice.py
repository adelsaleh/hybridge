"""Lattice orientation of triangle meshes (``edge_lattice_orientation``)."""

import math

import numpy as np
import pytest

from hybridge.core.mesh import DGMesh, edge_lattice_orientation, rectangle_mesh


def equilateral_lattice(n: int, angle_deg: float) -> DGMesh:
    """Equilateral triangles of side 1/n on an n x n rhombic patch, rotated by ``angle_deg``."""
    a, b = np.array([1.0, 0.0]), np.array([0.5, math.sqrt(3.0) / 2.0])
    index = {}
    nodes = []
    for i in range(n + 1):
        for j in range(n + 1):
            index[i, j] = len(nodes)
            nodes.append((i * a + j * b) / n)
    triangles = []
    for i in range(n):
        for j in range(n):
            triangles.append([index[i, j], index[i + 1, j], index[i, j + 1]])
            triangles.append([index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]])
    t = math.radians(angle_deg)
    rotation = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return DGMesh(np.asarray(nodes) @ rotation.T, np.asarray(triangles))


@pytest.mark.parametrize("angle, nearest", [(0.0, 0.0), (30.0, 30.0), (12.0, 12.0), (50.0, 10.0), (-7.0, 7.0)])
def test_equilateral_lattice_is_coherent_with_the_angle_of_its_nearest_edge_family(angle, nearest):
    result = edge_lattice_orientation(equilateral_lattice(8, angle))
    assert result["coherence"] == pytest.approx(1.0, abs=1e-12)
    assert result["angle"] == pytest.approx(nearest, abs=1e-9)
    assert result["equilateral"] == 1.0


def test_split_squares_have_one_third_coherence_and_no_equilateral_triangle():
    result = edge_lattice_orientation(rectangle_mesh(100), box=((-0.5, 0.5), (-0.5, 0.5)))
    assert result["coherence"] == pytest.approx(1.0 / 3.0, abs=0.01)
    assert result["equilateral"] == 0.0
    assert 0 < result["triangles"] < 2 * 100 * 100


def test_empty_box_is_rejected():
    with pytest.raises(ValueError, match="box"):
        edge_lattice_orientation(rectangle_mesh(4), box=((5.0, 6.0), (5.0, 6.0)))
