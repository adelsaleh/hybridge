"""Walking point evaluation of cellwise polynomials (core/transfer.CellwiseEvaluator); no solve."""
import numpy as np
import pytest

from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.core.transfer import CellwiseEvaluator, evaluate_field_at_points, triangle_neighbours


def random_field(basis="bernstein", order=4, n=6, seed=0):
    space = DGSpace(rectangle_mesh(n, n), order, basis_type=basis)
    rng = np.random.default_rng(seed)
    return space.field(rng.standard_normal((space.mesh.num_tri, space.el_dof)))


def points_in_cells(mesh, cells, rng):
    weights = rng.dirichlet(np.ones(3), len(cells))
    return np.einsum("nk,nkd->nd", weights, mesh.node_coords[mesh.triangles[cells]])


def test_neighbours_are_symmetric_and_share_the_opposite_edge():
    mesh = rectangle_mesh(5, 4)
    neighbours = triangle_neighbours(mesh.triangles)
    assert neighbours.shape == (mesh.num_tri, 3)
    for cell, row in enumerate(neighbours):
        for vertex, other in enumerate(row):
            if other < 0:
                continue
            assert cell in neighbours[other]
            edge = set(np.delete(mesh.triangles[cell], vertex))
            assert edge <= set(mesh.triangles[other])
    boundary_edges = np.count_nonzero(neighbours < 0)
    assert boundary_edges == 2 * (5 + 4)


@pytest.mark.parametrize("basis", ["bernstein", "dub_orth"])
def test_walk_matches_located_evaluation_from_any_nearby_start(basis):
    field = random_field(basis)
    mesh = field.space.mesh
    rng = np.random.default_rng(1)
    cells = rng.integers(0, mesh.num_tri, 4000)
    points = points_in_cells(mesh, cells, rng)
    reference = evaluate_field_at_points(field, points)
    evaluator = CellwiseEvaluator(mesh.node_coords, mesh.triangles, field.space.order, basis)
    neighbour_hints = np.where(triangle_neighbours(mesh.triangles)[cells, 0] >= 0,
                               triangle_neighbours(mesh.triangles)[cells, 0], cells)
    for hints in (cells, neighbour_hints, None):
        values, located = evaluator(field.coeffs, points, hints=hints)
        assert np.all(located >= 0)
        np.testing.assert_allclose(values, reference, rtol=0, atol=1e-12 * np.abs(reference).max())


def test_points_outside_the_mesh_are_reported_not_extrapolated():
    field = random_field()
    mesh = field.space.mesh
    evaluator = CellwiseEvaluator(mesh.node_coords, mesh.triangles, field.space.order)
    outside = np.array([[1.5, 0.0], [0.0, -1.2], [3.0, 3.0]])
    values, located = evaluator(field.coeffs, outside)
    assert np.all(located == -1) and np.all(np.isnan(values))
