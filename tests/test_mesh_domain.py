"""Triangle-area sampling and clearance on meshes with several boundaries."""

import numpy as np
import pytest

from hybridge.cases.profiles import sample_gaussian_blob_field
from hybridge.core.geometry import MeshDomain
from hybridge.core.mesh import DGMesh, rectangle_mesh


def square_with_hole():
    """Build four quadrilateral bands, split into eight triangles."""
    nodes = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2],
                      [-1, -1], [1, -1], [1, 1], [-1, 1]])
    triangles = []
    for i in range(4):
        j = (i+1) % 4
        triangles.extend([[i, j, j+4], [i, j+4, i+4]])
    return DGMesh(nodes, triangles)


def test_mesh_domain_membership_distance_and_shapes_include_hole():
    domain = MeshDomain(square_with_hole())
    assert domain.area == pytest.approx(12)
    np.testing.assert_array_equal(domain.bounds, [[-2, -2], [2, 2]])
    points = np.array([[[0, 0], [1.5, 0], [3, 0]],
                       [[1, 0], [2, 0], [1.5, 1.5]]])
    np.testing.assert_array_equal(domain.contains(points), [[False, True, False], [True, True, True]])
    np.testing.assert_allclose(domain.boundary_distance(points), [[1, .5, 1], [0, 0, .5]])
    assert domain.contains([0, 1.5]).shape == ()
    assert domain.contains(np.empty((0, 2))).shape == (0,)
    assert domain.boundary_distance(np.empty((0, 2))).shape == (0,)
    for bad in (0, [1, 2, 3], [[np.nan, 0]]):
        with pytest.raises(ValueError, match="points"):
            domain.contains(bad)


def test_mesh_domain_sample_is_area_uniform_across_unequal_disconnected_triangles():
    # Triangle areas are 1/2 and 2; equal element sampling would be incorrect.
    mesh = DGMesh([[0, 0], [1, 0], [0, 1], [10, 0], [12, 0], [10, 2]],
                  [[0, 1, 2], [3, 4, 5]])
    domain = MeshDomain(mesh)
    points = domain.sample_uniform(20000, np.random.default_rng(17))
    selected = points[:, 0] > 5
    assert selected.mean() == pytest.approx(.8, abs=.01)
    np.testing.assert_allclose(points[~selected].mean(axis=0), [1/3, 1/3], atol=.01)
    np.testing.assert_allclose(points[selected].mean(axis=0), [10+2/3, 2/3], atol=.02)
    assert domain.contains(points).all()
    assert not domain.contains([[5, .1]])[0]


def test_mesh_domain_reproducible_gaussians_keep_support_clear_of_all_walls():
    domain = MeshDomain(square_with_hole())
    first = sample_gaussian_blob_field(domain, (20, 20), (.02, .04), seed=17, wall_clearance=.05)
    second = sample_gaussian_blob_field(domain, (20, 20), (.02, .04), seed=17, wall_clearance=.05)
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.strengths, second.strengths)
    assert domain.contains(first.centers).all()
    assert np.all(domain.boundary_distance(first.centers) > first.cutoff*first.sigmas+.05)
    np.testing.assert_array_equal(first(np.array([1, 2]), np.array([0, 0])), 0)
    for sigma in (.02, .04):
        assert first.strengths[first.sigmas == sigma].sum() == pytest.approx(0, abs=1e-12)


def test_mesh_domain_chunked_boundary_queries_agree_with_rectangle():
    domain = MeshDomain(rectangle_mesh(300, 1))  # More than one boundary chunk.
    points = np.random.default_rng(3).uniform(-.9, .9, size=(700, 2))
    assert domain.contains(points).all()
    np.testing.assert_allclose(domain.boundary_distance(points), 1-np.max(abs(points), axis=1))


@pytest.mark.parametrize("count,clearance", [(0, 0), (1.5, 0), (np.nan, 0), (1, -.1), (1, np.inf), (1, 2)])
def test_mesh_domain_rejects_invalid_sampling_arguments(count, clearance):
    with pytest.raises(ValueError):
        MeshDomain(square_with_hole()).sample_uniform(count, np.random.default_rng(1), clearance=clearance)


def test_mesh_domain_rejects_impossible_clearance_with_a_bounded_search():
    with pytest.raises(ValueError, match="clearance"):
        MeshDomain(square_with_hole()).sample_uniform(1, np.random.default_rng(1), clearance=.9)
