"""Geometry/registry diagnostics only; no PDE solve or time integration."""
import json

import numpy as np
import pytest

from hybridge.core.geometry import PolygonDomain
from hybridge.core.mesh import gmsh_smooth_star_mesh
from scripts.n_gamma.cases.geometry import (
    BASELINE_SIZES, STRESS_POLYGONIZATIONS, build_case_mesh,
)


SHIFTS = {'cartesian': 0., 'axisymmetric': 3.}


def polygons(outer_count, hole_count, shift=3.):
    """Construct the specified walls independently of the meshing routine."""
    theta = np.arange(outer_count) * 2*np.pi/outer_count
    radius = .70 * (1 + .32*np.cos(5*theta))
    outer = np.column_stack((shift + radius*np.cos(theta), radius*np.sin(theta)))
    theta = np.arange(hole_count) * 2*np.pi/hole_count
    hole = np.column_stack((shift + .28 + .12*np.cos(theta), .10 + .12*np.sin(theta)))
    return PolygonDomain(outer), PolygonDomain(hole)


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
@pytest.mark.parametrize('counts,h', tuple(zip(STRESS_POLYGONIZATIONS, BASELINE_SIZES)))
def test_stress_polygon_walls_normals_and_mesh_record(counts, h, geometry, tmp_path):
    pytest.importorskip('gmsh')
    outer_count, hole_count = counts
    shift = SHIFTS[geometry]
    result = build_case_mesh('H', h, geometry=geometry, outer_vertices=outer_count,
                             hole_vertices=hole_count, cache_dir=tmp_path)
    mesh = result.mesh
    outer, hole = polygons(*counts, shift)
    if geometry == 'axisymmetric':
        assert np.all(mesh.node_coords[:, 0] > 0)
    assert np.all(mesh.aff_jacs > 0)
    # Every original CAD vertex survives; boundary segments may be subdivided.
    nodes = mesh.node_coords[np.unique(mesh.edges[mesh.bnd_edges_inds])]
    for wall in (outer, hole):
        distances = np.linalg.norm(wall.vertices[:, None, :] - nodes[None, :, :], axis=-1)
        assert np.max(distances.min(axis=1)) < 1.e-12
    midpoints = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]].mean(axis=1)
    hole_mask = hole.boundary_distance(midpoints) < 1.e-11
    assert hole_mask.sum() >= hole_count
    assert (~hole_mask).sum() >= outer_count
    assert np.max(outer.boundary_distance(midpoints[~hole_mask])) < 1.e-11
    assert np.max(np.minimum(outer.boundary_distance(nodes), hole.boundary_distance(nodes))) < 1.e-11
    assert outer.boundary_distance(np.array([[shift + .28, .10]]))[0] > .12
    # Element outward normals must point from the fluid into the inner hole.
    boundary_lookup = dict(zip(mesh.bnd_edges_inds, hole_mask))
    for element, face in np.argwhere(np.isin(mesh.loc2glob_edge, mesh.bnd_edges_inds)):
        edge = mesh.loc2glob_edge[element, face]
        if boundary_lookup[edge]:
            midpoint = mesh.node_coords[mesh.edges[edge]].mean(axis=0)
            assert np.dot(mesh.normals[element, face], midpoint - (shift + .28, .10)) < 0
    centroids = mesh.node_coords[mesh.triangles].mean(axis=1)
    assert not hole.contains(centroids).any()
    assert outer.contains(centroids).all()
    assert 2*mesh.aff_jacs.sum() == pytest.approx(outer.area - hole.area, abs=1.e-12)
    assert len(mesh.node_coords) - len(mesh.edges) + mesh.num_tri == 0
    assert result.metadata['actual_h'] == mesh.h
    assert result.metadata['outer_vertices'] == outer_count
    assert result.metadata['hole_vertices'] == hole_count
    assert result.metadata['geometry'] == geometry and result.metadata['domain'] == 'H'
    json.dumps(result.metadata)
    again = build_case_mesh('H', h, geometry=geometry, outer_vertices=outer_count,
                            hole_vertices=hole_count, cache_dir=tmp_path)
    np.testing.assert_array_equal(again.mesh.node_coords, mesh.node_coords)


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
@pytest.mark.parametrize('h', BASELINE_SIZES)
def test_baseline_meshes(h, geometry, tmp_path):
    pytest.importorskip('gmsh')
    shift = SHIFTS[geometry]
    mesh = build_case_mesh('B', h, geometry=geometry, cache_dir=tmp_path).mesh
    np.testing.assert_allclose(mesh.node_coords.min(axis=0), (shift - 1, -1))
    np.testing.assert_allclose(mesh.node_coords.max(axis=0), (shift + 1, 1))
    assert 2*mesh.aff_jacs.sum() == pytest.approx(4.)
    with pytest.raises(ValueError, match='geometry'):
        build_case_mesh('B', h, geometry='slab')


@pytest.mark.parametrize('options', [
    {'hole_center': (4., 0.)}, {'hole_center': (3.5, .2)},
    {'hole_radius': -.1}, {'hole_center': (np.nan, 0)},
    {'hole_boundary_points': 2}, {'hole_boundary_points': 20.5},
])
def test_invalid_hole_rejected_before_meshing(options, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid geometry reached Gmsh')
    monkeypatch.setattr('hybridge.core.mesh._generate_gmsh_mesh', forbidden)
    params = dict(center=(3., 0.), radius=.7, amplitude=.224,
                  boundary_points=80, hole_center=(3.28, .1),
                  hole_radius=.12, hole_boundary_points=20)
    params.update(options)
    with pytest.raises(ValueError):
        gmsh_smooth_star_mesh(.2, **params)


def test_hole_parameters_participate_in_cache_key(monkeypatch):
    keys = []
    def capture(*args, **kwargs):
        keys.append(kwargs['cache_key_data'])
    monkeypatch.setattr('hybridge.core.mesh._generate_gmsh_mesh', capture)
    for options in ({}, {'hole_center': (.1, 0.)}, {'hole_boundary_points': 20},
                    {'hole_boundary_points': 40}, {'boundary_points': 80},
                    {'hole_radius': .2}):
        gmsh_smooth_star_mesh(.2, **{'hole_radius': .1, **options})
    assert len({json.dumps(key, sort_keys=True) for key in keys}) == len(keys)


def test_case_registry_exact_boundary_data_in_both_geometries(tmp_path):
    pytest.importorskip('gmsh')
    from scripts.n_gamma.cases import forcing
    from scripts.n_gamma.cases.registry import CASE_NAMES, CASES, get_case

    assert set(CASE_NAMES) == {'stationary_baseline', 'stationary_stress',
                               'transient_baseline', 'transient_stress'}
    assert set(CASES) == {(g, n) for g in SHIFTS for n in CASE_NAMES}
    for (geometry, name), case in CASES.items():
        shift = SHIFTS[geometry]
        assert get_case(name, geometry=geometry) is case and case.geometry == geometry
        assert case.domain == ('B' if name.endswith('baseline') else 'H')
        assert case.boundary_mode == 'eliminate' and not case.bohm_conditions
        mesh = case.build_mesh(cache_dir=tmp_path).mesh
        first, second = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]].mean(axis=1).T
        if geometry == 'axisymmetric':
            assert (first > 0).all()
        x, y = first - shift, second
        for time in (.0, .41):
            kwargs = dict(stationary=case.stationary)
            np.testing.assert_allclose(case.density_boundary_at(time)(first, second),
                                       forcing.n_e(x, y, time, **kwargs))
            np.testing.assert_allclose(case.momentum_boundary_at(time)(first, second),
                                       forcing.Gamma_e(x, y, time, **kwargs))
            np.testing.assert_allclose(case.density_source_at(time)(first, second),
                                       forcing.S_n(x, y, time, geometry=geometry, **kwargs))
            np.testing.assert_allclose(case.momentum_source_at(time)(first, second),
                                       forcing.S_Gamma(x, y, time, geometry=geometry, **kwargs))
        b_1, b_2 = case.b_poloidal(first, second)
        np.testing.assert_allclose(np.stack((b_1, b_2), axis=-1), forcing.b_p(x, y))
    with pytest.raises(ValueError, match='unknown n–Gamma case'):
        get_case('missing', geometry='cartesian')
    with pytest.raises(ValueError, match='geometry'):
        get_case('transient_stress', geometry='slab')
    with pytest.raises(TypeError):
        get_case('transient_stress')
