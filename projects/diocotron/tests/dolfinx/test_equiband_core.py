from dataclasses import FrozenInstanceError, replace
import numpy as np
import pytest

from projects.diocotron.dolfinx.equiband.config import BandConfig, SolverConfig
from projects.diocotron.dolfinx.equiband.nonlinearities import Window
from projects.diocotron.dolfinx.equiband.geometry import (
    PackedMesh,
    SITES,
    VANDERMONDE_INVERSE,
    _deduplicate_vertices,
    _sample_torsion_slices,
    atlas_signature,
    basis,
    build_atlas,
)
from projects.diocotron.dolfinx.equiband.crossings import _crossings_numpy, _crossings_numba, BandObservableEvaluator


@pytest.mark.parametrize("kind", ["logistic", "mollified"])
def test_window_derivatives_and_primitive(kind):
    w = Window(BandConfig(0.1, 0.01, kind))
    q = np.linspace(-.1, .7, 241)
    h, m = 1e-7, .3
    assert np.all((w.value(q, m) >= 0) & (w.value(q, m) <= 1))
    np.testing.assert_allclose((w.value(q+h, m)-w.value(q-h, m))/(2*h), w.derivative(q, m), atol=1e-7)
    np.testing.assert_allclose((w.value(q, m+h)-w.value(q, m-h))/(2*h), w.midpoint_derivative(q, m), atol=1e-7)
    np.testing.assert_allclose((w.primitive(q+h, m)-w.primitive(q-h, m))/(2*h), w.value(q, m), atol=1e-7)
    assert w.primitive(0., m) == pytest.approx(0., abs=1e-15)
    with np.errstate(over="raise", invalid="raise"):
        assert w.primitive(1e100, m) == pytest.approx(.1, abs=1e-11)
        np.testing.assert_allclose(w.derivative(np.array([-1e100, 1e100]), m), 0.)


def test_config_fixed_width_and_validation(tmp_path):
    config = BandConfig()
    with pytest.raises(FrozenInstanceError):
        config.threshold_width_delta = .3
    for m in np.linspace(.02, .2, 20):
        lo, hi = config.thresholds(m)
        assert hi-lo == pytest.approx(config.threshold_width_delta, abs=5e-17)
    with pytest.raises(ValueError, match="relative smoothing"):
        BandConfig(smoothing_mode="relative_to_delta", relative_epsilon=0.)
    with pytest.raises(ValueError, match="positive"):
        BandConfig(epsilon=0.)
    with pytest.raises(ValueError, match="separate active-set"):
        BandConfig(kind="indicator")
    path = tmp_path / "old.json"
    path.write_text('{"physical_width": 0.1}')
    with pytest.raises(ValueError, match="obsolete"):
        SolverConfig.load(path)


def test_atlas_signature_is_semantic_not_a_raw_field_byte_hash():
    config = SolverConfig()
    first = atlas_signature("mesh", config.signature)
    assert first == atlas_signature("mesh", config.signature)
    assert first != atlas_signature("refined-mesh", config.signature)
    assert first != atlas_signature("mesh", replace(config, number_of_rays=128).signature)


def test_mesh_vertex_deduplication_crosses_spatial_hash_bin_boundaries():
    """Ulp-close shared corners must not become artificial mesh cracks.

    The first two points deliberately lie on opposite sides of a hash-bin
    boundary while remaining much closer than the topology tolerance.  This
    is the small deterministic regression for the fine curved-horseshoe mesh.
    """
    tolerance = 1e-6
    points = np.array([
        [tolerance*(1-1e-8), 0.],
        [tolerance*(1+1e-8), 0.],
        [4*tolerance, 0.],
        [4*tolerance, .25*tolerance],
    ])
    representatives, connectivity = _deduplicate_vertices(points, tolerance)
    np.testing.assert_array_equal(connectivity, [0, 0, 1, 1])
    assert len(representatives) == 2
    np.testing.assert_allclose(representatives[0], points[:2].mean(axis=0))
    np.testing.assert_allclose(representatives[1], points[2:].mean(axis=0))


@pytest.mark.parametrize("kernel", [_crossings_numpy, _crossings_numba])
def test_all_roots_not_just_sign_changes(kernel):
    # q(t)=(t-.2)(t-.8): equal signs at endpoints, TWO internal crossings.
    q = np.array([[.16, -1., 1.], [.16, -1., 1.]])
    counts, positions, slopes, _, _ = kernel(q, np.array([0, 1, 2]), np.zeros(2), np.ones(2), np.array([0.]), 1e-12)
    np.testing.assert_array_equal(counts, [[2], [2]])
    np.testing.assert_allclose(positions, .2)
    np.testing.assert_allclose(slopes, -.6)


def test_numpy_numba_crossing_parity():
    random = np.random.default_rng(41)
    q = random.normal(size=(200, 3))
    arguments = (q, np.arange(0, 201, 10), np.tile(np.arange(10), 20).astype(float), np.ones(200), np.array([-.2, 0., .3]), 1e-12)
    for left, right in zip(_crossings_numpy(*arguments), _crossings_numba(*arguments)):
        np.testing.assert_allclose(left, right, atol=2e-14)


def disk_audit_mesh(n=32):
    angles = np.arange(n)*2*np.pi/n
    boundary = np.column_stack((np.cos(angles), np.sin(angles)))
    triangles = np.stack((np.zeros_like(boundary), boundary, np.roll(boundary, -1, axis=0)), axis=1)
    mesh = PackedMesh.from_triangles(triangles)
    points = triangles[:, :1] + SITES[None, :, :1]*(triangles[:, 1:2]-triangles[:, :1]) + SITES[None, :, 1:]*(triangles[:, 2:3]-triangles[:, :1])
    coefficients = ((1-np.sum(points**2, axis=-1))/4) @ VANDERMONDE_INVERSE.T
    gradient = np.stack((-triangles[:, 0]/2, -(triangles[:, 1]-triangles[:, 0])/2, -(triangles[:, 2]-triangles[:, 0])/2), axis=1)
    return mesh, coefficients, gradient


def test_disk_rays_and_normalized_distance():
    from numba import set_num_threads
    set_num_threads(2)
    mesh, coefficients, gradient = disk_audit_mesh()
    config = SolverConfig(number_of_rays=16, samples_per_ray=64)
    atlas = build_atlas(mesh, coefficients, gradient, config)
    np.testing.assert_allclose(atlas.x_T, 0., atol=1e-14)
    assert np.max(atlas.endpoint_error) < config.center_stop_radius*mesh.diameter
    evaluator = BandObservableEvaluator(mesh, atlas, config, .25)
    metrics = evaluator.evaluate(coefficients, .16)
    assert metrics.admissible, metrics.reason
    np.testing.assert_allclose(metrics.s_crossings[:, 1], .6, atol=1e-9)
    assert metrics.distance == pytest.approx(np.dot(atlas.weights, .6/atlas.total_lengths), abs=1e-10)
    derivative = evaluator.derivative(coefficients, np.zeros_like(coefficients), .16)
    assert derivative == pytest.approx(np.dot(atlas.weights, -2/(.6*atlas.total_lengths)), abs=1e-9)
    collapsed = evaluator.evaluate(coefficients*.65, .16)
    assert not collapsed.admissible
    assert collapsed.reason == "NO_TWO_INTERFACE_BAND"
    stale = replace(atlas, mesh_signature="changed")
    with pytest.raises(ValueError, match="STALE"):
        BandObservableEvaluator(mesh, stale, config, .25)
    with pytest.raises(ValueError):
        atlas.weights[0] = 1.
    stringent = replace(config, distance_tolerance=1e-15, crossing_value_tolerance=1e-20)
    assert BandObservableEvaluator(mesh, atlas, stringent, .25).evaluate(coefficients, .16).reason == "UNRESOLVED_CROSSING"


def test_common_torsion_slices_remove_integrator_phase():
    """Equal-field slices must not depend on each ray's ODE output times.

    The two paths have the same geometric parameterization but deliberately
    use different node phases.  Comparing node index to node index would give
    different locations; interpolation at common torsion values recovers the
    same points.  This is the regression for the curved-horseshoe ordering
    failure that motivated the synchronized flow audit.
    """
    paths = np.zeros((2, 6, 2))
    first = np.array([0., .2, .4, .6, .8, 1.])
    second = np.array([0., .05, .31, .57, .91, 1.])
    paths[0, :, 0], paths[1, :, 0] = first, second
    paths[:, :, 1] = np.array([[0.], [1.]])
    values = np.stack((first, second))
    targets = np.linspace(0., 1., 11)
    sampled = _sample_torsion_slices(paths, values, np.array([6, 6]), targets)
    np.testing.assert_allclose(sampled[0, :, 0], targets, atol=1e-15)
    np.testing.assert_allclose(sampled[1, :, 0], targets, atol=1e-15)
    np.testing.assert_allclose(sampled[:, :, 1],
                               np.broadcast_to(np.array([0., 1.])[:, None], (2, 11)))


def test_unresolved_torsion_flow_core_rejects_inner_interface():
    """A full band may not enter the atlas region with unresolved ray labels."""
    mesh, coefficients, gradient = disk_audit_mesh()
    config = SolverConfig(number_of_rays=16, samples_per_ray=64)
    atlas = build_atlas(mesh, coefficients, gradient, config)
    # For phi=T and m=.16, c_+=.17 corresponds to T/Tmax=.68.  Marking
    # T/Tmax >= .60 unresolved must therefore reject this otherwise valid band.
    guarded = replace(atlas, resolved_torsion_fraction=.60,
                      flow_resolution=1e-12, unresolved_neighbor_pairs=1)
    evaluator = BandObservableEvaluator(
        mesh, guarded, config, .25, torsion_coefficients=coefficients)
    metrics = evaluator.evaluate(coefficients, .16)
    assert not metrics.admissible
    assert metrics.reason == "BAND_INSIDE_UNRESOLVED_TORSION_FLOW_CORE"
    assert metrics.flow_core_torsion_margin < 0
    assert metrics.inner_threshold_margin == metrics.core_margin


def test_distance_is_scale_invariant_and_not_density():
    mesh, coefficients, gradient = disk_audit_mesh()
    config = SolverConfig(number_of_rays=16, samples_per_ray=64)
    atlas = build_atlas(mesh, coefficients, gradient, config)
    base = BandObservableEvaluator(mesh, atlas, config, .25).evaluate(coefficients, .16)
    factor = 3.7
    scaled_mesh = PackedMesh.from_triangles(mesh.triangles*factor)
    scaled_config = replace(config, radius=factor, mesh_size=config.mesh_size*factor,
                            band=BandConfig(.02*factor**2, .002*factor**2))
    scaled_atlas = build_atlas(scaled_mesh, coefficients*factor**2, gradient*factor, scaled_config)
    scaled = BandObservableEvaluator(scaled_mesh, scaled_atlas, scaled_config, .25*factor**2).evaluate(coefficients*factor**2, .16*factor**2)
    assert scaled.distance == pytest.approx(base.distance, abs=1e-9)
    assert scaled.mean_physical_thickness == pytest.approx(factor*base.mean_physical_thickness, abs=1e-9)
    assert not hasattr(atlas, "rho")


def test_whole_mesh_audit_detects_two_components():
    from projects.diocotron.dolfinx.equiband.audit import ContourAudit
    x, y = np.meshgrid(np.linspace(-1, 1, 13), np.linspace(-1, 1, 13))
    vertices = np.column_stack((x.ravel(), y.ravel()))
    ll = (np.arange(12)[:, None]*13+np.arange(12)[None, :]).ravel()
    connectivity = np.vstack((np.column_stack((ll, ll+1, ll+13)), np.column_stack((ll+1, ll+14, ll+13))))
    triangles = vertices[connectivity]
    mesh = PackedMesh.from_triangles(triangles)
    points = triangles[:, :1]+SITES[None, :, :1]*(triangles[:, 1:2]-triangles[:, :1])+SITES[None, :, 1:]*(triangles[:, 2:3]-triangles[:, :1])
    values = np.exp(-30*((points[..., 0]-.4)**2+points[..., 1]**2))+np.exp(-30*((points[..., 0]+.4)**2+points[..., 1]**2))
    coefficients = values @ VANDERMONDE_INVERSE.T
    audit = ContourAudit(mesh)
    assert audit.check(coefficients, .45, 1e-12) == "MULTIPLE_MIDDLE_CONTOUR_COMPONENTS"
    assert len(audit.last_diagnostics) >= 2
    assert audit.last_diagnostics[-1]["components"] == 2
    assert audit.last_diagnostics[-1]["closed"] is True
