"""Tiny transfer/validation checks; fixtures are synthetic, not PDE equilibria."""
import json
import numpy as np
import pytest
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.core.projection import project_callable
from projects.diocotron.comparisons.hdg_projection import load_dolfinx_checkpoint
from projects.diocotron.dolfinx.checkpoint_data import checkpoint_array_sha256
from projects.diocotron.tests.comparisons.test_dolfinx_equilibrium_io import (
    _write_synthetic_checkpoint, _temperature)
from scripts.guiding_center.cases.imported_equilibrium import (
    WindowEquilibrium, boundary_polygon, imported_equilibrium)
from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset
from scripts.guiding_center.runtime.runner import _build_mesh, _project_case_density


def window_checkpoint(tmp_path, *, status='ADMISSIBLE_EQUILIBRIUM'):
    path = tmp_path/'synthetic.npz'
    arrays = _write_synthetic_checkpoint(path, format_name='hdgfem_dolfinx_fields_v2')
    arrays['field_names'] = np.array(['phi'])
    arrays['field_samples'] = arrays['field_samples'][:1]
    arrays['nodal_values'] = arrays['nodal_values'][:1]
    metadata = json.loads(str(arrays.pop('metadata')))
    metadata.update(num_fields=1, field_names=['phi'], geometry='synthetic_square',
                    equilibrium_status=status, state_id='synthetic-only',
                    window=dict(kind='logistic', epsilon=.1, threshold_width_delta=.4,
                                midpoint=1.5, smoothing_mode='absolute', relative_epsilon=None))
    metadata['checksums'] = {k: checkpoint_array_sha256(v) for k, v in arrays.items()}
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return path


def test_legacy_checkpoint_sampling_cw_cells_refinement_and_boundary_roundoff(tmp_path):
    source = load_dolfinx_checkpoint(window_checkpoint(tmp_path))
    points = np.random.default_rng(4).uniform(0, 1, (129, 2))
    points = np.vstack([points, [0, 0], [1, 1], [-np.finfo(float).eps, .3]])
    actual = source.evaluate_points('phi', points, chunk_size=17)
    np.testing.assert_allclose(actual, _temperature(points[:, 0], points[:, 1]), atol=2e-14)
    with pytest.raises(ValueError, match='outside'):
        source.evaluate_points('phi', np.array([[-1e-5, .3]]))
    with pytest.raises(ValueError, match='finite'):
        source.evaluate_points('phi', np.array([[np.nan, 0]]))


@pytest.mark.parametrize('basis', ['dub_orth', 'bernstein', 'hier_C0'])
def test_window_is_applied_after_cg_evaluation_before_dg_projection(tmp_path, basis):
    case = imported_equilibrium(checkpoint=window_checkpoint(tmp_path))
    mesh = rectangle_mesh(3, 2, xlim=(0, 1), ylim=(0, 1))
    space = DGSpace(mesh, 3, basis_type=basis, volume_quad_1d=16)
    config = GuidingCenterRunPreset(case='imported_equilibrium', description='synthetic',
                                    initial_projection_quad_1d=16)
    actual = _project_case_density(config, space, case, equilibrium=False, name='rho')
    window = case.imported_equilibrium.window
    expected = project_callable(lambda x, y: window.value(_temperature(x, y), 1.5), space)
    np.testing.assert_allclose(actual.coeffs, expected.coeffs, atol=2e-12)
    # Conservation of quadrature moments, no clipping of negative DG lobes.
    np.testing.assert_allclose(actual.values() @ space.quad_data.Krf_w,
                               expected.values() @ space.quad_data.Krf_w, atol=1e-13)
    source = case.imported_equilibrium.checkpoint
    source_mesh = source.to_mesh()
    source_space = DGSpace(source_mesh, 2)
    nonlinear_samples = window.value(source.field_samples[0], 1.5)
    nodal_interpolant = np.linalg.solve(source_space.basis_at(source.reference_points), nonlinear_samples.T).T
    wrong = source_space.field(nodal_interpolant)
    correct = project_callable(case.initial_density, source_space, volume_quad_1d=20)
    assert source_space.field(wrong.coeffs-correct.coeffs).l2_norm() > .01


def test_factory_validates_status_arguments_and_domain(tmp_path):
    with pytest.raises(ValueError, match='requires'):
        imported_equilibrium()
    with pytest.raises(ValueError, match='choose'):
        imported_equilibrium(checkpoint='x', archive='a', equilibrium='b')
    with pytest.raises(ValueError, match='accepted'):
        WindowEquilibrium(window_checkpoint(tmp_path, status='FAILED'))
    case = imported_equilibrium(checkpoint=window_checkpoint(tmp_path))
    with pytest.raises(ValueError, match='archived domain'):
        _build_mesh(GuidingCenterRunPreset(case=case.key, description='test', domain='disc'), case)
    assert len(boundary_polygon(case.imported_equilibrium.checkpoint)) == 4
    config = GuidingCenterRunPreset(case=case.key, description='test')
    assert _build_mesh(config, case).num_tri == 2
    assert not case.parameters['certificate_transferred']


def cubic_window_checkpoint(tmp_path):
    """Order-3 variant of ``window_checkpoint``: samples of a complete cubic on the CG3 lattice."""
    from projects.diocotron.dolfinx.checkpoint_data import dolfinx_lagrange_reference_points
    path = tmp_path/'synthetic_cubic.npz'
    arrays = dict(np.load(window_checkpoint(tmp_path), allow_pickle=False))
    metadata = json.loads(str(arrays.pop('metadata')))
    reference = dolfinx_lagrange_reference_points(3)
    nodes, cells = arrays['mesh_node_coords'], arrays['mesh_triangles']
    vertices = nodes[cells]
    l1, l2 = .5*(reference[:, 0]+1), .5*(reference[:, 1]+1)
    physical = (vertices[:, None, 0] + l1[None, :, None]*(vertices[:, None, 1]-vertices[:, None, 0])
                + l2[None, :, None]*(vertices[:, None, 2]-vertices[:, None, 0]))
    cubic = lambda x, y: _temperature(x, y) + .1*x**3 - .05*x*y**2
    arrays['reference_points'] = reference
    arrays['field_samples'] = cubic(physical[:, :, 0], physical[:, :, 1])[None]
    metadata.update(source_order=3, order=3)
    metadata['checksums'] = {k: checkpoint_array_sha256(v) for k, v in arrays.items()}
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return path, cubic


def test_higher_order_cg_potential_is_accepted(tmp_path):
    path, cubic = cubic_window_checkpoint(tmp_path)
    profile = WindowEquilibrium(path)
    assert profile.checkpoint.source_order == 3 and 'phi_CG3' in profile.provenance['density_transfer']
    points = np.random.default_rng(2).uniform(0, 1, (50, 2))
    np.testing.assert_allclose(profile.potential(points[:, 0], points[:, 1]), cubic(points[:, 0], points[:, 1]),
                               atol=1e-12)


def test_import_diagnostics_levels_and_seed_requires_stage(tmp_path):
    with pytest.raises(ValueError, match='import_diagnostics'):
        WindowEquilibrium(window_checkpoint(tmp_path), import_diagnostics='bogus')
    with pytest.raises(ValueError, match='band_stage'):
        imported_equilibrium(checkpoint=window_checkpoint(tmp_path), seed_mode=3)
    case = imported_equilibrium(checkpoint=window_checkpoint(tmp_path), import_diagnostics='none')
    assert case.initial_density == case.equilibrium_density and case.imported_equilibrium.band is None
    assert case.imported_equilibrium.initial_diagnostics(None, None, None, quadrature_1d=None,
                                                         trace_basis=None) == dict(
        equilibrium_import=case.imported_equilibrium.provenance)


def test_hint_cells_and_projection_layout_give_the_located_values(tmp_path):
    path, cubic = cubic_window_checkpoint(tmp_path)
    profile = WindowEquilibrium(path)
    checkpoint, cells = profile.checkpoint, profile.checkpoint.mesh_triangles
    rng = np.random.default_rng(4)
    owner = rng.integers(0, len(cells), 300)
    points = np.einsum('nk,nkd->nd', rng.dirichlet(np.ones(3), len(owner)), checkpoint.mesh_node_coords[cells[owner]])
    located = checkpoint._evaluate_points_located('phi', points, chunk_size=64)
    for hints in (owner, None, (owner + 1) % len(cells)):
        np.testing.assert_allclose(checkpoint.evaluate_points('phi', points, cells=hints), located, atol=1e-13)
    with pytest.raises(ValueError, match='one source cell id per point'):
        checkpoint.evaluate_points('phi', points, cells=owner[:-1])
    rows = np.broadcast_to(np.arange(len(cells))[:, None], (len(cells), 3))
    assert np.array_equal(profile.row_cells(rows.shape), rows.ravel())
    assert profile.row_cells((len(cells) + 1, 3)) is None and profile.row_cells((len(cells),)) is None
    grid = checkpoint.mesh_node_coords[cells].mean(axis=1)[:, None, :] + np.zeros((1, 3, 2))
    np.testing.assert_allclose(profile.potential(grid[..., 0], grid[..., 1]), cubic(grid[..., 0], grid[..., 1]),
                               atol=1e-12)


def test_parallel_window_matches_the_reference_window(tmp_path):
    profile = WindowEquilibrium(window_checkpoint(tmp_path))
    low, high = profile.window.config.thresholds(profile.midpoint)
    phi = np.concatenate([np.linspace(low - (high - low), high + (high - low), 2001), [low, high, profile.midpoint]])
    np.testing.assert_allclose(profile.window_value(phi.reshape(-1, 3)).ravel(),
                               profile.window.value(phi, profile.midpoint), rtol=0, atol=4e-16)
