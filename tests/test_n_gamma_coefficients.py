"""Axisymmetric n-Gamma coefficient builders (TODO L98 C1); no time integration."""
import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.mixed.adr_preparation import prepare_adr_data
from hybridge.mixed.coefficients import prepare_diffusion
from scripts.n_gamma import coefficients as nc

FLOOR = 1e-8


def b_poloidal(R, Z):
    x, y = R - 3., Z
    scale = 1./np.sqrt(1. + x*x + y*y)
    return -y*scale, x*scale


@pytest.fixture(scope='module')
def space():
    return DGSpace(rectangle_mesh(3, 2, xlim=(2., 4.), ylim=(-1., 1.)), 3, basis_type='dub_orth', volume_degree=14)


def fields(space, dip=False):
    """Density (optionally dipping below zero) and momentum DG fields."""
    n = space.project_callable((lambda R, Z: R - 3.2) if dip else (lambda R, Z: 2. + .2*np.sin(3*R)*np.cos(2*Z)))
    gamma = space.project_callable(lambda R, Z: .3 + .4*np.cos(2*R - Z))
    return n, gamma


def points(space):
    return space.mesh.map_reference_points(space.quad_data.Krf_quads)


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
def test_tensor_matches_plan_and_is_symmetric_variable(space, geometry):
    tensor = nc.diffusion_tensor(.02, b_poloidal, geometry=geometry)
    mapped = points(space)
    R, Z = mapped[..., 0], mapped[..., 1]
    x, y = R - 3., Z
    q = 1. + x*x + y*y
    weight = R if geometry == 'axisymmetric' else 1.
    np.testing.assert_allclose(tensor[0](R, Z), .02*weight*(1. + x*x)/q, rtol=1e-14)
    np.testing.assert_allclose(tensor[1](R, Z), .02*weight*x*y/q, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(tensor[2](R, Z), .02*weight*(1. + y*y)/q, rtol=1e-14)
    assert prepare_diffusion(tensor, space).counts['variable-symmetric'] == space.mesh.num_tri
    assert nc.reaction(3., geometry='cartesian') == 3.
    np.testing.assert_allclose(nc.reaction(3., geometry='axisymmetric')(R, Z), 3.*R)
    with pytest.raises(ValueError, match='positive'):
        nc.diffusion_tensor(0., b_poloidal, geometry=geometry)
    with pytest.raises(ValueError, match='geometry'):
        nc.diffusion_tensor(.02, b_poloidal, geometry='torus')


@pytest.mark.parametrize('dip', [False, True])
def test_advection_is_pointwise_clamped_quotient(space, dip):
    n, gamma = fields(space, dip)
    beta = nc.advection(space, n, gamma, b_poloidal, FLOOR, geometry='axisymmetric')
    quad = space.quad_data.Krf_quads
    mapped = points(space)
    R, Z = mapped[..., 0], mapped[..., 1]
    u = gamma.values_at_ref(quad)/np.maximum(n.values_at_ref(quad), FLOOR)
    b_r, b_z = b_poloidal(R, Z)
    values = beta.volume_values(space)
    np.testing.assert_allclose(values[..., 0], R*u*b_r, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(values[..., 1], R*u*b_z, rtol=1e-13, atol=1e-13)
    report = nc.density_floor_diagnostics(n, space, space.trace_space('legendre-modal'), FLOOR)
    assert report.volume_clamps == int((n.values_at_ref(quad) < FLOOR).sum())
    assert (report.volume_clamps > 0) == dip and (report.face_clamps > 0) == dip
    assert report.min_extrapolated_density <= n.values_at_ref(quad).min()


def test_floor_diagnostics_reject_nonfinite(space):
    n, _ = fields(space)
    bad = space.field(np.where(np.arange(n.coeffs.size).reshape(n.coeffs.shape) == 0, np.nan, n.coeffs))
    with pytest.raises(FloatingPointError, match='not finite'):
        nc.density_floor_diagnostics(bad, space, space.trace_space('legendre-modal'), FLOOR)


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
def test_sources_include_history_and_new_density_pressure(space, geometry):
    n, gamma = fields(space)
    source = lambda R, Z: np.cos(R)*Z
    quad = space.quad_data.Krf_quads
    mapped = points(space)
    R, Z = mapped[..., 0], mapped[..., 1]
    weight = R if geometry == 'axisymmetric' else 1.
    density = nc.density_source(space, source, gamma, geometry=geometry).volume_values(space)
    np.testing.assert_allclose(density, weight*(source(R, Z) + gamma.values_at_ref(quad)), rtol=1e-13)
    momentum = nc.momentum_source(space, source, gamma, n, b_poloidal, sound_speed=1.3,
                                  geometry=geometry).volume_values(space)
    dn_dr, dn_dz = n.grad_at_ref(quad)
    b_r, b_z = b_poloidal(R, Z)
    expected = weight*(source(R, Z) + gamma.values_at_ref(quad) - 1.69*(b_r*dn_dr + b_z*dn_dz))
    np.testing.assert_allclose(momentum, expected, rtol=1e-12, atol=1e-13)
    beta = nc.advection(space, n, gamma, b_poloidal, FLOOR, geometry=geometry).volume_values(space)
    u = gamma.values_at_ref(quad)/n.values_at_ref(quad)
    np.testing.assert_allclose(beta[..., 1], weight*u*b_z, rtol=1e-13, atol=1e-13)


def test_device_coefficients_match_host_and_stay_resident(space):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    from hybridge.mixed.coefficients_device import prepare_adr_data_cupy
    n, gamma = fields(space)
    trace = space.trace_space('legendre-modal')
    source = lambda R, Z: np.cos(R)*Z
    coefficients = (nc.momentum_source(space, source, gamma, n, b_poloidal, geometry='axisymmetric'),
                    nc.reaction(75., geometry='axisymmetric'),
                    nc.advection(space, n, gamma, b_poloidal, FLOOR, geometry='axisymmetric'))
    for coefficient in (coefficients[0], coefficients[2]):
        np.testing.assert_allclose(cp.asnumpy(coefficient.face_values(space, trace, xp=cp)),
                                   coefficient.face_values(space, trace), rtol=1e-12, atol=1e-13)
    kwargs = dict(diffusion=nc.diffusion_tensor(.03, b_poloidal, geometry='axisymmetric'), trace_space=trace)
    host = prepare_adr_data(coefficients[0], coefficients[1], coefficients[2], space, **kwargs)
    timings = {}
    device = prepare_adr_data_cupy(*coefficients, space, timings=timings, **kwargs)
    assert not any(key.endswith('host_fallback') for key in timings)
    for name in ('beta_dot_normal', 'beta_values', 'source_rhs', 'reaction_values', 'tau_total'):
        assert isinstance(getattr(device, name), cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(getattr(device, name)), getattr(host, name),
                                   rtol=1e-12, atol=1e-12, err_msg=name)
    host_report = nc.density_floor_diagnostics(n, space, trace, FLOOR)
    device_report = nc.density_floor_diagnostics(n, space, trace, FLOOR, device=True)
    assert device_report.volume_clamps == host_report.volume_clamps
    np.testing.assert_allclose(device_report.min_extrapolated_density, host_report.min_extrapolated_density,
                               rtol=1e-13)
