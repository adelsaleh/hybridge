"""Element-local ADR coefficients and device field evaluation (TODO L99--L101).

Bounded stationary checks only: no time integration.
"""
import numpy as np
import pytest

from hdgfem import (DGMesh, DGSpace, ElementCoefficient, field_gradient_at_ref, field_values_at_ref,
                    rectangle_mesh, solve_advection_diffusion_reaction_hdg)
from hdgfem.assembly.advection_diffusion_reaction import prepare_adr_data
from hdgfem.assembly.matrices_numpy import _face_quadrature_values_from_scalar_input

TABLES = ('beta_dot_normal', 'beta_values', 'source_rhs', 'reaction_values', 'tau_total', 'gamma')


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    return cupy


def distorted_space(order, nx=3, ny=2):
    mesh = rectangle_mesh(nx, ny)
    mesh = DGMesh.from_arrays(mesh.node_coords @ np.array([[1.2, .3], [-.1, .8]]) + [2., -.4], mesh.triangles)
    return DGSpace(mesh, order, basis_type='dub_orth')


def fields(space):
    """Smooth DG density/momentum-like fields and a DG(0) piecewise-constant velocity."""
    n = space.project_callable(lambda x, y: 2. + .2*np.sin(3*x)*np.cos(2*y), name='n')
    gamma = space.project_callable(lambda x, y: .3 + .4*np.cos(2*x - y), name='Gamma')
    space0 = DGSpace(space.mesh, 0, basis_type='dub_orth')
    elements = np.arange(space.mesh.num_tri, dtype=float)
    beta0 = (space0*space0).field((space0.field((.7 + .1*elements)[:, None]),
                                   space0.field((-.2 + .05*np.cos(elements))[:, None])))
    return n, gamma, beta0


def wrap_scalar(field):
    """Element coefficient that evaluates a DG field (host or device)."""
    return ElementCoefficient(lambda points, *, xp, t=None: field_values_at_ref(field, points, device=xp is not np),
                              field.space.mesh, name=field.name)


def wrap_vector(vector):
    def function(points, *, xp, t=None):
        return xp.stack([field_values_at_ref(c, points, device=xp is not np) for c in vector.components], axis=-1)
    return ElementCoefficient(function, vector.components[0].space.mesh, 2, 'beta')


def quotient_velocity(n, gamma, floor=1e-8):
    """beta = (Gamma/max(n, floor)) * b with b = (-y, x)/sqrt(1 + x^2 + y^2) evaluated pointwise."""
    space = n.space

    def function(points, *, xp, t=None):
        device = xp is not np
        u = field_values_at_ref(gamma, points, device=device) / xp.maximum(
            field_values_at_ref(n, points, device=device), floor)
        mapped = xp.asarray(space.mesh.map_reference_points(points))
        x, y = mapped[..., 0] - 3., mapped[..., 1]
        scale = u / xp.sqrt(1. + x*x + y*y)
        return xp.stack((-y*scale, x*scale), axis=-1)
    return ElementCoefficient(function, space.mesh, 2, 'u_star_b')


def test_element_coefficient_contract():
    space = distorted_space(2)
    trace = space.trace_space('legendre-modal')
    law = lambda x, y: x + 2.*y*y
    coefficient = ElementCoefficient(
        lambda points, *, xp, t=None: (lambda m: law(m[..., 0], m[..., 1]))(space.mesh.map_reference_points(points)),
        space.mesh, name='law')
    np.testing.assert_allclose(coefficient.face_values(space, trace),
                               _face_quadrature_values_from_scalar_input(law, space, 'law', trace_space=trace),
                               rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(coefficient.volume_values(space), law(*np.moveaxis(
        space.mesh.map_reference_points(space.quad_data.Krf_quads), -1, 0)), rtol=1e-14, atol=1e-14)
    assert not callable(coefficient)
    vector = wrap_vector((space*space).field((space.constant(1.), space.constant(2.))))
    np.testing.assert_array_equal(vector.component(1).volume_values(space), 2.)
    bad = ElementCoefficient(lambda points, *, xp, t=None: np.zeros((1, len(points))), space.mesh, name='bad')
    with pytest.raises(ValueError, match="bad.*shape"):
        bad.volume_values(space)
    nan = ElementCoefficient(lambda points, *, xp, t=None: np.full((space.mesh.num_tri, len(points)), np.nan),
                             space.mesh, name='nan')
    with pytest.raises(ValueError, match='finite'):
        nan.volume_values(space)
    other = distorted_space(2)
    with pytest.raises(ValueError, match='different mesh'):
        coefficient.volume_values(other)


@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
def test_host_preparation_matches_dg_inputs(basis):
    """Wrapped DG fields reproduce DG-field inputs; DG(0) beta.n jumps across faces."""
    space = distorted_space(3)
    trace = space.trace_space(basis)
    n, gamma, beta0 = fields(space)
    kwargs = dict(diffusion=.05, trace_space=trace)
    reference = prepare_adr_data(gamma, n, beta0, space, **kwargs)
    actual = prepare_adr_data(wrap_scalar(gamma), wrap_scalar(n), wrap_vector(beta0), space, **kwargs)
    for name in TABLES:
        np.testing.assert_allclose(getattr(actual, name), getattr(reference, name), rtol=1e-13, atol=1e-13,
                                   err_msg=name)
    assert np.ptp(actual.beta_values[..., 0]) > .1  # elementwise-constant, genuinely discontinuous


def test_pointwise_quotient_is_not_projected():
    space = distorted_space(2)
    n, gamma, _ = fields(space)
    beta = quotient_velocity(n, gamma)
    prepared = prepare_adr_data(space.zeros(), 0., beta, space, diffusion=.05)
    quad = space.quad_data.Krf_quads
    u = gamma.values_at_ref(quad) / n.values_at_ref(quad)
    mapped = space.mesh.map_reference_points(quad)
    x, y = mapped[..., 0] - 3., mapped[..., 1]
    np.testing.assert_allclose(prepared.beta_values[..., 0], -y*u/np.sqrt(1+x*x+y*y), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('assembly', ['numba'])
@pytest.mark.parametrize('flux_space', ['l2_closest', 'RT_projection'])
def test_host_solve_and_postprocessing_match_dg_inputs(assembly, flux_space):
    space = distorted_space(2)
    n, gamma, beta0 = fields(space)
    opts = dict(diffusion=.05, assembly_backend=assembly, solver='direct', hdg_postprocess='both',
                flux_postprocess_space=flux_space, verbose=False)
    boundary = lambda x, y: .2 + x - .3*y
    reference = solve_advection_diffusion_reaction_hdg(gamma, beta0, n, boundary, space, **opts)
    actual = solve_advection_diffusion_reaction_hdg(wrap_scalar(gamma), wrap_vector(beta0), wrap_scalar(n),
                                                   boundary, space, **opts)
    np.testing.assert_allclose(actual.local_unknowns, reference.local_unknowns, rtol=1e-11, atol=1e-12)
    for name in ('postprocessed_field',):
        np.testing.assert_allclose(getattr(actual, name).coeffs, getattr(reference, name).coeffs,
                                   rtol=1e-10, atol=1e-11)
    for component, expected in zip(actual.postprocessed_flux.components, reference.postprocessed_flux.components):
        np.testing.assert_allclose(component.coeffs, expected.coeffs, rtol=1e-10, atol=1e-11)


def test_device_field_evaluation_matches_host(cp):
    space = distorted_space(4)
    n, _, _ = fields(space)
    for points in (space.quad_data.Krf_quads, space.quad_data.pts_fc.reshape(-1, 2)):
        np.testing.assert_allclose(cp.asnumpy(field_values_at_ref(n, points, device=True)),
                                   n.values_at_ref(points), rtol=1e-13, atol=1e-13)
        for device, host in zip(field_gradient_at_ref(n, points, device=True), n.grad_at_ref(points)):
            assert isinstance(device, cp.ndarray)
            np.testing.assert_allclose(cp.asnumpy(device), host, rtol=1e-12, atol=1e-12)
    constant = space.constant(3.)
    assert float(cp.abs(field_gradient_at_ref(constant, space.quad_data.Krf_quads, device=True)[0]).max()) == 0.


@pytest.mark.parametrize('order', [2, 3])  # p=2: NQ == NEL, values must not pass as moments
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
def test_device_preparation_is_resident_and_matches_host(cp, basis, order):
    from hdgfem.backends.adr_coefficients_cupy import prepare_adr_data_cupy
    space = distorted_space(order)
    trace = space.trace_space(basis)
    n, gamma, _ = fields(space)
    beta = quotient_velocity(n, gamma)
    kwargs = dict(diffusion=.05, trace_space=trace)
    host = prepare_adr_data(wrap_scalar(gamma), wrap_scalar(n), beta, space, **kwargs)
    timings = {}
    device = prepare_adr_data_cupy(wrap_scalar(gamma), wrap_scalar(n), beta, space, timings=timings, **kwargs)
    assert not any(key.endswith('host_fallback') for key in timings)
    for name in TABLES:
        assert isinstance(getattr(device, name), cp.ndarray), name
        np.testing.assert_allclose(cp.asnumpy(getattr(device, name)), getattr(host, name),
                                   rtol=1e-12, atol=1e-12, err_msg=name)


def test_device_source_and_reaction_arrays(cp):
    from hdgfem.assembly import hdg
    from hdgfem.backends.adr_coefficients_cupy import _source_moments
    from hdgfem.backends.coefficients_cupy import volume_samples_cupy
    space = distorted_space(3)
    values = space.project_callable(lambda x, y: 1. + x*y).values()
    moments = hdg.source_moments(values, space)
    np.testing.assert_allclose(cp.asnumpy(_source_moments(cp, cp.asarray(values), space)), moments,
                               rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cp.asnumpy(_source_moments(cp, cp.asarray(moments), space)), moments,
                               rtol=0, atol=0)
    np.testing.assert_array_equal(cp.asnumpy(volume_samples_cupy(cp.asarray(values), space, label='r')), values)
    with pytest.raises(ValueError, match='shape'):
        _source_moments(cp, cp.zeros((1, 2)), space)
    with pytest.raises(ValueError, match='shape'):
        volume_samples_cupy(cp.zeros((1, 2)), space, label='reaction')


@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
def test_raw_cuda_solve_with_element_coefficients(cp, basis):
    pytest.importorskip('pyamgx')
    space = distorted_space(3)
    n, gamma, beta0 = fields(space)
    boundary = lambda x, y: .2 + x - .3*y
    common = dict(diffusion=.05, trace_basis=basis, hdg_postprocess='both', verbose=False)
    reference = solve_advection_diffusion_reaction_hdg(gamma, beta0, n, boundary, space,
                                                      assembly_backend='numba', solver='direct', **common)
    actual = solve_advection_diffusion_reaction_hdg(
        wrap_scalar(gamma), wrap_vector(beta0), wrap_scalar(n), boundary, space, assembly_backend='raw-cuda',
        solver='amgx', solver_rtol=1e-12, materialize_host_solution=False, **common)
    assert isinstance(actual.local_unknowns, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(actual.local_unknowns), reference.local_unknowns, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual.postprocessed_field.coeffs, reference.postprocessed_field.coeffs,
                               rtol=1e-8, atol=1e-9)
