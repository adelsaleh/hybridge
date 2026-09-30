"""Small-matrix recovery parity and explicit host-materialization contracts."""
from __future__ import annotations

import numpy as np
import pytest

from hdgfem import DGSpace, rectangle_mesh, solve_advection_diffusion_reaction_hdg
from hdgfem.mixed.adr_preparation import prepare_adr_data
from hdgfem.core.space import DGField, VectorDGField
from hdgfem.mixed.postprocess.total_flux import (
    _postprocess_total_flux,
    _postprocess_primal_from_total_flux,
)


def _cupy():
    """Require a usable device without building external libraries."""
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('CUDA device unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    return cp


def _problem(order):
    """Use variable velocity, nonzero boundary data and unequal side tau."""
    space = DGSpace(rectangle_mesh(2, 1, xlim=(-0.3, 1.2), ylim=(0.1, 0.8)), order, basis_type='dub_orth')
    beta_space = DGSpace(space.mesh, 1, basis_type='dub_orth')
    beta = (beta_space * beta_space).field((
        beta_space.project_callable(lambda x, y: 0.7 + 0.2*y),
        beta_space.project_callable(lambda x, y: -0.3 + 0.1*x)))
    tau = 0.4 + 0.07*np.arange(space.mesh.num_tri)[:, None] + 0.03*np.arange(3)[None, :]
    return space, beta, tau


@pytest.mark.parametrize('diffusion_kind', ['constant_isotropic', 'constant_diagonal', 'constant_full', 'variable_isotropic', 'variable_diagonal', 'variable_symmetric', 'variable_full', 'device_field'])
@pytest.mark.parametrize('order', [0, 1, 3, 6])
@pytest.mark.parametrize('trace_basis', ['legacy-lagrange', 'legendre-modal'])
@pytest.mark.parametrize('variant', ['l2_closest', 'RT_projection'])
def test_recovery_stays_device_resident(monkeypatch, order, trace_basis, variant, diffusion_kind):
    """Forbid downloads through both CuPy and lazy field access during recovery."""
    cp = _cupy()
    from hdgfem.core.device import field_from_cupy_coefficients

    space, beta, tau = _problem(order)
    from scripts.advection_diffusion_reaction.cases.tensor_cases import diffusion_cases
    diffusion = diffusion_cases().get(diffusion_kind)
    if diffusion_kind == 'device_field':
        coefficient_space = DGSpace(space.mesh, 1, basis_type='dub_orth')
        diffusion = (coefficient_space.project_callable(lambda x, y: 2.+.1*x),
                     coefficient_space.constant(.3), coefficient_space.constant(-.1),
                     coefficient_space.project_callable(lambda x, y: 1.+.1*y))
    trace_space = space.trace_space(trace_basis)
    prepared = prepare_adr_data(space.constant(1.0), space.constant(0.3), beta, space, diffusion=diffusion,
                               advection_stabilization=tau, diffusion_stabilization=0.6,
                               trace_space=trace_space)
    rng = np.random.default_rng(14)
    unknowns = rng.normal(size=(space.mesh.num_tri, 3*space.el_dof))
    trace = rng.normal(size=space.mesh.num_edg*trace_space.edg_dof)
    host_flux = _postprocess_total_flux(unknowns, trace, beta, prepared, space,
                                       trace_space, tau, variant, 'numba')
    host_primal = _postprocess_primal_from_total_flux(unknowns, host_flux, beta, prepared,
                                                     space, trace_space, tau, diffusion)
    device_beta = VectorDGField(tuple(field_from_cupy_coefficients(f.space, cp.asarray(f.coeffs))
                                      for f in beta.components))
    device_unknowns, device_trace = cp.asarray(unknowns), cp.asarray(trace)
    device_diffusion = diffusion
    if diffusion_kind == 'device_field':
        device_diffusion = tuple(field_from_cupy_coefficients(f.space, cp.asarray(f.coeffs)) for f in diffusion)

    def forbidden(*args, **kwargs):
        """Reject any hidden download of solution/coefficient arrays."""
        raise AssertionError('unexpected host materialization during recovery')

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', forbidden)
        patch.setattr(DGField, '_download_device_coefficients', forbidden)
        device_flux = _postprocess_total_flux(device_unknowns, device_trace, device_beta,
                                             prepared, space, trace_space, cp.asarray(tau), variant, 'cupy')
        device_primal = _postprocess_primal_from_total_flux(device_unknowns, device_flux,
                                                          device_beta, prepared, space, trace_space,
                                                          cp.asarray(tau), device_diffusion, 'cupy')
        for f in (*device_flux.components, device_primal):
            assert not f.coefficients_materialized
            assert f.device_coefficients_materialized()
        cp.cuda.get_current_stream().synchronize()
    for actual, expected in zip((*device_flux.components, device_primal), (*host_flux.components, host_primal)):
        np.testing.assert_allclose(actual.coeffs, expected.coeffs, rtol=2e-9, atol=2e-9)
    # The mean condition is independent of the recovered flux variant.
    q = device_primal.space.quad_data
    mean_post = q.Krf_w @ q.phi
    mean_base = q.Krf_w @ space.basis_at(q.Krf_quads)
    np.testing.assert_allclose(device_primal.coeffs @ mean_post,
                               unknowns.reshape(-1, 3, space.el_dof)[:, 0] @ mean_base,
                               rtol=2e-10, atol=2e-10)


@pytest.mark.parametrize('tensor', [False, True])
@pytest.mark.parametrize('materialize', [False, True])
@pytest.mark.parametrize('mode', ['none', 'primal', 'flux', 'both'])
@pytest.mark.parametrize('variant', ['l2_closest', 'RT_projection'])
@pytest.mark.parametrize('trace_basis', ['legacy-lagrange', 'legendre-modal'])
def test_raw_cuda_result_materialization(monkeypatch, materialize, mode, variant, trace_basis, tensor):
    """Exercise native AMGX solves and explicit versus lazy result downloads."""
    cp = _cupy()
    pytest.importorskip('pyamgx')
    space, beta, tau = _problem(1)
    boundary = lambda x, y: 0.2 + x - 0.3*y
    from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient
    diffusion = raw_cuda_coefficient('variable-full') if tensor else 0.2
    common = dict(diffusion=diffusion, advection_stabilization=tau, diffusion_stabilization=0.6,
                  trace_basis=trace_basis, flux_postprocess_space=variant,
                  hdg_postprocess=mode, scale_system=False, verbose=False)
    reference = solve_advection_diffusion_reaction_hdg(space.constant(1.0), beta, space.constant(0.3), boundary, space,
                                                      assembly_backend='numpy', solver='direct', **common)
    downloads = []
    original = cp.asnumpy

    def record(array, *args, **kwargs):
        """Account for every explicit CuPy host download."""
        downloads.append(array.size)
        return original(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', record)
        result = solve_advection_diffusion_reaction_hdg(space.constant(1.0), beta, space.constant(0.3), boundary, space,
                                                       assembly_backend='raw-cuda', solver='amgx',
                                                       solver_rtol=1e-11, solver_atol=1e-12,
                                                       materialize_host_solution=materialize, **common)
    fields = [result.field, *result.flux.components, *result.total_flux.components]
    expected = [reference.field, *reference.flux.components, *reference.total_flux.components]
    if mode in ('primal', 'both'):
        fields.append(result.postprocessed_field)
        expected.append(reference.postprocessed_field)
    if mode in ('flux', 'both'):
        fields.extend(result.postprocessed_flux.components)
        expected.extend(reference.postprocessed_flux.components)
    assert result.postprocessing_backend == ('none' if mode == 'none' else 'cupy')
    assert isinstance(result.trace, np.ndarray if materialize else cp.ndarray)
    assert isinstance(result.local_unknowns, np.ndarray if materialize else cp.ndarray)
    if not materialize:
        assert all(size <= 1 for size in downloads), downloads
        from hdgfem.core.field_ops import solution_trace
        assert solution_trace(result, space).data.ptr == result.trace.data.ptr
        interior = solution_trace(result, space, reduced=True)
        assert isinstance(interior, cp.ndarray)
        np.testing.assert_allclose(original(interior),
                                   reference.trace.reshape(space.mesh.num_edg, -1)[space.mesh.int_edges_inds].ravel(),
                                   rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(solution_trace(result, space, prefer_device=False),
                                   reference.trace, rtol=2e-9, atol=2e-10)
    for actual, host in zip(fields, expected):
        assert actual.coefficients_materialized == materialize
        np.testing.assert_allclose(actual.coeffs, host.coeffs, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(original(result.trace), reference.trace, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize('kind', ['upwind', 'scalar', 'callable', 'field', 'coefficients', 'quadrature'])
def test_device_stabilization_sampling(monkeypatch, kind):
    """Preserve supported stabilization inputs without hidden coefficient downloads."""
    cp = _cupy()
    from hdgfem.core.device import field_from_cupy_coefficients
    from hdgfem.mixed.postprocess.total_flux import _adr_postprocess_samples

    space, beta, _ = _problem(2)
    post = DGSpace(space.mesh, 3, basis_type='dub_orth')
    prepared = prepare_adr_data(space.constant(1.0), space.constant(0.3), beta, space, diffusion=0.2)
    tau_field = space.project_callable(lambda x, y: 0.5 + 0.01*x + 0.02*y)
    tau = {'upwind': None, 'scalar': 0.5,
           'callable': lambda x, y: 0.5 + 0.01*x + 0.02*y,
           'field': tau_field, 'coefficients': tau_field.coeffs,
           'quadrature': np.full((space.mesh.num_tri, 3, post.quad_data.weights_JGL.size), 0.5)}[kind]
    expected = _adr_postprocess_samples(beta, prepared, space, post, tau)
    if kind == 'field':
        tau = field_from_cupy_coefficients(space, cp.asarray(tau.coeffs))
    elif kind in ('coefficients', 'quadrature'):
        tau = cp.asarray(tau)

    def forbidden(*args, **kwargs):
        """Reject host staging of device stabilization coefficients."""
        raise AssertionError('unexpected coefficient download')

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', forbidden)
        patch.setattr(DGField, '_download_device_coefficients', forbidden)
        actual = _adr_postprocess_samples(beta, prepared, space, post, tau, xp=cp)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(cp.asnumpy(a), b, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('order', [0, 1, 3, 6])
def test_fused_primal_system_matches_independent_contractions(order):
    """Compare every mixed block and RHS, including the nonsymmetric cross terms."""
    cp = _cupy()
    from hdgfem.mixed.postprocess.flux_cupy import _primal_system_cupy
    from hdgfem.mixed.postprocess.primal_raw_cuda import primal_system_raw_cuda
    from hdgfem.mixed.coefficients import (
            sample_diffusion_tensor,
            inverse_diffusion_values,
        )
    from hdgfem.core.device import field_from_cupy_coefficients
    from hdgfem.mixed.postprocess.flux import _build_hdg_postprocess_cache
    from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient

    space, _, _ = _problem(order)
    cache = _build_hdg_postprocess_cache(space, space.trace_space('legendre-modal'),
                                       want_primal=False, want_flux=False)
    post, n = cache.post_space, space.mesh.num_tri
    q = post.quad_data
    rng = np.random.default_rng(17)
    flux = VectorDGField(tuple(field_from_cupy_coefficients(post, cp.asarray(rng.normal(size=post.shape)))
                               for _ in range(2)))
    local = cp.asarray(rng.normal(size=(n, 3*space.el_dof)))
    samples = (cp.asarray(rng.normal(size=(n, q.Krf_w.size, 2))),
               cp.asarray(rng.normal(size=(n, 3, q.weights_JGL.size, 2))),
               cp.asarray(1.+rng.random((n, 3, q.weights_JGL.size))))
    diffusion = raw_cuda_coefficient('variable-full')
    inverse = inverse_diffusion_values(sample_diffusion_tensor(diffusion, post, device=True))
    expected = _primal_system_cupy(local, flux, space, cache, samples, diffusion)
    actual = primal_system_raw_cuda(local, flux, space, cache, samples, inverse)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(cp.asnumpy(a), cp.asnumpy(b), atol=2e-12, rtol=2e-12)
