"""GPU gates for the isolated recovery/tuning experiment; independent CPU oracles."""
import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.conforming import oswald_average
from hybridge.mixed.postprocess.flux import _new_hdg_postprocess_cache, _postprocess_diffusion_solution
from hybridge.mixed.postprocess.flux_cupy import postprocess_diffusion_primal_cupy
from hybridge.mixed.postprocess.total_flux import _postprocess_total_flux, _postprocess_primal_from_total_flux
from hybridge.mixed.adr_preparation import prepare_adr_data
from hybridge.runtime.optional import require_cupy_device


@pytest.mark.parametrize('order', [4, 6])
def test_adr_tuned_recovery_matches_cpu_and_preserves_mean(order):
    cp = require_cupy_device()
    s = DGSpace(rectangle_mesh(3, 2), order, basis_type='dub_orth')
    trace = s.trace_space('legendre-modal')
    beta = (s*s).field((s.project_callable(lambda x,y:.2+.1*y), s.constant(-.1)))
    prepared = prepare_adr_data(s.constant(1.), s.constant(.3), beta, s,
        diffusion=.01/30., diffusion_stabilization='global_length', trace_space=trace)
    rng = np.random.default_rng(41)
    # Nonzero accepted perturbations, flux and trace make every constraint observable.
    unknowns = rng.normal(size=(s.mesh.num_tri, 3*s.el_dof))
    trace_values = rng.normal(size=s.mesh.num_edg*trace.edg_dof)
    host_flux = _postprocess_total_flux(unknowns, trace_values, beta, prepared, s, trace, None, 'l2_closest', 'numba')
    host = _postprocess_primal_from_total_flux(unknowns, host_flux, beta, prepared, s, trace, None, .01/30.)
    local, skeleton = cp.asarray(unknowns), cp.asarray(trace_values)
    flux = _postprocess_total_flux(local, skeleton, beta, prepared, s, trace, None, 'l2_closest', 'cupy')
    result = _postprocess_primal_from_total_flux(local, flux, beta, prepared, s, trace, None, .01/30., 'cupy')
    assert not result.coefficients_materialized and result.device_coefficients_materialized()
    np.testing.assert_allclose(result.coeffs, host.coeffs, rtol=2e-8, atol=2e-8)
    q = result.space.quad_data
    np.testing.assert_allclose(result.coeffs @ (q.Krf_w @ q.phi),
        unknowns[:, :s.el_dof] @ (q.Krf_w @ s.basis_at(q.Krf_quads)), rtol=2e-9, atol=2e-9)
    np.testing.assert_array_equal(cp.asnumpy(local), unknowns)
    np.testing.assert_array_equal(cp.asnumpy(skeleton), trace_values)


@pytest.mark.parametrize('order', [4, 6, 9])
def test_poisson_recovery_average_gradient_match_cpu(order):
    cp = require_cupy_device()
    s = DGSpace(rectangle_mesh(3, 2), order, basis_type='dub_orth')
    trace = s.trace_space('legendre-modal')
    rng = np.random.default_rng(29)
    local = rng.normal(size=(s.mesh.num_tri, 3*s.el_dof))
    host, _, _ = _postprocess_diffusion_solution(local,
        np.zeros(s.mesh.num_edg*trace.edg_dof), s, stabilization=1., diffusion=1., mode='primal', trace_space=trace)
    cache = _new_hdg_postprocess_cache(s, trace)
    device = postprocess_diffusion_primal_cupy(cp.asarray(local), s, cache)
    assert cache._diffusion_primal_info['postprocess.primal.operator'] == 2
    np.testing.assert_allclose(device.coeffs, host.coeffs, rtol=2e-10, atol=2e-10)
    # Fresh device-only field forces the fused path even after the comparison download.
    from hybridge.core.device import field_from_cupy_coefficients
    device = field_from_cupy_coefficients(device.space, cp.asarray(device.coeffs))
    a, b = oswald_average(host, boundary_values=0.), oswald_average(device, boundary_values=0., device=True)
    np.testing.assert_allclose(cp.asnumpy(b.dofs), a.dofs, rtol=2e-10, atol=2e-10)
    for x, y in zip(a.flux_field(s).components, b.flux_field(s).components):
        np.testing.assert_allclose(y.coeffs, x.coeffs, rtol=2e-10, atol=2e-10)
