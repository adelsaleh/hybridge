"""GPU gate for the tuned launches of the DR solver's recovery, Oswald and gradient; independent CPU oracles."""
import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.conforming import oswald_average
from hybridge.mixed.postprocess.flux import _new_hdg_postprocess_cache, _postprocess_diffusion_solution
from hybridge.mixed.postprocess.flux_cupy import postprocess_diffusion_primal_cupy
from hybridge.runtime.optional import require_cupy_device


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
    assert cache._diffusion_primal_info['postprocess.primal.launch_threads'] > 0
    np.testing.assert_allclose(device.coeffs, host.coeffs, rtol=2e-10, atol=2e-10)
    # Fresh device-only field forces the fused path even after the comparison download.
    from hybridge.core.device import field_from_cupy_coefficients
    device = field_from_cupy_coefficients(device.space, cp.asarray(device.coeffs))
    a, b = oswald_average(host, boundary_values=0.), oswald_average(device, boundary_values=0., device=True)
    np.testing.assert_allclose(cp.asnumpy(b.dofs), a.dofs, rtol=2e-10, atol=2e-10)
    for x, y in zip(a.flux_field(s).components, b.flux_field(s).components):
        np.testing.assert_allclose(y.coeffs, x.coeffs, rtol=2e-10, atol=2e-10)


def test_inverse_recovery_operator_is_tuned_and_matches_cpu(monkeypatch):
    cp = require_cupy_device()
    s = DGSpace(rectangle_mesh(3, 2), 6, basis_type='dub_orth')
    trace = s.trace_space('legendre-modal')
    local = np.random.default_rng(31).normal(size=(s.mesh.num_tri, 3*s.el_dof))
    host, _, _ = _postprocess_diffusion_solution(local,
        np.zeros(s.mesh.num_edg*trace.edg_dof), s, stabilization=1., diffusion=1., mode='primal', trace_space=trace)
    cache = _new_hdg_postprocess_cache(s, trace)
    d = cache.post_space.el_dof
    # Enough for the inverse form only; the stored P,Q form needs 2*el_dof+1 columns.
    monkeypatch.setenv('HYBRIDGE_PRIMAL_CACHE_MAX_BYTES', str(s.mesh.num_tri*d*(d+1)*8))
    device = postprocess_diffusion_primal_cupy(cp.asarray(local), s, cache)
    assert cache._diffusion_primal_info['postprocess.primal.operator'] == 1
    assert cache._diffusion_primal_info['postprocess.primal.launch_threads'] > 0
    np.testing.assert_allclose(device.coeffs, host.coeffs, rtol=2e-10, atol=2e-10)


def test_diffusion_solver_postprocess_reports_the_tuned_launch():
    require_cupy_device()
    from hybridge import DiffusionReactionHDGSolver
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGOptions

    s = DGSpace(rectangle_mesh(6, 6), 3, basis_type='dub_orth')
    solver = DiffusionReactionHDGSolver(
        s, source=s.constant(1.), reaction=s.zeros(), boundary_condition=0.,
        options=DiffusionReactionHDGOptions(stabilization=1., assembly_backend='raw-cuda', boundary_mode='eliminate',
                                            solver='amgx', solver_rtol=1e-8, hdg_postprocess='primal',
                                            verbose=False))
    result = solver.solve()
    details = result.timings.details
    assert details['postprocess.primal.launch_threads'] > 0
