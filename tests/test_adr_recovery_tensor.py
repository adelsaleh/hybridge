"""CPU oracles for the experimental tensor contraction equations."""
import numpy as np
import pytest

from campaigns.adr_recovery.problem import mesh, problem
from hybridge.mixed.postprocess.adr_recovery import TensorPrimalAssembly, primal_rows, recovery_working_bytes
from hybridge.mixed.postprocess.total_flux import (
    _adr_recovery_cache, _adr_postprocess_samples, _postprocess_total_flux, _postprocess_primal_from_total_flux)
from hybridge.mixed.coefficients import sample_diffusion_tensor, inverse_diffusion_values


@pytest.mark.parametrize('order', range(2, 10))
@pytest.mark.parametrize('variable', [False, True])
def test_tensor_equations_against_numba_recovery(order, variable):
    p = problem(order, mesh(2, 1), variable_tensor=variable)
    s, t = p['space'], p['trace_space']
    flux = _postprocess_total_flux(p['local'], p['trace'], p['beta'], p['prepared'], s, t, None)
    reference = _postprocess_primal_from_total_flux(p['local'], flux, p['beta'], p['prepared'], s, t,
                                                 None, p['diffusion'])
    cache = _adr_recovery_cache(s, t)
    samples = _adr_postprocess_samples(p['beta'], p['prepared'], s, cache.post_space, None)
    inverse = inverse_diffusion_values(sample_diffusion_tensor(p['diffusion'], cache.post_space))
    assembly = TensorPrimalAssembly(s, cache)
    a, b = assembly.assemble(p['local'], np.stack([f.coeffs for f in flux.components], axis=1), samples, inverse)
    x = np.linalg.solve(a, b[..., None])[..., 0]
    np.testing.assert_allclose(x[:, :cache.post_space.el_dof], reference.coeffs, rtol=3e-8, atol=3e-9)
    residual = np.max(abs(a @ x[..., None]-b[..., None]), axis=(1,2))
    scale = np.linalg.norm(a, np.inf, axis=(1,2))*np.max(abs(x), axis=1)+np.max(abs(b), axis=1)
    assert np.max(residual/scale) < 5e-13
    np.testing.assert_allclose(x[:, :cache.post_space.el_dof] @ cache.mean_post,
                              p['local'][:, :s.el_dof] @ cache.mean_base, rtol=2e-9, atol=2e-10)


def test_workspace_estimate_increases_with_degree_and_cells():
    assert primal_rows(4) == 82 and primal_rows(6) == 133
    assert all(recovery_working_bytes(p, 100) < recovery_working_bytes(p+1, 100) for p in range(2,9))
    assert recovery_working_bytes(9, 100) == 2*recovery_working_bytes(9, 50)
