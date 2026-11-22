"""Every-degree independent GPU gates for the recovery profiling campaign."""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('SLURM_JOB_ID'), reason='GPU gates require a Slurm allocation')


@pytest.mark.parametrize('order', range(2, 10))
@pytest.mark.parametrize('variable', [False, True])
def test_recovery_variants(order, variable):
    from campaigns.adr_recovery.problem import mesh, problem
    from hybridge.runtime.optional import require_cupy_device
    from hybridge.runtime.cuda_tuning import CandidateUnavailable
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space
    from hybridge.mixed.postprocess.adr_recovery import ADRRecoveryWorkspace
    from hybridge.mixed.postprocess.total_flux import (
        _adr_postprocess_samples, _postprocess_total_flux, _postprocess_primal_from_total_flux)
    from hybridge.mixed.postprocess.flux_cupy import _primal_system_cupy
    from hybridge.mixed.postprocess.primal_raw_cuda import primal_system_raw_cuda
    from hybridge.mixed.coefficients import sample_diffusion_tensor, inverse_diffusion_values
    from hybridge.linalg.gpu.fused_batched_solve import default_config
    from hybridge.mixed.adr_preparation import prepare_adr_data
    cp = require_cupy_device()
    p = problem(order, mesh(2, 1), variable_tensor=variable)
    s, t, local, trace = p['space'], p['trace_space'], cp.asarray(p['local']), cp.asarray(p['trace'])
    original_local, original_trace = local.copy(), trace.copy()
    workspace = ADRRecoveryWorkspace(s, t)
    for change in (0, 1):
        beta = p['beta'] if not change else (p['beta'].space).field(tuple(f*1.13 for f in p['beta'].components))
        diffusion=p['diffusion'] if not change else (tuple(f*1.07 for f in p['diffusion'])
                                                    if variable else p['diffusion']*1.07)
        prepared=prepare_adr_data(s.constant(1.),s.constant(.3),beta,s,diffusion=diffusion,
                                 diffusion_stabilization='global_length',trace_space=t)
        u = local if not change else local*.93
        tr = trace if not change else trace*1.07
        hf = _postprocess_total_flux(cp.asnumpy(u), cp.asnumpy(tr), beta, prepared, s, t, None)
        host = _postprocess_primal_from_total_flux(cp.asnumpy(u), hf, beta, prepared, s, t, None, diffusion)
        samples = _adr_postprocess_samples(beta, prepared, s, workspace.cache.post_space, None, xp=cp)
        previous = getattr(s, '_adr_recovery_cache', None)
        s._adr_recovery_cache = workspace.cache
        try:
            flux = _postprocess_total_flux(u, tr, beta, prepared, s, t, None, 'l2_closest', 'cupy',
                                           samples=samples, recovery=workspace.flux)
        finally:
            s._adr_recovery_cache = previous
        for actual, expected in zip(flux.components, hf.components):
            np.testing.assert_allclose(actual.coeffs, expected.coeffs, rtol=3e-8, atol=3e-9)
        inverse = inverse_diffusion_values(sample_diffusion_tensor(diffusion, workspace.cache.post_space, device=True))
        ref_a, ref_b = _primal_system_cupy(u, flux, s, workspace.cache, samples, diffusion)
        vals = cp.stack([as_cupy_coefficients(f, as_cupy_space(f.space)) for f in flux.components], axis=1)
        tensor_a, tensor_b = workspace.tensor.assemble(u, vals, samples, inverse)
        for value, ref in ((tensor_a, ref_a), (tensor_b, ref_b)):
            np.testing.assert_allclose(cp.asnumpy(value), cp.asnumpy(ref), rtol=2e-10, atol=2e-11)
        for compact in (False, True):
            for threads in (32, 64, 128, 256, 512):
                try:
                    a, b = primal_system_raw_cuda(u, flux, s, workspace.cache, samples, inverse,
                                                  compact=compact, threads=threads)
                except CandidateUnavailable:
                    assert threads >= 256
                    continue
                np.testing.assert_allclose(cp.asnumpy(a), cp.asnumpy(ref_a), rtol=2e-10, atol=2e-11)
                np.testing.assert_allclose(cp.asnumpy(b), cp.asnumpy(ref_b), rtol=2e-10, atol=2e-11)
        for solver in ('cublas', 'magma', 'fused:'+':'.join(map(str, default_config(workspace.tensor.rows)))):
            workspace.solver = solver
            actual = workspace.recover(u, tr, beta, prepared, None, diffusion)
            assert actual.device_coefficients_materialized() and not actual.coefficients_materialized
            np.testing.assert_allclose(actual.coeffs, host.coeffs, rtol=3e-8, atol=3e-9)
            np.testing.assert_allclose(actual.coeffs @ workspace.cache.mean_post,
                                      cp.asnumpy(u[:, :s.el_dof]) @ workspace.cache.mean_base, rtol=2e-9, atol=2e-10)
    np.testing.assert_array_equal(cp.asnumpy(local), cp.asnumpy(original_local))
    np.testing.assert_array_equal(cp.asnumpy(trace), cp.asnumpy(original_trace))


@pytest.mark.parametrize('order', [2, 5, 9])
def test_rebuild_matches_reuse(order):
    from campaigns.adr_recovery.problem import mesh, problem
    from hybridge.runtime.optional import require_cupy_device
    from hybridge.mixed.postprocess.adr_recovery import ADRRecoveryWorkspace
    cp = require_cupy_device()
    p = problem(order, mesh(2, 1))
    args = (cp.asarray(p['local']), cp.asarray(p['trace']), p['beta'], p['prepared'], None, p['diffusion'])
    retained = ADRRecoveryWorkspace(p['space'], p['trace_space'])
    for _ in range(2):
        fresh = ADRRecoveryWorkspace(p['space'], p['trace_space'],reference_cache=retained.cache)
        assert fresh.cache is not retained.cache
        assert fresh.cache.mean_base is retained.cache.mean_base
        assert fresh.flux is not retained.flux
        a, b = retained.recover(*args), fresh.recover(*args)
        np.testing.assert_allclose(a.coeffs, b.coeffs, rtol=2e-11, atol=2e-12)
