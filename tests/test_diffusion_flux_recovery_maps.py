"""Stationary small-matrix recovery checks; run with NUMBA_DISABLE_JIT=1.

No PDE solves, time steps, CUDA compilation, or kernel launches are needed.
"""
import numpy as np
import pytest
import hdgfem.mixed.postprocess.flux as postprocess_flux
import hdgfem.runtime.optional as runtime_optional
from hdgfem.mixed.postprocess.flux_recovery import build_flux_recovery_reference
from hdgfem.core.mesh import DGMesh, rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.mixed.postprocess.flux import _postprocess_diffusion_solution
from hdgfem.solvers.diffusion_reaction import _resolve_diffusion_postprocessing_backend
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.configuration import _validate_config


def apply_maps(reference, space, trace_space, local, trace, tau):
    """Evaluate the reference/Piola mathematics with NumPy for parity checks."""
    result = np.empty((2, space.mesh.num_tri, reference.post_space.el_dof))
    n = reference.post_space.el_dof
    local_trace = trace_space.element_coefficients(trace).reshape(space.mesh.num_tri, 3, -1)
    for k, affine in enumerate(space.mesh.aff_mats):
        gap = np.einsum('fai,i->fa', reference.face_moments, local[k,:space.el_dof])
        gap -= np.einsum('ai,fi->fa', reference.trace_moments, local_trace[k])
        tau_face = tau if np.isscalar(tau) else np.asarray(tau)[k,:,None]
        gap = (gap * space.mesh.jacs_el_fc[k,:,None] * tau_face).ravel()
        correction = reference.lift @ gap
        if reference.nullspace.shape[1]:
            metric = affine.T @ affine
            g = np.array([metric[0,0],metric[0,1],metric[1,1]])
            gram = np.einsum('c,cij->ij',g,reference.gram)
            rhs = -np.einsum('c,cij,j->i',g,reference.cross,gap)
            correction += reference.nullspace @ np.linalg.solve(gram,rhs)
        raw = local[k,space.el_dof:].reshape(2,-1) @ reference.embedding.T
        result[:,k] = raw + affine @ correction.reshape(2,n) / space.mesh.aff_jacs[k]
    return result


@pytest.mark.parametrize('degree,trace_kind', [(0,'legacy-lagrange'),(1,'bernstein'),(2,'legendre-modal'),(5,'legendre-modal')])
@pytest.mark.parametrize('variant', ['RT_projection','l2_closest'])
def test_reference_lifting_matches_existing_postprocessor_on_skew_triangles(degree, trace_kind, variant):
    mesh = rectangle_mesh(1,1)
    nodes = mesh.node_coords @ np.array([[1.8,.4],[-.2,.7]]) + [.2,-.1]
    mesh = DGMesh.from_arrays(nodes,mesh.triangles)
    space = DGSpace(mesh,degree,basis_type='dub_orth')
    trace_space = space.trace_space(trace_kind)
    rng = np.random.default_rng(530)
    local = rng.normal(size=(mesh.num_tri,3*space.el_dof))
    trace = rng.normal(size=mesh.num_edg*trace_space.edg_dof)
    reference = build_flux_recovery_reference(space,trace_space,l2_closest=variant=='l2_closest')
    for tau in (.4,1.7,np.array([[.4,.8,1.2],[.5,.9,1.3]])):
        # A retry or face policy changes tau, not cached geometry maps.
        expected = _postprocess_diffusion_solution(local,trace,space,tau,1.,'flux',
            trace_space=trace_space,flux_postprocess_space=variant,postprocessing_backend='numba')[1]
        actual = apply_maps(reference,space,trace_space,local,trace,tau)
        np.testing.assert_allclose(actual,expected.as_component_first(),atol=2.e-9,rtol=2.e-9)
    # At zero stabilization the raw flux is unchanged apart from degree embedding.
    lifted = apply_maps(reference,space,trace_space,local,trace,0.)
    raw = np.stack([local[:,(c+1)*space.el_dof:(c+2)*space.el_dof] @ reference.embedding.T for c in range(2)])
    np.testing.assert_allclose(lifted,raw,atol=1.e-12)
    assert reference.nullspace.shape[1] == (degree if variant=='l2_closest' else 0)


@pytest.mark.parametrize('suffix,variant', [('rt','RT_projection'),('l2_closest','l2_closest')])
def test_disk_bdf2_presets_require_continuous_raw_cuda_recovered_drift(suffix,variant):
    config = preset_by_key(f'euler_vortex_gas_si_bdf2_p6_poisson_p5_{suffix}_p6')
    _validate_config(config)
    assert config.case == 'euler_vortex_gas'
    assert config.time_scheme == 'si-bdf2'
    assert config.order == 6 and config.poisson_order_offset == -1
    assert config.transport_electric_field == 'postprocessed'
    assert config.poisson_hdg_postprocess == 'flux'
    assert config.poisson_flux_postprocess_space == variant
    assert config.poisson_flux_postprocess_every == 0
    assert config.poisson_postprocessing_backend == 'raw-cuda'
    assert config.poisson_assembly_backend == config.transport_assembly_backend == 'raw-cuda'
    assert _resolve_diffusion_postprocessing_backend('raw-cuda','flux',variant,'auto') == 'raw-cuda'


@pytest.mark.parametrize('variant', ['RT_projection', 'l2_closest'])
def test_raw_dispatch_keeps_inputs_on_device_and_reuses_cache(monkeypatch, variant):
    from types import SimpleNamespace
    import hdgfem.mixed.postprocess.flux_recovery_raw_cuda as raw
    import hdgfem.transport.cupy as cupy_backend
    import hdgfem.solvers.diffusion_reaction as diffusion
    space = DGSpace(rectangle_mesh(1,1),2,basis_type='dub_orth')
    trace_space = space.trace_space('legendre-modal')
    class Resident:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Device input was materialized on the host')
    local, trace = Resident(), Resident()
    sentinel, cached = object(), object()
    calls=[]
    def recover(*args,cache=None):
        assert args[0] is local and args[1] is trace
        calls.append(cache)
        return sentinel,cached
    monkeypatch.setattr(raw,'recover_diffusion_flux_raw_cuda',recover)
    monkeypatch.setattr(cupy_backend,'require_cupy',lambda: SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda:None))))
    monkeypatch.setattr(runtime_optional,'require_cupy',lambda: SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda:None))))
    monkeypatch.setattr(runtime_optional,'require_cupy',lambda: SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda:None))))
    monkeypatch.setattr(runtime_optional,'require_cupy',lambda: SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda:None))))
    monkeypatch.setattr(runtime_optional,'require_cupy',lambda: SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda:None))))
    monkeypatch.setattr(runtime_optional,'require_cupy',lambda: SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda:None))))
    monkeypatch.setattr(runtime_optional,'require_cupy',lambda: SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda:None))))
    monkeypatch.setattr(runtime_optional,'require_cupy',lambda: SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda:None))))
    cache=None
    for tau in (1.,2.):
        primal,flux,cache=diffusion._postprocess_diffusion_solution(local,trace,space,tau,1.,'flux',
            trace_space=trace_space,cache=cache,flux_postprocess_space=variant,postprocessing_backend='raw-cuda')
        assert primal is None and flux is sentinel
        assert cache.flux_schur_lu is None  # No large host factorization.
    assert calls == [None,cached]


# Explicit opt-in: this test compiles CUDA kernels and must be run by the user.
import os
@pytest.mark.skipif(os.environ.get('HDGFEM_RUN_CUDA_RECOVERY_TESTS')!='1',
                    reason='CUDA compilation requires explicit opt-in')
@pytest.mark.parametrize('degree', [0,2,5])
@pytest.mark.parametrize('variant', ['RT_projection','l2_closest'])
def test_cuda_recovery_matches_host_and_reuses_geometry(degree,variant,monkeypatch):
    import cupy as cp
    from hdgfem.mixed.postprocess.flux_recovery_raw_cuda import (
            recover_diffusion_flux_raw_cuda,
        )
    mesh=rectangle_mesh(1,1)
    mesh=DGMesh.from_arrays(mesh.node_coords @ np.array([[1.8,.4],[-.2,.7]]),mesh.triangles)
    space=DGSpace(mesh,degree,basis_type='dub_orth')
    trace_space=space.trace_space('legendre-modal')
    rng=np.random.default_rng(43)
    cache=None
    for tau in (.4,1.7,np.array([[.4,.8,1.2],[.5,.9,1.3]])):
        local=rng.normal(size=(mesh.num_tri,3*space.el_dof))
        trace=rng.normal(size=mesh.num_edg*trace_space.edg_dof)
        expected=_postprocess_diffusion_solution(local,trace,space,tau,1.,'flux',
            trace_space=trace_space,flux_postprocess_space=variant,postprocessing_backend='numba')[1]
        previous=cache
        with monkeypatch.context() as guard:
            def reject(*args,**kwargs):
                raise AssertionError('Unexpected field transfer to host during CUDA recovery')
            guard.setattr(cp,'asnumpy',reject)
            field,cache=recover_diffusion_flux_raw_cuda(cp.asarray(local),cp.asarray(trace),
                space,trace_space,tau,variant,cache=cache)
            cp.cuda.get_current_stream().synchronize()
        if previous is not None:
            assert cache is previous
        np.testing.assert_allclose(field.as_component_first(),expected.as_component_first(),atol=2.e-9,rtol=2.e-9)


@pytest.mark.parametrize('preset', [
    'euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6_fast',
    'euler_vortex_gas_si_bdf2_p6_poisson_p5_l2_closest_p6_fast',
    'positive_turbulence_iter_si_bdf2_p6_poisson_p5_rt_p6_fast',
])
def test_solver_hands_device_buffers_to_recovery_and_reuses_cache_on_rhs_updates(monkeypatch, preset):
    """Exercise the runner's public source/boundary-update path without solving."""
    import hdgfem.solvers.diffusion_reaction as diffusion
    from scripts.guiding_center.runtime.configuration import _make_poisson_options
    space=DGSpace(rectangle_mesh(1,1),1,basis_type='dub_orth')
    local,trace,flux,cache=object(),object(),object(),object()
    options = _make_poisson_options(preset_by_key(preset))
    assert options.cache_device_matrix
    solver = diffusion.DiffusionReactionHDGSolver(
        space, source=space.zeros(), reaction=space.zeros(), boundary_condition=0., options=options)
    calls = []
    def recover(actual_local,actual_trace,*args,**kwargs):
        assert actual_local is local and actual_trace is trace
        assert kwargs['postprocessing_backend']=='raw-cuda'
        assert kwargs['flux_postprocess_space']==options.flux_postprocess_space
        calls.append(kwargs['cache'])
        return None,flux,cache
    monkeypatch.setattr(diffusion,'_postprocess_diffusion_solution',recover)
    monkeypatch.setattr(postprocess_flux,'_postprocess_diffusion_solution',recover)
    timings=diffusion.DiffusionReactionTimings(0.,0.,0.,0.,0.,0.,0.)
    for source in (0., 1., 2.):
        solver.set_source(space.constant(source))
        solver.set_boundary_condition(source)
        if calls:
            assert solver._hdg_postprocess_cache is cache
        result=diffusion.DiffusionReactionResult(field=space.zeros(),flux=None,trace=None,
            timings=timings,trace_device=trace,local_unknowns_device=local)
        recovered=solver._postprocess_result(result)
        assert recovered.postprocessed_flux is flux
        assert recovered.trace_device is recovered.local_unknowns_device is None
        assert solver._hdg_postprocess_cache is cache
    assert calls == [None, cache, cache]
