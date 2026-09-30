"""Saved-face rank and local stabilization checks; no time integration."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from hdgfem.solvers import ScaledUpwind
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.hdg.stabilization import advection_trace_stabilization_values
from hdgfem.transport.numba import (
    _advection_stabilization_coefficients,
    _advection_trace_weight_tables,
)
from hdgfem.transport.raw_cuda import _kernel_source, _RAW_FUSED_TEMPLATE
from hdgfem.transport.diagnostics import transport_rank_failure_details
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _runtime_config, _validate_config
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key

DATA=json.loads((Path(__file__).parent/'data/recovered_drift_rank_failures.json').read_text())


@pytest.mark.parametrize('case',DATA,ids=lambda d:d['variant'])
def test_saved_face_has_full_trace_support_with_lax_friedrichs(case):
    pair=np.array(case['normal_samples']);basis=np.array(case['trace_basis']);weights=np.array(case['weights'])
    space=DGSpace(rectangle_mesh(1,1),6,basis_type='dub_orth',edge_quad_1d=13)
    trace=space.trace_space('legacy-lagrange')
    normal=np.zeros((2,3,13));normal[:,0]=pair
    ranks=[]
    for policy in (None,'lax-friedrichs'):
        tau=advection_trace_stabilization_values(space,normal,policy,trace_space=trace)
        gamma=(tau-normal)[:,0].sum(axis=0)
        weighted=np.sqrt(weights*gamma/gamma.max())[:,None]*basis.T
        singular=np.linalg.svd(weighted,compute_uv=False)
        ranks.append(np.count_nonzero(singular>max(weighted.shape)*np.finfo(float).eps*singular[0]))
        if policy:
            assert singular[-1]/singular[0]>.25
    assert ranks==[6,7]
    assert case['column_panel_rank']==6  # Independently measured original assembled block.


@pytest.mark.parametrize('policy', [None, 'lax-friedrichs', ScaledUpwind(1), ScaledUpwind(1.25), ScaledUpwind(3.5)])
def test_numpy_numba_and_cupy_policy_weights_agree_without_cuda(monkeypatch, policy):
    import hdgfem.transport.cupy as backend
    space=DGSpace(rectangle_mesh(1,1),3,basis_type='dub_orth')
    trace=space.trace_space('legendre-modal')
    beta=np.random.default_rng(13).normal(size=(2,space.mesh.num_tri,space.el_dof))
    normal=np.einsum('dki,kfd,fiq->kfq',beta,space.mesh.normals,trace.bas_of_bd_quads)
    expected=advection_trace_stabilization_values(space,normal,policy,trace_space=trace)
    kind,scalar,coeff=_advection_stabilization_coefficients(policy,space)
    tau,gamma=_advection_trace_weight_tables(space,beta,kind,scalar,coeff,trace_space=trace)
    np.testing.assert_allclose(tau,expected,rtol=2.e-13,atol=2.e-13)
    np.testing.assert_allclose(gamma,expected-normal,rtol=2.e-13,atol=2.e-13)
    monkeypatch.setattr(backend,'require_cupy',lambda:np)
    tau_cp,gamma_cp=backend._advection_trace_weights_cupy(policy,None,normal,None)
    np.testing.assert_allclose(tau_cp,expected)
    np.testing.assert_allclose(gamma_cp,expected-normal)


def test_raw_source_specializes_both_assembly_and_reconstruction():
    for policy,factor in [(None,1),('lax-friedrichs',2),(ScaledUpwind(1.25),1.25),(ScaledUpwind(3.5),3.5)]:
        source=_kernel_source(_RAW_FUSED_TEMPLATE,nel=28,ntr=7,ncols=22,nqf=13,
                              advection_stabilization=policy)
        assert f'#define RAW_ADVECTION_TAU_FACTOR {factor}\n' in source
        assert 'RAW_ADVECTION_TAU_FACTOR * fabs(normal_flux)' in source
        assert 'reconstruct_advection_raw_fused' in source
    with pytest.raises(ValueError):
        _kernel_source(_RAW_FUSED_TEMPLATE,nel=28,ntr=7,ncols=22,nqf=13,advection_stabilization='typo')


@pytest.mark.parametrize('variant',['rt','l2_closest'])
def test_only_recovered_presets_select_new_flux_and_cli_can_restore_upwind(variant):
    name=f'euler_vortex_gas_si_bdf2_p6_poisson_p5_{variant}_p6'
    config=preset_by_key(name)
    assert config.transport_advection_stabilization=='conflict-averaged-upwind'
    _validate_config(config)
    args=build_parser().parse_args([name,'--transport-advection-stabilization','upwind','--dry-run'])
    assert _runtime_config(config,args).transport_advection_stabilization is None
    assert preset_by_key('euler_vortex_gas_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr').transport_advection_stabilization is None


def test_inflow_only_rank_claim_does_not_trigger_lf_poisson_retry():
    error=RuntimeError('transport failure')
    error.transport_diagnostics={'advection_stabilization':'lax-friedrichs',
        'trace_inflow_diagnostics':{'trace_dofs_per_face':7,'worst_faces':[
            {'edge':1,'inflow_nodes':6,'outward_normal_samples':[[1.],[-1.]]}]}}
    assert transport_rank_failure_details(error) is None
    error.transport_diagnostics['advection_stabilization']='upwind'
    assert transport_rank_failure_details(error)['edges']==[1]


def test_raw_policy_preflight_accepts_scaled_assembly_modes():
    from hdgfem.solvers.capabilities import validate_advection_backend_configuration
    common=dict(operation='solve',assembly_backend='raw-cuda',solver='amgx',cupyx_solver='bicgstab',
        boundary_mode='zero-flux',trace_basis='legacy-lagrange',trace_ordering='none',
        materialize_host_solution=False,raw_lu_mode='coop',raw_matrix_format='bsr',
        requires_host_system=False,advection_stabilization_is_default=False,
        advection_stabilization_is_lax_friedrichs=True)
    validate_advection_backend_configuration(raw_local_assembly='fused',**common)
    validate_advection_backend_configuration(raw_local_assembly='split3',**common)


import os
@pytest.mark.skipif(os.environ.get('HDGFEM_RUN_CUDA_TRANSPORT_TESTS')!='1',
                   reason='CUDA compilation requires explicit opt-in')
@pytest.mark.parametrize('policy',[ScaledUpwind(1.25), ScaledUpwind(3.5)])
@pytest.mark.parametrize('assembly_mode',['fused','split3'])
@pytest.mark.parametrize('cache_response',[False,True])
def test_cuda_lax_friedrichs_local_assembly_and_reconstruction(cache_response, assembly_mode, policy):
    import cupy as cp
    from scipy.sparse import coo_matrix,bsr_matrix
    from hdgfem.core.device import as_cupy_space, as_cupy_vector_coefficients
    from hdgfem.core.device import as_cupy_trace_space
    from hdgfem.transport.cuda import (
            assemble_reduced_system_cuda,
            reconstruct_advection_field_cuda,
        )
    from hdgfem.transport.numba import (
            assemble_projected_trace_system_zero_flux_numba,
            reconstruct_projected_field_numba,
        )
    space=DGSpace(rectangle_mesh(1,1),6,basis_type='dub_orth')
    trace=space.trace_space('legacy-lagrange')
    rng=np.random.default_rng(44)
    beta=space.vector_field([space.field(.01*rng.normal(size=space.shape)) for _ in range(2)])
    source,reaction=space.constant(1.),space.constant(1.)
    host=assemble_projected_trace_system_zero_flux_numba(source,beta,reaction,space,
        advection_stabilization=policy,trace_space=trace).reduction
    cspace=as_cupy_space(space)
    coeff=as_cupy_vector_coefficients(beta,cspace)
    device=assemble_reduced_system_cuda(source,reaction,None,coeff,cspace,as_cupy_trace_space(trace),
        backend='raw-cuda',raw_local_assembly=assembly_mode,raw_lu_mode='coop',raw_matrix_format='bsr',
        zero_boundary_flux=True,raw_cache_local_response=cache_response,advection_stabilization=policy)
    shape=(host.rhs.size,host.rhs.size)
    a=coo_matrix((host.data,(host.rows,host.cols)),shape=shape).toarray()
    b=bsr_matrix((cp.asnumpy(device.data),cp.asnumpy(device.indices),cp.asnumpy(device.indptr)),shape=shape).toarray()
    np.testing.assert_allclose(a,b,rtol=2.e-10,atol=2.e-11)
    np.testing.assert_allclose(host.rhs,cp.asnumpy(device.rhs),rtol=2.e-10,atol=2.e-11)
    full_trace=rng.normal(size=space.mesh.num_edg*trace.edg_dof)
    expected=reconstruct_projected_field_numba(full_trace,source,beta,reaction,space,
        advection_stabilization=policy,zero_boundary_flux=True,trace_space=trace)
    actual,_=reconstruct_advection_field_cuda(cp.asarray(full_trace),source,reaction,coeff,device)
    np.testing.assert_allclose(expected.coeffs,cp.asnumpy(actual),rtol=2.e-10,atol=2.e-11)


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_invalid_upwind_factor(factor):
    with pytest.raises(ValueError, match="finite and positive"):
        ScaledUpwind(factor)


def test_cli_multiplier_overrides_preset():
    name = "euler_vortex_gas_si_bdf2_p6_poisson_p5_rt_p6"
    args = build_parser().parse_args([name, "--transport-upwind-factor", "1.25", "--dry-run"])
    config = _runtime_config(preset_by_key(name), args)
    assert config.transport_advection_stabilization == ScaledUpwind(1.25)
    _validate_config(config)


@pytest.mark.parametrize("factor", [1., 1.25, 3.5])
def test_cuda_precomputed_face_matrices_with_numpy_standin(monkeypatch, factor):
    import hdgfem.transport.cuda as backend
    from hdgfem.hdg import matrices as reference
    space = DGSpace(rectangle_mesh(1, 1), 3, basis_type="dub_orth")
    trace = space.trace_space("legendre-modal")
    normal = np.random.default_rng(70).normal(size=(space.mesh.num_tri, 3, trace.weights.size))
    tau = factor * np.abs(normal)
    gamma = tau - normal
    monkeypatch.setattr(backend, "require_cupy", lambda: np)
    monkeypatch.setattr(backend, "oriented_trace_basis_cupy",
                        lambda _space, _trace: reference._oriented_trace_basis_on_element_sides(space, trace_space=trace))
    cspace = SimpleNamespace(mesh=space.mesh, el_dof=space.el_dof, edg_dof=trace.edg_dof)
    pairs = [
        (backend.boundary_mass_cupy, reference.boundary_mass_from_trace_stabilization, tau),
        (backend.element_boundary_mats_cupy, reference.element_boundary_mats_from_trace_weight, gamma),
        (backend.trace_lift_cupy, reference.advection_trace_lift_from_stabilization, tau),
        (backend.interior_trace_mass_blocks_cupy, reference.advection_interior_trace_mass_blocks_from_weight, gamma),
    ]
    for actual, expected, weight in pairs:
        np.testing.assert_allclose(actual(normal, cspace, trace, upwind_scale=factor),
                                   expected(space, weight, trace_space=trace), atol=1.e-13)
