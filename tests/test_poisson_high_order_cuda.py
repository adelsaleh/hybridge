"""Small stationary matrix/field checks; no simulations or AMGX builds."""
import numpy as np
import pytest
import scipy.sparse as sparse

from hybridge import DGMesh, DGSpace, DiffusionReactionHDGSolver
from hybridge.core.device import as_cupy_space
from hybridge.hdg.condensation_device import reconstruct_trace_cupy
from hybridge.mixed.cupy import (
    assemble_projected_diffusion_trace_system_eliminated_cupy,
    assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
    assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy,
    assemble_projected_diffusion_trace_rhs_cached_cupy,
    attach_schur_cholesky_cache_cupy,
    build_trace_reference,
    reconstruct_compact_diffusion_field_cupy,
    solve_local_mats,
    solve_mixed_from_scalar_cholesky_cupy,
)
from hybridge.mixed.numba import reconstruct_projected_diffusion_local_unknowns_numba
from hybridge.mixed.raw_cuda.identity import reconstruct_projected_diffusion_field_raw_cuda


@pytest.fixture(scope='module')
def cp():
    cupy=pytest.importorskip('cupy')
    try:
        if cupy.cuda.runtime.getDeviceCount()<1:
            pytest.skip('CUDA device unavailable')
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    return cupy


def mesh():
    # Distorted cells with both face orientations and boundary/interior faces.
    return DGMesh.from_arrays(np.array([[-1.,-.9],[1.2,-.7],[-.7,1.1],[.05,-.03]]),
                             np.array([[0,1,3],[1,2,3],[2,0,3]],dtype=np.int64))


def canonical(assembled,cp):
    data=cp.asnumpy(assembled.data)
    size=int(assembled.rhs.size)
    if assembled.matrix_format=='coo':
        matrix=sparse.coo_matrix((data,(cp.asnumpy(assembled.rows),cp.asnumpy(assembled.cols))),shape=(size,size))
    else:
        constructor=sparse.bsr_matrix if assembled.matrix_format=='bsr' else sparse.csr_matrix
        matrix=constructor((data,cp.asnumpy(assembled.indices),cp.asnumpy(assembled.indptr)),shape=(size,size))
    return matrix.tocsr().toarray()


def cupy_reconstruct(assembled,trace,device):
    """Use the existing arbitrary-order dense CuPy/cuBLAS reconstruction."""
    traces=trace.reshape((device.mesh.num_edg,device.edg_dof))[device.mesh.loc2glob_edge]
    traces=traces.reshape((device.mesh.num_tri,3*device.edg_dof))
    local_rhs=assembled.source_rhs[...,None]+assembled.element_boundary_mats@traces[...,None]
    if assembled.schur_cholesky_cache is not None:
        return solve_mixed_from_scalar_cholesky_cupy(assembled.schur_cholesky_cache,local_rhs).squeeze(-1)
    return solve_local_mats(assembled.local_lhs,local_rhs).squeeze(-1)


@pytest.mark.parametrize('order',range(10))
@pytest.mark.parametrize('basis',['legacy-lagrange','legendre-modal'])
@pytest.mark.parametrize('fmt',['coo','csr','bsr'])
@pytest.mark.parametrize('policy',['none','schur-lu','schur-cholesky'])
def test_direct_formats_and_mixed_reconstruction(cp,order,basis,fmt,policy):
    space=DGSpace(mesh(),order,basis_type='dub_orth')
    source=space.project_callable(lambda x,y:1.+.2*x-.1*y)
    reaction=space.zeros()
    boundary=lambda x,y:.3+x-.5*y+.1*x*y
    device=as_cupy_space(space)
    tr=build_trace_reference(device,basis)
    raw=assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source,reaction,boundary,
        1.3,space,trace_basis=basis,matrix_format=fmt,block_size=128,trace_ref=tr,
        cache_local_factors=policy!='none',local_factor_kind=policy if policy!='none' else 'schur-lu')
    oracle=DiffusionReactionHDGSolver(space,source=source,reaction=reaction,
        boundary_condition=boundary,stabilization=1.3,trace_basis=basis,
        assembly_backend='numpy',boundary_mode='eliminate',verbose=0).assemble_global_matrix()
    expected=sparse.coo_matrix((oracle.data,(oracle.rows,oracle.cols)),shape=(oracle.rhs.size,)*2).toarray()
    matrix=canonical(raw,cp)
    cupy_oracle=assemble_projected_diffusion_trace_system_eliminated_cupy(source,reaction,boundary,
        1.3,space,trace_basis=basis,trace_ref=tr,use_schur_cholesky=policy=='schur-cholesky')
    np.testing.assert_allclose(matrix,canonical(cupy_oracle,cp),rtol=2.e-11,atol=3.e-11)
    cp.testing.assert_allclose(raw.rhs,cupy_oracle.rhs,rtol=2.e-11,atol=3.e-11)
    np.testing.assert_allclose(matrix,expected,rtol=2.e-11,atol=3.e-11)
    np.testing.assert_allclose(cp.asnumpy(raw.rhs),oracle.rhs,rtol=2.e-11,atol=3.e-11)
    solution=np.linalg.solve(expected,oracle.rhs)
    trace=reconstruct_trace_cupy(cp.asarray(solution),raw.boundary_trace,device)
    expected_local=reconstruct_projected_diffusion_local_unknowns_numba(cp.asnumpy(trace),source,
        reaction,1.3,space,trace_space=space.trace_space(basis))
    r=raw.raw_assembly
    if policy=='schur-cholesky':
        raw=attach_schur_cholesky_cache_cupy(raw,reaction,device,tr,1.3)
        uh,local,_=reconstruct_compact_diffusion_field_cupy(trace,raw.schur_cholesky_cache,device)
    else:
        uh,local,_=reconstruct_projected_diffusion_field_raw_cuda(trace=trace,source_rhs=raw.source_rhs,
            cspace=device,trace_ref=tr,d0_reference=r.d0_reference,d1_reference=r.d1_reference,
            face_element_mass=r.face_element_mass,tau=1.3,block_size=128,return_local_unknowns=True,
            cached_factors=r if policy=='schur-lu' else None)
    np.testing.assert_allclose(cp.asnumpy(local),expected_local,rtol=2.e-10,atol=3.e-11)
    cp.testing.assert_allclose(local,cupy_reconstruct(cupy_oracle,trace,device),rtol=2.e-10,atol=3.e-11)
    assert np.linalg.norm(matrix@solution-cp.asnumpy(raw.rhs))<=2.e-10*max(1.,np.linalg.norm(oracle.rhs))


@pytest.mark.parametrize('element_basis',['bernstein','hier_C0'])
@pytest.mark.parametrize('basis',['legacy-lagrange','legendre-modal'])
@pytest.mark.parametrize('policy',['none','schur-lu','schur-cholesky'])
def test_high_order_other_element_bases(cp,element_basis,basis,policy):
    """Use normwise CuPy parity for the less well-conditioned element bases."""
    space=DGSpace(mesh(),9,basis_type=element_basis)
    device=as_cupy_space(space)
    tr=build_trace_reference(device,basis)
    source,reaction=space.constant(1.),space.zeros()
    boundary=lambda x,y:.3+x-.5*y
    raw=assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source,reaction,boundary,
        1.3,space,trace_basis=basis,matrix_format='bsr',block_size=128,trace_ref=tr,
        cache_local_factors=policy!='none',local_factor_kind=policy if policy!='none' else 'schur-lu')
    oracle=assemble_projected_diffusion_trace_system_eliminated_cupy(source,reaction,boundary,
        1.3,space,trace_basis=basis,trace_ref=tr,use_schur_cholesky=policy=='schur-cholesky')
    matrix,expected=canonical(raw,cp),canonical(oracle,cp)
    assert np.linalg.norm(matrix-expected)/max(1.,np.linalg.norm(expected))<5.e-11
    assert float((cp.linalg.norm(raw.rhs-oracle.rhs)/cp.maximum(cp.linalg.norm(oracle.rhs),1.)).item())<5.e-11
    trace=cp.sin(cp.arange(space.mesh.num_edg*device.edg_dof,dtype=cp.float64))
    r=raw.raw_assembly
    if policy=='schur-cholesky':
        raw=attach_schur_cholesky_cache_cupy(raw,reaction,device,tr,1.3)
        _,local,_=reconstruct_compact_diffusion_field_cupy(trace,raw.schur_cholesky_cache,device)
    else:
        _,local,_=reconstruct_projected_diffusion_field_raw_cuda(trace=trace,source_rhs=raw.source_rhs,
            cspace=device,trace_ref=tr,d0_reference=r.d0_reference,d1_reference=r.d1_reference,
            face_element_mass=r.face_element_mass,tau=1.3,block_size=128,return_local_unknowns=True,
            cached_factors=r if policy=='schur-lu' else None)
    expected_local=cupy_reconstruct(oracle,trace,device)
    # Degree-nine Bernstein coefficients reach O(1e4) for this arbitrary trace;
    # use the same 2e-10 relative reconstruction target as the mixed-field checks.
    assert float((cp.linalg.norm(local-expected_local)/cp.maximum(cp.linalg.norm(expected_local),1.)).item())<2.e-10


@pytest.mark.parametrize('order',[7,8,9])
@pytest.mark.parametrize('basis',['legacy-lagrange','legendre-modal'])
@pytest.mark.parametrize('fmt',['csr','bsr'])
@pytest.mark.parametrize('policy',['none','schur-lu','schur-cholesky'])
def test_changing_source_and_nonzero_boundary_reuses_operator(cp,order,basis,fmt,policy):
    space=DGSpace(mesh(),order,basis_type='dub_orth')
    device=as_cupy_space(space)
    tr=build_trace_reference(device,basis)
    reaction=space.zeros()
    def assemble(source,boundary):
        result=assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source,reaction,boundary,
            1.3,space,trace_basis=basis,matrix_format=fmt,block_size=64,trace_ref=tr,
            cache_local_factors=policy!='none',local_factor_kind=policy if policy!='none' else 'schur-lu')
        return attach_schur_cholesky_cache_cupy(result,reaction,device,tr,1.3) if policy=='schur-cholesky' else result
    source=space.constant(1.)
    cached=assemble(source,lambda x,y:0.*x)
    matrix=canonical(cached,cp)
    pointer=cached.data.data.ptr
    factor=cached.raw_assembly.poisson_context.factors
    for amplitude in (1.,-.7,0.):
        source=space.project_callable(lambda x,y:1.+amplitude*(.2+x*y))
        boundary=lambda x,y:amplitude*(1.+x-.3*y)
        if policy=='schur-cholesky' and amplitude!=-.7:
            cached=assemble_projected_diffusion_trace_rhs_cached_cupy(source,boundary,device,tr,cached)
        else:
            cached=assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy(source,reaction,boundary,
                1.3,space,cached_raw=cached.raw_assembly,trace_basis=basis,block_size=64,trace_ref=tr)
            if policy=='schur-cholesky':
                cached=attach_schur_cholesky_cache_cupy(cached,reaction,device,tr,1.3)
        fresh=assemble(source,boundary)
        cupy_oracle=assemble_projected_diffusion_trace_system_eliminated_cupy(source,reaction,boundary,
            1.3,space,trace_basis=basis,trace_ref=tr,use_schur_cholesky=policy=='schur-cholesky')
        assert cached.matrix_format==fmt
        assert cached.data.data.ptr==pointer
        if policy!='none':
            assert cached.raw_assembly.poisson_context.factors is factor
        np.testing.assert_allclose(cp.asnumpy(cached.rhs),cp.asnumpy(fresh.rhs),rtol=2.e-10,atol=3.e-11)
        cp.testing.assert_allclose(cached.rhs,cupy_oracle.rhs,rtol=2.e-10,atol=3.e-11)
        solution=np.linalg.solve(matrix,cp.asnumpy(fresh.rhs))
        trace=reconstruct_trace_cupy(cp.asarray(solution),cached.boundary_trace,device)
        expected=reconstruct_projected_diffusion_local_unknowns_numba(cp.asnumpy(trace),source,
            reaction,1.3,space,trace_space=space.trace_space(basis))
        if policy=='schur-cholesky':
            _,local,_=reconstruct_compact_diffusion_field_cupy(trace,cached.schur_cholesky_cache,device)
        else:
            r=cached.raw_assembly
            _,local,_=reconstruct_projected_diffusion_field_raw_cuda(trace=trace,source_rhs=cached.source_rhs,
                cspace=device,trace_ref=tr,d0_reference=r.d0_reference,d1_reference=r.d1_reference,
                face_element_mass=r.face_element_mass,tau=1.3,block_size=64,return_local_unknowns=True,
                cached_factors=r if policy=='schur-lu' else None)
        np.testing.assert_allclose(cp.asnumpy(local),expected,rtol=2.e-10,atol=3.e-11)
        cp.testing.assert_allclose(local,cupy_reconstruct(cupy_oracle,trace,device),rtol=2.e-10,atol=3.e-11)


def test_automatic_tuning_persists_and_does_not_modify_cached_factors(cp,tmp_path,monkeypatch):
    from hybridge.hdg.cuda.tuning import clear_tuning_cache
    from hybridge.mixed.raw_cuda import poisson
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR',str(tmp_path))
    clear_tuning_cache()
    space=DGSpace(mesh(),9,basis_type='dub_orth')
    source,reaction=space.constant(1.),space.zeros()
    def assemble():
        return assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source,reaction,
            lambda x,y:1.+x,1.3,space,trace_basis='legendre-modal',matrix_format='bsr',
            block_size='auto',cache_local_factors=True)
    first=assemble()
    assert first.raw_assembly.poisson_context.reports['assembly']['source']=='measured'
    assert list(tmp_path.glob('*.json'))
    clear_tuning_cache()
    def no_trials(*args,**kwargs):
        raise AssertionError('a persistent cache hit must not allocate tuning scratch')
    monkeypatch.setattr(poisson,'sample_elements',no_trials)
    second=assemble()
    assert second.raw_assembly.poisson_context.reports['assembly']['source']=='disk'
    np.testing.assert_allclose(cp.asnumpy(first.data),cp.asnumpy(second.data),rtol=1.e-13,atol=1.e-13)
    np.testing.assert_array_equal(cp.asnumpy(first.raw_assembly.schur_lu),cp.asnumpy(second.raw_assembly.schur_lu))


@pytest.mark.parametrize('order',[7,9])
@pytest.mark.parametrize('basis',['legacy-lagrange','legendre-modal'])
@pytest.mark.parametrize('fmt',['csr','bsr'])
@pytest.mark.parametrize('scale',[False,True])
@pytest.mark.parametrize('policy',['none','schur-lu','schur-cholesky'])
def test_public_solver_preserves_format_and_cache_with_changing_boundary(cp,order,basis,fmt,scale,policy):
    import json
    from pathlib import Path
    pytest.importorskip('pyamgx')
    config=json.loads(Path('configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json').read_text())
    config['solver'].update(print_solve_stats=0,monitor_residual=0)
    config['solver']['preconditioner'].update(print_grid_stats=0)
    if fmt=='bsr':
        config['solver']['preconditioner']['classical_bsr_hierarchy']='scalar_expand'
    space=DGSpace(mesh(),order,basis_type='dub_orth')
    solver=DiffusionReactionHDGSolver(space,source=space.constant(1.),reaction=space.zeros(),
        boundary_condition=0.,stabilization=1.3,assembly_backend='raw-cuda',
        solver='amgx',amgx_config=config,solver_rtol=1.e-9,maxiter=200,scale_system=scale,
        trace_basis=basis,raw_matrix_format=fmt,raw_block_size=128,
        cache_local_factors=policy,boundary_mode='eliminate',hdg_postprocess='none',verbose=0)
    try:
        pointer=None
        for amplitude in (0.,1.,-.7,0.):
            source=space.project_callable(lambda x,y:1.+amplitude*(.2+x*y))
            boundary=lambda x,y:amplitude*(1.+x-.3*y)
            solver.set_source(source)
            solver.set_boundary_condition(boundary)
            result=solver.solve()
            cached=solver._raw_cuda_assembly_cache
            assert cached.matrix_format==fmt
            if pointer is None:
                pointer=cached.data.data.ptr
            assert cached.data.data.ptr==pointer
            assert result.global_solve_result.physical_residual_target_met
            matrix=canonical(cached,cp)
            solution=np.linalg.solve(matrix,cp.asnumpy(cached.rhs))
            trace=reconstruct_trace_cupy(cp.asarray(solution),cached.boundary_trace,as_cupy_space(space))
            expected=reconstruct_projected_diffusion_local_unknowns_numba(cp.asnumpy(trace),source,
                space.zeros(),1.3,space,trace_space=space.trace_space(basis))
            np.testing.assert_allclose(result.field.coeffs,expected[:,:space.el_dof],rtol=2.e-8,atol=2.e-9)
            np.testing.assert_allclose(result.flux.as_component_first(),
                np.stack((expected[:,space.el_dof:2*space.el_dof],expected[:,2*space.el_dof:])),
                rtol=2.e-8,atol=2.e-9)
    finally:
        solver.clear_cache()


def test_cached_factor_owner_and_single_dof_accounting(cp):
    from hybridge.mixed.raw_cuda import poisson
    triangle=DGMesh.from_arrays(np.array([[0.,0.],[1.,0.],[0.,1.]]),np.array([[0,1,2]],dtype=np.int64))
    space=DGSpace(triangle,0)
    assembled=assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
        space.constant(1.),space.zeros(),lambda x,y:0.*x,1.3,space,matrix_format='csr',block_size=32,
        cache_local_factors=True)
    context=assembled.raw_assembly.poisson_context
    assert context.local_factor_bytes==context.factors.nbytes+context.pivots.nbytes
    with pytest.raises(ValueError,match='different space, trace basis, or stabilization'):
        poisson._validate_context(context,context.cspace,context.trace_ref,1.4,None)
    with pytest.raises(ValueError,match='factor key'):
        poisson._validate_context(context,context.cspace,context.trace_ref,1.3,'different')
