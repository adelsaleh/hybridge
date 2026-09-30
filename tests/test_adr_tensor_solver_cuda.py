"""Bounded stationary tensor ADR solves and recovery checks; no time stepping."""
import numpy as np
import pytest
from hdgfem import DGSpace, rectangle_mesh, solve_advection_diffusion_reaction_hdg
from scripts.advection_diffusion_reaction.cases.tensor_cases import manufactured_raw_tensor


@pytest.fixture(scope='module')
def cp():
    cp=pytest.importorskip('cupy')
    pytest.importorskip('pyamgx')
    if cp.cuda.runtime.getDeviceCount()==0:
        pytest.skip('No CUDA device')
    return cp


@pytest.mark.parametrize('order',[2,4,6])
@pytest.mark.parametrize('basis',['legacy-lagrange','legendre-modal'])
@pytest.mark.parametrize('fmt',['csr','bsr'])
def test_native_tensor_solve_and_device_reconstruction(cp,monkeypatch,order,basis,fmt):
    from hdgfem.backends import advection_cuda
    def forbidden(*args,**kwargs):
        raise AssertionError('native BSR solve unexpectedly scalarized')
    monkeypatch.setattr(advection_cuda,'_scalarize_device_bsr_matrix',forbidden)
    space=DGSpace(rectangle_mesh(2,2),order,basis_type='dub_orth')
    beta=(space*space).field((space.constant(.7),space.constant(-.2)))
    problem, exact, flux = manufactured_raw_tensor('affine')
    diffusion, source = problem['diffusion'], problem['source']
    opts=dict(diffusion=diffusion,hdg_postprocess='none',trace_basis=basis,
              diffusion_stabilization=lambda x,y: 1.+.05*x+.02*y,
              scale_system=True,verbose=False)
    reference=solve_advection_diffusion_reaction_hdg(source,beta,.3,exact,space,
        assembly_backend='numpy',solver='direct',**opts)
    actual=solve_advection_diffusion_reaction_hdg(source,beta,.3,exact,space,
        assembly_backend='raw-cuda',solver='amgx',raw_matrix_format=fmt,
        materialize_host_solution=False,solver_rtol=1e-10,**opts)
    assert isinstance(actual.local_unknowns,cp.ndarray)
    assert isinstance(actual.trace,cp.ndarray)
    assert isinstance(actual.matrix_data,cp.ndarray)
    assert actual.matrix_format==fmt
    assert actual.matrix_rows is None and actual.matrix_cols is None
    assert actual.diffusion_structure['variable-full']==space.mesh.num_tri
    assert actual.postprocessed_field is None and actual.postprocessed_flux is None
    np.testing.assert_allclose(cp.asnumpy(actual.local_unknowns),reference.local_unknowns,rtol=2e-8,atol=2e-8)
    np.testing.assert_allclose(cp.asnumpy(actual.trace),reference.trace,rtol=2e-9,atol=2e-9)
    assert actual.field.l2_error(exact)<2e-9
    assert actual.flux.l2_error(flux)<2e-8
    assert not getattr(actual.global_solve_result,'amgx_bsr_scalarized',False)


@pytest.mark.parametrize('mode', ['flux', 'primal', 'both'])
def test_tensor_postprocessing_enabled(mode):
    """The CUDA capability gate accepts qualified tensor recoveries."""
    from hdgfem.backends.capabilities import validate_advection_diffusion_backend_configuration
    validate_advection_diffusion_backend_configuration(operation='solve',assembly_backend='raw-cuda',
        solver='amgx',cupyx_solver='bicgstab',boundary_mode='eliminate',trace_basis='legendre-modal',
        postprocess_mode=mode,scalar_diffusion=False)


@pytest.mark.parametrize('variant', ['l2_closest', 'RT_projection'])
@pytest.mark.parametrize('order',[1,2])
def test_stationary_tensor_manufactured_convergence(cp,order,variant):
    problem, exact, _ = manufactured_raw_tensor('sine')
    diffusion, source = problem['diffusion'], problem['source']
    errors=[]
    recovered=[]
    for n in (2,4,8):
        space=DGSpace(rectangle_mesh(n,n,xlim=(0.,1.),ylim=(0.,1.)),order,basis_type='dub_orth',volume_quad_1d=order+4)
        beta=(space*space).field((space.constant(.7),space.constant(-.2)))
        result=solve_advection_diffusion_reaction_hdg(source,beta,.3,exact,space,
            diffusion=diffusion,assembly_backend='raw-cuda',solver='amgx',hdg_postprocess='both', flux_postprocess_space=variant,
            raw_matrix_format='bsr',trace_basis='legendre-modal',solver_rtol=1e-10,verbose=False)
        errors.append(result.field.l2_error(exact))
        recovered.append(result.postprocessed_field.l2_error(exact))
    assert np.log2(errors[-2]/errors[-1])>order+.7, errors

    assert np.log2(recovered[-2]/recovered[-1])>order+1.5, recovered
