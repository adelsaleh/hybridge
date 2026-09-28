"""Assembly-only tensor ADR qualification: no AMGX, solves, or time integration."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, bsr_matrix

from hdgfem import DGMesh, DGSpace, rectangle_mesh
from hdgfem.assembly.advection_diffusion_reaction import prepare_adr_data, assemble_numpy
from hdgfem.assembly.diffusion_coefficients import prepare_diffusion
from hdgfem.backends.advection_diffusion_reaction_numba import assemble_projected_adr_trace_system_eliminated_numba
from hdgfem.backends.advection_diffusion_reaction_raw_cuda import assemble_projected_adr_trace_operator_raw_cuda
from test_adr_tensor_numba import diffusion_cases


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    return cupy


def dense_host(system):
    return coo_matrix((system.data, (system.rows, system.cols)), shape=(system.rhs.size,)*2).toarray()


def dense_device(cp, system):
    shape = (system.rhs.size,)*2
    if system.matrix_format == 'coo':
        return coo_matrix((cp.asnumpy(system.data), (cp.asnumpy(system.rows), cp.asnumpy(system.cols))), shape=shape).toarray()
    constructor = bsr_matrix if system.matrix_format == 'bsr' else csr_matrix
    return constructor((cp.asnumpy(system.data), cp.asnumpy(system.indices), cp.asnumpy(system.indptr)), shape=shape).toarray()


def velocity(space, x=.7, y=-.2):
    return (space*space).field((space.constant(x), space.constant(y)))


def tau_law(x, y, *, element, local_face, normal, t=None):
    """Vary along a face and independently on both incidences."""
    return 2. + .05*x + .03*y + .1*element + .02*local_face + .01*normal[..., 0]


@pytest.mark.parametrize('order', range(7))
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
@pytest.mark.parametrize('diffusion,kind', diffusion_cases())
def test_tensor_assembly_all_shapes(cp, order, basis, diffusion, kind):
    mesh = rectangle_mesh(2, 1)
    mesh = DGMesh.from_arrays(mesh.node_coords @ np.array([[1.2,.3],[-.1,.8]]), mesh.triangles)
    space = DGSpace(mesh, order, basis_type='dub_orth', volume_quad_1d=order+3)
    trace = space.trace_space(basis)
    kwargs = dict(diffusion=diffusion, diffusion_stabilization=tau_law, trace_space=trace)
    source, beta, reaction = (lambda x,y: 1.+.1*x*y), velocity(space), (lambda x,y: .4+.05*x)
    prepared = prepare_adr_data(source, reaction, beta, space, **kwargs)
    light = prepare_adr_data(source, reaction, beta, space, dense_local_matrices=False, **kwargs)
    assert light.element_boundary is None and light.u_boundary_mass is None
    boundary = lambda x,y: .2+x-.3*y
    numpy_assembly = assemble_numpy(prepared, boundary, space, diffusion=diffusion, trace_space=trace)
    numpy = numpy_assembly.trace_system
    numba = assemble_projected_adr_trace_system_eliminated_numba(
        prepared, boundary, space, diffusion_data=prepare_diffusion(diffusion,space), trace_space=trace).trace_system
    expected = dense_host(numpy)
    np.testing.assert_allclose(dense_host(numba), expected, rtol=2e-10, atol=3e-10)
    np.testing.assert_allclose(numba.rhs, numpy.rhs, rtol=2e-10, atol=3e-10)
    for fmt in ('coo','csr','bsr'):
        operator = assemble_projected_adr_trace_operator_raw_cuda(
            light, boundary, space, diffusion=diffusion, trace_space=trace, matrix_format=fmt)
        actual = operator.assembly
        np.testing.assert_allclose(dense_device(cp, actual), expected, rtol=2e-10, atol=3e-10)
        np.testing.assert_allclose(cp.asnumpy(actual.rhs), numpy.rhs, rtol=2e-10, atol=3e-10)
        assert operator.diffusion_structure[kind] == mesh.num_tri
        assert actual.timings['raw.shared_bytes'] <= 48*1024
        assert actual.timings['raw.batch_columns'] <= 8
        assert actual.timings['raw.conversion'] == 0
        if fmt != 'coo':
            assert actual.rows is None and actual.cols is None
            assert operator.csr_pattern is not None
        if fmt == 'bsr':
            assert actual.data.shape[1:] == (order+1, order+1)
        if fmt == 'csr':
            from hdgfem.assembly import hdg
            from hdgfem.backends.advection_diffusion_reaction_numba import reconstruct_projected_adr_local_unknowns_numba
            from hdgfem.backends.advection_diffusion_reaction_raw_cuda import reconstruct_projected_adr_local_unknowns_raw_cuda
            trace_values=np.sin(np.arange(mesh.num_edg*(order+1))+.2)
            source_block=np.zeros((mesh.num_tri,3*space.el_dof))
            source_block[:,:space.el_dof]=prepared.source_rhs
            reference=hdg.reconstruct_local_unknowns(trace_values,source_block,numpy_assembly.local_solver,
                prepared.element_boundary,space,trace_space=trace)
            numba_local=reconstruct_projected_adr_local_unknowns_numba(trace_values,prepared,space,
                trace_space=trace,diffusion=diffusion)
            unknowns,timings=reconstruct_projected_adr_local_unknowns_raw_cuda(operator,cp.asarray(trace_values))
            assert isinstance(unknowns,cp.ndarray)
            np.testing.assert_allclose(numba_local,reference,rtol=3e-10,atol=3e-10)
            np.testing.assert_allclose(cp.asnumpy(unknowns),reference,rtol=3e-10,atol=3e-10)
            assert timings['raw.reconstruction.device']>0



@pytest.mark.parametrize('block_size', [1,32,64,128])
@pytest.mark.parametrize('fmt', ['coo','csr','bsr'])
def test_mixed_cross_space_and_launch_sizes(cp, block_size, fmt):
    mesh = rectangle_mesh(2,1,xlim=(-1.,1.))
    space = DGSpace(mesh,4,basis_type='dub_orth')
    coefficients = DGSpace(mesh,1,basis_type='dub_orth')
    diffusion = (coefficients.project_callable(lambda x,y: 2.+.1*x), .2,
                 coefficients.project_callable(lambda x,y: 1.+.1*y))
    beta = (coefficients*coefficients).field((coefficients.constant(.7),coefficients.constant(-.2)))
    tau = DGSpace(mesh,0).field(np.array([[1.],[2.],[3.],[4.]]))
    trace = space.trace_space('legendre-modal')
    prep = prepare_adr_data(coefficients.constant(1.), coefficients.constant(.2), beta, space,
                            diffusion=diffusion, diffusion_stabilization=tau, trace_space=trace)
    expected = assemble_numpy(prep, .3, space, diffusion=diffusion, trace_space=trace).trace_system
    result = assemble_projected_adr_trace_operator_raw_cuda(prep,.3,space,diffusion=diffusion,
                        trace_space=trace,matrix_format=fmt,block_size=block_size)
    np.testing.assert_allclose(dense_device(cp,result.assembly),dense_host(expected),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(cp.asnumpy(result.assembly.rhs),expected.rhs,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('basis', ['legacy-lagrange','legendre-modal'])
def test_mixed_element_classifications(cp,basis):
    space=DGSpace(rectangle_mesh(2,1,xlim=(-1.,1.)),3,basis_type='dub_orth')
    diffusion=(lambda x,y: np.where(x<0.,2.,2.+.1*x),0.,2.)
    trace=space.trace_space(basis)
    prep=prepare_adr_data(space.constant(1.),.3,velocity(space),space,diffusion=diffusion,trace_space=trace)
    expected=assemble_numpy(prep,.2,space,diffusion=diffusion,trace_space=trace).trace_system
    result=assemble_projected_adr_trace_operator_raw_cuda(prep,.2,space,diffusion=diffusion,trace_space=trace)
    assert result.diffusion_structure['constant-isotropic']==2
    assert result.diffusion_structure['variable-diagonal']==2
    np.testing.assert_allclose(dense_device(cp,result.assembly),dense_host(expected),rtol=1e-10,atol=1e-10)


def test_local_failures_are_reported(cp):
    space=DGSpace(rectangle_mesh(1,1),0)
    trace=space.trace_space('legacy-lagrange')
    prep=prepare_adr_data(space.constant(1.),0.,velocity(space,0.,0.),space,trace_space=trace)
    singular=replace(prep,tau_total=np.zeros_like(prep.tau_total))
    with pytest.raises(np.linalg.LinAlgError,match='scalar Schur.*element'):
        assemble_projected_adr_trace_operator_raw_cuda(singular,0.,space,trace_space=trace)
    bad=replace(prep,beta_values=np.full_like(prep.beta_values,np.nan))
    with pytest.raises(ValueError,match='finite'):
        assemble_projected_adr_trace_operator_raw_cuda(bad,0.,space,trace_space=trace)


def test_resource_and_coefficient_preflight(cp):
    space=DGSpace(rectangle_mesh(1,1),1)
    trace=space.trace_space('legacy-lagrange')
    prep=prepare_adr_data(space.constant(1.),.3,velocity(space),space,trace_space=trace)
    for diffusion in (0.,np.nan,(1.,2.,1.)):
        with pytest.raises(ValueError,match='diffusion'):
            assemble_projected_adr_trace_operator_raw_cuda(prep,0.,space,diffusion=diffusion,trace_space=trace)
    for kwargs in ({'matrix_format':'csc'},{'block_size':256}):
        with pytest.raises(ValueError):
            assemble_projected_adr_trace_operator_raw_cuda(prep,0.,space,trace_space=trace,**kwargs)
    high=DGSpace(space.mesh,7)
    with pytest.raises(ValueError,match='p=0--6'):
        assemble_projected_adr_trace_operator_raw_cuda(prep,0.,high,trace_space=high.trace_space('legacy-lagrange'))


def test_workspace_specializations():
    from hdgfem.backends.adr_tensor_raw_cuda import tensor_workspace
    for p in range(7):
        n=(p+1)*(p+2)//2
        sizes=[tensor_workspace(n,np.array([kind]))[2] for kind in range(7)]
        assert max(sizes)<=48*1024
        if p>=3:
            assert sizes[0]<=sizes[3]<=sizes[4]<sizes[5]
            assert sizes[0]<sizes[5]


def test_scalar_serial_diagnostic_is_retained(cp):
    space=DGSpace(rectangle_mesh(2,1),2)
    trace=space.trace_space('legendre-modal')
    prep=prepare_adr_data(space.constant(1.),.3,velocity(space),space,trace_space=trace)
    serial=assemble_projected_adr_trace_operator_raw_cuda(prep,.2,space,trace_space=trace,block_size=1)
    cooperative=assemble_projected_adr_trace_operator_raw_cuda(prep,.2,space,trace_space=trace)
    np.testing.assert_allclose(dense_device(cp,serial.assembly),dense_device(cp,cooperative.assembly),rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(cp.asnumpy(serial.assembly.rhs),cp.asnumpy(cooperative.assembly.rhs),rtol=1e-11,atol=1e-11)


def test_mass_factor_failure_and_trace_validation(cp):
    from hdgfem.backends.advection_diffusion_reaction_raw_cuda import reconstruct_projected_adr_local_unknowns_raw_cuda
    space=DGSpace(rectangle_mesh(1,1),1)
    trace=space.trace_space('legacy-lagrange')
    prep=prepare_adr_data(space.constant(1.),.3,velocity(space),space,trace_space=trace,dense_local_matrices=False)
    large=(lambda x,y: 1e40*(2.+.1*x),.3e40,-.1e40,1e40)
    with pytest.raises(np.linalg.LinAlgError,match='inverse-diffusion mass'):
        assemble_projected_adr_trace_operator_raw_cuda(prep,0.,space,trace_space=trace,diffusion=large)
    operator=assemble_projected_adr_trace_operator_raw_cuda(prep,0.,space,trace_space=trace,diffusion=(2.,1e-15,2.))
    assert operator.diffusion_structure['constant-full']==space.mesh.num_tri
    with pytest.raises(ValueError,match='full trace'):
        reconstruct_projected_adr_local_unknowns_raw_cuda(operator,cp.zeros(1))
    with pytest.raises(ValueError,match='finite'):
        reconstruct_projected_adr_local_unknowns_raw_cuda(operator,cp.full(space.mesh.num_edg*2,cp.nan))
