"""Small-matrix ADR tensor parity and structural specialization checks."""
import numpy as np
import pytest
from scipy.sparse import coo_matrix

from hdgfem import DGSpace, DGMesh, rectangle_mesh, solve_advection_diffusion_reaction_hdg
from hdgfem.assembly.diffusion_coefficients import prepare_diffusion, normal_diffusivity_on_faces
from hdgfem.solvers.stabilization import GlobalLengthDiffusion
from hdgfem.backends.capabilities import UnsupportedBackendConfigurationError


def diffusion_cases():
    """Exercise every structural path, including nonsymmetric elliptic tensors."""
    return [
        pytest.param(2., 'constant-isotropic', id='scalar'),
        pytest.param(np.diag([2., 1.]), 'constant-diagonal', id='diagonal'),
        pytest.param((2., .2, 1.), 'constant-full', id='symmetric-constant'),
        pytest.param((2., .3, -.1, 1.), 'constant-full', id='general-constant'),
        pytest.param(lambda x,y: 2. + .1*x, 'variable-isotropic', id='variable-scalar'),
        pytest.param((lambda x,y: 2.+.1*x, 0., lambda x,y: 1.+.1*y), 'variable-diagonal', id='variable-diagonal'),
        pytest.param((lambda x,y: 2.+.1*x, lambda x,y: .2+.03*y, lambda x,y: 1.+.1*y), 'variable-symmetric', id='variable-symmetric'),
        pytest.param((lambda x,y: 2.+.1*x, lambda x,y: .3+.02*y, lambda x,y: -.1+.01*x, 1.), 'variable-full', id='variable-general'),
    ]


@pytest.mark.parametrize('diffusion,kind', diffusion_cases())
@pytest.mark.parametrize('basis', ['legacy-lagrange','legendre-modal'])
@pytest.mark.parametrize('order', [1, 3])
def test_tensor_solve_matches_numpy(diffusion, kind, basis, order):
    mesh = rectangle_mesh(2, 2)
    mesh = DGMesh.from_arrays(mesh.node_coords @ np.array([[1.2,.3],[-.1,.8]]), mesh.triangles)
    space = DGSpace(mesh, order, basis_type='dub_orth')
    source = space.project_callable(lambda x,y: 1.+x*y)
    reaction = space.project_callable(lambda x,y: .4+.05*x)
    beta = (space*space).field((space.constant(.7), space.constant(-.2)))
    kwargs = dict(diffusion=diffusion, trace_basis=basis, solver='direct',
                  hdg_postprocess='none', verbose=False)
    boundary = lambda x,y: .2+x-.3*y
    reference = solve_advection_diffusion_reaction_hdg(source,beta,reaction,boundary,space,
                                                       assembly_backend='numpy', **kwargs)
    actual = solve_advection_diffusion_reaction_hdg(source,beta,reaction,boundary,space,
                                                    assembly_backend='numba', **kwargs)
    def matrix(result):
        """Canonicalize duplicate COO contributions for an independent residual."""
        return coo_matrix((result.matrix_data,(result.matrix_rows,result.matrix_cols)),
                          shape=(result.rhs.size, result.rhs.size)).toarray()
    np.testing.assert_allclose(matrix(actual),matrix(reference),rtol=3e-11,atol=3e-11)
    np.testing.assert_allclose(actual.rhs,reference.rhs,rtol=3e-11,atol=3e-11)
    np.testing.assert_allclose(actual.trace,reference.trace,rtol=3e-10,atol=3e-10)
    np.testing.assert_allclose(actual.local_unknowns,reference.local_unknowns,rtol=3e-10,atol=3e-10)
    residual = matrix(reference) @ actual.trace[actual.reduction.free_mask] - reference.rhs
    assert np.linalg.norm(residual) / np.linalg.norm(reference.rhs) < 1e-10
    assert actual.diffusion_structure[kind] == mesh.num_tri
    assert sum(actual.diffusion_structure.values()) == mesh.num_tri


def test_exact_classification_does_not_drop_small_couplings():
    space = DGSpace(rectangle_mesh(1,1), 1)
    assert prepare_diffusion((2.,1e-15,2.),space).counts['constant-full'] == 2
    assert prepare_diffusion((2.,0.,2.),space).counts['constant-isotropic'] == 2
    assert prepare_diffusion(space.constant(2.),space).counts['constant-isotropic'] == 2


@pytest.mark.parametrize('diffusion',[0.,-1.,np.nan,(1.,2.,1.),(1.,np.inf,1.)])
def test_invalid_diffusion_rejected(diffusion):
    space = DGSpace(rectangle_mesh(1,1),1)
    with pytest.raises(ValueError,match='diffusion'):
        prepare_diffusion(diffusion,space)


def test_normal_diffusivity_keeps_both_incidences():
    mesh = rectangle_mesh(1,1)
    constants = DGSpace(mesh,0)
    field = constants.field(np.array([[1.],[3.]]))
    space = DGSpace(mesh,2)
    normal = normal_diffusivity_on_faces(field,space)
    np.testing.assert_allclose(normal, np.array([[1.,1.,1.],[3.,3.,3.]]))
    np.testing.assert_allclose(GlobalLengthDiffusion(domain_length=2.).resolve(field,space),normal/2.)
    k = np.array([[2.,.2],[.2,1.]])
    expected = np.einsum('Kfi,ij,Kfj->Kf',mesh.normals,k,mesh.normals)
    np.testing.assert_allclose(normal_diffusivity_on_faces(k,space),expected)


@pytest.mark.parametrize('kind', ['scalar','diagonal','symmetric','general'])
@pytest.mark.parametrize('order', [1,2])
def test_manufactured_tensor_convergence(kind,order):
    pi=np.pi
    def exact(x,y):
        """Smooth manufactured primal field."""
        return np.sin(pi*x)*np.sin(pi*y)
    def components(x,y):
        """Smooth elliptic scalar, diagonal, symmetric or general tensor."""
        a=1.+.2*x
        if kind=='scalar':
            return a,0.*x,0.*x,a
        d=2.+.1*y
        if kind=='diagonal':
            return a,0.*x,0.*x,d
        b=.15+.03*x
        c=b if kind=='symmetric' else -.05+.02*y
        return a,b,c,d
    diffusion=tuple((lambda x,y,j=j: components(x,y)[j]) for j in range(4))
    def derivatives(x,y):
        """Return analytic gradient and Hessian entries."""
        return (pi*np.cos(pi*x)*np.sin(pi*y),pi*np.sin(pi*x)*np.cos(pi*y),
                -pi*pi*exact(x,y),pi*pi*np.cos(pi*x)*np.cos(pi*y))
    def source(x,y):
        """div(beta*u-kappa*grad(u)) + reaction*u with analytic derivatives."""
        ux,uy,uxx,uxy=derivatives(x,y)
        a,b,c,d=components(x,y)
        divx=.22 if kind=='general' else .2
        divy=0. if kind=='scalar' else (.1 if kind=='diagonal' else .13)
        return .7*ux-.2*uy-(a+d)*uxx-(b+c)*uxy-divx*ux-divy*uy+.5*exact(x,y)
    def flux(x,y):
        """Conservative physical diffusive flux."""
        ux,uy,_,_=derivatives(x,y)
        a,b,c,d=components(x,y)
        return -a*ux-b*uy,-c*ux-d*uy
    errors=[]
    for n in [2,4,8]:
        space=DGSpace(rectangle_mesh(n,n,xlim=(0.,1.),ylim=(0.,1.)),order,
                      basis_type='dub_orth',volume_quad_1d=order+5)
        beta=(space*space).field((space.constant(.7),space.constant(-.2)))
        result=solve_advection_diffusion_reaction_hdg(
            source,beta,.5,exact,space,diffusion=diffusion,assembly_backend='numba',
            solver='direct',hdg_postprocess='none',trace_basis='legendre-modal',verbose=False)
        errors.append((result.field.l2_error(exact),result.flux.l2_error(flux)))
    rates=np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:]))
    assert np.all(rates[-1] > order+.65), (kind,order,errors,rates)


@pytest.mark.parametrize('cross_space', [False, True])
def test_mixed_element_paths_and_cross_space_fields(cross_space):
    from dataclasses import replace
    from hdgfem.assembly.advection_diffusion_reaction import prepare_adr_data, assemble_numpy
    from hdgfem.backends.advection_diffusion_reaction_numba import assemble_projected_adr_trace_system_eliminated_numba
    mesh=rectangle_mesh(2,1,xlim=(-1.,1.))
    space=DGSpace(mesh,3,basis_type='dub_orth')
    coefficient_space=DGSpace(mesh,1,basis_type='dub_orth')
    def coefficient(x,y):
        """Exactly constant on left elements and variable on right elements."""
        return np.where(x<0.,2.,2.+.1*x)
    diffusion=(coefficient,0.,2.)
    if cross_space:
        # Projection may introduce roundoff variation in nominally constant
        # elements; preserve that variation rather than rounding coefficients.
        diffusion=(coefficient_space.project_callable(coefficient),
                   coefficient_space.zeros(),coefficient_space.constant(2.))
    tensor=prepare_diffusion(diffusion,space)
    if not cross_space:
        assert tensor.counts['variable-diagonal'] == 2
        assert tensor.counts['constant-isotropic'] == 2
    source=space.constant(1.)
    beta=(space*space).field((space.constant(.5),space.constant(.2)))
    prepared=prepare_adr_data(source,.3,beta,space,diffusion=diffusion)
    boundary=lambda x,y: x-y
    actual=assemble_projected_adr_trace_system_eliminated_numba(prepared,boundary,space,diffusion_data=tensor)
    forced=assemble_projected_adr_trace_system_eliminated_numba(
        prepared,boundary,space,diffusion_data=replace(tensor,kinds=np.full_like(tensor.kinds,6)))
    reference=assemble_numpy(prepared,boundary,space,diffusion=diffusion)
    for other in [forced,reference]:
        shape=(actual.trace_system.rhs.size,)*2
        lhs=coo_matrix((actual.trace_system.data,(actual.trace_system.rows,actual.trace_system.cols)),shape=shape).toarray()
        rhs=coo_matrix((other.trace_system.data,(other.trace_system.rows,other.trace_system.cols)),shape=shape).toarray()
        np.testing.assert_allclose(lhs,rhs,rtol=3e-11,atol=3e-11)
        np.testing.assert_allclose(actual.trace_system.rhs,other.trace_system.rhs,rtol=3e-11,atol=3e-11)


@pytest.mark.parametrize('assembly,reconstruction', [('numpy','numba'),('numba','numpy'),('numba','numba')])
@pytest.mark.parametrize('flux_space', ['l2_closest','RT_projection'])
def test_tensor_flux_postprocessing_and_mixed_stages(assembly,reconstruction,flux_space):
    space=DGSpace(rectangle_mesh(2,2),2,basis_type='dub_orth')
    diffusion=(lambda x,y: 2.+.1*x,.2,lambda x,y: 1.+.1*y)
    beta=(space*space).field((space.constant(.7),space.constant(-.2)))
    kwargs=dict(diffusion=diffusion,solver='direct',hdg_postprocess='flux',
                flux_postprocess_space=flux_space,verbose=False)
    reference=solve_advection_diffusion_reaction_hdg(
        space.constant(1.),beta,.3,0.,space,assembly_backend='numpy',reconstruction_backend='numpy',**kwargs)
    result=solve_advection_diffusion_reaction_hdg(
        space.constant(1.),beta,.3,0.,space,assembly_backend=assembly,reconstruction_backend=reconstruction,**kwargs)
    np.testing.assert_allclose(result.local_unknowns,reference.local_unknowns,rtol=1e-10,atol=1e-10)
    assert result.postprocessed_field is None
    for actual,expected in zip(result.postprocessed_flux.components,reference.postprocessed_flux.components):
        np.testing.assert_allclose(actual.coeffs,expected.coeffs,rtol=1e-9,atol=1e-9)


def test_tensor_primal_postprocess_rejected_before_preparation(monkeypatch):
    import hdgfem.solvers.advection_diffusion_reaction as adr
    def forbidden(*args,**kwargs):
        """Preparation must not run for a known unsupported recovery stage."""
        raise AssertionError('prepared before preflight')
    monkeypatch.setattr(adr,'prepare_adr_data',forbidden)
    space=DGSpace(rectangle_mesh(1,1),1)
    with pytest.raises(UnsupportedBackendConfigurationError,match='primal postprocessing'):
        solve_advection_diffusion_reaction_hdg(1.,(1.,0.),.1,0.,space,
                                               diffusion=(2.,.2,1.),solver='direct',verbose=False)
