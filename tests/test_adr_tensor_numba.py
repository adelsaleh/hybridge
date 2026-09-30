"""Small-matrix ADR tensor parity and structural specialization checks."""
import numpy as np
import pytest
from scipy.sparse import coo_matrix

from hdgfem import DGSpace, DGMesh, rectangle_mesh, solve_advection_diffusion_reaction_hdg
from hdgfem.mixed.coefficients import prepare_diffusion, normal_diffusivity_on_faces
from hdgfem.mixed.stabilization import GlobalLengthDiffusion
from hdgfem.runtime.errors import UnsupportedBackendConfigurationError
from scripts.advection_diffusion_reaction.cases.tensor_cases import (
    diffusion_cases as coefficient_cases, manufactured_tensor, raw_cuda_coefficient,
)


def diffusion_cases():
    """Exercise every structural path, including nonsymmetric elliptic tensors."""
    coefficients = coefficient_cases()
    return [
        pytest.param(coefficients['constant_isotropic'], 'constant-isotropic', id='scalar'),
        pytest.param(np.diag([2., 1.]), 'constant-diagonal', id='diagonal'),
        pytest.param(coefficients['constant_full'], 'constant-full', id='symmetric-constant'),
        pytest.param(raw_cuda_coefficient('constant-full'), 'constant-full', id='general-constant'),
        pytest.param(coefficients['variable_isotropic'], 'variable-isotropic', id='variable-scalar'),
        pytest.param(coefficients['variable_diagonal'], 'variable-diagonal', id='variable-diagonal'),
        pytest.param(coefficients['variable_symmetric'], 'variable-symmetric', id='variable-symmetric'),
        pytest.param(coefficients['variable_full'], 'variable-full', id='variable-general'),
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
    problem, exact, flux = manufactured_tensor(kind)
    source, diffusion = problem['source'], problem['diffusion']
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
    from hdgfem.mixed.adr_preparation import prepare_adr_data
    from hdgfem.mixed.adr_numpy import assemble_numpy
    from hdgfem.mixed.adr_numba import (
            assemble_projected_adr_trace_system_eliminated_numba,
        )
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


@pytest.mark.parametrize('mode', ['primal', 'both'])
def test_tensor_primal_postprocess_enabled(mode):
    """Public recovery accepts elliptic tensors and preserves the element mean."""
    space=DGSpace(rectangle_mesh(1,1),1)
    result=solve_advection_diffusion_reaction_hdg(space.constant(1.),(space.constant(1.),space.zeros()),.1,0.,space,
        diffusion=(2.,.2,1.),solver='direct',hdg_postprocess=mode,verbose=False)
    post=result.postprocessed_field
    q=post.space.quad_data
    np.testing.assert_allclose(post.coeffs @ (q.Krf_w @ q.phi),
        result.field.coeffs @ (q.Krf_w @ space.basis_at(q.Krf_quads)), atol=1e-12)


@pytest.mark.parametrize('kind', ['scalar', 'diagonal', 'symmetric', 'general'])
@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('variant', ['l2_closest', 'RT_projection'])
def test_tensor_primal_manufactured_convergence(kind, order, variant):
    """Continuous forcing checks convergence of both recovered fields on small meshes."""
    problem, exact, flux = manufactured_tensor(kind)
    errors = []
    for n in [2, 4, 8]:
        space = DGSpace(rectangle_mesh(n, n, xlim=(0., 1.), ylim=(0., 1.)), order,
                        basis_type='dub_orth', volume_quad_1d=order+5)
        beta = (space*space).field((space.constant(.7), space.constant(-.2)))
        result = solve_advection_diffusion_reaction_hdg(
            problem['source'], beta, .5, exact, space, diffusion=problem['diffusion'],
            assembly_backend='numba', solver='direct', hdg_postprocess='both',
            flux_postprocess_space=variant, trace_basis='legendre-modal', verbose=False)
        def total_flux(x, y):
            """Exact conservative total flux, including advection."""
            qx, qy = flux(x, y)
            return qx + .7*exact(x, y), qy - .2*exact(x, y)
        errors.append((result.postprocessed_field.l2_error(exact),
                       result.postprocessed_flux.l2_error(total_flux)))
    rates = np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:]))
    assert rates[-1, 0] > order+1.5, (kind, order, variant, errors, rates)
    assert rates[-1, 1] > order+.65, (kind, order, variant, errors, rates)
