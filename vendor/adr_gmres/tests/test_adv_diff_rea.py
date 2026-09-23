import numpy as np
import pytest
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.adv_diff_rea import solve_advection_diffusion_reaction, assemble_advection_diffusion_reaction
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.adv_diff_rea_cases import get_case


@pytest.mark.parametrize('basis',['dub_orth','bernstein'])
@pytest.mark.parametrize('case',['quadratic','variable_velocity'])
def test_polynomial_exactness(case,basis):
    kw,exact = get_case(case)
    s = DGSpace(rectangle_mesh(4,3),3,basis_type=basis)
    r = solve_advection_diffusion_reaction(space=s,**kw)
    assert r.field.l2_error(exact)<2e-10
    assert r.relative_residual<1e-11


def test_zero_velocity_matches_diffusion():
    kw,exact = get_case('quadratic')
    kw.update(beta=(0.,0.),source=lambda x,y: -8+.8*exact(x,y))
    s = DGSpace(rectangle_mesh(4,4),3,basis_type='dub_orth')
    r = solve_advection_diffusion_reaction(space=s,**kw)
    ref = solve_diffusion_face_dense_direct(kw['source'],kw['reaction'],exact,s,diffusion=kw['diffusion'])
    np.testing.assert_allclose(r.trace,ref.trace,rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(r.assembly.element_blocks,ref.assembly.element_blocks,rtol=1e-10,atol=1e-11)


@pytest.mark.parametrize('case',['trigonometric','advection_dominated','anisotropic'])
def test_mesh_convergence(case):
    kw,exact = get_case(case)
    errors=[]
    for n in (2,4,8):
        s=DGSpace(rectangle_mesh(n,n),3,basis_type='dub_orth')
        errors.append(solve_advection_diffusion_reaction(space=s,**kw).field.l2_error(exact))
    assert errors[2]<errors[1]<errors[0]
    assert np.log2(errors[1]/errors[2])>1.5


def test_cpu_gmres_matches_direct():
    kw,_=get_case('variable_velocity')
    s=DGSpace(rectangle_mesh(4,3),3,basis_type='dub_orth')
    ref=solve_advection_diffusion_reaction(space=s,**kw)
    r=solve_advection_diffusion_reaction(space=s,solver='gmres',rtol=1e-12,**kw)
    np.testing.assert_allclose(r.trace,ref.trace,rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('case',['quadratic','variable_velocity','advection_dominated'])
def test_gpu_assembly_matches_cpu(case):
    from hdgfem.backends.cupy import require_cupy_device
    try:
        require_cupy_device()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    kw,exact=get_case(case)
    s=DGSpace(rectangle_mesh(4,3),3,basis_type='dub_orth')
    a=assemble_advection_diffusion_reaction(space=s,**kw)
    b=assemble_advection_diffusion_reaction(space=s,backend='cupy',**kw)
    for attr in ('local_solver','element_blocks'):
        np.testing.assert_allclose(getattr(b,attr),getattr(a,attr),rtol=2e-10,atol=2e-11)
    np.testing.assert_allclose(b.system.blocks,a.system.blocks,rtol=2e-10,atol=2e-11)
    np.testing.assert_allclose(b.system.rhs,a.system.rhs,rtol=2e-10,atol=2e-11)


def test_bad_backend_rejected():
    kw,_=get_case('quadratic')
    s=DGSpace(rectangle_mesh(1,1),1)
    with pytest.raises(ValueError):
        assemble_advection_diffusion_reaction(space=s,backend='invalid',**kw)


@pytest.mark.parametrize('basis',['dub_orth','bernstein'])
@pytest.mark.parametrize('order',[1,3,6])
def test_numba_tensor_adr_matches_numpy(basis,order):
    """Variable anisotropy/velocity, nonpolynomial nonzero BC, both orientations."""
    angle=lambda x,y: .7*x-.3*y
    epsilon=1e-6
    k00=lambda x,y: np.cos(angle(x,y))**2+epsilon*np.sin(angle(x,y))**2
    k01=lambda x,y: (1-epsilon)*np.cos(angle(x,y))*np.sin(angle(x,y))
    k11=lambda x,y: np.sin(angle(x,y))**2+epsilon*np.cos(angle(x,y))**2
    kw=dict(source=lambda x,y: np.cos(2*x+y), reaction=.4,
            beta=(lambda x,y: 2+np.sin(y),lambda x,y: -1+np.cos(x)),
            diffusion=((k00,k01),(k01,k11)),
            boundary_condition=lambda x,y: np.exp(.2*x-.1*y),stabilization=.7)
    space=DGSpace(rectangle_mesh(2,2),order,basis_type=basis,
                  volume_quad_1d=2*order+4,edge_quad_1d=2*order+4)
    a=assemble_advection_diffusion_reaction(space=space,backend='numpy',**kw)
    b=assemble_advection_diffusion_reaction(space=space,backend='numba',**kw)
    assert np.any(space.mesh.orientations) and np.any(~space.mesh.orientations)
    for av,bv in [(a.local_matrix,b.local_matrix),(a.local_solver,b.local_solver),
                  (a.element_blocks,b.element_blocks),(a.system.blocks,b.system.blocks),
                  (a.system.rhs,b.system.rhs),(a.boundary,b.boundary),(a.source_rhs,b.source_rhs)]:
        assert np.linalg.norm(av-bv)/max(np.linalg.norm(av),1e-300)<1e-9
        assert np.all(np.isfinite(bv))


def test_exact_diffusive_flux():
    kw,_=get_case('quadratic')
    s=DGSpace(rectangle_mesh(4,3),3,basis_type='dub_orth')
    r=solve_advection_diffusion_reaction(space=s,**kw)
    coeff=r.flux.as_component_first()
    assert s.field(coeff[0]).l2_error(lambda x,y: -4*x-1.2*y)<2e-10
    assert s.field(coeff[1]).l2_error(lambda x,y: -.6*x-4*y)<2e-10


@pytest.mark.parametrize('diffusion',[-1.,0.,np.diag([1.,-1.])])
def test_invalid_diffusion_rejected(diffusion):
    kw,_=get_case('quadratic')
    kw['diffusion']=diffusion
    with pytest.raises(ValueError,match='positive definite'):
        assemble_advection_diffusion_reaction(space=DGSpace(rectangle_mesh(1,1),1),**kw)


@pytest.mark.parametrize('preconditioner',['none','poly','block_jacobi','block_jacobi_poly','asm','asm_poly'])
def test_gpu_gmres_preconditioners(preconditioner):
    from hdgfem.backends.cupy import require_cupy_device
    try:
        require_cupy_device()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    kw,_=get_case('variable_velocity')
    s=DGSpace(rectangle_mesh(4,3),3,basis_type='dub_orth')
    ref=solve_advection_diffusion_reaction(space=s,**kw)
    r=solve_advection_diffusion_reaction(space=s,solver='gpu',assembly_backend='cupy',rtol=1e-12,
        gpu_options=dict(preconditioner=preconditioner,polynomial_degree=6,restart=75,
                         orthogonalization='cgs2',autotune=False,operator='raw_fused',
                         asm_application='fused',block_jacobi_application='raw'),**kw)
    np.testing.assert_allclose(r.trace,ref.trace,rtol=1e-9,atol=1e-10)
