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


def _variable_tensor_case():
    from scripts.compare_cupy_adr_routes import variable_tensor_case
    return variable_tensor_case()


def _shuffled_mesh(n, seed=7):
    """Rectangle triangles in random order with cyclically rotated vertices (positive orientation kept)."""
    from hdgfem.core.mesh import rectangle_mesh
    mesh = rectangle_mesh(n, n)
    rng = np.random.default_rng(seed)
    triangles = mesh.triangles[rng.permutation(mesh.num_tri)]
    shift = rng.integers(0, 3, len(triangles))
    triangles = np.stack([np.roll(t, k) for t, k in zip(triangles, shift)])
    return mesh.node_coords.copy(), triangles


def _relative(a, b):
    return np.linalg.norm((np.asarray(a)-np.asarray(b)).ravel())/max(np.linalg.norm(np.asarray(b).ravel()), 1e-300)


_DEVICE_FIELDS = ('local_solver', 'boundary', 'source_rhs', 'element_blocks')


def _inverse_tolerance(case):
    # Local matrices agree to ~1e-16. Fields downstream of the local inverse carry
    # cond(A) times that; the 1e-6 anisotropy of variable_tensor makes A so badly
    # conditioned that two correct BLAS orderings differ by ~1e-10 there (the
    # host-prep CuPy and NumPy routes differ by 6e-11 on the same data).
    return 1e-9 if case in ('variable_tensor', 'host_only') else 1e-12


def _local_matrix_tolerance(case):
    # Device and host libm differ by an ulp in sin/cos; the variable_tensor
    # inverse-diffusion determinant cancels 1 - O(1e-6), amplifying that to ~3e-12.
    return 1e-10 if case in ('variable_tensor', 'host_only') else 1e-14


def _assert_system_close(left, right, tol):
    for name in ('blocks', 'rhs'):
        assert _relative(getattr(left, name), getattr(right, name)) < tol, name
    # Dirichlet traces can vanish (sin(pi x) on the boundary of [-1,1]^2): then compare absolutely.
    difference = np.linalg.norm(left.boundary_trace-right.boundary_trace)
    assert difference < 1e-14 or _relative(left.boundary_trace, right.boundary_trace) < tol
    for name in ('neighbors', 'global_to_local', 'global_faces'):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize('mesh', ['rectangle', 'shuffled'])
def test_device_face_topology_matches_host(mesh):
    """Device sort-based topology equals the host loops (neighbors and gather map)."""
    from hdgfem.assembly.face_dense import build_face_topology
    from hdgfem.backends.cupy_assembly import prepare_face_assembly_contributions
    from hdgfem.backends.cupy_adr_assembly import build_device_face_topology
    from hdgfem.core.mesh import rectangle_mesh
    s = DGSpace(rectangle_mesh(5, 4) if mesh == 'rectangle' else _shuffled_mesh(5), 1)
    topology = build_face_topology(s.mesh.loc2glob_edge)
    layout = prepare_face_assembly_contributions(s.mesh.loc2glob_edge, topology,
                                                 active_row_faces=s.mesh.interior_face_mask)
    neighbors, element_ids, rows, columns, count, _, _ = build_device_face_topology(np, s.mesh.loc2glob_edge)
    np.testing.assert_array_equal(neighbors, topology.neighbors)
    np.testing.assert_array_equal(count, topology.incidence_count)
    np.testing.assert_array_equal(element_ids, layout.element_ids)
    np.testing.assert_array_equal(rows, layout.row_local_faces)
    np.testing.assert_array_equal(columns, layout.column_local_faces)


@pytest.mark.parametrize('chunk', [None, 5])
@pytest.mark.parametrize('order', [1, 4, 7])
@pytest.mark.parametrize('case', ['trigonometric', 'anisotropic', 'variable_tensor', 'shuffled'])
def test_device_route_algorithm_matches_numpy_on_cpu(case, order, chunk):
    """The device route's array code, run with NumPy, reproduces the NumPy assembly."""
    from hdgfem.backends.cupy_adr_assembly import assemble_adr_device
    from hdgfem.core.mesh import rectangle_mesh
    kw = _variable_tensor_case() if case == 'variable_tensor' else get_case('variable_velocity' if case == 'shuffled' else case)[0]
    s = DGSpace(_shuffled_mesh(3) if case == 'shuffled' else rectangle_mesh(3, 2), order, basis_type='dub_orth')
    ref = assemble_advection_diffusion_reaction(space=s, **kw)
    out = assemble_adr_device(space=s, xp=np, element_chunk=chunk, return_local_matrix=True, **kw)
    assert _relative(out['local_matrix'], ref.local_matrix) < 1e-14
    tol = _inverse_tolerance(case)
    for name in _DEVICE_FIELDS:
        assert _relative(out[name], getattr(ref, name)) < tol, name
    _assert_system_close(out['system'], ref.system, tol)


def _require_gpu():
    from hdgfem.backends.cupy import require_cupy_device
    try:
        require_cupy_device()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.mark.parametrize('order', list(range(1, 10)))
@pytest.mark.parametrize('case', ['trigonometric', 'advection_dominated', 'variable_tensor'])
def test_cupy_device_route_matches_host_route(case, order):
    _require_gpu()
    kw = _variable_tensor_case() if case == 'variable_tensor' else get_case(case)[0]
    s = DGSpace(rectangle_mesh(4, 3), order, basis_type='dub_orth')
    host = assemble_advection_diffusion_reaction(space=s, backend='cupy', cupy_route='host', **kw)
    for chunk in (None, 7):
        dev = assemble_advection_diffusion_reaction(space=s, backend='cupy', cupy_route='device',
                                                    element_chunk=chunk, return_local_matrix=True, **kw)
        assert dev.info['route'] == 'device' and not dev.info['host_callable_fallbacks']
        assert _relative(dev.local_matrix, host.local_matrix) < _local_matrix_tolerance(case)
        tol = _inverse_tolerance(case)
        for name in _DEVICE_FIELDS:
            assert _relative(getattr(dev, name), getattr(host, name)) < tol, name
        _assert_system_close(dev.system, host.system, tol)


def test_cupy_device_route_host_callable_fallback():
    """A callable that rejects CuPy arrays is evaluated on the host, with the same result."""
    _require_gpu()
    from scripts.compare_cupy_adr_routes import host_only_case
    kw = host_only_case()
    s = DGSpace(rectangle_mesh(3, 3), 3, basis_type='dub_orth')
    host = assemble_advection_diffusion_reaction(space=s, backend='cupy', cupy_route='host', **kw)
    dev = assemble_advection_diffusion_reaction(space=s, backend='cupy', cupy_route='device', **kw)
    assert dev.info['host_callable_fallbacks'] == ['source']
    assert dev.local_matrix is None
    _assert_system_close(dev.system, host.system, _inverse_tolerance('host_only'))


def test_cupy_route_option_validated():
    kw, _ = get_case('quadratic')
    with pytest.raises(ValueError, match='cupy_route'):
        assemble_advection_diffusion_reaction(space=DGSpace(rectangle_mesh(1, 1), 1), backend='cupy',
                                              cupy_route='invalid', **kw)
