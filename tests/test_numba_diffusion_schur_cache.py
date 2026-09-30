"""Compiled small-matrix checks for persistent host diffusion Schur factors."""
import numpy as np
import pytest
from scipy.sparse import coo_matrix

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.backends.numba import (
    build_diffusion_schur_cache_numba,
    assemble_projected_diffusion_trace_system_eliminated_numba,
    assemble_projected_diffusion_trace_rhs_eliminated_numba,
    reconstruct_projected_diffusion_local_unknowns_numba,
)
from hdgfem.hdg.numba_common import cholesky_factor_inplace, cholesky_solve_inplace
from hdgfem.linalg import expand_known_dofs


def boundary(x, y):
    """Nonzero boundary data exercises eliminated trace contributions."""
    return 0.3 + x - 0.5 * y


@pytest.mark.parametrize('kind', ['schur-lu', 'schur-cholesky'])
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
@pytest.mark.parametrize('order', [0, 1, 3, 6])
def test_factor_cache_assembly_rhs_reconstruction(kind, basis, order):
    space = DGSpace(rectangle_mesh(2, 2), order)
    trace_space = space.trace_space(basis)
    source = space.project_callable(lambda x, y: 1 + x * y)
    reaction = space.zeros()
    cache = build_diffusion_schur_cache_numba(reaction, 1., space, factor_kind=kind)
    kwargs = dict(trace_space=trace_space)
    fresh = assemble_projected_diffusion_trace_system_eliminated_numba(
        source, reaction, boundary, 1., space, **kwargs)
    cached = assemble_projected_diffusion_trace_system_eliminated_numba(
        source, reaction, boundary, 1., space, cached_factors=cache, **kwargs)
    n = fresh.trace_system.rhs.size
    def matrix(result):
        """Sum element COO contributions before comparing or solving."""
        system = result.trace_system
        return coo_matrix((system.data, (system.rows, system.cols)), shape=(n, n)).toarray()
    a = matrix(fresh)
    np.testing.assert_allclose(matrix(cached), a, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(cached.trace_system.rhs, fresh.trace_system.rhs, rtol=2e-10, atol=2e-11)
    changed_source = space.project_callable(lambda x, y: 2 - x + y)
    rhs, _, reduction, _ = assemble_projected_diffusion_trace_rhs_eliminated_numba(
        changed_source, reaction, boundary, 1., space, cached_factors=cache, **kwargs)
    changed = assemble_projected_diffusion_trace_system_eliminated_numba(
        changed_source, reaction, boundary, 1., space, **kwargs)
    np.testing.assert_allclose(rhs, changed.trace_system.rhs, rtol=2e-10, atol=2e-11)
    solution = np.linalg.solve(a, rhs)
    assert np.linalg.norm(a @ solution - rhs) / np.linalg.norm(rhs) < 1e-10
    trace = expand_known_dofs(solution, reduction)
    expected = reconstruct_projected_diffusion_local_unknowns_numba(
        trace, changed_source, reaction, 1., space, **kwargs)
    actual = reconstruct_projected_diffusion_local_unknowns_numba(
        trace, changed_source, reaction, 1., space, cached_factors=cache, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)
    with pytest.raises(ValueError, match='stale'):
        assemble_projected_diffusion_trace_rhs_eliminated_numba(
            source, reaction, boundary, 2., space, cached_factors=cache, **kwargs)


def test_cholesky_factor_action_and_rejection():
    rng = np.random.default_rng(82)
    m = rng.normal(size=(8, 8))
    a = m @ m.T + np.eye(8)
    factor = a.copy()
    assert cholesky_factor_inplace(factor) == 0
    rhs = rng.normal(size=(8, 4))
    solved = rhs.copy()
    cholesky_solve_inplace(factor, solved)
    np.testing.assert_allclose(a @ solved, rhs, rtol=2e-13, atol=2e-13)
    assert cholesky_factor_inplace(np.diag([1., -1.])) == 2
    assert cholesky_factor_inplace(np.array([[1., 3.], [0., 1.]])) == -1
    assert cholesky_factor_inplace(np.array([[np.nan]])) == -1


@pytest.mark.parametrize('kind', ['schur-lu', 'schur-cholesky'])
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
@pytest.mark.parametrize('order', [0, 2, 6])
@pytest.mark.parametrize('assemble_first', [False, True])
def test_stateful_solver_numpy_parity_and_invalidation(kind, basis, order, assemble_first):
    from hdgfem import DiffusionReactionHDGSolver, DGMesh
    from hdgfem.solvers.diffusion_reaction import hdg_residual
    mesh = rectangle_mesh(2, 2)
    mesh = DGMesh.from_arrays(mesh.node_coords @ np.array([[1.8, .3], [-.2, .8]]), mesh.triangles)
    space = DGSpace(mesh, order)
    source = space.project_callable(lambda x, y: 1 + x * y)
    reaction = space.project_callable(lambda x, y: 0.5 + x * x)
    options = dict(stabilization=1., trace_basis=basis, solver='direct',
                   boundary_mode='eliminate', verbose=0, hdg_postprocess='none')
    host = DiffusionReactionHDGSolver(space, assembly_backend='numba',
                                     cache_local_factors=kind, **options)
    reference = DiffusionReactionHDGSolver(space, assembly_backend='numpy', **options)
    host.set_problem(source, reaction, boundary)
    reference.set_problem(source, reaction, boundary)
    ha, na = host.assemble_global_matrix(), reference.assemble_global_matrix()
    shape = (ha.rhs.size, ha.rhs.size)
    hm = coo_matrix((ha.data, (ha.rows, ha.cols)), shape=shape).toarray()
    nm = coo_matrix((na.data, (na.rows, na.cols)), shape=shape).toarray()
    np.testing.assert_allclose(hm, nm, rtol=3e-9, atol=3e-10)
    np.testing.assert_allclose(ha.rhs, na.rhs, rtol=3e-9, atol=3e-10)
    if not assemble_first:
        host.clear_cache()
        reference.clear_cache()
    actual = host.solve()
    expected = reference.solve()
    reduced_trace = actual.trace[ha.reduction.free_mask]
    assert np.linalg.norm(nm @ reduced_trace - na.rhs) / np.linalg.norm(na.rhs) < 1e-9
    np.testing.assert_allclose(actual.trace, expected.trace, rtol=3e-9, atol=3e-10)
    # At p=6 this nodal/sheared reference's explicit inverse loses ~1e-8
    # in flux coefficients; the independent mixed-equation residual is tighter.
    np.testing.assert_allclose(actual.local_unknowns, expected.local_unknowns,
                               rtol=3e-9, atol=2e-8 if order == 6 else 3e-10)
    flux_coeffs = actual.local_unknowns[:, space.el_dof:].reshape(space.mesh.num_tri, 2, space.el_dof).transpose(1, 0, 2)
    residual = hdg_residual(actual.field, flux_coeffs, actual.trace,
                           source_values=host.source.values_at_ref(space.quad_data.Krf_quads),
                           stabilization=1., reaction=host.reaction,
                           trace_space=space.trace_space(basis))
    assert np.linalg.norm(residual) < 1e-9
    factors = host._numba_local_factors
    host.set_source(space.project_callable(lambda x, y: 2 - x))
    reference.set_source(host.source)
    host.set_boundary_condition(lambda x, y: x + y)
    reference.set_boundary_condition(host.boundary_condition)
    actual, expected = host.solve(), reference.solve()
    assert host._numba_local_factors is factors
    np.testing.assert_allclose(actual.trace, expected.trace, rtol=3e-9, atol=3e-10)
    # At p=6 this nodal/sheared reference's explicit inverse loses ~1e-8
    # in flux coefficients; the independent mixed-equation residual is tighter.
    np.testing.assert_allclose(actual.local_unknowns, expected.local_unknowns,
                               rtol=3e-9, atol=2e-8 if order == 6 else 3e-10)
    flux_coeffs = actual.local_unknowns[:, space.el_dof:].reshape(space.mesh.num_tri, 2, space.el_dof).transpose(1, 0, 2)
    residual = hdg_residual(actual.field, flux_coeffs, actual.trace,
                           source_values=host.source.values_at_ref(space.quad_data.Krf_quads),
                           stabilization=1., reaction=host.reaction,
                           trace_space=space.trace_space(basis))
    assert np.linalg.norm(residual) < 1e-9
    host.set_reaction(space.zeros())
    assert host._numba_local_factors is None
    host.solve()
    assert host._numba_local_factors is not factors
    host.with_options(stabilization=2.)
    assert host._numba_local_factors is None
    host.solve()
    host.set_space(space)
    assert host._numba_local_factors is None


@pytest.mark.parametrize('kind', ['schur-lu', 'schur-cholesky'])
@pytest.mark.parametrize('order', [1, 3, 6])
def test_factor_action_against_numpy_schur(kind, order):
    from hdgfem.solvers.diffusion_reaction import _local_solver_pre_mats
    from hdgfem.hdg.numba_common import lu_solve_inplace
    space = DGSpace(rectangle_mesh(2, 1, xlim=(-2., 1.), ylim=(-0.3, 0.7)), order)
    reaction = space.project_callable(lambda x, y: 0.1 + x * x)
    cache = build_diffusion_schur_cache_numba(reaction, 0.7, space, factor_kind=kind)
    dx, dy, mt, nx, ny, ji = _local_solver_pre_mats(reaction, 0.7, space)
    mi = space.quad_data.MKrf_inv
    schur = mt + ji * (nx - dx) @ mi @ dx + ji * (ny - dy) @ mi @ dy
    rng = np.random.default_rng(47)
    for element in range(space.mesh.num_tri):
        rhs = rng.normal(size=(space.quad_data.el_dof, 3))
        solution = rhs.copy()
        if kind == 'schur-lu':
            lu_solve_inplace(cache.factors[element], cache.pivots[element], solution)
        else:
            cholesky_solve_inplace(cache.factors[element], solution)
        np.testing.assert_allclose(schur[element] @ solution, rhs, rtol=2e-8, atol=2e-8)


def test_invalid_and_stale_operator_cache():
    space = DGSpace(rectangle_mesh(1, 1), 2)
    reaction = space.project_callable(lambda x, y: 1 + x * x)
    cache = build_diffusion_schur_cache_numba(reaction, 1., space)
    reaction.coeffs[0, 0] += 0.5
    with pytest.raises(ValueError, match='stale'):
        assemble_projected_diffusion_trace_rhs_eliminated_numba(
            space.zeros(), reaction, boundary, 1., space, cached_factors=cache)
    with pytest.raises(ValueError, match='finite'):
        build_diffusion_schur_cache_numba(reaction, np.nan, space)
    with pytest.raises(ValueError, match='positive'):
        build_diffusion_schur_cache_numba(reaction, -1., space, factor_kind='schur-cholesky')
