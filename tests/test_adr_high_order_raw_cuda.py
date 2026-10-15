"""Higher-order ADR qualification with independent dense CuPy local algebra."""
import numpy as np
import pytest

from hybridge import DGMesh, DGSpace, rectangle_mesh
from hybridge.hdg import condensation as hdg
from hybridge.mixed.adr_preparation import prepare_adr_data
from hybridge.mixed.coefficients import inverse_diffusion_values, sample_diffusion_tensor
from hybridge.mixed.raw_cuda import tensor as raw
from hybridge.mixed.raw_cuda.adr_operator import (
    assemble_projected_adr_trace_operator_raw_cuda as assemble,
    reconstruct_projected_adr_local_unknowns_raw_cuda as reconstruct,
)
from test_adr_tensor_numba import diffusion_cases
from test_adr_tensor_raw_cuda import cp, dense_device, tau_law, velocity


def cupy_reference(cp, prepared, diffusion, space, trace, boundary, trace_values):
    """Assemble full mixed 3*NEL systems by quadrature, solve, and scatter on GPU.

    ADR has no general CuPy assembly backend. This test oracle uses CuPy einsum
    and dense solves, independently of the raw scalar Schur/mass factorization,
    its structural specializations, and its COO/CSR/BSR scattering.
    """
    mesh, q = space.mesh, space.quad_data
    ne, n, nt = mesh.num_tri, space.el_dof, trace.edg_dof
    basis, weights = cp.asarray(q.bas_of_quads), cp.asarray(q.Krf_w)
    jac = cp.asarray(mesh.aff_jacs)
    gradients = cp.einsum('Kab,biq->Kaiq', cp.asarray(mesh.inv_aff_mats_t), cp.asarray(q.dbas_of_quads))
    derivative = cp.einsum('Kaiq,jq,q,K->Kaij', gradients, basis, weights, jac)
    inverse = cp.asarray(inverse_diffusion_values(sample_diffusion_tensor(diffusion, space)))
    mass = cp.einsum('Kqc,iq,jq,q,K->Kcij', inverse, basis, basis, weights, jac)
    scalar = cp.einsum('Kq,iq,jq,q,K->Kij', cp.asarray(prepared.reaction_values), basis, basis, weights, jac)
    scalar -= cp.einsum('Kqa,Kaiq,jq,q,K->Kij', cp.asarray(prepared.beta_values), gradients, basis, weights, jac)

    phi, fw = cp.asarray(trace.bas_of_bd_quads), cp.asarray(trace.weights)
    psi = cp.asarray(trace.oriented_basis_table[(~mesh.orientations).astype(np.int64)])
    fj, normals = cp.asarray(mesh.jacs_el_fc), cp.asarray(mesh.normals)
    tau, gamma = cp.asarray(prepared.tau_total), cp.asarray(prepared.gamma)
    normal_mass = cp.einsum('q,fiq,fjq,Kfa,Kf->Kaij', fw, phi, phi, normals, fj)
    scalar += cp.einsum('q,fiq,fjq,Kfq,Kf->Kij', fw, phi, phi, tau, fj)
    matrix = cp.zeros((ne, 3*n, 3*n), dtype=cp.float64)
    blocks = matrix.reshape(ne, 3, n, 3, n)
    blocks[:, 0, :, 0] = scalar
    for dim in range(2):
        blocks[:, 0, :, dim+1] = normal_mass[:, dim] - derivative[:, dim]
        blocks[:, dim+1, :, 0] = derivative[:, dim]
        for other in range(2):
            blocks[:, dim+1, :, other+1] = -mass[:, 2*dim+other]

    coupling = cp.einsum('q,fiq,Kfdq,Kf->Kifd', fw, phi, psi, fj)
    gamma_coupling = cp.einsum('q,fiq,Kfdq,Kfq,Kf->Kifd', fw, phi, psi, gamma, fj)
    boundary_matrix = cp.concatenate((gamma_coupling,
        coupling*normals[:, None, :, 0, None], coupling*normals[:, None, :, 1, None]), axis=1).reshape(ne, 3*n, 3*nt)
    lift = cp.concatenate((cp.einsum('q,fiq,Kfdq,Kfq,Kf->Kfdi', fw, phi, psi, tau, fj),
        (coupling*normals[:, None, :, 0, None]).transpose(0, 2, 3, 1),
        (coupling*normals[:, None, :, 1, None]).transpose(0, 2, 3, 1)), axis=-1).reshape(ne, 3*nt, 3*n)
    source = cp.zeros((ne, 3*n, 1), dtype=cp.float64)
    source[:, :n, 0] = cp.asarray(prepared.source_rhs)
    response = cp.linalg.solve(matrix, cp.concatenate((boundary_matrix, source), axis=-1))
    condensed = -lift @ response[:, :, :-1]
    gamma_mass = cp.einsum('q,Kfdq,Kfeq,Kfq,Kf->Kfde', fw, psi, psi, gamma, fj)
    for face in range(3):
        index = slice(face*nt, (face+1)*nt)
        condensed[:, index, index] += gamma_mass[:, face]
    local_rhs = (lift @ response[:, :, -1:])[:, :, 0]
    dofs = cp.asarray((mesh.loc2glob_edge[..., None]*nt+np.arange(nt)).reshape(ne, 3*nt))
    full = cp.zeros((mesh.num_edg*nt,)*2, dtype=cp.float64)
    rhs = cp.zeros(mesh.num_edg*nt, dtype=cp.float64)
    cp.add.at(full, (dofs[:, :, None], dofs[:, None, :]), condensed)
    cp.add.at(rhs, dofs, local_rhs)
    known = cp.asarray(hdg.boundary_trace_coefficients(boundary, space, trace_space=trace)).reshape(-1)
    free = cp.asarray((mesh.int_edges_inds[:, None]*nt+np.arange(nt)).reshape(-1))
    reduced_rhs = (rhs - full @ known)[free]
    local_trace = cp.asarray(trace_values)[dofs]
    unknowns = response[:, :, -1] + cp.einsum('Kij,Kj->Ki', response[:, :, :-1], local_trace)
    return full[free[:, None], free[None, :]], reduced_rhs, unknowns


@pytest.mark.parametrize('order', [7, 8, 9])
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
@pytest.mark.parametrize('diffusion,kind', diffusion_cases())
def test_high_order_cupy_reference_and_cache_policies(cp, order, basis, diffusion, kind):
    """Check all formats/caches against independent assembly and reconstruction."""
    mesh = rectangle_mesh(2, 1)
    mesh = DGMesh.from_arrays(mesh.node_coords @ np.array([[1.2, .3], [-.1, .8]]), mesh.triangles)
    space = DGSpace(mesh, order, basis_type='dub_orth', volume_quad_1d=order+3)
    trace = space.trace_space(basis)
    prepared = prepare_adr_data(lambda x, y: 1.+.1*x*y, lambda x, y: .4+.05*x,
        velocity(space), space, diffusion=diffusion, diffusion_stabilization=tau_law,
        trace_space=trace, dense_local_matrices=False)
    boundary = lambda x, y: .2+x-.3*y
    values = np.sin(np.arange(mesh.num_edg*(order+1))+.2)
    expected, rhs, local = [cp.asnumpy(value) for value in
        cupy_reference(cp, prepared, diffusion, space, trace, boundary, values)]
    for policy in raw.CACHE_LOCAL_FACTORS:
        for fmt in ('coo', 'csr', 'bsr'):
            operator = assemble(prepared, boundary, space, diffusion=diffusion, trace_space=trace,
                                cache_local_factors=policy, matrix_format=fmt)
            actual, timings = reconstruct(operator, cp.asarray(values))
            np.testing.assert_allclose(dense_device(cp, operator.assembly), expected, rtol=5e-10, atol=5e-10)
            np.testing.assert_allclose(cp.asnumpy(operator.assembly.rhs), rhs, rtol=5e-10, atol=5e-10)
            np.testing.assert_allclose(cp.asnumpy(actual), local, rtol=5e-10, atol=5e-10)
            assert operator.diffusion_structure[kind] == mesh.num_tri
            assert operator.assembly.timings['raw.conversion'] == 0.
            assert operator.assembly.timings['raw.mass_scratch_bytes'] <= raw.TENSOR_MASS_SCRATCH_LIMIT
            assert timings['raw.reconstruction.mass_scratch_bytes'] <= raw.TENSOR_MASS_SCRATCH_LIMIT
            assert bool(operator.reconstruction_data['cached_factors']) == (policy != 'none')
            assert (operator.mass_factors is not None) == (policy == 'schur-lu+mass' and kind.startswith('variable'))


@pytest.mark.parametrize('order', [7, 8, 9])
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
@pytest.mark.parametrize('reaction_value', [0., .4])
def test_high_order_zero_advection_matches_existing_cupy_paths(cp, order, basis, reaction_value):
    """Compare the common DR subset with existing dense/Schur-Cholesky CuPy paths."""
    from hybridge.core.device import as_cupy_space
    from hybridge.mixed.cupy import assemble_projected_diffusion_trace_system_eliminated_cupy
    from test_poisson_high_order_cuda import cupy_reconstruct

    space = DGSpace(rectangle_mesh(2, 1), order, basis_type='dub_orth')
    trace = space.trace_space(basis)
    device = as_cupy_space(space)
    source = space.project_callable(lambda x, y: 1.+.2*x-.1*y)
    reaction = space.constant(reaction_value)
    boundary = lambda x, y: .3+x-.5*y+.1*x*y
    values = cp.sin(cp.arange(space.mesh.num_edg*(order+1), dtype=cp.float64)+.2)
    prepared = prepare_adr_data(source, reaction, velocity(space, 0., 0.), space,
                                trace_space=trace, diffusion_stabilization=1.3,
                                dense_local_matrices=False)
    references = []
    for cholesky in (False, True):
        oracle = assemble_projected_diffusion_trace_system_eliminated_cupy(
            source, reaction, boundary, 1.3, space, trace_basis=basis, use_schur_cholesky=cholesky)
        references.append((dense_device(cp, oracle), cp.asnumpy(oracle.rhs),
                           cp.asnumpy(cupy_reconstruct(oracle, values, device))))
    for policy in raw.CACHE_LOCAL_FACTORS:
        for fmt in ('coo', 'csr', 'bsr'):
            operator = assemble(prepared, boundary, space, trace_space=trace,
                                cache_local_factors=policy, matrix_format=fmt)
            actual = (dense_device(cp, operator.assembly), cp.asnumpy(operator.assembly.rhs),
                      cp.asnumpy(reconstruct(operator, values)[0]))
            for expected in references:
                for result, reference in zip(actual, expected):
                    np.testing.assert_allclose(result, reference, rtol=5e-10, atol=5e-10)


@pytest.mark.parametrize('element_basis', ['bernstein', 'hier_C0'])
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
def test_high_order_other_element_bases(cp, element_basis, basis):
    """Use normwise parity for the less well-conditioned degree-nine bases."""
    from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient

    space = DGSpace(rectangle_mesh(2, 1), 9, basis_type=element_basis)
    trace = space.trace_space(basis)
    diffusion = raw_cuda_coefficient('variable-full')
    prepared = prepare_adr_data(space.constant(1.), .4, velocity(space), space,
        diffusion=diffusion, diffusion_stabilization=tau_law, trace_space=trace,
        dense_local_matrices=False)
    boundary = lambda x, y: .2+x-.3*y
    values = cp.sin(cp.arange(space.mesh.num_edg*10, dtype=cp.float64)+.2)
    expected = [cp.asnumpy(v) for v in cupy_reference(cp, prepared, diffusion, space, trace, boundary, values)]
    for policy in raw.CACHE_LOCAL_FACTORS:
        operator = assemble(prepared, boundary, space, trace_space=trace, diffusion=diffusion,
                            cache_local_factors=policy, matrix_format='bsr')
        actual = (dense_device(cp, operator.assembly), cp.asnumpy(operator.assembly.rhs),
                  cp.asnumpy(reconstruct(operator, values)[0]))
        for result, reference in zip(actual, expected):
            assert np.linalg.norm(result-reference)/max(1., np.linalg.norm(reference)) < 2e-10


@pytest.mark.parametrize('order', [7, 8, 9])
@pytest.mark.parametrize('kind', range(7))
@pytest.mark.parametrize('shared_limit', [48*1024, 64*1024, 99*1024, 163*1024])
def test_high_order_resource_planning(order, kind, shared_limit):
    """All classifications fit with default and overintegrated volume rules."""
    n = (order+1)*(order+2)//2
    for nq in ((order+2)**2, (order+5)**2):
        plan = raw.tensor_execution_workspace(n, np.array([0, kind]), nq, 2*order+1,
                                               order=order, shared_limit=shared_limit)
        assert plan.compact and 1 <= plan.batch <= 8 and plan.shared_bytes <= shared_limit
        if plan.external_mass:
            with pytest.raises(raw.TensorWorkspaceError):
                raw.tensor_workspace(n, np.array([kind]), nq, 2*order+1, order=order,
                                     compact=True, shared_limit=shared_limit)


def test_high_order_excessive_quadrature_still_has_a_bounded_error():
    """External factors cannot hide unbounded coefficient scratch."""
    with pytest.raises(raw.TensorWorkspaceError, match='p=9, NEL=55.*NQ=10000'):
        raw.tensor_execution_workspace(55, np.array([6]), 10000, 19, order=9, shared_limit=48*1024)


@pytest.mark.parametrize('cache', raw.CACHE_LOCAL_FACTORS)
@pytest.mark.parametrize('block_size,fmt', [(1, 'csr'), (32, 'coo'), (64, 'bsr'), (128, 'csr')])
@pytest.mark.parametrize('kind', [4, 5, 6])
def test_high_order_external_mass_chunks_and_reuse(cp, monkeypatch, cache, block_size, fmt, kind):
    """Force a 48 KiB device budget, two-element scratch chunks, and mixed kinds."""
    from hybridge.hdg.cuda import tuning
    from hybridge.mixed.coefficients_device import prepare_adr_data_cupy

    monkeypatch.setattr(tuning, 'device_signature', lambda _cp: {'sharedMemPerBlockOptin': 48*1024})
    monkeypatch.setattr(raw, 'TENSOR_MASS_SCRATCH_LIMIT', 2*raw.mass_factor_size(55, kind)*8)
    space = DGSpace(rectangle_mesh(2, 1, xlim=(-1., 1.)), 9, basis_type='dub_orth', volume_quad_1d=14)
    trace = space.trace_space('legendre-modal')
    coefficients = DGSpace(space.mesh, 1, basis_type='dub_orth')
    # Left elements take the constant path alongside diagonal, symmetric or full tensors.
    off_diagonal = (lambda x, y: np.where(x < 0., 0., .3+.02*y)) if kind >= 5 else 0.
    lower = (lambda x, y: np.where(x < 0., 0., -.1+.01*x)) if kind == 6 else off_diagonal
    diffusion = (lambda x, y: np.where(x < 0., 2., 2.+.1*x), off_diagonal, lower,
                 lambda x, y: np.where(x < 0., 2., 1.+.05*y))
    values = np.sin(np.arange(space.mesh.num_edg*10)+.2)
    old = None
    for value in (.7, -.4):
        kwargs = dict(diffusion=diffusion, diffusion_stabilization=tau_law, trace_space=trace)
        source, reaction = coefficients.constant(1.+value), coefficients.constant(.4+.1*value)
        beta = velocity(coefficients, value, -.2)
        prepared = prepare_adr_data(source, reaction, beta, space, dense_local_matrices=False, **kwargs)
        device = prepare_adr_data_cupy(source, reaction, beta, space, **kwargs)
        expected, rhs, local = [cp.asnumpy(v) for v in
            cupy_reference(cp, prepared, diffusion, space, trace, value, values)]
        operator = assemble(device, value, space, diffusion=diffusion, trace_space=trace,
                            cache_local_factors=cache, matrix_format=fmt, block_size=block_size,
                            mass_factors=old.mass_factors if old is not None else None)
        actual, timings = reconstruct(operator, values, block_size=block_size)
        assert operator.assembly.timings['raw.external_mass'] == 1.
        expected_launches = 1 if cache == 'schur-lu+mass' else 2
        assert operator.assembly.timings['raw.launches'] == expected_launches
        assert timings['raw.reconstruction.launches'] == expected_launches
        assert operator.diffusion_structure['constant-isotropic'] == 2
        classification = {4: 'variable-diagonal', 5: 'variable-symmetric', 6: 'variable-full'}[kind]
        assert operator.diffusion_structure[classification] == 2
        if old is not None and cache == 'schur-lu+mass':
            assert operator.assembly.timings['raw.mass_factors.cached'] == 1.
        np.testing.assert_allclose(dense_device(cp, operator.assembly), expected, rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(cp.asnumpy(operator.assembly.rhs), rhs, rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(cp.asnumpy(actual), local, rtol=5e-10, atol=5e-10)
        old = operator


@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
def test_high_order_all_dirichlet_triangle(cp, basis):
    """An empty reduced trace system still reconstructs every local coefficient."""
    mesh = DGMesh.from_arrays(np.array([[0., 0.], [1.2, .1], [.2, .9]]), np.array([[0, 1, 2]]))
    space = DGSpace(mesh, 9, basis_type='dub_orth')
    trace = space.trace_space(basis)
    boundary = lambda x, y: .2+x-.3*y
    prepared = prepare_adr_data(space.constant(1.), .4, velocity(space), space, trace_space=trace,
                                dense_local_matrices=False)
    values = hdg.boundary_trace_coefficients(boundary, space, trace_space=trace).reshape(-1)
    expected, rhs, local = cupy_reference(cp, prepared, 1., space, trace, boundary, values)
    assert expected.shape == (0, 0) and rhs.size == 0
    for fmt in ('coo', 'csr', 'bsr'):
        operator = assemble(prepared, boundary, space, trace_space=trace, matrix_format=fmt)
        assert operator.assembly.data.size == operator.assembly.rhs.size == 0
        actual, _ = reconstruct(operator, values)
        cp.testing.assert_allclose(actual, local, rtol=5e-10, atol=5e-10)


@pytest.mark.parametrize('block_size', [1, 32, 128])
def test_mass_lu_pivots_cross_four_warp_row_groups(cp, block_size):
    """Force swaps across rows 32, 64 and 96, independently of PDE conditioning."""
    from hybridge.runtime.precision import real_raw_module

    n, batch = 110, 8
    rng = np.random.default_rng(1717)
    matrix = np.eye(n)*4. + rng.normal(scale=.02, size=(n, n))
    matrix = matrix[np.roll(np.arange(n), 73)]
    exact = rng.normal(size=(n, batch))
    lu, rhs = cp.asarray(matrix), cp.asarray(matrix @ exact)
    pivots, status = cp.empty(n, dtype=cp.int32), cp.zeros(1, dtype=cp.int32)
    code = raw.kernel_source(55, 10, 1, 1, 1, 'csr', batch, n*n, compact=True, external_mass=True)
    code += r'''
extern "C" __global__ void check_mass_solve(double* lu, int* piv, double* rhs, int* status) {
    factor_mass(lu,piv,status);
    if(*status) return;
    invert_diagonal(lu,2*NEL);
    triangular_warp<2*NEL,false>(lu,piv,rhs,RAW_BATCH_COLS);
}
'''
    kernel = real_raw_module(code=code, options=('--std=c++11',)).get_function('check_mass_solve')
    kernel((1,), (block_size,), (lu, pivots, rhs, status))
    assert int(status[0]) == 0
    assert bool(cp.any(cp.abs(pivots-cp.arange(n)) >= 64))
    np.testing.assert_allclose(cp.asnumpy(rhs), exact, rtol=2e-12, atol=2e-12)
