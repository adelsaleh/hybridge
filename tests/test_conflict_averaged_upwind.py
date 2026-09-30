"""Face repair and small operators; run with NUMBA_DISABLE_JIT=1 to avoid compilation."""
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix

import hdgfem.runtime.optional as runtime_optional
from hdgfem import DGSpace, rectangle_mesh
from hdgfem.assembly import matrices_numpy as mats
from hdgfem.assembly.advection_residual import UpwindHDGTransportResidual
from hdgfem.backends import numba as nb
from hdgfem.kernels.advection_reaction_fused import _assemble_conflict_face_trace_weights
from hdgfem.linalg.transport_diagnostics import trace_inflow_diagnostics
from hdgfem.solvers.advection_reaction import AdvectionReactionHDGSolver
from hdgfem.solvers.stabilization import (
    conflict_averaged_normal_pair, effective_advection_normal_flux,
    gauge_inactive_advection_trace_blocks,
)

POLICY = "conflict-averaged-upwind"
CASES = json.loads((Path(__file__).parent / "data/recovered_drift_rank_failures.json").read_text())


def face_samples(space, left, right):
    mesh = space.mesh
    normal = np.zeros((mesh.num_tri, 3, len(left)))
    slots = mesh.edge_side_indices[mesh.int_edges_inds[0]]
    for side, samples in zip(slots, (left, right)):
        element, face = divmod(side, 3)
        normal[element, face] = samples if mesh.orientations[element, face] else samples[::-1]
    return normal


def constant_pair_problem(a=.1, b=.2, order=2):
    space = DGSpace(rectangle_mesh(1, 1), order, basis_type="dub_orth")
    slots = space.mesh.edge_side_indices[space.mesh.int_edges_inds[0]]
    values = np.zeros((2, space.mesh.num_tri))
    for side, speed in zip(slots, (a, b)):
        element, face = divmod(side, 3)
        values[:, element] = speed * space.mesh.normals[element, face]
    one = np.zeros(space.shape)
    one[:, 0] = 1.0  # Exact constant mode; do not project and retain roundoff modes.
    beta = space.vector_field([space.field(one * component[:, None]) for component in values])
    return space, beta


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["variant"])
@pytest.mark.parametrize("scale", [1., 1.e-12, 1.e12])
def test_saved_failure_recovers_full_weighted_rank(case, scale):
    a, b = np.asarray(case["normal_samples"]) * scale
    space = DGSpace(rectangle_mesh(1, 1), 6, basis_type="dub_orth", edge_quad_1d=13)
    trace = space.trace_space("legacy-lagrange")
    normal = face_samples(space, a, b)
    report = trace_inflow_diagnostics(normal, space.mesh.loc2glob_edge, space.mesh.orientations,
        space.mesh.int_edges_inds, np.asarray(case["trace_basis"]), np.asarray(case["weights"]),
        stabilization=POLICY)
    face = report["worst_faces"][0]
    assert face["sampling_rank"] == 7
    assert face["effective_trace_support"] >= 7
    assert face["sampling_rcond"] > 1.e-3
    tau, gamma = mats.advection_trace_weights_from_normal_flux(space, normal, POLICY, trace_space=trace)
    effective = effective_advection_normal_flux(normal, space.mesh, POLICY)
    np.testing.assert_array_equal(tau, np.abs(effective))
    np.testing.assert_array_equal(gamma, tau - effective)
    np.testing.assert_array_equal(normal, face_samples(space, a, b))


def test_node_rules_exact_and_near_cancellation_and_scaling():
    a = np.array([2., -2., -1., 2., 0., 0., 1., 1., 0.])
    b = np.array([-2., 2., -3., 4., 2., -2., 1., np.nextafter(1., 2.), 0.])
    left, right = conflict_averaged_normal_pair(a, b)
    expected = np.array([2., -2., -1., -1., -1., 0., 0., -np.finfo(float).eps / 2, 0.])
    np.testing.assert_array_equal(left, expected)
    np.testing.assert_array_equal(right, [-2., 2., -3., 1., 1., -2., 0., np.finfo(float).eps / 2, 0.])
    for scale in (2.**-40, 2.**40):
        x, y = conflict_averaged_normal_pair(a * scale, b * scale)
        np.testing.assert_array_equal(x, left * scale)
        np.testing.assert_array_equal(y, right * scale)
    # A constant state has balanced pair flux at repaired and physical nodes.
    conflict = (a >= 0) & (b >= 0) & (a + b > 0)
    flux = (abs(left) - (abs(left)-left)) + (abs(right) - (abs(right)-right))
    np.testing.assert_array_equal(flux[conflict | (a == -b)], 0.)


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
def test_numba_owned_weights_match_numpy_and_cupy_reference(monkeypatch, basis):
    import hdgfem.backends.cupy as cp_backend
    space = DGSpace(rectangle_mesh(2, 1), 3, basis_type="dub_orth")
    trace = space.trace_space(basis)
    beta = np.random.default_rng(27).normal(size=(2, *space.shape))
    raw = np.einsum("dki,kfd,fiq->kfq", beta, space.mesh.normals, trace.bas_of_bd_quads)
    expected = mats.advection_trace_weights_from_normal_flux(space, raw, POLICY, trace_space=trace)
    tau, gamma = np.empty_like(raw), np.empty_like(raw)
    slots = space.mesh.edge_side_indices
    # Reverse execution order to expose any dependency on neighbor weight writes.
    for element in reversed(range(space.mesh.num_tri)):
        _assemble_conflict_face_trace_weights(tau, gamma, element, space.mesh.normals,
            trace.bas_of_bd_quads, beta, space.mesh.loc2glob_edge, space.mesh.orientations, slots, False)
    np.testing.assert_allclose(tau, expected[0], atol=2.e-13)
    np.testing.assert_allclose(gamma, expected[1], atol=2.e-13)
    monkeypatch.setattr(cp_backend, "require_cupy", lambda: np)
    monkeypatch.setattr(runtime_optional, "require_cupy", lambda: np)
    actual = cp_backend._advection_trace_weights_cupy(POLICY, SimpleNamespace(mesh=space.mesh), raw, trace)
    np.testing.assert_allclose(actual, expected, atol=2.e-13)
    assert space.mesh.edge_side_indices is slots
    np.testing.assert_allclose((tau-gamma)[~space.mesh.interior_face_mask], raw[~space.mesh.interior_face_mask])


@pytest.mark.parametrize("a,b", [(.1, .2), (.1, -.1), (.1, .1), (0., 0.), (0., -.1)])
@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_numpy_numba_matrices_reconstruction_and_gauge(a, b, basis, boundary_mode, monkeypatch):
    space, beta = constant_pair_problem(a, b)
    kwargs = dict(source=space.constant(1), beta=beta, reaction=space.constant(3),
        boundary_condition=1., boundary_mode=boundary_mode, trace_basis=basis,
        advection_stabilization=POLICY, solver="direct", verbose=0)
    ref_solver = AdvectionReactionHDGSolver(space, assembly_backend="numpy", **kwargs)
    expected = ref_solver.solve()
    # The new policy must bypass the old separate weight-precomputation pass.
    monkeypatch.setattr(nb, "assemble_face_trace_weights_kernel",
                        lambda *args: pytest.fail("unexpected separate face-weight pass"))
    actual_solver = AdvectionReactionHDGSolver(space, assembly_backend="numba", **kwargs)
    actual = actual_solver.solve()
    n = ref_solver.solve_rhs.size
    def matrix(solver):
        return coo_matrix((solver.solve_data, (solver.solve_rows, solver.solve_cols)), shape=(n, n)).toarray()
    np.testing.assert_allclose(matrix(actual_solver), matrix(ref_solver), atol=3.e-13)
    np.testing.assert_allclose(actual_solver.solve_rhs, ref_solver.solve_rhs, atol=3.e-13)
    np.testing.assert_allclose(actual.field.coeffs, expected.field.coeffs, atol=3.e-12)
    np.testing.assert_allclose(actual.trace, expected.trace, atol=3.e-12)
    if a == b:
        dofs = (space.mesh.int_edges_inds[:, None] * space.trace_space(basis).edg_dof + np.arange(space.trace_space(basis).edg_dof)).ravel()
        matrix_dofs = np.arange(len(dofs)) if boundary_mode == "eliminate" else dofs
        np.testing.assert_array_equal(matrix(actual_solver)[np.ix_(matrix_dofs, matrix_dofs)], np.eye(len(dofs)))
        np.testing.assert_array_equal(actual.trace[dofs], 0)
        np.testing.assert_array_equal(actual_solver.solve_rhs[matrix_dofs], 0)
    if b == -a and a != 0:
        healthy_solver = AdvectionReactionHDGSolver(space, assembly_backend="numpy",
            **dict(kwargs, advection_stabilization=None))
        healthy_solver.solve()
        np.testing.assert_array_equal(matrix(healthy_solver), matrix(ref_solver))


@pytest.mark.parametrize("a,b", [(.1,.2), (.1,.1), (0.,0.), (.1,np.nextafter(.1, 1.))])
@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
def test_zero_flux_residual_matches_implicit_operator_and_conserves_mass(a, b, basis):
    space, beta = constant_pair_problem(a, b)
    source, reaction = space.constant(1), space.constant(3)
    solver = AdvectionReactionHDGSolver(space, source=source, beta=beta, reaction=reaction,
        boundary_condition=None, assembly_backend="numba", boundary_mode="zero-flux",
        trace_basis=basis, advection_stabilization=POLICY, solver="direct", verbose=0)
    solved = solver.solve()
    residual = UpwindHDGTransportResidual(space, trace_basis=basis, advection_stabilization=POLICY)
    value, trace = residual.evaluate(solved.field, beta)
    np.testing.assert_allclose(value.coeffs, 3 * solved.field.coeffs - source.coeffs, atol=3.e-12)
    interior = space.mesh.int_edges_inds
    np.testing.assert_allclose(trace, solved.trace.reshape(-1, space.trace_space(basis).edg_dof)[interior].ravel(), atol=2.e-11)
    mass_moments = value.coeffs @ space.quad_data.MKrf
    integral = np.einsum("ki,ki,k->", space.constant(1).coeffs, mass_moments, space.mesh.aff_jacs)
    assert abs(integral) < 2.e-13
    # COO and face-block COO carry exactly the same algebraic gauge.
    assembly = nb.assemble_projected_trace_system_zero_flux_numba(source, beta, reaction, space,
        advection_stabilization=POLICY, trace_space=space.trace_space(basis), return_block_coo=True)
    blocks = np.zeros((len(interior), len(interior), space.trace_space(basis).edg_dof, space.trace_space(basis).edg_dof))
    np.add.at(blocks, (assembly.block_rows, assembly.block_cols), assembly.block_data)
    dense = blocks.transpose(0, 2, 1, 3).reshape(trace.size, trace.size)
    reduction = assembly.reduction
    np.testing.assert_allclose(dense, coo_matrix((reduction.data, (reduction.rows, reduction.cols)), shape=dense.shape).toarray())


def test_partial_support_is_not_gauged_or_clipped():
    space = DGSpace(rectangle_mesh(1, 1), 2)
    a, b = np.ones(5), np.ones(5)
    b[0] = np.nextafter(1., 2.)
    normal = face_samples(space, a, b)
    effective = effective_advection_normal_flux(normal, space.mesh, POLICY)
    blocks = np.zeros((2, 3, 3))
    gauge_inactive_advection_trace_blocks(blocks, abs(effective), space.mesh)
    np.testing.assert_array_equal(blocks, 0)
    assert np.count_nonzero(effective) == 2
    report = trace_inflow_diagnostics(normal, space.mesh.loc2glob_edge, space.mesh.orientations,
        space.mesh.int_edges_inds, np.polynomial.legendre.legvander(np.linspace(-1, 1, 5), 2).T,
        np.ones(5), stabilization=POLICY)
    assert report["worst_faces"][0]["sampling_rank"] == 1


def test_operator_reuse_keeps_policy_and_option_change_invalidates():
    space, beta = constant_pair_problem()
    solver = AdvectionReactionHDGSolver(space, source=space.constant(1), beta=beta,
        reaction=space.constant(3), boundary_condition=1., assembly_backend="numpy",
        boundary_mode="eliminate", advection_stabilization=POLICY, cache_operator=True,
        solver="direct", verbose=0)
    solver.solve()
    cached = solver._operator_cache["matrix"]
    solver.set_source(space.constant(2))
    actual = solver.solve()
    assert solver._operator_cache["matrix"] is cached
    solver.clear_cache()
    fresh = solver.solve()
    np.testing.assert_allclose(actual.field.coeffs, fresh.field.coeffs, atol=1.e-13)
    solver.with_options(advection_stabilization="lax-friedrichs")
    assert not solver._operator_cache
    solver.solve()
    assert solver._operator_cache["matrix"] is not cached


def test_raw_specializations_keep_launches_and_connectivity_static():
    from hdgfem.backends import advection_raw_cuda as raw
    from hdgfem.backends import advection_tsle_bsr as split
    templates = [raw._RAW_FUSED_TEMPLATE, raw._raw_fused_csr_template(), raw._raw_fused_bsr_template(),
                 raw._RAW_FUSED_TEMPLATE + split._TSLE_KERNEL_TEMPLATE]
    for template in templates:
        source = raw._kernel_source(template, nel=6, ntr=3, ncols=10, nqf=5, advection_stabilization=POLICY)
        standard = raw._kernel_source(template, nel=6, ntr=3, ncols=10, nqf=5)
        assert source.count('__global__') == standard.count('__global__')
        assert '#define RAW_CONFLICT_AVERAGED_UPWIND 1' in source
        assert 'double mass_value = (gauge_face && row_dof == col_dof)' in source
        assert 'orientations[side] == orientations[other]' in source
        assert 'edge_side_indices' in source
    # Existing launch sites, with a cached mesh pointer passed directly.
    assembly_source = inspect.getsource(raw.assemble_projected_advection_trace_system_eliminated_raw_cuda_fused)
    calls = [node for node in ast.walk(ast.parse(assembly_source)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'kernel']
    assert len(calls) == 2  # mutually exclusive COO and CSR/BSR branches
    assert assembly_source.count('cspace.mesh.edge_side_indices,') == 2
    assert 'asarray(mesh_h.edge_side_indices' not in assembly_source


import os


@pytest.mark.skipif(os.environ.get("HDGFEM_RUN_CUDA_TRANSPORT_TESTS") != "1",
                   reason="CUDA compilation requires explicit opt-in")
@pytest.mark.parametrize("mode,fmt", [("fused", "coo"), ("fused", "csr"), ("fused", "bsr"), ("split3", "bsr")])
@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("cache_response", [False, True])
@pytest.mark.parametrize("pair", [(0.1, 0.2), (0.1, 0.1), (0., 0.), (0., -0.1), "variable"])
@pytest.mark.parametrize("boundary_mode", ["eliminate", "zero-flux"])
@pytest.mark.parametrize("order", [2, 6], ids=["p2", "p6"])
def test_authorized_cuda_matrices_and_reconstruction(mode, fmt, basis, cache_response, pair, boundary_mode, order):
    import cupy as cp
    from scipy.sparse import csr_matrix, bsr_matrix
    from hdgfem.core.device import as_cupy_space, as_cupy_vector_coefficients
    from hdgfem.core.device import as_cupy_trace_space
    from hdgfem.backends.advection_cuda import (
            assemble_reduced_system_cuda,
            reconstruct_advection_field_cuda,
        )
    space, beta = constant_pair_problem(*(pair if pair != "variable" else (.1, .2)), order=order)
    if pair == "variable":
        rng = np.random.default_rng(81)
        beta = space.vector_field([space.field(.01*rng.normal(size=space.shape)) for _ in range(2)])
    trace = space.trace_space(basis)
    source, reaction = space.constant(1), space.constant(3)
    boundary = None if boundary_mode == "zero-flux" else 1.
    if boundary_mode == "zero-flux":
        ref = nb.assemble_projected_trace_system_zero_flux_numba(source, beta, reaction, space,
            advection_stabilization=POLICY, trace_space=trace).reduction
        rhs = ref.rhs
        shape = (rhs.size,) * 2
        expected = coo_matrix((ref.data, (ref.rows, ref.cols)), shape=shape).toarray()
    else:
        # Independent NumPy contraction path for the condensed matrix.
        ref = AdvectionReactionHDGSolver(space, source=source, beta=beta, reaction=reaction,
            boundary_condition=boundary, assembly_backend="numpy", boundary_mode=boundary_mode,
            trace_basis=basis, advection_stabilization=POLICY, verbose=0)
        ref.assemble_trace_system()
        rhs = ref.solve_rhs
        shape = (rhs.size,) * 2
        expected = coo_matrix((ref.solve_data, (ref.solve_rows, ref.solve_cols)), shape=shape).toarray()
    cspace = as_cupy_space(space)
    coeffs = as_cupy_vector_coefficients(beta, cspace)
    device = assemble_reduced_system_cuda(source, reaction, boundary, coeffs, cspace,
        as_cupy_trace_space(trace), backend="raw-cuda", raw_local_assembly=mode,
        raw_lu_mode="coop", raw_matrix_format=fmt, raw_cache_local_response=cache_response,
        zero_boundary_flux=boundary_mode == "zero-flux", advection_stabilization=POLICY)
    if fmt == "coo":
        matrix = coo_matrix((cp.asnumpy(device.data), (cp.asnumpy(device.rows), cp.asnumpy(device.cols))), shape=shape)
    else:
        cls = bsr_matrix if fmt == "bsr" else csr_matrix
        matrix = cls((cp.asnumpy(device.data), cp.asnumpy(device.indices), cp.asnumpy(device.indptr)), shape=shape)
    np.testing.assert_allclose(matrix.toarray(), expected, rtol=2.e-10, atol=2.e-12)
    np.testing.assert_allclose(cp.asnumpy(device.rhs), rhs, rtol=2.e-10, atol=2.e-12)
    full_trace = np.random.default_rng(44).normal(size=space.mesh.num_edg*trace.edg_dof)
    host = nb.reconstruct_projected_field_numba(full_trace, source, beta, reaction, space,
        trace_space=trace, advection_stabilization=POLICY, zero_boundary_flux=boundary_mode == "zero-flux")
    actual, _ = reconstruct_advection_field_cuda(cp.asarray(full_trace), source, reaction, coeffs, device)
    np.testing.assert_allclose(cp.asnumpy(actual), host.coeffs, rtol=2.e-10, atol=2.e-12)


def test_reversing_global_orientation_preserves_effective_local_samples():
    space = DGSpace(rectangle_mesh(2, 1), 2)
    mesh = space.mesh
    normal = np.random.default_rng(77).normal(size=(mesh.num_tri, 3, 7))
    reversed_mesh = SimpleNamespace(edge_side_indices=mesh.edge_side_indices[:, ::-1],
        orientations=~mesh.orientations, int_edges_inds=mesh.int_edges_inds)
    expected = effective_advection_normal_flux(normal, mesh, POLICY)
    actual = effective_advection_normal_flux(normal, reversed_mesh, POLICY)
    np.testing.assert_array_equal(actual, expected)


def test_active_deficient_residual_retains_strict_support_check():
    from hdgfem.linalg.transport_diagnostics import UpwindHDGTraceRankError
    space = DGSpace(rectangle_mesh(1, 1), 2)
    residual = UpwindHDGTransportResidual(space, advection_stabilization=POLICY)
    count = residual.normal_flux.shape[-1]
    a, b = np.ones(count), np.ones(count)
    b[0] = np.nextafter(1., 2.)
    residual.normal_flux[:] = effective_advection_normal_flux(face_samples(space, a, b), space.mesh, POLICY)
    with pytest.raises(UpwindHDGTraceRankError):
        residual._check_trace_support()


def test_cupy_reference_mass_gauge_matches_numpy(monkeypatch):
    from hdgfem.backends import cupy as backend
    space, _ = constant_pair_problem()
    trace = space.trace_space("legendre-modal")
    tau = np.zeros((space.mesh.num_tri, 3, trace.weights.size))
    monkeypatch.setattr(backend, "require_cupy", lambda: np)
    monkeypatch.setattr(runtime_optional, "require_cupy", lambda: np)
    monkeypatch.setattr(backend, "_oriented_trace_basis_cupy",
        lambda *_: mats._oriented_trace_basis_on_element_sides(space, trace_space=trace))
    actual = backend._advection_interior_trace_mass_blocks_cupy(
        SimpleNamespace(mesh=space.mesh), tau, trace, inactive_tau=tau)
    expected = mats.advection_interior_trace_mass_blocks_from_weight(space, tau, trace_space=trace, inactive_tau=tau)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.sum(axis=0), np.eye(trace.edg_dof))


def test_raw_kernel_argument_counts_without_compilation():
    import re
    from hdgfem.backends import advection_raw_cuda as raw
    from hdgfem.backends import advection_tsle_bsr as split
    source = raw._RAW_FUSED_TEMPLATE + split._TSLE_KERNEL_TEMPLATE
    local = re.findall(r'assemble_projected_local_advection_raw\((.*?)\)', source, flags=re.S)
    assert len(local) == 4  # definition, fused build, reconstruction, split build
    assert len({len(arguments.split(',')) for arguments in local}) == 1
    # Check Python launch argument arity against every changed CUDA entry point.
    def c_arity(template, name):
        return len(re.search(r'void '+name+r'\((.*?)\)', template, flags=re.S).group(1).split(','))
    def py_launches(function):
        tree = ast.parse(inspect.getsource(function))
        return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'kernel']
    def arity(call):
        return sum(4 if isinstance(arg, ast.Starred) else 1 for arg in call.args[2].elts)
    calls = py_launches(raw.assemble_projected_advection_trace_system_eliminated_raw_cuda_fused)
    assert sorted(map(arity, calls)) == sorted([
        c_arity(raw._RAW_FUSED_TEMPLATE, 'assemble_advection_raw_fused'),
        c_arity(raw._raw_fused_csr_template(), 'assemble_advection_raw_fused_csr')])
    call, = py_launches(raw.reconstruct_projected_advection_field_raw_cuda_fused)
    assert arity(call) == c_arity(raw._RAW_FUSED_TEMPLATE, 'reconstruct_advection_raw_fused')
    tree = ast.parse(inspect.getsource(split.assemble_projected_advection_trace_system_eliminated_tsle_bsr))
    for variable, kernel in [('build_args', 'advection_tsle_build'), ('scatter_args', 'advection_tsle_scatter_bsr')]:
        values = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets)]
        assert len(values) == 1
        assert len(values[0].elts) == c_arity(split._TSLE_KERNEL_TEMPLATE, kernel)


def test_requested_snapshot_contains_raw_and_effective_samples(tmp_path):
    from hdgfem.linalg.transport_diagnostics import save_transport_failure_snapshot, analyze_transport_snapshot
    space, beta = constant_pair_problem()
    trace = space.trace_space("legacy-lagrange")
    matrix = coo_matrix(np.eye(trace.edg_dof))
    raw = SimpleNamespace(beta_coeffs=np.stack([c.coeffs for c in beta.components]),
        source_coeffs=None, reaction_coeffs=None, reaction_scalar=3., reaction_is_scalar=True,
        zero_boundary_flux=True, advection_stabilization=POLICY)
    assembly = SimpleNamespace(raw=raw, cspace=SimpleNamespace(host=space), trace_ref=SimpleNamespace(host=trace),
        matrix_format="coo", data=matrix.data, rows=matrix.row, cols=matrix.col, rhs=np.zeros(trace.edg_dof))
    path = tmp_path / "conflict.npz"
    report = save_transport_failure_snapshot(path, assembly)
    assert "snapshot_analysis_error" not in report
    with np.load(path) as saved:
        expected = effective_advection_normal_flux(saved["normal_flux"], space.mesh, POLICY)
        expected[~space.mesh.interior_face_mask] = 0
        np.testing.assert_array_equal(saved["effective_normal_flux"], expected)
        np.testing.assert_array_equal(saved["edge_side_indices"], space.mesh.edge_side_indices)
        assert analyze_transport_snapshot(saved)["advection_stabilization"] == POLICY
    face = report["trace_inflow_diagnostics"]["worst_faces"][0]
    assert face["sampling_rank"] == trace.edg_dof
    assert face["outward_normal_samples"] != face["effective_outward_normal_samples"]
