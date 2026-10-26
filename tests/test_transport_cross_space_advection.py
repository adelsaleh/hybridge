"""Transport with a advection field from another DG space (degree q != p): Numba, CuPy and raw CUDA.

The advection field is contracted with cross-space reference tables
(``hybridge.transport.advection_field``) at the solution quadrature points, without
projection. References: the NumPy assembly (which samples any-space DG fields
at the same points) and, for zero-flux, a degree-p advection field elevated exactly to
degree q, which must reproduce the same-space solve.

GPU tests compile CUDA and need ``HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.sparse import bsr_matrix, coo_matrix, csr_matrix

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.space import VectorDGField
from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver
from hybridge.transport.advection_field import advection_field_reference, advection_field_space, same_advection_basis

pytest.importorskip("numba")

BETA = (lambda x, y: np.sin(2.0 * x + y) + 0.3, lambda x, y: np.cos(x - 2.0 * y) * x)
SOURCE = lambda x, y: np.exp(x) * y  # noqa: E731
REACTION = lambda x, y: 1.0 + 0.2 * x * y  # noqa: E731
BOUNDARY = lambda x, y: x + y * y  # noqa: E731
CUDA = os.environ.get("HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS") == "1"


def _problem(order, beta_order, *, basis="dub_orth", beta_basis=None, mesh=None):
    mesh = rectangle_mesh(4, 3) if mesh is None else mesh
    space = DGSpace(mesh, order, basis_type=basis)
    beta_space = DGSpace(mesh, beta_order, basis_type=beta_basis or basis)
    beta = VectorDGField(BETA, beta_space)
    return space, beta, space.project_callable(SOURCE), space.project_callable(REACTION)


def _elevated(beta_p, space, beta_order):
    """Return ``beta_p`` (in ``space``) represented exactly in degree ``beta_order``."""
    target = DGSpace(space.mesh, beta_order, basis_type=space.reference.basis_type)
    elevation = target.degree_elevation_matrix_from(space)
    return VectorDGField(tuple(target.field(c.coeffs @ elevation) for c in beta_p.components))


def _matrix(solver):
    n = solver.solve_rhs.size
    return coo_matrix((solver.solve_data, (solver.solve_rows, solver.solve_cols)), shape=(n, n)).toarray()


def test_same_basis_drift_reuses_solution_tables():
    space = DGSpace(rectangle_mesh(2, 2), 3, basis_type="dub_orth")
    trace = space.trace_space("legendre-modal")
    twin = DGSpace(space.mesh, 3, basis_type="dub_orth", volume_degree=9)
    for beta_space in (space, twin):
        assert same_advection_basis(space, beta_space)
        tables = advection_field_reference(space, beta_space, trace)
        assert tables.same_basis and tables.face_basis is trace.bas_of_bd_quads
        assert tables.advection_tensor is advection_field_reference(space, space, trace).advection_tensor


@pytest.mark.parametrize("beta_order,beta_basis", [(5, "dub_orth"), (2, "dub_orth"), (4, "bernstein")])
def test_cross_advection_tables_match_direct_sampling(beta_order, beta_basis):
    from hybridge.hdg.coefficients import _basis_on_test_faces

    space = DGSpace(rectangle_mesh(2, 2), 3, basis_type="dub_orth")
    beta_space = DGSpace(space.mesh, beta_order, basis_type=beta_basis)
    trace = space.trace_space("legacy-lagrange")
    tables = advection_field_reference(space, beta_space, trace)
    assert not tables.same_basis and tables.el_dof == beta_space.el_dof
    np.testing.assert_allclose(tables.face_basis, _basis_on_test_faces(beta_space, space, trace_space=trace),
                               rtol=0, atol=1e-14)
    q = space.quad_data
    psi = beta_space.basis_at(q.Krf_quads)
    expected = np.einsum("q,qk,qj,qiD->Dkij", q.Krf_w, psi, q.phi, q.gphi)
    np.testing.assert_allclose(tables.advection_tensor, expected, rtol=1e-13, atol=1e-14)
    assert advection_field_reference(space, beta_space, trace).advection_tensor is tables.advection_tensor


def test_advection_field_space_validation():
    mesh = rectangle_mesh(2, 2)
    space = DGSpace(mesh, 2)
    other_mesh = DGSpace(rectangle_mesh(2, 2), 3)
    with pytest.raises(ValueError, match="same mesh"):
        advection_field_space(VectorDGField(BETA, other_mesh), space)
    mixed = VectorDGField((DGSpace(mesh, 2).project_callable(BETA[0]), DGSpace(mesh, 3).project_callable(BETA[1])))
    with pytest.raises(ValueError, match="one DG space"):
        advection_field_space(mixed, space)
    assert advection_field_space(VectorDGField(BETA, DGSpace(mesh, 4)), space).order == 4


@pytest.mark.parametrize("order,beta_order", [(1, 3), (2, 4), (3, 2), (2, 2)])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("policy", [None, "upwind", "lax-friedrichs"])
@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
@pytest.mark.parametrize("basis", ["dub_orth", "bernstein"])
@pytest.mark.parametrize("volume_operator", ["tensor", "quadrature"])
def test_numba_matches_numpy_for_cross_space_drift(order, beta_order, trace_basis, policy, boundary_mode, basis,
                                                   volume_operator, monkeypatch):
    import hybridge.transport.numba as transport_numba

    monkeypatch.setattr(transport_numba, "_use_volume_quadrature",
                        lambda *args: volume_operator == "quadrature")
    space, beta, source, reaction = _problem(order, beta_order, basis=basis,
                                             beta_basis="dub_orth" if basis == "bernstein" else basis)
    kwargs = dict(source=source, beta=beta, reaction=reaction, boundary_condition=BOUNDARY,
                  boundary_mode=boundary_mode, trace_basis=trace_basis, advection_stabilization=policy,
                  solver="direct", verbose=0)
    reference = AdvectionReactionHDGSolver(space, assembly_backend="numpy", **kwargs)
    expected = reference.solve()
    actual_solver = AdvectionReactionHDGSolver(space, assembly_backend="numba", **kwargs)
    actual = actual_solver.solve()
    scale = np.abs(_matrix(reference)).max()
    np.testing.assert_allclose(_matrix(actual_solver), _matrix(reference), rtol=0, atol=1e-13 * scale)
    np.testing.assert_allclose(actual_solver.solve_rhs, reference.solve_rhs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.field.coeffs, expected.field.coeffs, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("order,beta_order", [(1, 3), (2, 4), (3, 5)])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("policy", [None, "upwind"])
@pytest.mark.parametrize("volume_operator", ["tensor", "quadrature"])
def test_numba_zero_flux_elevated_drift_matches_same_space(order, beta_order, trace_basis, policy, volume_operator,
                                                           monkeypatch):
    import hybridge.transport.numba as transport_numba

    monkeypatch.setattr(transport_numba, "_use_volume_quadrature",
                        lambda *args: volume_operator == "quadrature")
    space, beta_p, source, reaction = _problem(order, order)
    results = []
    for beta in (beta_p, _elevated(beta_p, space, beta_order)):
        solver = AdvectionReactionHDGSolver(
            space, source=source, beta=beta, reaction=reaction, boundary_condition=None,
            boundary_mode="zero-flux", trace_basis=trace_basis, advection_stabilization=policy,
            assembly_backend="numba", solver="direct", solver_rtol=1e-9, verbose=0)
        results.append(solver.solve().field.coeffs)
    np.testing.assert_allclose(results[1], results[0], rtol=0, atol=1e-10 * np.abs(results[0]).max())


def test_volume_operator_rules():
    """The auto rules follow the measured break-even work ratios (GPU 1.25, CPU 2.0)."""
    from types import SimpleNamespace
    import hybridge.transport.numba as transport_numba
    from hybridge.transport.cuda import use_volume_quadrature

    def device_space(order, nqv):
        nel = (order + 1) * (order + 2) // 2
        return SimpleNamespace(el_dof=nel, host=SimpleNamespace(quad_data=SimpleNamespace(Krf_w=np.ones(nqv))))

    # Default rules: p=6 (33 points) quadrature; p=8 (324-point legacy rule) tensor.
    assert use_volume_quadrature(device_space(6, 33), 28, reaction_is_scalar=False)
    assert use_volume_quadrature(device_space(6, 33), 45, reaction_is_scalar=True)
    assert not use_volume_quadrature(device_space(8, 324), 45, reaction_is_scalar=False)
    # Measured near break-even: p=4, q=6 with a 42-point rule is faster as a tensor.
    assert not use_volume_quadrature(device_space(4, 42), 28, reaction_is_scalar=False)
    assert use_volume_quadrature(device_space(8, 324), 45, reaction_is_scalar=False, requested="quadrature")
    assert not use_volume_quadrature(device_space(2, 6), 6, reaction_is_scalar=False, requested="tensor")
    with pytest.raises(ValueError):
        use_volume_quadrature(device_space(2, 6), 6, reaction_is_scalar=False, requested="exact")
    # CPU: the vectorized tensor contraction wins up to a ~2x work ratio.
    p6 = DGSpace(rectangle_mesh(1, 1), 6, basis_type="dub_orth")
    assert p6.quad_data.Krf_w.size == 33
    assert not transport_numba._use_volume_quadrature(p6, 28, True)
    assert transport_numba._use_volume_quadrature(p6, 45, True)
    assert transport_numba._use_volume_quadrature(p6, 28, False)


@pytest.mark.parametrize("order,beta_order", [(2, 4), (3, 2)])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("boundary_mode", ["zero-flux", "eliminate"])
def test_residual_with_cross_space_drift_inverts_the_cross_space_solve(order, beta_order, trace_basis,
                                                                       boundary_mode):
    """Residual F(rho) and the assembled solve agree for a advection field of another degree.

    With source rho - alpha F(rho), the steady solve of u + alpha beta.grad u
    recovers rho; both sides evaluate the degree-q advection field at the same points.
    """
    from hybridge.core.field_ops import field_linear_combination
    from hybridge.solvers.advection_reaction import solve_advection_reaction_hdg
    from hybridge.transport.residual import UpwindHDGTransportResidual

    space, beta, _, _ = _problem(order, beta_order, mesh=rectangle_mesh(3, 2))
    beta_space = beta.components[0].space
    density = space.field(np.random.default_rng(5).normal(size=space.shape))
    boundary = None if boundary_mode == "zero-flux" else BOUNDARY
    residual = UpwindHDGTransportResidual(space, trace_basis=trace_basis, boundary_mode=boundary_mode)
    rhs, _ = residual.evaluate(density, beta, boundary)
    alpha = 0.03
    source = field_linear_combination(space, [(1.0, density), (-alpha, rhs)])
    scaled = VectorDGField(tuple(field_linear_combination(beta_space, [(alpha, c)]) for c in beta.components))
    result = solve_advection_reaction_hdg(
        source, scaled, space.constant(1.0), boundary, space, boundary_mode=boundary_mode,
        trace_basis=trace_basis, assembly_backend="numba", solver="direct", verbose=False, return_=("result",))
    np.testing.assert_allclose(result.field.coeffs, density.coeffs, rtol=1e-10, atol=1e-10)


def test_residual_elevated_drift_matches_same_space_drift():
    from hybridge.transport.residual import UpwindHDGTransportResidual

    space, beta_p, _, _ = _problem(3, 3)
    density = space.field(np.random.default_rng(6).normal(size=space.shape))
    residual = UpwindHDGTransportResidual(space, boundary_mode="zero-flux")
    expected, expected_trace = residual.evaluate(density, beta_p)
    expected, expected_trace = expected.coeffs.copy(), expected_trace.copy()
    actual, actual_trace = residual.evaluate(density, _elevated(beta_p, space, 6))
    scale = np.abs(expected).max()
    np.testing.assert_allclose(actual.coeffs, expected, rtol=0, atol=1e-11 * scale)
    np.testing.assert_allclose(actual_trace, expected_trace, rtol=0, atol=1e-11 * np.abs(expected_trace).max())


# --------------------------------------------------------------------------
# GPU: CuPy and raw CUDA against the Numba/NumPy references.
# --------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not (CUDA and _cuda_available()),
                               reason="CUDA compilation requires HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1 and a GPU")

RAW_PATHS = [("fused", "coo"), ("fused", "csr"), ("fused", "bsr"), ("split3", "bsr"), ("precomputed", "coo")]


def _device_matrix(device, fmt, shape):
    import cupy as cp

    if fmt == "coo":
        return coo_matrix((cp.asnumpy(device.data), (cp.asnumpy(device.rows), cp.asnumpy(device.cols))),
                          shape=shape).toarray()
    cls = bsr_matrix if fmt == "bsr" else csr_matrix
    return cls((cp.asnumpy(device.data), cp.asnumpy(device.indices), cp.asnumpy(device.indptr)),
               shape=shape).toarray()


@cuda_only
@pytest.mark.parametrize("mode,fmt", RAW_PATHS)
@pytest.mark.parametrize("order,beta_order", [(2, 4), (3, 2), (6, 5), (4, 6)])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("boundary_mode", ["eliminate", "zero-flux"])
@pytest.mark.parametrize("policy", [None, "upwind"])
@pytest.mark.parametrize("volume_operator", ["tensor", "quadrature"])
@pytest.mark.parametrize("constant_reaction", [False, True])
def test_raw_cuda_cross_space_matrices_and_reconstruction(mode, fmt, order, beta_order, trace_basis,
                                                          boundary_mode, policy, volume_operator,
                                                          constant_reaction):
    """Raw CUDA vs Numba for a advection field of another degree, every raw path.

    (6, 5) has fewer advection field than solution modes: the fused BSR kernel then
    stages its diagonal Schur rows in a coefficient block that must still
    hold 3*NTR*NTR doubles. Both volume operators (tensor contraction and
    quadrature) and the scalar path of a constant reaction field are covered.
    """
    import cupy as cp
    import hybridge.transport.numba as transport_numba
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space, as_cupy_vector_coefficients
    from hybridge.hdg.stabilization import resolve_transport_stabilization
    from hybridge.transport.cuda import assemble_reduced_system_cuda, reconstruct_advection_field_cuda

    if mode == "precomputed" and (boundary_mode == "zero-flux" or volume_operator == "quadrature"):
        pytest.skip("precomputed raw assembly: eliminate, CuPy-built local matrices only")
    space, beta, source, reaction = _problem(order, beta_order, mesh=rectangle_mesh(3, 2))
    if constant_reaction:
        # Coercive (r - div(beta)/2 > 0, |div(beta)| <= 4 here): a smaller
        # constant makes some zero-flux local matrices nearly singular (cond
        # ~1e7) and the backends then differ by cond * eps, not by a bug.
        reaction = space.constant(6.0)
    trace = space.trace_space(trace_basis)
    zero_flux = boundary_mode == "zero-flux"
    boundary = None if zero_flux else BOUNDARY
    stabilization = resolve_transport_stabilization(policy, beta)
    reference = transport_numba.assemble_projected_trace_system_eliminated_numba(
        source, beta, reaction, boundary, space, advection_stabilization=stabilization,
        zero_boundary_flux=zero_flux, trace_space=trace).reduction
    shape = (reference.rhs.size,) * 2
    expected = coo_matrix((reference.data, (reference.rows, reference.cols)), shape=shape).toarray()

    cspace = as_cupy_space(space)
    coeffs = as_cupy_vector_coefficients(beta, cspace)
    assert coeffs.shape == (2, space.mesh.num_tri, beta.components[0].space.el_dof)
    device = assemble_reduced_system_cuda(
        source, reaction, boundary, coeffs, cspace, as_cupy_trace_space(trace), backend="raw-cuda",
        raw_local_assembly=mode, raw_matrix_format=fmt, raw_cache_local_response=False,
        zero_boundary_flux=zero_flux, advection_stabilization=stabilization,
        beta_space=beta.components[0].space, raw_volume_operator=volume_operator)
    if mode != "precomputed":
        assert device.raw.reaction_is_scalar == constant_reaction
        assert (device.raw.volume_table is not None) == (volume_operator == "quadrature")
    scale = np.abs(expected).max()
    # Same bounds as the same-space raw parity tests (test_conflict_averaged_upwind).
    np.testing.assert_allclose(_device_matrix(device, fmt, shape), expected, rtol=2e-10, atol=2e-12 * scale)
    np.testing.assert_allclose(cp.asnumpy(device.rhs), reference.rhs, rtol=2e-10,
                               atol=2e-12 * max(1.0, np.abs(reference.rhs).max()))

    full_trace = np.random.default_rng(7).normal(size=space.mesh.num_edg * trace.edg_dof)
    host = transport_numba.reconstruct_projected_field_numba(
        full_trace, source, beta, reaction, space, trace_space=trace,
        advection_stabilization=stabilization, zero_boundary_flux=zero_flux)
    actual, _ = reconstruct_advection_field_cuda(cp.asarray(full_trace), source, reaction, coeffs, device)
    np.testing.assert_allclose(cp.asnumpy(actual), host.coeffs, rtol=2e-10,
                               atol=2e-12 * np.abs(host.coeffs).max())


@cuda_only
@pytest.mark.parametrize("backend", ["raw-cuda", "cupy"])
@pytest.mark.parametrize("order,beta_order", [(2, 4), (3, 2)])
def test_device_solve_matches_numba_for_cross_space_drift(backend, order, beta_order):
    space, beta, source, reaction = _problem(order, beta_order)
    kwargs = dict(source=source, beta=beta, reaction=reaction, boundary_condition=BOUNDARY,
                  boundary_mode="eliminate", trace_basis="legendre-modal", solver="direct", verbose=0)
    expected = AdvectionReactionHDGSolver(space, assembly_backend="numba", **kwargs).solve()
    extra = {"raw_local_assembly": "fused"} if backend == "raw-cuda" else {}
    actual = AdvectionReactionHDGSolver(space, assembly_backend=backend, **extra, **kwargs).solve()
    np.testing.assert_allclose(np.asarray(actual.field.coeffs), expected.field.coeffs, rtol=0,
                               atol=1e-10 * np.abs(expected.field.coeffs).max())


@cuda_only
def test_raw_cuda_rejects_cross_space_coefficients_without_their_space():
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space, as_cupy_vector_coefficients
    from hybridge.transport.cuda import assemble_reduced_system_cuda

    space, beta, source, reaction = _problem(2, 4)
    cspace = as_cupy_space(space)
    with pytest.raises(ValueError, match="beta_space"):
        assemble_reduced_system_cuda(
            source, reaction, BOUNDARY, as_cupy_vector_coefficients(beta, cspace), cspace,
            as_cupy_trace_space(space.trace_space("legendre-modal")), backend="raw-cuda",
            raw_local_assembly="fused")


@cuda_only
@pytest.mark.parametrize("order,beta_order", [(2, 4), (3, 2)])
def test_device_residual_matches_host_for_cross_space_drift(order, beta_order):
    import cupy as cp
    from hybridge.transport.residual import UpwindHDGTransportResidual

    space, beta, _, _ = _problem(order, beta_order)
    density = space.field(np.random.default_rng(8).normal(size=space.shape))
    host_rhs, host_trace = UpwindHDGTransportResidual(space, boundary_mode="zero-flux").evaluate(density, beta)
    device_rhs, device_trace = UpwindHDGTransportResidual(
        space, boundary_mode="zero-flux", backend="device").evaluate(density, beta)
    np.testing.assert_allclose(np.asarray(device_rhs.coeffs), host_rhs.coeffs, rtol=1e-11,
                               atol=1e-12 * np.abs(host_rhs.coeffs).max())
    np.testing.assert_allclose(cp.asnumpy(device_trace), host_trace, rtol=1e-11,
                               atol=1e-12 * np.abs(host_trace).max())


@cuda_only
def test_split3_memory_sizes_count_an_existing_workspace():
    import cupy as cp
    from hybridge.transport.tsle_bsr import RawAdvectionTsleWorkspace, split3_memory_sizes, split3_workspace_bytes

    space = DGSpace(rectangle_mesh(4, 4), 6, basis_type="dub_orth")
    trace = space.trace_space("legendre-modal")
    needed, available = split3_memory_sizes(space, trace)
    assert needed == split3_workspace_bytes(space.mesh.num_tri, space.el_dof, 7, trace.weights.size)
    assert available > 0
    workspace = RawAdvectionTsleWorkspace()
    workspace.ensure(cp, num_elements=space.mesh.num_tri, nel=space.el_dof, ncols=22, nqf=trace.weights.size)
    assert workspace.nbytes == needed
    _, without_workspace = split3_memory_sizes(space, trace)
    _, with_workspace = split3_memory_sizes(space, trace, workspace)
    # A workspace already held for this signature counts as available.
    assert with_workspace == without_workspace + workspace.nbytes
    other = RawAdvectionTsleWorkspace()
    other.ensure(cp, num_elements=space.mesh.num_tri, nel=space.el_dof, ncols=22, nqf=trace.weights.size + 1)
    assert split3_memory_sizes(space, trace, other)[1] == split3_memory_sizes(space, trace)[1]


@cuda_only
@pytest.mark.parametrize("order,expected", [(5, "fused"), (6, "split3")])
def test_auto_local_assembly_picks_split3_from_p6_when_memory_allows(order, expected):
    space, beta, source, reaction = _problem(order, order + 1, mesh=rectangle_mesh(3, 2))
    solver = AdvectionReactionHDGSolver(
        space, source=source, beta=beta, reaction=reaction, boundary_condition=None,
        boundary_mode="zero-flux", trace_basis="legendre-modal", assembly_backend="raw-cuda",
        raw_local_assembly="auto", solver="amgx", verbose=0)
    assembly = solver.assemble_tangent_boundary_raw_cuda_bsr()
    used = "split3" if any(key.startswith("raw.raw.tsle.") or key.startswith("raw.tsle.") for key in assembly.timings) else "fused"
    assert used == expected
