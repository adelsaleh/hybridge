"""Advection-reaction HDG with coefficients given at the quadrature points (Numba, NumPy, raw CUDA).

``QuadratureVectorField`` / ``QuadratureScalar`` (``hybridge.core.quadrature_coefficients``) give the
advection field, reaction and source as values at the volume and face quadrature points. Sampling a DG advection field
and reaction at those points reproduces the DG-coefficient assembly with the per-point volume
operator, so the Numba DG path is the reference for the Numba pointwise path, and the Numba
pointwise path is the reference for the raw-CUDA one (project rule: Numba host references).

GPU tests compile CUDA and need ``HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1``.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.quadrature_coefficients import QuadratureVectorField, QuadratureScalar
from hybridge.core.space import VectorDGField

pytest.importorskip("numba")

import hybridge.transport.numba as transport_numba  # noqa: E402
from hybridge.transport.numba import (  # noqa: E402
    assemble_projected_trace_system_eliminated_numba,
    assemble_projected_trace_system_numba,
    reconstruct_projected_field_numba,
)

# A advection field that is tangent to no face of the test meshes, so per-side sampled and DG
# normal fluxes take the same conflict and gauge decisions.
BETA = (lambda x, y: np.sin(2.0 * x + y) + 1.7, lambda x, y: np.cos(x - 2.0 * y) * x + 0.4)
SOURCE = lambda x, y: np.exp(x) * y  # noqa: E731
REACTION = lambda x, y: 1.0 + 0.2 * x * y  # noqa: E731
BOUNDARY = lambda x, y: x + y * y  # noqa: E731
CUDA = os.environ.get("HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS") == "1"


def _cuda_available() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - no usable CUDA runtime
        return False


cuda_only = pytest.mark.skipif(not (CUDA and _cuda_available()),
                               reason="set HYBRIDGE_RUN_CUDA_TRANSPORT_TESTS=1 on a CUDA machine")


def _problem(order, *, beta_order=None, basis="dub_orth", trace_basis="legacy-lagrange", mesh=None,
             constant_reaction=False):
    """DG coefficients and their quadrature-point samples on one small mesh."""
    mesh = rectangle_mesh(4, 3) if mesh is None else mesh
    space = DGSpace(mesh, order, basis_type=basis)
    trace = space.trace_space(trace_basis)
    beta_space = DGSpace(mesh, order if beta_order is None else beta_order, basis_type=basis)
    beta = VectorDGField(BETA, beta_space)
    source = space.project_callable(SOURCE)
    # A coercive constant (r - div(beta)/2 > 0 here) keeps zero-flux local matrices well conditioned.
    reaction = space.constant(6.0) if constant_reaction else space.project_callable(REACTION)
    sampled_beta = QuadratureVectorField.from_vector_field(beta, space, trace_space=trace)
    sampled_reaction = reaction if constant_reaction else QuadratureScalar.from_field(reaction, space)
    sampled_source = QuadratureScalar.from_field(source, space)
    return space, trace, (source, beta, reaction), (sampled_source, sampled_beta, sampled_reaction)


def _dense(system):
    """Dense trace matrix of a Numba trace system."""
    n = system.rhs.size
    return coo_matrix((system.data, (system.rows, system.cols)), shape=(n, n)).toarray()


def _close(actual, expected, rtol, atol_scale):
    """Assert closeness relative to the largest reference entry."""
    scale = max(float(np.max(np.abs(expected))), 1.0e-300)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol_scale * scale)


@pytest.fixture
def forced_quadrature(monkeypatch):
    """Force the per-point volume operator for the DG reference."""
    monkeypatch.setattr(transport_numba, "_use_volume_quadrature", lambda *args: True)


def _assemble(mode, coefficients, space, trace, policy):
    """Assemble one Numba trace system in boundary ``mode``."""
    source, beta, reaction = coefficients
    if mode == "penalty":
        return assemble_projected_trace_system_numba(
            source, beta, reaction, BOUNDARY, space, advection_stabilization=policy, trace_space=trace).trace_system
    if mode == "eliminate":
        return assemble_projected_trace_system_eliminated_numba(
            source, beta, reaction, BOUNDARY, space, advection_stabilization=policy, trace_space=trace).trace_system
    return assemble_projected_trace_system_eliminated_numba(
        source, beta, reaction, None, space, advection_stabilization=policy, trace_space=trace,
        zero_boundary_flux=True).trace_system


@pytest.mark.parametrize("basis", ["dub_orth", "bernstein"])
@pytest.mark.parametrize("mode", ["penalty", "eliminate", "zero-flux"])
@pytest.mark.parametrize("policy", [None, "upwind", "lax-friedrichs"])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_numba_pointwise_matches_numba_dg(order, trace_basis, policy, mode, basis, forced_quadrature):
    space, trace, dg, pointwise = _problem(order, basis=basis, trace_basis=trace_basis)
    expected = _assemble(mode, dg, space, trace, policy)
    actual = _assemble(mode, pointwise, space, trace, policy)
    _close(_dense(actual), _dense(expected), 0, 1e-12)
    _close(actual.rhs, expected.rhs, 1e-11, 1e-12)
    trace_values = np.random.default_rng(order).normal(size=space.mesh.num_edg * trace.edg_dof)
    zero_flux = mode == "zero-flux"
    expected_field = reconstruct_projected_field_numba(trace_values, *dg, space, advection_stabilization=policy,
                                                       zero_boundary_flux=zero_flux, trace_space=trace)
    actual_field = reconstruct_projected_field_numba(trace_values, *pointwise, space, advection_stabilization=policy,
                                                     zero_boundary_flux=zero_flux, trace_space=trace)
    _close(actual_field.coeffs, expected_field.coeffs, 0, 1e-11)


@pytest.mark.parametrize("beta_order", [2, 4])
def test_cross_space_drift_and_constant_reaction_samples(beta_order, forced_quadrature):
    space, trace, dg, pointwise = _problem(2, beta_order=beta_order, constant_reaction=True)
    expected = _assemble("eliminate", dg, space, trace, None)
    actual = _assemble("eliminate", pointwise, space, trace, None)
    _close(_dense(actual), _dense(expected), 0, 1e-12)
    _close(actual.rhs, expected.rhs, 1e-11, 1e-12)


def test_pointwise_matches_the_tensor_volume_operator():
    """Without forcing, the DG reference uses its default operator; both integrate the same form."""
    space, trace, dg, pointwise = _problem(3)
    expected = _assemble("eliminate", dg, space, trace, None)
    actual = _assemble("eliminate", pointwise, space, trace, None)
    _close(_dense(actual), _dense(expected), 0, 1e-12)
    _close(actual.rhs, expected.rhs, 1e-12, 1e-12)


# --------------------------------------------------------------------------------------------
# Solver plumbing (host)
# --------------------------------------------------------------------------------------------

from hybridge.hdg.stabilization import resolve_transport_stabilization  # noqa: E402
from hybridge.runtime.errors import UnsupportedBackendConfigurationError  # noqa: E402
from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver  # noqa: E402


def _exact_coefficients(space, trace, *, device=False):
    """Exact (single-valued) advection field, reaction and source at the quadrature points."""
    xp = np
    if device:
        import cupy as xp  # noqa: F811
    beta = QuadratureVectorField.from_callable(
        lambda x, y: (xp.sin(2.0 * x + y) + 1.7, xp.cos(x - 2.0 * y) * x + 0.4), space, trace_space=trace,
        device=device)
    reaction = QuadratureScalar.from_callable(lambda x, y: 1.0 + 0.2 * x * y, space, device=device)
    source = QuadratureScalar.from_callable(lambda x, y: xp.exp(x) * y, space, device=device)
    return source, beta, reaction


def _host_solver(backend, coefficients, space, trace_basis="legacy-lagrange", **options):
    source, beta, reaction = coefficients
    return AdvectionReactionHDGSolver(space, assembly_backend=backend, source=source, beta=beta, reaction=reaction,
                                      boundary_condition=BOUNDARY, boundary_mode="eliminate",
                                      trace_basis=trace_basis, solver="direct", verbose=0, **options)


@pytest.mark.parametrize("order", [1, 3])
def test_numpy_pointwise_solve_matches_numba_pointwise(order):
    space = DGSpace(rectangle_mesh(4, 3), order, basis_type="dub_orth")
    coefficients = _exact_coefficients(space, space.trace_space("legacy-lagrange"))
    expected = _host_solver("numba", coefficients, space).solve()
    actual = _host_solver("numpy", coefficients, space).solve()
    _close(actual.field.coeffs, expected.field.coeffs, 0, 1e-10)


def test_cached_operator_set_source_matches_fresh_solve():
    space = DGSpace(rectangle_mesh(4, 3), 2, basis_type="dub_orth")
    source, beta, reaction = _exact_coefficients(space, space.trace_space("legacy-lagrange"))
    solver = _host_solver("numpy", (source, beta, reaction), space, cache_operator=True)
    solver.solve()
    new_source = QuadratureScalar.from_callable(lambda x, y: np.cos(3.0 * x) - y, space)
    reused = solver.set_source(new_source).solve()
    fresh = _host_solver("numpy", (new_source, beta, reaction), space).solve()
    _close(reused.field.coeffs, fresh.field.coeffs, 0, 1e-12)


def test_quadrature_source_enters_through_its_moments():
    space = DGSpace(rectangle_mesh(4, 3), 3, basis_type="dub_orth")
    _, beta, reaction = _exact_coefficients(space, space.trace_space("legacy-lagrange"))
    source = QuadratureScalar.from_callable(lambda x, y: np.exp(x) * y, space)
    with_values = _host_solver("numba", (source, beta, reaction), space).solve()
    with_field = _host_solver("numba", (source.project(), beta, reaction), space).solve()
    _close(with_values.field.coeffs, with_field.field.coeffs, 0, 1e-13)


def test_quadrature_drift_defaults_to_conflict_averaged_upwind():
    space = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    _, beta, _ = _exact_coefficients(space, space.trace_space("legacy-lagrange"))
    assert resolve_transport_stabilization(None, beta) == "conflict-averaged-upwind"


def test_unsupported_backends_reject_quadrature_coefficients():
    space = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
    coefficients = _exact_coefficients(space, space.trace_space("legacy-lagrange"))
    with pytest.raises(UnsupportedBackendConfigurationError, match="quadrature-point coefficients"):
        _host_solver("cupy", coefficients, space).solve()
    with pytest.raises(UnsupportedBackendConfigurationError, match="quadrature-point coefficients"):
        _host_solver("raw-cuda", coefficients, space, raw_local_assembly="precomputed").solve()
    with pytest.raises(ValueError, match="different mesh"):
        other = DGSpace(rectangle_mesh(3, 2), 2, basis_type="dub_orth")
        _host_solver("numba", coefficients, other).solve()


# --------------------------------------------------------------------------------------------
# Explicit residual (host)
# --------------------------------------------------------------------------------------------

from hybridge.transport.residual import UpwindHDGTransportResidual  # noqa: E402


@pytest.mark.parametrize("boundary_mode", ["eliminate", "zero-flux"])
@pytest.mark.parametrize("beta_order", [3, 5])
def test_residual_with_quadrature_drift_matches_dg_drift(boundary_mode, beta_order):
    space, trace, dg, pointwise = _problem(3, beta_order=beta_order)
    residual = UpwindHDGTransportResidual(space, trace_basis="legacy-lagrange", backend="host",
                                          boundary_mode=boundary_mode)
    density = space.project_callable(lambda x, y: np.sin(x) * np.cos(2.0 * y) + 1.0)
    boundary = BOUNDARY if boundary_mode == "eliminate" else None
    field, trace_values = residual.evaluate(density, dg[1], boundary)
    expected_field, expected_trace = np.array(field.coeffs, copy=True), np.array(trace_values, copy=True)
    field, trace_values = residual.evaluate(density, pointwise[1], boundary)
    _close(field.coeffs, expected_field, 1e-12, 1e-12)
    _close(trace_values, expected_trace, 1e-12, 1e-12)


# --------------------------------------------------------------------------------------------
# Raw CUDA (GPU)
# --------------------------------------------------------------------------------------------

RAW_POINTWISE_PATHS = [("fused", "coo"), ("fused", "csr"), ("fused", "bsr"), ("split3", "bsr")]


def _device_matrix(device, fmt, shape):
    """Dense host copy of a device trace matrix in ``fmt``."""
    import cupy as cp
    from scipy.sparse import bsr_matrix, csr_matrix

    if fmt == "coo":
        return coo_matrix((cp.asnumpy(device.data), (cp.asnumpy(device.rows), cp.asnumpy(device.cols))),
                          shape=shape).toarray()
    cls = bsr_matrix if fmt == "bsr" else csr_matrix
    return cls((cp.asnumpy(device.data), cp.asnumpy(device.indices), cp.asnumpy(device.indptr)),
               shape=shape).toarray()


@cuda_only
@pytest.mark.parametrize("mode,fmt", RAW_POINTWISE_PATHS)
@pytest.mark.parametrize("order", [2, 3, 6])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("boundary_mode", ["eliminate", "zero-flux"])
@pytest.mark.parametrize("policy", [None, "upwind"])
@pytest.mark.parametrize("reaction_kind", ["pointwise", "constant"])
def test_raw_cuda_pointwise_matches_numba_pointwise(mode, fmt, order, trace_basis, boundary_mode, policy,
                                                    reaction_kind):
    """Raw CUDA vs Numba with the advection field (and reaction) given at the quadrature points.

    p=6 exercises the BSR diagonal staging in the coefficient block; a constant
    reaction field takes the scalar reaction path next to the pointwise advection field.
    """
    import cupy as cp
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space
    from hybridge.transport.cuda import assemble_reduced_system_cuda, reconstruct_advection_field_cuda

    space, trace, dg, pointwise = _problem(order, trace_basis=trace_basis, mesh=rectangle_mesh(3, 2),
                                           constant_reaction=reaction_kind == "constant")
    source, beta, reaction = dg[0], pointwise[1], pointwise[2]
    zero_flux = boundary_mode == "zero-flux"
    boundary = None if zero_flux else BOUNDARY
    stabilization = resolve_transport_stabilization(policy, beta)
    reference = assemble_projected_trace_system_eliminated_numba(
        source, beta, reaction, boundary, space, advection_stabilization=stabilization,
        zero_boundary_flux=zero_flux, trace_space=trace).reduction
    shape = (reference.rhs.size,) * 2
    expected = coo_matrix((reference.data, (reference.rows, reference.cols)), shape=shape).toarray()
    device = assemble_reduced_system_cuda(
        source, reaction, boundary, None, as_cupy_space(space), as_cupy_trace_space(trace), backend="raw-cuda",
        raw_local_assembly=mode, raw_matrix_format=fmt, raw_cache_local_response=False,
        zero_boundary_flux=zero_flux, advection_stabilization=stabilization, beta_values=beta.to_device())
    assert device.raw.pointwise_beta and device.raw.pointwise_reaction == (reaction_kind == "pointwise")
    assert device.raw.reaction_is_scalar == (reaction_kind == "constant")
    assert device.raw.volume_table is not None
    scale = np.abs(expected).max()
    np.testing.assert_allclose(_device_matrix(device, fmt, shape), expected, rtol=2e-10, atol=2e-12 * scale)
    np.testing.assert_allclose(cp.asnumpy(device.rhs), reference.rhs, rtol=2e-10,
                               atol=2e-12 * max(1.0, np.abs(reference.rhs).max()))
    full_trace = np.random.default_rng(7).normal(size=space.mesh.num_edg * trace.edg_dof)
    host = reconstruct_projected_field_numba(full_trace, source, beta, reaction, space, trace_space=trace,
                                             advection_stabilization=stabilization, zero_boundary_flux=zero_flux)
    actual, _ = reconstruct_advection_field_cuda(cp.asarray(full_trace), source, reaction, None, device)
    np.testing.assert_allclose(cp.asnumpy(actual), host.coeffs, rtol=2e-10, atol=2e-12 * np.abs(host.coeffs).max())


@cuda_only
@pytest.mark.parametrize("order", [2, 3])
def test_raw_cuda_single_valued_drift_solve_matches_numba(order):
    """Exact device advection field, reaction and source: raw-CUDA fused assembly and reconstruction vs Numba.

    Both solve the trace system directly, so the comparison sees the assembly alone.
    """
    space = DGSpace(rectangle_mesh(4, 3), order, basis_type="dub_orth")
    trace = space.trace_space("legacy-lagrange")
    device_coefficients = _exact_coefficients(space, trace, device=True)
    host_coefficients = tuple(c.to_host() for c in device_coefficients)
    expected = _host_solver("numba", host_coefficients, space).solve()
    source, beta, reaction = device_coefficients
    actual = AdvectionReactionHDGSolver(
        space, assembly_backend="raw-cuda", source=source, beta=beta, reaction=reaction,
        boundary_condition=BOUNDARY, boundary_mode="eliminate", trace_basis="legacy-lagrange", solver="direct",
        raw_local_assembly="fused", verbose=0).solve()
    import cupy as cp
    np.testing.assert_allclose(cp.asnumpy(cp.asarray(actual.field.coeffs)), expected.field.coeffs, rtol=0,
                               atol=1e-10 * np.abs(expected.field.coeffs).max())


def test_raw_cuda_entry_rejects_tensor_operator_precomputed_and_cupy_assembly():
    """The device entry point rejects quadrature-point inputs before any device work."""
    from hybridge.transport.cuda import assemble_reduced_system_cuda

    space, trace, dg, pointwise = _problem(2, mesh=rectangle_mesh(2, 2))
    args = (dg[0], pointwise[2], BOUNDARY, None, space, trace)
    with pytest.raises(ValueError, match="tensor"):
        assemble_reduced_system_cuda(*args, backend="raw-cuda", raw_local_assembly="fused",
                                     raw_volume_operator="tensor", beta_values=pointwise[1])
    for backend, local_assembly in (("raw-cuda", "precomputed"), ("cupy", "fused")):
        with pytest.raises(UnsupportedBackendConfigurationError, match="quadrature-point coefficients"):
            assemble_reduced_system_cuda(*args, backend=backend, raw_local_assembly=local_assembly,
                                         beta_values=pointwise[1])


@cuda_only
@pytest.mark.parametrize("fmt", ["coo", "bsr"])
def test_raw_cuda_pointwise_kernels_use_no_local_memory(fmt, monkeypatch):
    """p=6 fused kernels: the pointwise variant adds no thread-local memory to the DG variant.

    Below p=8 the DG assembly kernel already keeps one NEL-entry Schur lift vector per thread in
    local memory (NEL * 8 bytes, staged in shared memory only from p=8); reconstruction uses none.
    """
    import cupy as cp
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space, as_cupy_vector_coefficients
    from hybridge.transport import raw_cuda as raw
    from hybridge.transport.cuda import assemble_reduced_system_cuda, reconstruct_advection_field_cuda

    compiled = {}
    compile_timed, compile_plain = raw._compile_kernel_timed, raw._compile_kernel

    def record_timed(cupy, source, name, shared):
        kernel, seconds = compile_timed(cupy, source, name, shared)
        compiled[name, "#define RAW_POINTWISE_BETA 1" in source] = kernel
        return kernel, seconds

    def record_plain(cupy, source, name, shared):
        kernel = compile_plain(cupy, source, name, shared)
        compiled[name, "#define RAW_POINTWISE_BETA 1" in source] = kernel
        return kernel

    monkeypatch.setattr(raw, "_compile_kernel_timed", record_timed)
    monkeypatch.setattr(raw, "_compile_kernel", record_plain)
    space, trace, dg, pointwise = _problem(6, mesh=rectangle_mesh(2, 2))
    cspace, ctrace = as_cupy_space(space), as_cupy_trace_space(trace)
    full_trace = cp.zeros(space.mesh.num_edg * trace.edg_dof)
    for beta_coeffs, beta_values, reaction in ((as_cupy_vector_coefficients(dg[1], cspace), None, dg[2]),
                                               (None, pointwise[1].to_device(), pointwise[2])):
        device = assemble_reduced_system_cuda(
            dg[0], reaction, BOUNDARY, beta_coeffs, cspace, ctrace, backend="raw-cuda", raw_local_assembly="fused",
            raw_matrix_format=fmt, raw_cache_local_response=False, raw_volume_operator="quadrature",
            beta_values=beta_values)
        reconstruct_advection_field_cuda(full_trace, dg[0], reaction, beta_coeffs, device)
    assert {pointwise for _, pointwise in compiled} == {False, True}
    sizes = {key: int(kernel.attributes["local_size_bytes"]) for key, kernel in compiled.items()}
    for name in {name for name, _ in compiled}:
        assert sizes[name, True] <= sizes[name, False], sizes
        if name.startswith("reconstruct"):
            assert sizes[name, True] == 0, sizes


@cuda_only
@pytest.mark.parametrize("boundary_mode", ["eliminate", "zero-flux"])
def test_device_residual_with_quadrature_drift_matches_host(boundary_mode):
    import cupy as cp

    space, trace, dg, pointwise = _problem(3, beta_order=5)
    density = space.project_callable(lambda x, y: np.sin(x) * np.cos(2.0 * y) + 1.0)
    boundary = BOUNDARY if boundary_mode == "eliminate" else None
    host = UpwindHDGTransportResidual(space, trace_basis="legacy-lagrange", backend="host", boundary_mode=boundary_mode)
    expected, expected_trace = host.evaluate(density, pointwise[1], boundary)
    expected, expected_trace = np.array(expected.coeffs, copy=True), np.array(expected_trace, copy=True)
    device = UpwindHDGTransportResidual(space, trace_basis="legacy-lagrange", backend="device",
                                        boundary_mode=boundary_mode)
    field, trace_values = device.evaluate(density, pointwise[1].to_device(), boundary)
    assert not field.coefficients_materialized
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space
    actual = cp.asnumpy(as_cupy_coefficients(field, as_cupy_space(space)))
    _close(actual, expected, 1e-11, 1e-12)
    _close(cp.asnumpy(trace_values), expected_trace, 1e-11, 1e-12)
    dg_field, _ = device.evaluate(density, dg[1], boundary)
    _close(cp.asnumpy(as_cupy_coefficients(dg_field, as_cupy_space(space))), expected, 1e-11, 1e-12)
