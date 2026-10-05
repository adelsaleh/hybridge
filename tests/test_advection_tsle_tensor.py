from __future__ import annotations

import numpy as np
import pytest

from hybridge import DGSpace, VectorDGField, rectangle_mesh


def _cupy_runtime_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    return True


def _cutensor_available() -> bool:
    try:
        from hybridge.runtime.optional import require_cutensor

        require_cutensor()
    except Exception:
        return False
    return True


def _magma_available() -> bool:
    try:
        from hybridge.linalg.gpu.magma_batched import magma_available

        return magma_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cupy_runtime_available(), reason="CuPy CUDA runtime is unavailable")


def _split3_inputs(order: int, trace_basis: str, *, zero_boundary_flux: bool, stabilization=None) -> dict:
    """Return the keyword inputs that assemble_reduced_system_cuda passes to TSLE-BSR."""
    import hybridge.transport.cuda as transport_cuda
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space, as_cupy_vector_coefficients

    mesh = rectangle_mesh(3, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, order, basis_type="dub_orth", volume_quad_1d=2 * order + 2, edge_quad_1d=order + 2)
    source_h = space.project_callable(lambda x, y: 1.0 + 0.2 * x - 0.1 * y, name="source_h")
    reaction_h = space.project_callable(lambda x, y: 2.0 + 0.1 * x * y, name="reaction_h")
    # The velocity changes sign inside the domain, so upwind directions and
    # (for conflict averaging) outflow/outflow faces both occur.
    beta_h = VectorDGField(
        (lambda x, y: 0.3 + y, lambda x, y: -0.25 + 0.5 * x * x - 0.1 * y), space, name="beta_h")
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(space.trace_space(trace_basis), device=cspace.device_id)
    captured: dict = {}
    original = transport_cuda.assemble_projected_advection_trace_system_eliminated_tsle_bsr

    def capture(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    transport_cuda.assemble_projected_advection_trace_system_eliminated_tsle_bsr = capture
    try:
        transport_cuda.assemble_reduced_system_cuda(
            source_h, reaction_h, None if zero_boundary_flux else (lambda x, y: x + 0.5 * y),
            as_cupy_vector_coefficients(beta_h, cspace), cspace, trace_ref,
            backend="raw-cuda", raw_local_assembly="split3", raw_block_size=32, raw_matrix_format="bsr",
            zero_boundary_flux=zero_boundary_flux, advection_stabilization=stabilization)
    finally:
        transport_cuda.assemble_projected_advection_trace_system_eliminated_tsle_bsr = original
    captured.pop("workspace", None)
    return captured


def _assert_matches_split3(inputs: dict, *, rtol: float, **options) -> None:
    import cupy as cp
    from hybridge.transport.tsle_bsr import assemble_projected_advection_trace_system_eliminated_tsle_bsr
    from hybridge.transport.tsle_tensor import assemble_projected_advection_trace_system_eliminated_tsle_tensor

    reference = assemble_projected_advection_trace_system_eliminated_tsle_bsr(**inputs)
    result = assemble_projected_advection_trace_system_eliminated_tsle_tensor(**inputs, **options)
    cp.testing.assert_array_equal(result.indptr, reference.indptr)
    cp.testing.assert_array_equal(result.indices, reference.indices)
    assert result.data.dtype == result.rhs.dtype == result.local_response.dtype == np.float64
    for name in ("data", "rhs", "local_response"):
        expected = cp.asarray(getattr(reference, name))
        actual = cp.asarray(getattr(result, name))
        scale = float(cp.abs(expected).max())
        assert float(cp.abs(actual - expected).max()) <= rtol * scale, name
    assert result.timings["raw.tsle_tensor.device"] > 0.0


@pytest.mark.parametrize("order,trace_basis", ((1, "legacy-lagrange"), (4, "legendre-modal"), (7, "legacy-lagrange")))
@pytest.mark.parametrize("contraction", ("cublas", "cutensor"))
@pytest.mark.parametrize("local_solver", ("coop", "cublas"))
def test_fp64_tensor_tsle_matches_split3(order, trace_basis, contraction, local_solver) -> None:
    if contraction == "cutensor" and not _cutensor_available():
        pytest.skip("cuTENSOR is unavailable")
    inputs = _split3_inputs(order, trace_basis, zero_boundary_flux=False)
    _assert_matches_split3(inputs, rtol=1.0e-12, precision="float64", contraction=contraction,
                           local_solver=local_solver)


@pytest.mark.parametrize("trace_basis", ("legacy-lagrange", "legendre-modal"))
def test_fp64_tensor_tsle_zero_flux_conflict_averaged_matches_split3(trace_basis) -> None:
    inputs = _split3_inputs(5, trace_basis, zero_boundary_flux=True, stabilization="conflict-averaged-upwind")
    _assert_matches_split3(inputs, rtol=1.0e-12, precision="float64", contraction="cublas", local_solver="coop")


@pytest.mark.skipif(not _magma_available(), reason="MAGMA is unavailable (set HYBRIDGE_MAGMA_ROOT)")
@pytest.mark.parametrize("precision,rtol", (("float64", 1.0e-12), ("float32", 1.0e-4)))
def test_magma_tensor_tsle_matches_split3(precision, rtol) -> None:
    inputs = _split3_inputs(6, "legendre-modal", zero_boundary_flux=True, stabilization="conflict-averaged-upwind")
    _assert_matches_split3(inputs, rtol=rtol, precision=precision, contraction="cublas", local_solver="magma")


def test_fp32_tensor_tsle_tracks_fp64_split3() -> None:
    inputs = _split3_inputs(3, "legacy-lagrange", zero_boundary_flux=False)
    _assert_matches_split3(inputs, rtol=1.0e-5, precision="float32", contraction="cublas", local_solver="coop")


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("trans", (False, True))
def test_lu_solve_batched_cublas_matches_numpy(dtype, trans) -> None:
    import cupy as cp
    from hybridge.linalg.gpu.cublas_batched import BatchedLUWorkspace, lu_solve_batched_cublas

    rng = np.random.default_rng(3)
    matrices = rng.standard_normal((5, 7, 7)) + 7.0 * np.eye(7)
    rhs = rng.standard_normal((5, 3, 7))
    # Column-major storage: the C-order transpose unless trans selects the stored matrix itself.
    stored = matrices if trans else matrices.transpose(0, 2, 1)
    a = cp.asarray(np.ascontiguousarray(stored), dtype=dtype)
    b = cp.asarray(rhs, dtype=dtype)
    workspace = lu_solve_batched_cublas(a, b, trans=trans, workspace=BatchedLUWorkspace())
    expected = np.linalg.solve(matrices, rhs.transpose(0, 2, 1)).transpose(0, 2, 1)
    tolerance = 1.0e-4 if dtype == np.float32 else 1.0e-12
    np.testing.assert_allclose(cp.asnumpy(b), expected, rtol=tolerance, atol=tolerance)
    assert int(cp.count_nonzero(workspace.info)) == 0


def test_specialize_real_source_targets_float32_only_on_request() -> None:
    from hybridge.runtime.precision import specialize_real_source

    source = "double x = fabs(y) * 0.5 + 1.0e-30;"
    assert specialize_real_source(source, np.float64) == source
    assert specialize_real_source(source, np.float32) == "float x = fabsf(y) * 0.5f + 1.0e-30f;"
    with pytest.raises(ValueError):
        specialize_real_source(source, np.float16)
