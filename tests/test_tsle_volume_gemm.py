"""Element-batched split3 volume operator (HYBRIDGE_RAW_SPLIT3_VOLUME) against the inline build."""

from __future__ import annotations

import pytest

from hybridge import DGSpace, VectorDGField, rectangle_mesh
from hybridge.transport import tsle_volume_gemm


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _cuda_available(), reason="CuPy CUDA runtime is unavailable")


def test_volume_mode_switch(monkeypatch):
    monkeypatch.delenv("HYBRIDGE_RAW_SPLIT3_VOLUME", raising=False)
    assert tsle_volume_gemm.volume_mode() == "inline"
    for mode in ("dmma", "cublas", "inline"):
        monkeypatch.setenv("HYBRIDGE_RAW_SPLIT3_VOLUME", mode)
        assert tsle_volume_gemm.volume_mode() == mode
    monkeypatch.setenv("HYBRIDGE_RAW_SPLIT3_VOLUME", "tensor-core")
    with pytest.raises(ValueError, match="HYBRIDGE_RAW_SPLIT3_VOLUME"):
        tsle_volume_gemm.volume_mode()


def test_prepass_targets_the_dense_tensor_operator_only():
    assert tsle_volume_gemm.volume_prepass_applies(nqv=None, pointwise_beta=False, pointwise_reaction=False)
    assert not tsle_volume_gemm.volume_prepass_applies(nqv=42, pointwise_beta=False, pointwise_reaction=False)
    assert not tsle_volume_gemm.volume_prepass_applies(nqv=None, pointwise_beta=True, pointwise_reaction=False)


def test_dims_pad_rows_and_columns():
    assert tsle_volume_gemm._dims(55, 55, True) == (111, 112, 3025, 3040)
    assert tsle_volume_gemm._dims(55, 55, False) == (165, 168, 3025, 3040)
    assert tsle_volume_gemm._dims(3, 6, True) == (13, 16, 9, 32)


def _assemble(order, trace_basis, *, field_reaction, beta_order, block_size, zero_boundary_flux=True):
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space, as_cupy_vector_coefficients
    from hybridge.transport.cuda import assemble_reduced_system_cuda
    from hybridge.transport.tsle_bsr import RawAdvectionTsleWorkspace

    mesh = rectangle_mesh(4, 3, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, order, basis_type="dub_orth")
    beta_space = space if beta_order is None else DGSpace(mesh, beta_order, basis_type="dub_orth")
    beta = VectorDGField((lambda x, y: 0.3 + y, lambda x, y: -0.25 + 0.5 * x * x - 0.1 * y), beta_space)
    source = space.project_callable(lambda x, y: 1.0 + 0.2 * x - 0.1 * y)
    reaction = (space.project_callable(lambda x, y: 2.0 + 0.1 * x * y) if field_reaction else space.constant(1.5))
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(space.trace_space(trace_basis), device=cspace.device_id)
    return assemble_reduced_system_cuda(
        source, reaction, None if zero_boundary_flux else (lambda x, y: x + 0.5 * y),
        as_cupy_vector_coefficients(beta, cspace), cspace, trace_ref, backend="raw-cuda",
        raw_local_assembly="split3", raw_block_size=block_size, raw_matrix_format="bsr",
        zero_boundary_flux=zero_boundary_flux, raw_tsle_workspace=RawAdvectionTsleWorkspace(),
        raw_volume_operator="tensor", beta_space=None if beta_order is None else beta_space)


@cuda_only
@pytest.mark.parametrize("mode", ["dmma", "cublas"])
@pytest.mark.parametrize("order,trace_basis,field_reaction,beta_order,block_size", [
    (2, "legacy-lagrange", False, None, 32),
    (5, "legendre-modal", True, None, 64),
    (8, "legacy-lagrange", False, None, "auto"),
    (9, "legendre-modal", True, None, 128),
    (9, "legacy-lagrange", False, None, "auto"),
    (6, "legendre-modal", False, 7, 64),   # advection field from another space (NB != NEL)
])
def test_volume_prepass_matches_inline(monkeypatch, mode, order, trace_basis, field_reaction, beta_order, block_size):
    import cupy as cp
    from hybridge.transport.tsle_bsr import clear_tsle_runtime_caches

    if mode == "dmma" and int(cp.cuda.Device().compute_capability) < 80:
        pytest.skip("FP64 mma needs compute capability 8.0+")
    clear_tsle_runtime_caches()
    monkeypatch.setenv("HYBRIDGE_RAW_SPLIT3_VOLUME", "inline")
    reference = _assemble(order, trace_basis, field_reaction=field_reaction, beta_order=beta_order,
                          block_size=block_size)
    monkeypatch.setenv("HYBRIDGE_RAW_SPLIT3_VOLUME", mode)
    result = _assemble(order, trace_basis, field_reaction=field_reaction, beta_order=beta_order,
                       block_size=block_size)
    assert result.timings["raw.tsle.volume.preloaded"] == 1.0
    assert reference.timings["raw.tsle.volume.preloaded"] == 0.0
    assert result.timings["raw.tsle.volume"] > 0.0
    for name in ("data", "rhs"):
        expected, actual = cp.asarray(getattr(reference, name)), cp.asarray(getattr(result, name))
        assert float(cp.abs(actual - expected).max()) <= 1e-12 * float(cp.abs(expected).max()), name
    expected, actual = reference.raw.local_response, result.raw.local_response
    assert float(cp.abs(actual - expected).max()) <= 1e-11 * float(cp.abs(expected).max())


@cuda_only
def test_volume_prepass_with_dirichlet_data(monkeypatch):
    import cupy as cp

    monkeypatch.setenv("HYBRIDGE_RAW_SPLIT3_VOLUME", "inline")
    reference = _assemble(7, "legendre-modal", field_reaction=False, beta_order=None, block_size=64,
                          zero_boundary_flux=False)
    monkeypatch.setenv("HYBRIDGE_RAW_SPLIT3_VOLUME", "dmma")
    result = _assemble(7, "legendre-modal", field_reaction=False, beta_order=None, block_size=64,
                       zero_boundary_flux=False)
    for name in ("data", "rhs"):
        expected, actual = cp.asarray(getattr(reference, name)), cp.asarray(getattr(result, name))
        assert float(cp.abs(actual - expected).max()) <= 1e-12 * float(cp.abs(expected).max()), name
