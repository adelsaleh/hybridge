"""Tensor AR assembly and measured, same-precision automatic pipeline selection."""

from __future__ import annotations

import numpy as np
import pytest

from hybridge import DGSpace, VectorDGField, rectangle_mesh
from hybridge.core.quadrature_coefficients import QuadratureVectorField
from hybridge.solvers import advection_reaction as solver_module
from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver
from hybridge.solvers.capabilities import (
    RAW_TENSOR_MIN_ORDER,
    resolve_raw_local_assembly,
    tensor_memory_check_needed,
)
from hybridge.transport.tsle_tensor import tensor_workspace_bytes

GIB = 2 ** 30
BETA = (lambda x, y: 0.3 + y, lambda x, y: -0.25 + 0.5 * x * x - 0.1 * y)


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _cuda_available(), reason="CuPy CUDA runtime is unavailable")


@pytest.mark.parametrize("order,tensor_bytes,expected", [
    (9, 7 * GIB, "tensor"),
    (6, 1 * GIB, "tensor"),
    (9, 50 * GIB, "split3"),       # does not fit: the split3/fused rule decides
    (7, 50 * GIB, "fused"),
    (5, 1 * GIB, "tensor"),
    (4, 1 * GIB, "fused"),         # below RAW_TENSOR_MIN_ORDER
    (9, None, "split3"),           # not eligible: no size supplied
])
def test_auto_policy(order, tensor_bytes, expected):
    assert resolve_raw_local_assembly("auto", order=order, bsr_output=True, tensor_workspace_bytes=tensor_bytes,
                                      available_device_bytes=140 * GIB) == expected


def test_auto_policy_takes_tensor_when_it_needs_less_memory_than_split3():
    # 2M elements at p=9: chunked tensor 44 GB > 25% of 140 GB, but split3 would take 77 GB unchecked.
    sizes = dict(tensor_workspace_bytes=44 * 10 ** 9, available_device_bytes=140 * 10 ** 9)
    assert resolve_raw_local_assembly("auto", order=9, bsr_output=True, split3_workspace_bytes=77 * 10 ** 9,
                                      **sizes) == "tensor"
    assert resolve_raw_local_assembly("auto", order=9, bsr_output=True, split3_workspace_bytes=40 * 10 ** 9,
                                      **sizes) == "split3"
    # Below RAW_SPLIT3_MIN_ORDER the alternative is fused (no workspace): only the memory fraction counts.
    assert resolve_raw_local_assembly("auto", order=7, bsr_output=True, split3_workspace_bytes=77 * 10 ** 9,
                                      **sizes) == "fused"


def test_auto_policy_keeps_split3_requirements_and_explicit_values():
    sizes = dict(tensor_workspace_bytes=GIB, available_device_bytes=140 * GIB)
    assert resolve_raw_local_assembly("auto", order=9, bsr_output=False, **sizes) == "fused"
    assert resolve_raw_local_assembly("auto", order=9, bsr_output=True, raw_lu_mode="safe", **sizes) == "fused"
    assert resolve_raw_local_assembly("auto", order=9, bsr_output=True, cache_operator=True, **sizes) == "fused"
    assert resolve_raw_local_assembly("split3", order=9, bsr_output=True, **sizes) == "split3"
    assert resolve_raw_local_assembly("tensor", order=2, bsr_output=True) == "tensor"


def test_tensor_check_needs_sm90_tensor_inputs_and_order():
    common = dict(order=RAW_TENSOR_MIN_ORDER, bsr_output=True, tensor_inputs=True)
    assert tensor_memory_check_needed("auto", compute_capability="90", **common)
    for capability in ("80", "89", "120", None):
        assert not tensor_memory_check_needed("auto", compute_capability=capability, **common)
    assert not tensor_memory_check_needed("auto", compute_capability="90", **dict(common, tensor_inputs=False))
    assert not tensor_memory_check_needed("auto", compute_capability="90", **dict(common, order=4))
    assert not tensor_memory_check_needed("fused", compute_capability="90", **common)


def test_tensor_workspace_bytes_tracks_the_measured_h200_workspace():
    # Q1 (H200, p=9, 117,439 triangles): 7.03 GiB workspace without the FP64 response copy.
    without_copy = tensor_workspace_bytes(117_439, 55, 10, 10, cache_local_response=False) / GIB
    assert 6.5 < without_copy < 7.5
    assert tensor_workspace_bytes(117_439, 55, 10, 10) > tensor_workspace_bytes(117_439, 55, 10, 10,
                                                                                cache_local_response=False)


def test_tensor_inputs_need_dg_fields_in_the_solution_basis():
    mesh = rectangle_mesh(2, 2)
    space = DGSpace(mesh, 3, basis_type="dub_orth")
    source = space.project_callable(lambda x, y: 1.0 + x)
    reaction = space.project_callable(lambda x, y: 2.0 + y)
    assert solver_module._tensor_inputs(space, source, VectorDGField(BETA, space), reaction)
    compatible = DGSpace(mesh, 3, basis_type="dub_orth", volume_quad_1d=9)
    assert solver_module._tensor_inputs(space, source, VectorDGField(BETA, compatible), reaction)
    assert not solver_module._tensor_inputs(space, source, VectorDGField(BETA, DGSpace(mesh, 4, basis_type="dub_orth")),
                                            reaction)
    quadrature_beta = QuadratureVectorField.from_callable(BETA, space, trace_space=space.trace_space("legendre-modal"))
    assert not solver_module._tensor_inputs(space, source, quadrature_beta, reaction)


def _solver(order: int, trace_basis: str, local_assembly: str, *, beta_order: int | None = None):
    mesh = rectangle_mesh(3, 2, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    space = DGSpace(mesh, order, basis_type="dub_orth")
    beta_space = space if beta_order is None else DGSpace(mesh, beta_order, basis_type="dub_orth")
    return AdvectionReactionHDGSolver(
        space,
        source=space.project_callable(lambda x, y: 1.0 + 0.2 * x - 0.1 * y),
        beta=VectorDGField(BETA, beta_space),
        reaction=space.project_callable(lambda x, y: 2.0 + 0.1 * x * y),
        boundary_condition=None, boundary_mode="zero-flux", trace_basis=trace_basis,
        assembly_backend="raw-cuda", raw_local_assembly=local_assembly, solver="amgx", verbose=0)


def _path(assembly) -> str:
    keys = assembly.timings
    if any("tsle_tensor" in key for key in keys):
        return "tensor"
    return "split3" if any(".tsle." in f".{key}" or key.startswith("raw.tsle.") for key in keys) else "fused"


@cuda_only
@pytest.mark.parametrize("order", [6, 7])
@pytest.mark.parametrize("trace_basis", ["legacy-lagrange", "legendre-modal"])
def test_solver_tensor_matches_split3(order, trace_basis):
    import cupy as cp

    reference = _solver(order, trace_basis, "split3").assemble_tangent_boundary_raw_cuda_bsr()
    result = _solver(order, trace_basis, "tensor").assemble_tangent_boundary_raw_cuda_bsr()
    assert _path(reference) == "split3" and _path(result) == "tensor"
    cp.testing.assert_array_equal(result.indptr, reference.indptr)
    cp.testing.assert_array_equal(result.indices, reference.indices)
    for name in ("data", "rhs"):
        expected, actual = cp.asarray(getattr(reference, name)), cp.asarray(getattr(result, name))
        assert float(cp.abs(actual - expected).max()) <= 1e-11 * float(cp.abs(expected).max()), name


@cuda_only
@pytest.mark.parametrize("order", [4, 5, 6, 7, 9])
def test_auto_pipeline_matches_reference_and_reuses_disk(monkeypatch, tmp_path, order):
    import json
    import cupy as cp
    from hybridge.runtime.cuda_tuning import clear_tuning_cache
    from hybridge.transport import tuning_sample
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    clear_tuning_cache()
    reference = _solver(order, "legendre-modal", "fused").assemble_tangent_boundary_raw_cuda_bsr(
        cache_local_response=True)
    solver = _solver(order, "legendre-modal", "auto")
    assembly = solver.assemble_tangent_boundary_raw_cuda_bsr(cache_local_response=True)
    assert _path(assembly) in {"fused", "split3", "tensor"}
    cp.testing.assert_array_equal(assembly.indptr, reference.indptr)
    cp.testing.assert_array_equal(assembly.indices, reference.indices)
    for actual, expected in ((assembly.data, reference.data), (assembly.rhs, reference.rhs),
                             (assembly.raw.local_response, reference.raw.local_response)):
        cp.testing.assert_allclose(actual, expected, rtol=5e-10, atol=5e-11)
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    records = [record for record in records if record["workload"]["family"] == "ar-full-pipeline"]
    assert len(records) == 1 and len(records[0]["candidates"]) <= 8
    assert all(record["configuration"]["local_solver"] != "auto"
               for record in records[0]["candidates"] if "configuration" in record)
    clear_tuning_cache()
    monkeypatch.setattr(tuning_sample, "sample_raw_options", lambda *a, **kw: pytest.fail("cache hit sampled"))
    reused = solver.assemble_tangent_boundary_raw_cuda_bsr(cache_local_response=True)
    assert reused.timings["raw.pipeline.autotune.reused"] == 1.
    cp.testing.assert_allclose(reused.data, reference.data, rtol=5e-10, atol=5e-11)


@cuda_only
def test_auto_excludes_tensor_for_a_cross_space_advection_field(monkeypatch, tmp_path):
    import cupy as cp
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    assembly = _solver(6, "legendre-modal", "auto", beta_order=7).assemble_tangent_boundary_raw_cuda_bsr()
    reference = _solver(6, "legendre-modal", "fused", beta_order=7).assemble_tangent_boundary_raw_cuda_bsr()
    assert _path(assembly) in {"fused", "split3"}
    cp.testing.assert_allclose(assembly.data, reference.data, rtol=5e-10, atol=5e-11)
    with pytest.raises(NotImplementedError, match="tensor"):
        _solver(6, "legendre-modal", "tensor", beta_order=7).assemble_tangent_boundary_raw_cuda_bsr()


@cuda_only
@pytest.mark.parametrize("zero_flux", [False, True])
def test_auto_preserves_pointwise_coefficients_and_boundary_data(monkeypatch, tmp_path, zero_flux):
    import cupy as cp
    from hybridge.core.quadrature_coefficients import QuadratureScalar
    from hybridge.transport.cuda import assemble_reduced_system_cuda, reconstruct_advection_field_cuda
    from test_advection_tsle_bsr import _projected_problem
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    space, cspace, trace, source, reaction, _, boundary = _projected_problem(3, "legendre-modal")
    beta = QuadratureVectorField.from_callable(BETA, space, trace_space=space.trace_space("legendre-modal"))
    reaction = QuadratureScalar.from_field(reaction, space)
    results = []
    for path in ("fused", "auto"):
        results.append(assemble_reduced_system_cuda(
            source, reaction, None if zero_flux else boundary, None, cspace, trace,
            beta_values=beta, raw_local_assembly=path, raw_matrix_format="bsr",
            zero_boundary_flux=zero_flux, raw_cache_local_response=True))
    expected, actual = results
    assert _path(actual) in {"fused", "split3"}
    for name in ("data", "rhs", "boundary_trace"):
        cp.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=5e-10, atol=5e-11)
    trace_values = cp.asarray(np.random.default_rng(42).normal(size=(space.mesh.num_edg, cspace.edg_dof)))
    reference_field, _ = reconstruct_advection_field_cuda(trace_values, source, reaction, None, expected)
    actual_field, _ = reconstruct_advection_field_cuda(trace_values, source, reaction, None, actual)
    cp.testing.assert_allclose(actual_field, reference_field, rtol=5e-10, atol=5e-11)


@cuda_only
def test_tensor_workspace_is_reused_and_cleared():
    solver = _solver(6, "legendre-modal", "tensor")
    solver.assemble_tangent_boundary_raw_cuda_bsr()
    workspace = solver._raw_cuda_tsle_workspace
    assert workspace.tensor is not None and workspace.tensor.nbytes > 0
    buffers = dict(workspace.tensor.buffers)
    solver.assemble_tangent_boundary_raw_cuda_bsr()
    assert all(workspace.tensor.buffers[name] is buffers[name] for name in buffers)
    workspace.clear()
    assert workspace.tensor.nbytes == 0


def test_tensor_resolves_the_default_lu_mode_to_coop():
    from hybridge.transport.raw_cuda import resolve_raw_lu_mode

    assert resolve_raw_lu_mode(None, "tensor") == "coop"
    assert resolve_raw_lu_mode(None, "precomputed") == "safe"


@pytest.mark.parametrize("capability,magma,expected", [
    ("90", True, "auto"), ("90", False, "auto"), ("120", True, "auto"), ("80", False, "auto"), (None, True, "auto"),
])
def test_default_tensor_local_solver_calibrates_on_every_architecture(monkeypatch, capability, magma, expected):
    from hybridge.linalg.gpu import magma_batched
    from hybridge.transport import tsle_tensor

    def load_magma():
        if not magma:
            raise OSError("no MAGMA")

    monkeypatch.delenv("HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER", raising=False)
    monkeypatch.setattr(tsle_tensor, "_device_compute_capability", lambda: capability)
    monkeypatch.setattr(magma_batched, "load_magma", load_magma)
    assert tsle_tensor.default_tensor_engines()[1] == expected
    monkeypatch.setenv("HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER", "magma")
    assert tsle_tensor.default_tensor_engines()[1] == "magma"


@cuda_only
@pytest.mark.parametrize("solver", ["fused", "magma", "coop"])
def test_tensor_chunks_match_one_pass(monkeypatch, solver):
    """HYBRIDGE_RAW_TENSOR_CHUNK_ELEMENTS splits stages 1-3 and the scatter into uneven chunks."""
    import cupy as cp

    monkeypatch.setenv("HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER", solver)
    reference = _solver(7, "legendre-modal", "tensor").assemble_tangent_boundary_raw_cuda_bsr()
    monkeypatch.setenv("HYBRIDGE_RAW_TENSOR_CHUNK_ELEMENTS", "5")
    result = _solver(7, "legendre-modal", "tensor").assemble_tangent_boundary_raw_cuda_bsr()
    assert reference.timings["raw.tsle_tensor.chunks"] == 1.0
    assert result.timings["raw.tsle_tensor.chunks"] == 3.0 and result.timings["raw.tsle_tensor.chunk_elements"] == 5.0
    cp.testing.assert_array_equal(result.indices, reference.indices)
    for name in ("data", "rhs"):
        expected, actual = cp.asarray(getattr(reference, name)), cp.asarray(getattr(result, name))
        assert float(cp.abs(actual - expected).max()) <= 1e-13 * float(cp.abs(expected).max()), name


def test_chunk_elements_split_evenly(monkeypatch):
    from hybridge.transport.tsle_tensor import tensor_chunk_elements

    monkeypatch.delenv("HYBRIDGE_RAW_TENSOR_CHUNK_ELEMENTS", raising=False)
    monkeypatch.delenv("HYBRIDGE_RAW_TENSOR_CHUNK_GIB", raising=False)
    assert tensor_chunk_elements(117_439, 55, 10, 10) == 117_439
    assert tensor_chunk_elements(2_000_000, 55, 10, 10) == 250_000
    assert tensor_chunk_elements(10, 55, 10, 10, chunk_elements=4) == 4
    monkeypatch.setenv("HYBRIDGE_RAW_TENSOR_CHUNK_GIB", "1")
    assert tensor_chunk_elements(117_439, 55, 10, 10) < 20_000
    with pytest.raises(ValueError):
        tensor_chunk_elements(10, 55, 10, 10, chunk_elements=0)


@cuda_only
@pytest.mark.parametrize('order', [2, 6, 9])
def test_calibrated_tensor_solve_matches_explicit_reference_and_reuses_cache(order, monkeypatch, tmp_path):
    import cupy as cp
    from hybridge.runtime.cuda_tuning import clear_tuning_cache
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER', 'cublas')
    reference = _solver(order, 'legendre-modal', 'tensor').assemble_tangent_boundary_raw_cuda_bsr(cache_local_response=True)
    monkeypatch.setenv('HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER', 'auto')
    clear_tuning_cache()
    solver = _solver(order, 'legendre-modal', 'tensor')
    first = solver.assemble_tangent_boundary_raw_cuda_bsr(cache_local_response=True)
    assert first.timings['raw.tsle_tensor.autotune.reused'] == 0.
    clear_tuning_cache()
    second = solver.assemble_tangent_boundary_raw_cuda_bsr(cache_local_response=True)
    assert second.timings['raw.tsle_tensor.autotune.reused'] == 1.
    for result in (first, second):
        cp.testing.assert_array_equal(result.indices, reference.indices)
        for name in ('data', 'rhs', 'local_response'):
            actual = cp.asarray(getattr(result.raw, name))
            expected = cp.asarray(getattr(reference.raw, name))
            cp.testing.assert_allclose(actual, expected, rtol=5e-10, atol=5e-11)
