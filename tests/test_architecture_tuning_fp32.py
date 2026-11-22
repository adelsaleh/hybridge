"""Small actual-workload qualification in a separate FP32 process.

Run with HYBRIDGE_PRECISION=float32 before importing Hybridge. These checks
exercise assembly/reconstruction and persistent cache hits without a global
linear solver or FP64-specific test helpers.
"""
import json
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("HYBRIDGE_PRECISION") != "float32",
    reason="requires a separate HYBRIDGE_PRECISION=float32 process",
)


@pytest.fixture
def fp32_cuda(tmp_path, monkeypatch):
    """Give each workload a fresh bounded cache and retain process precision."""
    import numpy as np
    from hybridge.runtime.precision import REAL_DTYPE
    from hybridge.runtime.cuda_tuning import clear_tuning_cache

    assert np.dtype(REAL_DTYPE) == np.dtype("float32"), "set precision before importing Hybridge"
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    if not available:
        pytest.skip("CUDA device unavailable")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING", "auto")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_SECONDS", "30")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CANDIDATES", "3")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYBRIDGE_RAW_SPLIT3_VOLUME", "inline")
    clear_tuning_cache()
    yield cp, tmp_path
    clear_tuning_cache()


def _assert_fp32_close(cp, actual, expected, *, tolerance=6.e-5):
    """Check dtype and finite values before same-precision numerical parity."""
    import numpy as np

    assert actual.dtype == np.dtype("float32")
    assert bool(cp.all(cp.isfinite(actual)).item())
    reference = cp.asnumpy(expected) if isinstance(expected, cp.ndarray) else np.asarray(expected)
    assert np.all(np.isfinite(reference))
    scale = max(float(np.max(np.abs(reference), initial=0.)), 1.)
    np.testing.assert_allclose(cp.asnumpy(actual), reference,
                               rtol=tolerance, atol=tolerance * scale)


def _ar_problem():
    """Use projected FP32 coefficients on a p3 mesh with interior/boundary edges."""
    from hybridge import DGSpace, VectorDGField, rectangle_mesh
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space, as_cupy_vector_coefficients

    space = DGSpace(rectangle_mesh(2, 1, xlim=(-1., 1.)), 3, basis_type="dub_orth",
                    volume_quad_1d=8, edge_quad_1d=5)
    cspace = as_cupy_space(space)
    trace = as_cupy_trace_space(space.trace_space("legendre-modal"))
    source = space.project_callable(lambda x, y: 1. + .1*x - .05*y)
    reaction = space.project_callable(lambda x, y: 2. + .1*x*y)
    beta = VectorDGField((lambda x, y: .8 + .1*y, lambda x, y: -.2 + .05*x), space)
    coefficients = as_cupy_vector_coefficients(beta, cspace)
    return space, cspace, trace, source, reaction, coefficients


def _ar_assemble(problem, mode, block):
    """Keep every numerical input fixed while changing execution selection."""
    from hybridge.transport.cuda import assemble_reduced_system_cuda

    _, cspace, trace, source, reaction, coefficients = problem
    return assemble_reduced_system_cuda(
        source, reaction, lambda x, y: .2 + .1*x - .05*y, coefficients, cspace, trace,
        backend="raw-cuda", raw_local_assembly=mode, raw_block_size=block,
        raw_lu_mode="coop", raw_matrix_format="bsr", raw_cache_local_response=True,
    )


def _check_ar(cp, problem, actual, expected):
    """Compare BSR assembly, retained response and reconstructed FP32 field."""
    from hybridge.transport.cuda import reconstruct_advection_field_cuda

    space, _, trace, source, reaction, coefficients = problem
    cp.testing.assert_array_equal(actual.indptr, expected.indptr)
    cp.testing.assert_array_equal(actual.indices, expected.indices)
    for name in ("data", "rhs"):
        _assert_fp32_close(cp, getattr(actual, name), getattr(expected, name))
    _assert_fp32_close(cp, actual.raw.local_response, expected.raw.local_response)
    values = cp.sin(cp.arange(space.mesh.num_edg * (space.order + 1), dtype=cp.float32))
    actual_field, _ = reconstruct_advection_field_cuda(values, source, reaction, coefficients, actual)
    reference_field, _ = reconstruct_advection_field_cuda(values, source, reaction, coefficients, expected)
    _assert_fp32_close(cp, actual_field, reference_field)


def _forbid_trial(*args, **kwargs):
    """Fail if a persistent production cache hit tries preparing calibration."""
    raise AssertionError("cached production call attempted calibration allocation")


def test_fp32_full_ar_auto_matches_fused_and_excludes_tensor(fp32_cuda, monkeypatch):
    """The complete AR selector preserves FP32 and never offers an FP64 tensor path."""
    import numpy as np
    from hybridge.runtime.cuda_tuning import clear_tuning_cache
    from hybridge.transport import tsle_tensor, tuning_sample

    cp, cache = fp32_cuda
    problem = _ar_problem()
    expected = _ar_assemble(problem, "fused", 32)
    monkeypatch.setattr(tsle_tensor, "assemble_projected_advection_trace_system_eliminated_tsle_tensor",
                        _forbid_trial)
    actual = _ar_assemble(problem, "auto", "auto")
    _check_ar(cp, problem, actual, expected)
    assert actual.timings["raw.pipeline.algorithm_id"] in (0., 1.)
    records = [json.loads(path.read_text()) for path in cache.glob("*.json")]
    record, = [entry for entry in records if entry["workload"].get("family") == "ar-full-pipeline"]
    assert np.dtype(record["workload"]["precision"]) == np.dtype("float32")
    assert record["workload"]["tensor_inputs"] is False
    assert 1 <= len(record["candidates"]) <= 3
    assert all(not candidate["variant"].startswith("tensor:") for candidate in record["candidates"])
    assert "rejected" not in record["candidates"][0]
    clear_tuning_cache()
    monkeypatch.setattr(tuning_sample, "sample_raw_options", _forbid_trial)
    cached = _ar_assemble(problem, "auto", "auto")
    assert cached.timings["raw.pipeline.autotune.reused"] == 1.
    _check_ar(cp, problem, cached, expected)


def test_fp32_split3_auto_preserves_precision_reconstruction_and_disk_cache(fp32_cuda, monkeypatch):
    """All three conservative stage trials share the three-candidate budget."""
    from hybridge.runtime.cuda_tuning import clear_tuning_cache
    from hybridge.transport import tuning_sample

    cp, cache = fp32_cuda
    problem = _ar_problem()
    expected = _ar_assemble(problem, "split3", 32)
    actual = _ar_assemble(problem, "split3", "auto")
    _check_ar(cp, problem, actual, expected)
    records = [json.loads(path.read_text()) for path in cache.glob("*.json")]
    assert {record["workload"]["stage"] for record in records} == {"build", "solve", "scatter"}
    assert all(record["workload"]["precision"] == "float32" for record in records)
    assert sum(len(record["candidates"]) for record in records) == 3
    assert all("rejected" not in record["candidates"][0] for record in records)
    clear_tuning_cache()
    monkeypatch.setattr(tuning_sample, "sample_raw_options", _forbid_trial)
    cached = _ar_assemble(problem, "split3", "auto")
    assert cached.timings["raw.tsle.autotune.reused"] == 1.
    assert cached.timings["raw.tsle.autotune.sample_elements"] == 0.
    _check_ar(cp, problem, cached, expected)


def test_fp32_adr_auto_matches_explicit_and_independent_numpy_reconstruction(fp32_cuda, monkeypatch):
    """Check selected ADR kernels against the full host mixed local equations."""
    import numpy as np
    from scipy.sparse import bsr_matrix, coo_matrix
    from hybridge import DGSpace, rectangle_mesh
    from hybridge.hdg import condensation as hdg
    from hybridge.mixed.adr_preparation import prepare_adr_data
    from hybridge.mixed.adr_numpy import assemble_numpy
    from hybridge.mixed.raw_cuda.adr_operator import (
        assemble_projected_adr_trace_operator_raw_cuda as assemble,
        reconstruct_projected_adr_local_unknowns_raw_cuda as reconstruct,
    )
    from hybridge.mixed.raw_cuda import tensor_tuning
    from hybridge.runtime.cuda_tuning import clear_tuning_cache

    cp, _ = fp32_cuda
    space = DGSpace(rectangle_mesh(2, 1), 3, basis_type="dub_orth", volume_quad_1d=6)
    trace = space.trace_space("legendre-modal")
    beta = (space * space).field((space.constant(.6), space.constant(-.2)))
    diffusion = (lambda x, y: 1.5 + .1*x, lambda x, y: .05 + .01*y,
                 lambda x, y: .02 + .01*x, lambda x, y: 1.2 + .1*y)
    prepared = prepare_adr_data(lambda x, y: 1. + .1*x*y, 2., beta, space,
                               diffusion=diffusion, trace_space=trace)
    common = dict(diffusion=diffusion, trace_space=trace, matrix_format="bsr",
                  cache_local_factors="schur-lu+mass")
    explicit = assemble(prepared, .2, space, block_size=64, **common)
    automatic = assemble(prepared, .2, space, **common)
    host = assemble_numpy(prepared, .2, space, diffusion=diffusion, trace_space=trace)
    system = host.trace_system
    host_matrix = coo_matrix((system.data, (system.rows, system.cols)),
                             shape=(system.rhs.size,) * 2).toarray()
    values = np.sin(np.arange(space.mesh.num_edg * trace.edg_dof, dtype=np.float32) + np.float32(.2))
    source = np.zeros((space.mesh.num_tri, 3*space.el_dof), dtype=np.float32)
    source[:, :space.el_dof] = prepared.source_rhs
    host_local = hdg.reconstruct_local_unknowns(values, source, host.local_solver,
        prepared.element_boundary, space, trace_space=trace)

    def check(operator):
        """Check numerical outputs, including the selected reconstruction cache."""
        assembly = operator.assembly
        _assert_fp32_close(cp, assembly.data, explicit.assembly.data)
        _assert_fp32_close(cp, assembly.rhs, explicit.assembly.rhs)
        dense = bsr_matrix((cp.asnumpy(assembly.data), cp.asnumpy(assembly.indices),
                            cp.asnumpy(assembly.indptr)), shape=host_matrix.shape).toarray()
        _assert_fp32_close(cp, cp.asarray(dense), host_matrix, tolerance=3.e-4)
        _assert_fp32_close(cp, assembly.rhs, system.rhs, tolerance=3.e-4)
        local, _ = reconstruct(operator, cp.asarray(values))
        explicit_local, _ = reconstruct(explicit, cp.asarray(values))
        _assert_fp32_close(cp, local, explicit_local)
        _assert_fp32_close(cp, local, host_local, tolerance=3.e-4)

    check(automatic)
    record = automatic.reconstruction_data["tuning"]
    assert record["source"] == "measured" and 1 <= len(record["candidates"]) <= 3
    assert "rejected" not in record["candidates"][0]
    assert automatic.assembly.timings["raw.tuning.sample_elements"] > 0.
    clear_tuning_cache()
    monkeypatch.setattr(tensor_tuning, "sample_element_indices", _forbid_trial)
    cached = assemble(prepared, .2, space, **common)
    assert cached.reconstruction_data["tuning"]["source"] == "disk"
    assert cached.assembly.timings["raw.tuning.sample_elements"] == 0.
    check(cached)
