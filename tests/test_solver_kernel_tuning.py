"""Solver calibration sampling, explicit controls, parity and residual gates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.linalg.gpu import solver_tuning


def test_bsr_sampling_preserves_rows_neighbors_and_own_correction(monkeypatch):
    """Compressed trials retain actual blocks and every sampled gather value."""
    from test_hp_mg_packed_kernels import _spd_face_bsr

    pointers, columns, data, _ = _spd_face_bsr(37, 3, 17)
    inverse = np.arange(37 * 9, dtype=float).reshape(37, 3, 3)
    snapshot = [x.copy() for x in (pointers, columns, data, inverse)]
    monkeypatch.setattr(solver_tuning, "MAX_SAMPLE_ROWS", 7)
    sampled = solver_tuning.sample_bsr(np, pointers, columns, data, inverse)
    selected = solver_tuning.sample_group_indices(37, 100)
    original_x = solver_tuning.trial_vector(np, np.arange(37), 3, data.dtype).reshape(37, 3)
    sampled_x = sampled["x"].reshape(-1, 3)
    for row, original_row in enumerate(selected):
        start, end = pointers[original_row:original_row + 2]
        trial_start, trial_end = sampled["indptr"][row:row + 2]
        np.testing.assert_array_equal(sampled["data"][trial_start:trial_end], data[start:end])
        np.testing.assert_array_equal(sampled_x[sampled["indices"][trial_start:trial_end]],
                                      original_x[columns[start:end]])
        np.testing.assert_array_equal(sampled_x[row], original_x[original_row])
        np.testing.assert_array_equal(sampled["inverse"][row], inverse[original_row])
    for actual, expected in zip((pointers, columns, data, inverse), snapshot):
        np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(sampled["data"], data)
    assert not np.shares_memory(sampled["inverse"], inverse)


def test_sampling_caps_matrix_bytes_and_spans_the_workload(monkeypatch):
    monkeypatch.setattr(solver_tuning, "MAX_SAMPLE_ROWS", 100)
    monkeypatch.setattr(solver_tuning, "MAX_SAMPLE_MATRIX_BYTES", 4096)
    rows = solver_tuning.sample_group_indices(10000, 512)
    assert len(rows) == 8
    assert rows[0] == 0 and rows[-1] == 9999
    assert np.all(np.diff(rows) > 0)


def test_auto_preset_preserves_transfer_graph_and_krylov_policy(monkeypatch):
    from hybridge.linalg.multigrid.face_hp import face_pmg_kernel_preset

    monkeypatch.delenv("HYBRIDGE_HP_MG_KERNELS", raising=False)
    monkeypatch.delenv("HDGFEM_HP_MG_KERNELS", raising=False)
    assert face_pmg_kernel_preset() == ("auto", "cupy", False)
    assert face_pmg_kernel_preset("legacy") == ("fused-raw-cuda", "cupy", False)
    assert face_pmg_kernel_preset("packed") == ("packed-raw-cuda", "cupy", False)
    monkeypatch.setenv("HYBRIDGE_HP_MG_KERNELS", "graph")
    assert face_pmg_kernel_preset() == ("packed-raw-cuda", "fused", True)


def test_gemv_cache_hit_has_no_trials_and_keys_precision_shape_reuse(monkeypatch):
    """Resource-only cache preparation must not allocate sampled matrices."""
    from hybridge.runtime.cuda_tuning import fingerprint

    class Kernel:
        attributes = {"max_threads_per_block": 1024, "num_regs": 16, "shared_size_bytes": 0}

    captured = []
    def cached_selection(**kwargs):
        captured.append(kwargs)
        name = kwargs["candidates"][0]
        return name, kwargs["prepare"](name), {"reused": True}

    monkeypatch.setattr(solver_tuning, "device_signature", lambda cp: {})
    monkeypatch.setattr(solver_tuning, "select_variant", cached_selection)
    monkeypatch.setattr(solver_tuning, "sample_group_indices",
                        lambda *args: pytest.fail("cache hit allocated trial inputs"))
    monkeypatch.setenv("HYBRIDGE_FACE_DENSE_THREADS", "128")
    for shape, dtype, reuse in (((10, 3, 9), np.float64, "iteration"),
                               ((11, 3, 9), np.float64, "iteration"),
                               ((10, 3, 9), np.float32, "iteration"),
                               ((10, 3, 9), np.float64, "single-apply")):
        result = solver_tuning.select_gather_gemv(
            np, np.empty(shape, dtype=dtype), None, operation="test", block=3,
            baseline="raw", baseline_source="reference", compile_baseline=Kernel,
            run_baseline=lambda *args: pytest.fail("cache hit ran a trial"), reuse_regime=reuse)
        assert result[:2] == ("raw", 128)
    assert len({fingerprint(entry["identity"]) for entry in captured}) == 4
    assert captured[0]["candidates"] == ["raw:128", "interleaved:128", "coalesced:128"]
    assert all("f32" not in candidate for candidate in captured[0]["candidates"])


def test_bsr_auto_dispatch_is_integrated_and_explicit_backends_bypass_it(monkeypatch):
    from hybridge.linalg.gpu import legendre_face_bsr as bsr

    selected, packed = [], []
    class Implementation:
        def __init__(self, *args, **kwargs):
            self.packed = SimpleNamespace(nbytes=72)
    def choose(operator):
        selected.append(operator)
        return "packed:128", {"source": "measured"}
    def make_packed(operator, *, threads):
        packed.append(threads)
        return Implementation()

    monkeypatch.setattr(bsr, "require_cupy", lambda: np)
    monkeypatch.setattr(bsr, "select_face_bsr_kernels", choose)
    monkeypatch.setattr(bsr, "_PackedFaceBsrOperator", make_packed)
    monkeypatch.setattr(bsr, "_RawFaceBsrOperator", Implementation)
    arguments = (np.array([0, 1], np.int32), np.array([0], np.int32), np.eye(3)[None])
    automatic = bsr.LegendreFaceBsrOperator(*arguments)
    explicit = bsr.LegendreFaceBsrOperator(*arguments, backend="raw-cuda")
    assert selected == [automatic]
    assert packed == [128]
    assert automatic.backend_used == "packed-raw-cuda"
    assert automatic.tuning_report["persistent_layout_bytes"] == 72
    assert explicit.backend_used == "raw-cuda" and explicit.tuning_report is None


def _cupy_or_skip():
    from hybridge.runtime.optional import require_cupy_device
    try:
        return require_cupy_device()
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def tuning_cache(monkeypatch, tmp_path):
    from hybridge.runtime.cuda_tuning import clear_tuning_cache
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING", "auto")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_SECONDS", "30")
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CANDIDATES", "8")
    monkeypatch.delenv("HYBRIDGE_FACE_DENSE_THREADS", raising=False)
    monkeypatch.delenv("HYBRIDGE_FACE_DENSE_GROUP_LANES", raising=False)
    clear_tuning_cache()
    yield
    clear_tuning_cache()


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("block", (3, 7))
def test_face_dense_auto_measures_preserves_inputs_and_reuses_cache(dtype, block, tuning_cache):
    cp = _cupy_or_skip()
    from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator

    rows, slots = 517, 5
    rng = np.random.default_rng(73)
    values = cp.asarray(rng.standard_normal((rows, block, slots * block)), dtype=dtype)
    neighbors = cp.asarray(rng.integers(0, rows, (rows, slots)), dtype=cp.int32)
    neighbors[::3, -1] = -1
    original = values.copy()
    x = cp.asarray(rng.standard_normal(rows * block), dtype=dtype)
    kw = dict(matrix_batches=values, neighbors=neighbors, device_id=int(cp.cuda.runtime.getDevice()))
    reference = CuPyFaceDenseOperator(**kw, implementation="raw_fused")
    automatic = CuPyFaceDenseOperator(**kw, implementation="auto")
    assert automatic.tuning_report["source"] == "measured"
    assert automatic.implementation in {"raw_fused", "coalesced", "interleaved"}
    cp.testing.assert_allclose(automatic.matvec(x), reference.matvec(x),
                              rtol=3e-5 if dtype == np.float32 else 2e-12,
                              atol=3e-5 if dtype == np.float32 else 2e-12)
    assert cp.array_equal(values, original)
    repeated = CuPyFaceDenseOperator(**kw, implementation="auto")
    assert repeated.tuning_report["reused"]
    assert repeated.implementation == automatic.implementation
    assert reference.tuning_report is None


@pytest.mark.parametrize("block", (3, 7))
def test_inverse_preconditioner_auto_preserves_full_application(block, tuning_cache):
    cp = _cupy_or_skip()
    from hybridge.linalg.gpu.preconditioners import (
        CuPyFaceAdditiveSchwarzPreconditioner, CuPyFaceBlockJacobiPreconditioner,
    )

    rows = 517
    rng = np.random.default_rng(21)
    device = int(cp.cuda.runtime.getDevice())
    inverse = cp.asarray(rng.standard_normal((rows, block, block)))
    x = cp.asarray(rng.standard_normal(rows * block))
    reference = CuPyFaceBlockJacobiPreconditioner(inverse, device, application="raw")
    automatic = CuPyFaceBlockJacobiPreconditioner(inverse, device)
    cp.testing.assert_allclose(automatic.apply(x), reference.apply(x), rtol=2e-12, atol=2e-12)
    assert automatic.tuning_report["source"] == "measured"
    assert automatic.application != "interleaved_f32"

    faces = np.column_stack((np.arange(rows - 1), np.arange(1, rows), -np.ones(rows - 1)))
    local_inverse = cp.asarray(rng.standard_normal((rows - 1, 3 * block, 3 * block)))
    kw = dict(inverse_matrices=local_inverse, element_system_faces=cp.asarray(faces, dtype=cp.int32),
              block_size=block, num_system_faces=rows, device_id=device)
    reference = CuPyFaceAdditiveSchwarzPreconditioner(**kw, application="fused")
    automatic = CuPyFaceAdditiveSchwarzPreconditioner(**kw)
    cp.testing.assert_allclose(automatic.apply(x), reference.apply(x), rtol=2e-12, atol=2e-12)
    assert automatic.tuning_report["source"] == "measured"
    assert automatic.application != "interleaved_f32"


@pytest.mark.parametrize("block", (1, 3, 7, 10))
def test_bsr_auto_spmv_and_smoother_match_legacy(block, tuning_cache):
    cp = _cupy_or_skip()
    from test_hp_mg_packed_kernels import _spd_face_bsr
    from hybridge.linalg.gpu.legendre_face_bsr import (
        LegendreFaceBsrOperator, PackedFaceBsrKernels, select_face_bsr_kernels,
    )
    from hybridge.runtime.precision import REAL_DTYPE

    ptr, col, data, diagonal = _spd_face_bsr(517, block, 33)
    data = cp.asarray(data, dtype=REAL_DTYPE)
    args = (cp.asarray(ptr), cp.asarray(col), data)
    reference = LegendreFaceBsrOperator(*args, backend="raw-cuda")
    automatic = LegendreFaceBsrOperator(*args)
    repeated = LegendreFaceBsrOperator(*args)
    x = cp.asarray(np.random.default_rng(2).standard_normal(517 * block), dtype=REAL_DTYPE)
    rhs = cp.asarray(np.random.default_rng(3).standard_normal(517 * block), dtype=REAL_DTYPE)
    inverse = cp.ascontiguousarray(cp.linalg.inv(data[cp.asarray(diagonal)]))
    try:
        cp.testing.assert_allclose(automatic.matvec(x), reference.matvec(x), rtol=2e-12, atol=2e-12)
        assert automatic.tuning_report["source"] == "measured"
        assert repeated.tuning_report["reused"]
        variant, report = select_face_bsr_kernels(reference, inverse)
        assert report["source"] == "measured"
        expected = reference.fused_block_jacobi_step(rhs, x, inverse, .375)
        if variant.startswith("packed:"):
            packed = PackedFaceBsrKernels(reference, inverse, threads=int(variant.split(":")[1]))
            assert cp.array_equal(packed.step(rhs, x, .375, cp.empty_like(rhs)), expected)
        assert reference.tuning_report is None
    finally:
        repeated.close()
        automatic.close()
        reference.close()


@pytest.mark.parametrize("degree", (2, 6))
def test_pmg_auto_preserves_full_cycle_and_true_residual(degree, tuning_cache):
    """Test the full auto hierarchy with fixed recurrence and a true residual."""
    cp = _cupy_or_skip()
    from test_hp_mg_packed_kernels import _cycle
    from hybridge.linalg.multigrid.face_hp import solve_pcg_prototype

    legacy = _cycle(degree, "legacy")
    automatic = _cycle(degree, "auto")
    try:
        rhs = cp.asarray(np.random.default_rng(12).standard_normal(automatic.fine_operator.shape[0]))
        cp.testing.assert_allclose(automatic.apply(rhs), legacy.apply(rhs), rtol=2e-12, atol=2e-13)
        assert automatic.transfer_backend == "cupy" and not automatic.cuda_graphs
        assert automatic.degree_schedule == legacy.degree_schedule
        assert automatic.levels[0].tuning_report is not None
        result = solve_pcg_prototype(automatic.fine_operator, rhs,
                                     preconditioner=automatic, rtol=1e-10, maxiter=200)
        assert result.converged
        assert result.relative_residual <= 1e-10
    finally:
        automatic.close()
        legacy.close()
