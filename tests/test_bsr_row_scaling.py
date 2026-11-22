"""Face-BSR left row scaling: every launch shape and p = 1..9 against an exact NumPy reference."""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    cp.cuda.runtime.getDeviceCount()
except Exception:  # pragma: no cover - host-only machines
    pytest.skip("no CUDA device", allow_module_level=True)

from hybridge.linalg.gpu import sparse as gpu_sparse
from hybridge.linalg.gpu.sparse import _DeviceBsrMatrixView


def _face_bsr(order, block_rows=97, seed=0):
    """A face-BSR pattern like an HDG trace system: the diagonal block plus 2-4 neighbours per row.

    Row 3 has a diagonal ten orders below its row maximum (row-maximum fallback) and one row of block row 5 is zero
    (unit fallback), the two branches of the scaling rule.
    """
    rng = np.random.default_rng(seed + order)
    b = order + 1
    indptr, indices = [0], []
    for row in range(block_rows):
        neighbours = rng.choice([c for c in range(block_rows) if c != row], size=int(rng.integers(2, 5)),
                                replace=False)
        columns = sorted([row, *neighbours.tolist()])
        indices += columns
        indptr.append(len(indices))
    data = rng.standard_normal((len(indices), b, b))
    for row in range(block_rows):
        for k in range(indptr[row], indptr[row + 1]):
            if indices[k] == row:
                data[k] += np.eye(b) * (4.0 + b)
    diag3 = next(k for k in range(indptr[3], indptr[4]) if indices[k] == 3)
    data[diag3, 1, 1] = 1.0e-14
    data[indptr[5]:indptr[6], min(2, b - 1), :] = 0.0
    return np.asarray(indptr, np.int32), np.asarray(indices, np.int32), data


def _reference(indptr, indices, data, rhs):
    """The rule of the scalar CSR scaling, with the kernel's summation order."""
    b = data.shape[1]
    data, rhs = data.copy(), rhs.copy()
    diagonal = np.empty(rhs.size)
    for block_row in range(indptr.size - 1):
        for r in range(b):
            value_sum, row_max = 0.0, 0.0
            for k in range(indptr[block_row], indptr[block_row + 1]):
                for c in range(b):
                    entry = data[k, r, c]
                    row_max = max(row_max, abs(entry))
                    if indices[k] == block_row and c == r:
                        value_sum += entry
            value = value_sum
            if not np.isfinite(value) or abs(value) <= 1.0e-10 * row_max:
                value = row_max
            if not np.isfinite(value) or value == 0.0:
                value = 1.0
            row = block_row * b + r
            diagonal[row] = value
            rhs[row] /= value
            inverse = 1.0 / value
            data[indptr[block_row]:indptr[block_row + 1], r, :] *= inverse
    return data, rhs, diagonal


@pytest.mark.parametrize("order", range(1, 10))
@pytest.mark.parametrize("shape", gpu_sparse._BSR_SCALE_SHAPES, ids=gpu_sparse._bsr_scale_shape_name)
def test_every_launch_shape_scales_and_restores_bitwise(order, shape):
    indptr, indices, data = _face_bsr(order)
    b = order + 1
    rhs = np.linspace(-2.0, 3.0, (indptr.size - 1) * b)
    expected_data, expected_rhs, expected_diagonal = _reference(indptr, indices, data, rhs)

    d_indptr, d_indices, d_data, d_rhs = map(cp.asarray, (indptr, indices, data, rhs))
    d_diagonal = cp.empty(rhs.size)
    gpu_sparse._launch_bsr_row_scale(shape, d_indptr, d_indices, d_data, d_rhs, d_diagonal,
                                     indptr.size - 1, b)
    np.testing.assert_array_equal(cp.asnumpy(d_diagonal), expected_diagonal)
    np.testing.assert_array_equal(cp.asnumpy(d_rhs), expected_rhs)
    np.testing.assert_array_equal(cp.asnumpy(d_data), expected_data)

    grid, block = gpu_sparse._bsr_scale_launch(indptr.size - 1, shape)
    _, restore = gpu_sparse._bsr_row_scale_kernels(shape)
    restore(grid, block, (d_indptr, d_data, d_diagonal, np.int32(indptr.size - 1), np.int32(b)))
    restored = expected_data.copy()
    for block_row in range(indptr.size - 1):
        for r in range(b):
            restored[indptr[block_row]:indptr[block_row + 1], r, :] *= expected_diagonal[block_row * b + r]
    np.testing.assert_array_equal(cp.asnumpy(d_data), restored)


@pytest.mark.parametrize("order", range(1, 10))
def test_tuned_public_path_matches_the_reference(order, monkeypatch, tmp_path):
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    indptr, indices, data = _face_bsr(order, block_rows=513, seed=7)
    b = order + 1
    rhs = np.cos(np.arange((indptr.size - 1) * b))
    expected_data, expected_rhs, expected_diagonal = _reference(indptr, indices, data, rhs)
    matrix = _DeviceBsrMatrixView(data=cp.asarray(data), indices=cp.asarray(indices),
                                  indptr=cp.asarray(indptr), shape=(rhs.size, rhs.size), block_size=b)
    d_rhs = cp.asarray(rhs)
    diagonal = gpu_sparse._diagonal_scale_bsr_rows_in_place(matrix, d_rhs)
    assert gpu_sparse._bsr_row_scale_choice(matrix) in gpu_sparse._BSR_SCALE_SHAPES
    np.testing.assert_array_equal(cp.asnumpy(diagonal), expected_diagonal)
    np.testing.assert_array_equal(cp.asnumpy(d_rhs), expected_rhs)
    np.testing.assert_array_equal(cp.asnumpy(matrix.data), expected_data)
    gpu_sparse._restore_left_scaled_bsr_rows_in_place(matrix, diagonal)
    np.testing.assert_allclose(cp.asnumpy(matrix.data), data, rtol=4.0e-16 * 4, atol=0.0)


def test_block_sizes_beyond_32_are_rejected():
    matrix = _DeviceBsrMatrixView(data=cp.zeros((1, 33, 33)), indices=cp.zeros(1, np.int32),
                                  indptr=cp.asarray([0, 1], np.int32), shape=(33, 33), block_size=33)
    with pytest.raises(ValueError, match="block sizes 1..32"):
        gpu_sparse._diagonal_scale_bsr_rows_in_place(matrix, cp.zeros(33))


@pytest.mark.parametrize("sample_bytes", [None, 1])
def test_tuning_never_downloads_a_full_array(sample_bytes, monkeypatch, tmp_path):
    """The device AMGX path promises no full-array downloads; the tuner reads only scalars.

    ``sample_bytes=1`` shrinks the trial sample to its one-block-row minimum.
    """
    monkeypatch.setenv("HYBRIDGE_CUDA_TUNING_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(gpu_sparse, "_BSR_ROW_SCALE_CHOICES", {})
    if sample_bytes is not None:
        monkeypatch.setattr(gpu_sparse, "_BSR_SCALE_SAMPLE_BYTES", sample_bytes)
    indptr, indices, data = _face_bsr(6, block_rows=257, seed=3)
    rhs = np.sin(np.arange((indptr.size - 1) * 7))
    expected_data, expected_rhs, _ = _reference(indptr, indices, data, rhs)
    matrix = _DeviceBsrMatrixView(data=cp.asarray(data), indices=cp.asarray(indices),
                                  indptr=cp.asarray(indptr), shape=(rhs.size, rhs.size), block_size=7)
    d_rhs = cp.asarray(rhs)

    def no_download(*args, **kwargs):
        raise AssertionError("BSR row scaling downloaded a full array")

    with monkeypatch.context() as patched:
        patched.setattr(cp, "asnumpy", no_download)
        gpu_sparse._diagonal_scale_bsr_rows_in_place(matrix, d_rhs)  # cache miss: tunes
        gpu_sparse._BSR_ROW_SCALE_CHOICES.clear()
        assert gpu_sparse._bsr_row_scale_choice(matrix) in gpu_sparse._BSR_SCALE_SHAPES  # disk hit
    np.testing.assert_array_equal(cp.asnumpy(d_rhs), expected_rhs)
    np.testing.assert_array_equal(cp.asnumpy(matrix.data), expected_data)
