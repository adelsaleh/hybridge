from __future__ import annotations

import numpy as np
import pytest

from hdgfem.backends.cublas_batched import invert_batched_cublas
from hdgfem.runtime.optional import require_cupy_device


def _cupy_or_skip():
    try:
        cp = require_cupy_device()
    except RuntimeError as error:
        pytest.skip(str(error))
    if bool(getattr(cp.cuda.runtime, "is_hip", False)):
        pytest.skip("explicit cuBLAS tests require an NVIDIA CUDA build")
    return cp


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cublas_batched_inverse_matches_numpy(dtype: type) -> None:
    cp = _cupy_or_skip()
    rng = np.random.default_rng(20260724)
    matrices = rng.standard_normal((7, 5, 5)).astype(dtype)
    matrices += 6.0 * np.eye(5, dtype=dtype)[None, :, :]
    matrices_gpu = cp.asarray(matrices, order="C")
    original = matrices_gpu.copy()

    result = invert_batched_cublas(matrices_gpu, label="test batch")
    cp.cuda.get_current_stream().synchronize()

    tolerance = 2.0e-5 if dtype is np.float32 else 2.0e-13
    np.testing.assert_allclose(
        cp.asnumpy(result.inverse_matrices),
        np.linalg.inv(matrices),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(cp.asnumpy(matrices_gpu), cp.asnumpy(original))
    np.testing.assert_array_equal(result.factorization_info, 0)
    np.testing.assert_array_equal(result.inversion_info, 0)
    assert result.factorization_info.dtype == np.int32
    assert result.inversion_info.dtype == np.int32
    assert result.maximum_inverse_residual < (3.0e-4 if dtype is np.float32 else 2.0e-12)


def test_cublas_batched_inverse_reports_singular_batch() -> None:
    cp = _cupy_or_skip()
    matrices = np.array(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[1.0, 2.0], [2.0, 4.0]],
            [[3.0, -1.0], [0.5, 2.0]],
        ],
        dtype=np.float64,
    )
    with pytest.raises(np.linalg.LinAlgError, match="batch 1"):
        invert_batched_cublas(cp.asarray(matrices), label="singular test")


def test_cublas_batched_inverse_rejects_noncontiguous_input() -> None:
    cp = _cupy_or_skip()
    matrices = cp.eye(4, dtype=cp.float64)[None, :, :]
    noncontiguous = cp.broadcast_to(matrices, (3, 4, 4))
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        invert_batched_cublas(noncontiguous)


def test_cublas_batched_inverse_rejects_integer_input() -> None:
    cp = _cupy_or_skip()
    with pytest.raises(TypeError, match="float32 or float64"):
        invert_batched_cublas(cp.eye(3, dtype=cp.int32)[None, :, :])
