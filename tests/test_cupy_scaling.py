from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from hybridge.linalg.gpu.sparse import (
    diagonal_scale_cupy_csr_rows_in_place,
    symmetric_scale_cupy_csr_in_place,
)


pytest.importorskip("cupy")
pytest.importorskip("cupyx")


def _cupy_modules():
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    return cp, cpsp


def _reference_system():
    dense = np.array(
        [
            [4.0, -1.0, 0.5, 0.0],
            [-1.0, 9.0, 2.0, 0.25],
            [0.5, 2.0, 16.0, -3.0],
            [0.0, 0.25, -3.0, 25.0],
        ],
        dtype=np.float64,
    )
    rhs = np.array([1.5, -2.0, 3.0, 4.0], dtype=np.float64)
    return sp.csr_matrix(dense), rhs


def test_symmetric_scale_cupy_csr_matches_numpy_reference():
    cp, cpsp = _cupy_modules()
    matrix_host, rhs_host = _reference_system()
    matrix = cpsp.csr_matrix(matrix_host)
    rhs = cp.asarray(rhs_host)

    inverse_sqrt_diagonal = symmetric_scale_cupy_csr_in_place(matrix, rhs)
    cp.cuda.get_current_stream().synchronize()

    scale = 1.0 / np.sqrt(np.diag(matrix_host.toarray()))
    expected_matrix = (scale[:, None] * matrix_host.toarray()) * scale[None, :]
    expected_rhs = scale * rhs_host
    np.testing.assert_allclose(cp.asnumpy(inverse_sqrt_diagonal), scale, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(matrix.get().toarray(), expected_matrix, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(cp.asnumpy(rhs), expected_rhs, rtol=1e-15, atol=1e-15)

    scaled_solution = np.array([0.2, -0.3, 0.7, 1.1], dtype=np.float64)
    physical_solution = scale * scaled_solution
    scaled_residual = expected_matrix @ scaled_solution - expected_rhs
    physical_residual_from_scaled = scaled_residual / scale
    physical_residual = matrix_host.toarray() @ physical_solution - rhs_host
    np.testing.assert_allclose(physical_residual_from_scaled, physical_residual, rtol=1e-15, atol=1e-15)


def test_left_scale_cupy_csr_matches_numpy_reference():
    cp, cpsp = _cupy_modules()
    matrix_host, rhs_host = _reference_system()
    matrix = cpsp.csr_matrix(matrix_host)
    rhs = cp.asarray(rhs_host)

    diagonal = diagonal_scale_cupy_csr_rows_in_place(matrix, rhs)
    cp.cuda.get_current_stream().synchronize()

    diagonal_host = np.diag(matrix_host.toarray())
    np.testing.assert_allclose(cp.asnumpy(diagonal), diagonal_host, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(matrix.get().toarray(), matrix_host.toarray() / diagonal_host[:, None], rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(cp.asnumpy(rhs), rhs_host / diagonal_host, rtol=1e-15, atol=1e-15)


def _weak_diagonal_system():
    dense = np.array(
        [
            [4.0, -1.0, 0.0],
            [2.0, 1.0e-14, -5.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rhs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return sp.csr_matrix(dense), rhs


def test_left_scale_falls_back_to_row_max_for_tiny_or_empty_diagonal():
    """A tiny diagonal uses the row maximum and an all-zero row is unscaled."""
    cp, cpsp = _cupy_modules()
    matrix_host, rhs_host = _weak_diagonal_system()
    matrix = cpsp.csr_matrix(matrix_host)
    rhs = cp.asarray(rhs_host)

    diagonal = diagonal_scale_cupy_csr_rows_in_place(matrix, rhs)
    cp.cuda.get_current_stream().synchronize()

    expected = np.array([4.0, 5.0, 1.0])
    np.testing.assert_array_equal(cp.asnumpy(diagonal), expected)
    np.testing.assert_allclose(matrix.get().toarray(), matrix_host.toarray() / expected[:, None], rtol=1e-15)
    np.testing.assert_allclose(cp.asnumpy(rhs), rhs_host / expected, rtol=1e-15)


def test_symmetric_scale_falls_back_to_row_max_for_tiny_or_empty_diagonal():
    """Symmetric scaling uses the same robust diagonal estimate as left scaling."""
    cp, cpsp = _cupy_modules()
    matrix_host, rhs_host = _weak_diagonal_system()
    matrix = cpsp.csr_matrix(matrix_host)
    rhs = cp.asarray(rhs_host)

    inverse_sqrt_diagonal = symmetric_scale_cupy_csr_in_place(matrix, rhs)
    cp.cuda.get_current_stream().synchronize()

    scale = 1.0 / np.sqrt(np.array([4.0, 5.0, 1.0]))
    np.testing.assert_allclose(cp.asnumpy(inverse_sqrt_diagonal), scale, rtol=1e-15)
    assert np.isfinite(matrix.get().toarray()).all()
    np.testing.assert_allclose(cp.asnumpy(rhs), scale * rhs_host, rtol=1e-15)
