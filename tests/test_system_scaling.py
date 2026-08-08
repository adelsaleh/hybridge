from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from hdgfem.linalg import scale_sparse_system


def test_scale_sparse_system_left_matches_jacobi_rows() -> None:
    matrix = scipy.sparse.csr_array([[4.0, 2.0], [1.0, 3.0]])
    rhs = np.array([2.0, 6.0])

    scaled_matrix, scaled_rhs, transform = scale_sparse_system(matrix, rhs, "left")

    np.testing.assert_allclose(scaled_matrix.toarray(), [[1.0, 0.5], [1.0 / 3.0, 1.0]])
    np.testing.assert_allclose(scaled_rhs, [0.5, 2.0])
    assert transform is None


def test_scale_sparse_system_symmetric_preserves_physical_solution() -> None:
    matrix = scipy.sparse.csr_array([[4.0, 2.0], [2.0, 9.0]])
    rhs = np.array([2.0, 7.0])

    scaled_matrix, scaled_rhs, transform = scale_sparse_system(matrix, rhs, "symmetric")
    scaled_solution = np.linalg.solve(scaled_matrix.toarray(), scaled_rhs)

    assert transform is not None
    np.testing.assert_allclose(transform * scaled_solution, np.linalg.solve(matrix.toarray(), rhs))
    np.testing.assert_allclose(scaled_matrix.diagonal(), np.ones(2))


def test_scale_sparse_system_rejects_nonpositive_symmetric_diagonal() -> None:
    matrix = scipy.sparse.eye(2, format="csr")
    matrix[1, 1] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        scale_sparse_system(matrix, np.ones(2), "symmetric")
