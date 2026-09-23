"""CPU checks of block-preserving BSR principal submatrix extraction."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import bsr_matrix, csr_matrix

from hdgfem.linalg.bsr import principal_bsr_submatrix


def _matrix(block_size=2, dtype=np.float64):
    # Include asymmetric graph edges, an empty block row, a zero diagonal
    # block, and a zero off-diagonal block to distinguish structure from values.
    indices = np.array([0, 1, 3, 0, 1, 2, 1, 2, 4, 0, 3, 4], dtype=np.int32)
    indptr = np.array([0, 3, 6, 9, 12, 12], dtype=np.int32)
    rng = np.random.default_rng(971 + block_size)
    data = rng.standard_normal((indices.size, block_size, block_size))
    if np.issubdtype(dtype, np.complexfloating):
        data = data + 1j * rng.standard_normal(data.shape)
    data = data.astype(dtype)
    data[1] = 0
    data[7] = 0
    return bsr_matrix((data, indices, indptr), shape=(5 * block_size, 5 * block_size))


@pytest.mark.parametrize("block_size", [2, 7])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.parametrize("selected", [[3, 1, 0], [4, 2, 1], [2], []])
def test_principal_bsr_preserves_values_order_and_structural_blocks(block_size, dtype, selected):
    matrix = _matrix(block_size, dtype)
    original_data = matrix.data.copy()
    original_indices = matrix.indices.copy()
    original_indptr = matrix.indptr.copy()
    actual = principal_bsr_submatrix(matrix, selected)
    scalar_rows = (
        np.asarray(selected, dtype=np.int64)[:, None] * block_size + np.arange(block_size)
    ).ravel()
    expected = matrix.toarray()[np.ix_(scalar_rows, scalar_rows)]
    np.testing.assert_array_equal(actual.toarray(), expected)
    assert actual.blocksize == (block_size, block_size)
    assert actual.dtype == matrix.dtype
    assert actual.has_canonical_format

    # The explicit block-level oracle retains zero-valued stored blocks.
    old_to_new = {old: new for new, old in enumerate(selected)}
    expected_data = []
    expected_indices = []
    expected_indptr = [0]
    for old_row in selected:
        entries = []
        for slot in range(matrix.indptr[old_row], matrix.indptr[old_row + 1]):
            old_column = int(matrix.indices[slot])
            if old_column in old_to_new:
                entries.append((old_to_new[old_column], matrix.data[slot]))
        for new_column, block in sorted(entries, key=lambda item: item[0]):
            expected_indices.append(new_column)
            expected_data.append(block)
        expected_indptr.append(len(expected_indices))
    np.testing.assert_array_equal(actual.indices, expected_indices)
    np.testing.assert_array_equal(actual.indptr, expected_indptr)
    np.testing.assert_array_equal(
        actual.data, np.asarray(expected_data, dtype=dtype).reshape(-1, block_size, block_size)
    )
    np.testing.assert_array_equal(matrix.data, original_data)
    np.testing.assert_array_equal(matrix.indices, original_indices)
    np.testing.assert_array_equal(matrix.indptr, original_indptr)
    assert not np.shares_memory(actual.data, matrix.data)
    assert not np.shares_memory(actual.indices, matrix.indices)
    assert not np.shares_memory(actual.indptr, matrix.indptr)


def test_principal_bsr_uses_no_scalar_sparse_conversion(monkeypatch):
    matrix = _matrix(7)

    def forbidden(*args, **kwargs):
        raise AssertionError("scalar sparse/dense conversion is forbidden")

    monkeypatch.setattr(bsr_matrix, "tocsr", forbidden)
    monkeypatch.setattr(bsr_matrix, "tocoo", forbidden)
    monkeypatch.setattr(bsr_matrix, "toarray", forbidden)
    result = principal_bsr_submatrix(matrix, np.array([3, 0, 1], dtype=np.uint64))
    assert result.shape == (21, 21)
    assert result.blocksize == (7, 7)
    assert result.has_canonical_format


@pytest.mark.parametrize(
    ("selected", "exception", "message"),
    [
        ([1, 1], ValueError, "duplicates"),
        ([-1], ValueError, "out-of-range"),
        ([5], ValueError, "out-of-range"),
        (np.array([2**64 - 1], dtype=np.uint64), ValueError, "out-of-range"),
        ([[0, 1]], ValueError, "one-dimensional"),
        ([1.0], TypeError, "integer block row IDs"),
        ([True], TypeError, "integer block row IDs"),
    ],
)
def test_principal_bsr_rejects_invalid_selection(selected, exception, message):
    with pytest.raises(exception, match=message):
        principal_bsr_submatrix(_matrix(), selected)


@pytest.mark.parametrize("defect", ["csr", "nonsquare", "rectangular_blocks", "unsorted", "duplicate"])
def test_principal_bsr_rejects_incompatible_storage(defect):
    matrix = _matrix()
    exception = ValueError
    if defect == "csr":
        matrix = csr_matrix(np.eye(6))
        exception = TypeError
        message = "SciPy BSR"
    elif defect == "nonsquare":
        matrix = bsr_matrix(np.ones((4, 6)), blocksize=(2, 2))
        message = "must be square"
    elif defect == "rectangular_blocks":
        matrix = bsr_matrix(np.ones((6, 6)), blocksize=(2, 3))
        message = "must be square"
    elif defect == "unsorted":
        indices = matrix.indices.copy()
        data = matrix.data.copy()
        indices[:3] = indices[:3][::-1]
        data[:3] = data[:3][::-1]
        matrix = bsr_matrix((data, indices, matrix.indptr.copy()), shape=matrix.shape)
        message = "sorted BSR indices"
    else:
        indptr = matrix.indptr.copy()
        indptr[1:] += 1
        matrix = bsr_matrix(
            (
                np.concatenate((matrix.data[:1], matrix.data)),
                np.concatenate((matrix.indices[:1], matrix.indices)),
                indptr,
            ),
            shape=matrix.shape,
        )
        message = "sorted BSR indices"
    with pytest.raises(exception, match=message):
        principal_bsr_submatrix(matrix, [0])


def test_principal_bsr_full_identity_selection_is_independent_copy():
    matrix = _matrix(7)
    actual = principal_bsr_submatrix(matrix, np.arange(5))
    np.testing.assert_array_equal(actual.data, matrix.data)
    np.testing.assert_array_equal(actual.indices, matrix.indices)
    np.testing.assert_array_equal(actual.indptr, matrix.indptr)
    actual.data.fill(0)
    assert np.any(matrix.data != 0)
