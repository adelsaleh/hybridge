"""Parallel COO-to-CSR pattern kernels and the pattern-reusing PARDISO solver (small matrices only)."""
import numpy as np
import pytest
import scipy.sparse

from hdgfem.linalg import sparse_pattern as sp_


def coo(size, count, *, long_row=False, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = rng.integers(0, size, count), rng.integers(0, size, count)
    if long_row:
        rows[:count//2] = 3     # longer than the insertion-sort limit
    rows = np.concatenate([rows, np.arange(size)])      # every row nonempty
    cols = np.concatenate([cols, np.arange(size)])
    return rows, cols, rng.standard_normal(rows.size)


@pytest.mark.parametrize('size,count,long_row', [(40, 2000, False), (500, 30000, True), (5, 0, False)])
def test_pattern_matches_numpy_scipy_and_bincount_bitwise(size, count, long_row):
    rows, cols, data = coo(size, count, long_row=long_row)
    pattern = sp_.build_coo_csr_pattern(rows, cols, size, chunks=7)
    reference = sp_._build_coo_csr_pattern_numpy(rows, cols, size, sp_.index_dtype(rows.size + 1))
    for actual, expected in zip(pattern, reference):
        np.testing.assert_array_equal(actual, expected)
    indptr, indices, order, segments = pattern
    matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    matrix.sum_duplicates()
    np.testing.assert_array_equal(indptr, matrix.indptr)
    np.testing.assert_array_equal(indices, matrix.indices)
    values, nonfinite = sp_.csr_values_from_coo(data, order, segments)
    entry = np.empty(order.size, dtype=np.int64)
    entry[order] = np.repeat(np.arange(indices.size), np.diff(segments))
    assert nonfinite == 0
    np.testing.assert_array_equal(values, np.bincount(entry, weights=data, minlength=indices.size))
    x, rhs = np.linspace(-1, 1, size), np.cos(np.arange(size))
    residual = sp_.csr_residual_norm(indptr.astype(np.int32) + 1, indices.astype(np.int32) + 1, values, x, rhs, 1)
    assert residual == pytest.approx(np.linalg.norm(matrix @ x - rhs), rel=1e-12)


def test_pattern_errors_mismatches_and_nonfinite_counts():
    with pytest.raises(ValueError, match=r'\[0, 4\)'):
        sp_.build_coo_csr_pattern(np.array([0, 4]), np.array([0, 1]), 4)
    rows, cols, data = coo(30, 400)
    assert sp_.coo_pattern_mismatches(rows, cols, rows.astype(np.int32), cols.astype(np.int32)) == 0
    changed = cols.copy()
    changed[[3, 17]] = (changed[[3, 17]] + 1) % 30
    assert sp_.coo_pattern_mismatches(rows, changed, rows, cols) == 2
    _, _, order, segments = sp_.build_coo_csr_pattern(rows, cols, 30)
    data[5] = np.nan
    assert sp_.csr_values_from_coo(data, order, segments)[1] == 1


def test_reusable_pardiso_reuses_analysis_and_detects_pattern_changes():
    pytest.importorskip('pypardiso')
    from hdgfem.linalg.pardiso_runtime import ReusablePardisoSolver

    size = 120
    matrix = (scipy.sparse.random(size, size, density=.05, random_state=3) + 6*scipy.sparse.eye(size)).tocoo()
    rows = np.concatenate([matrix.row, matrix.row[:40]])       # duplicates summed (zero weights)
    cols = np.concatenate([matrix.col, matrix.col[:40]])
    data = np.concatenate([matrix.data, np.zeros(40)])
    rhs = np.sin(np.arange(size, dtype=float))
    solver = ReusablePardisoSolver(threads=2)
    try:
        first = solver.solve_coo(rows, cols, data, rhs, size)
        second = solver.solve_coo(rows, cols, 2*data, rhs, size)
        assert not first.pardiso_analysis_reused and second.pardiso_analysis_reused and solver.analysis_count == 1
        np.testing.assert_allclose(2*second.x, first.x, rtol=1e-12)
        np.testing.assert_allclose(matrix.tocsr() @ first.x, rhs, atol=1e-12)
        third = solver.solve_coo(rows[:-1], cols[:-1], data[:-1], rhs, size)
        assert not third.pardiso_analysis_reused and solver.analysis_count == 2
        with pytest.raises(ValueError, match='non-finite'):
            solver.solve_coo(rows, cols, np.where(np.arange(rows.size) == 2, np.inf, data), rhs, size)
        empty = rows != 7
        with pytest.raises(ValueError, match='empty row'):
            solver.solve_coo(rows[empty], cols[empty], data[empty], rhs, size)
    finally:
        solver.close()
