"""AMGX borrowing a scalar device CSR (``attach_CSR`` of the CUDA-13 pyamgx/AMGX fork).

The qualification check runs with NumPy here; the borrowed solve itself needs CuPy and pyamgx
and is skipped without them.
"""
from __future__ import annotations

import numpy as np
import pytest

from hybridge.linalg.amgx.device_solver import _csr_rows_are_canonical_with_diagonal


def _laplacian(n):
    indptr = [0]
    indices, data = [], []
    for i in range(n):
        for j, v in ((i-1, -1.), (i, 2.), (i+1, -1.)):
            if 0 <= j < n:
                indices.append(j)
                data.append(v)
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32), np.array(data)


def test_canonical_csr_with_diagonal_qualifies():
    indptr, indices, _ = _laplacian(6)
    assert _csr_rows_are_canonical_with_diagonal(np, indptr, indices, 6) == (True, None)


def test_unsorted_duplicate_and_missing_diagonal_rows_are_refused():
    indptr, indices, _ = _laplacian(6)
    unsorted = indices.copy()
    unsorted[1], unsorted[2] = unsorted[2], unsorted[1]
    assert _csr_rows_are_canonical_with_diagonal(np, indptr, unsorted, 6)[1] == 'columns are not strictly increasing within rows'
    duplicate = indices.copy()
    duplicate[2] = duplicate[1]
    assert _csr_rows_are_canonical_with_diagonal(np, indptr, duplicate, 6)[1] == 'columns are not strictly increasing within rows'
    off = indices.copy()
    off[0] = 1  # row 0 keeps two entries, columns 1 and 1 -> duplicate; make it (1, 2) instead
    off[1] = 2
    assert _csr_rows_are_canonical_with_diagonal(np, indptr, off, 6)[1] == 'a row lacks an explicit diagonal entry'
    assert _csr_rows_are_canonical_with_diagonal(np, indptr[:-1], indices, 6)[1] == 'indptr does not describe the matrix'


@pytest.mark.skipif(pytest.importorskip('cupy', reason='CuPy') is None, reason='CuPy')
def test_borrowed_csr_solve_matches_uploaded_solve():
    cp = pytest.importorskip('cupy')
    sparse = pytest.importorskip('cupyx.scipy.sparse')
    pytest.importorskip('pyamgx')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # no CUDA device
        pytest.skip('no CUDA device')
    from hybridge.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver
    n = 300
    indptr, indices, data = _laplacian(n)
    matrix = sparse.csr_matrix((cp.asarray(data), cp.asarray(indices), cp.asarray(indptr)), shape=(n, n))
    rhs = cp.asarray(np.random.default_rng(3).standard_normal(n))
    config = {'config_version': 2, 'solver': {'solver': 'PCG', 'preconditioner': {'solver': 'JACOBI_L1'},
                                              'convergence': 'RELATIVE_INI_CORE', 'tolerance': 1e-12,
                                              'max_iters': 500, 'monitor_residual': 1}}
    results = {}
    for borrow in (False, True):
        solver = PyAMGXCsrDeviceSolver(config=config, tolerance=1e-12, maxiter=500, borrow_csr=borrow)
        try:
            solver.setup(matrix)
            assert solver.csr_borrowed is borrow, solver.borrow_note
            assert solver.mat.is_attached is borrow  # pyamgx exposes it as a property
            x, info = solver.solve(rhs)
            results[borrow] = cp.asnumpy(x)
            if borrow:
                # New values in the borrowed buffer: AMGX repeats its setup and solves the scaled system.
                scaled = sparse.csr_matrix((matrix.data*2., matrix.indices, matrix.indptr), shape=(n, n))
                solver.replace_coefficients(scaled)
                assert solver.setup_count == 2
                y, _ = solver.solve(rhs)
                assert np.allclose(cp.asnumpy(y), results[True]/2., rtol=0, atol=1e-10)
        finally:
            solver.close()
        assert solver.closed and not solver.csr_borrowed
    assert np.allclose(results[True], results[False], rtol=0, atol=1e-12)
    # A non-canonical matrix is uploaded, not borrowed.
    shuffled = sparse.csr_matrix((cp.asarray(data), cp.asarray(indices[::-1].copy()), cp.asarray(indptr)), shape=(n, n))
    solver = PyAMGXCsrDeviceSolver(config=config, tolerance=1e-12, maxiter=500, borrow_csr=True)
    try:
        solver.setup(shuffled)
        assert not solver.csr_borrowed and solver.borrow_note.startswith('uploaded:')
    finally:
        solver.close()


@pytest.mark.skipif(pytest.importorskip('cupy', reason='CuPy') is None, reason='CuPy')
def test_borrowed_bsr_solve_matches_uploaded_solve():
    """Borrowed square-block BSR: needs the extended fork (``attach_BSR``); otherwise the solver
    uploads, says why, and this test skips. JACOBI_L1 is scalar-only in the fork, so the block
    systems use a block-capable preconditioner."""
    cp = pytest.importorskip('cupy')
    pytest.importorskip('pyamgx')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # no CUDA device
        pytest.skip('no CUDA device')
    from types import SimpleNamespace
    from hybridge.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver
    n = 300
    rhs = cp.asarray(np.random.default_rng(3).standard_normal(n))
    bsr_config = {'config_version': 2, 'solver': {'solver': 'PBICGSTAB', 'preconditioner': {'solver': 'BLOCK_JACOBI'},
                                                  'convergence': 'RELATIVE_INI_CORE', 'tolerance': 1e-12,
                                                  'max_iters': 500, 'monitor_residual': 1}}
    for b in (2, 3):
        nb = n//b
        bindptr, bindices, _ = _laplacian(nb)
        nnz = bindices.size
        blocks = np.zeros((nnz, b, b))
        for k, (i, j) in enumerate(zip(np.repeat(np.arange(nb), np.diff(bindptr)), bindices)):
            blocks[k] = (4.*np.eye(b)+0.5*np.ones((b, b))) if i == j else -np.eye(b)
        bsr = SimpleNamespace(indptr=cp.asarray(bindptr), indices=cp.asarray(bindices), data=cp.asarray(blocks),
                              shape=(nb*b, nb*b), block_size=b)
        brhs = rhs[:nb*b]
        bsr_results = {}
        for borrow in (False, True):
            solver = PyAMGXCsrDeviceSolver(config=bsr_config, tolerance=1e-12, maxiter=500, borrow_csr=borrow)
            try:
                solver.setup(bsr)
                if borrow and not solver.csr_borrowed:
                    pytest.skip(f'borrowed BSR unavailable: {solver.borrow_note}')
                assert solver.csr_borrowed is borrow, solver.borrow_note
                assert solver.mat.is_attached is borrow
                x, _ = solver.solve(brhs)
                bsr_results[borrow] = cp.asnumpy(x)
                if borrow:
                    scaled = SimpleNamespace(indptr=bsr.indptr, indices=bsr.indices, data=bsr.data*2.,
                                             shape=bsr.shape, block_size=b)
                    solver.replace_coefficients(scaled)
                    assert solver.setup_count == 2
                    y, _ = solver.solve(brhs)
                    assert np.allclose(cp.asnumpy(y), bsr_results[True]/2., rtol=0, atol=1e-10)
            finally:
                solver.close()
            assert solver.closed and not solver.csr_borrowed
        assert np.allclose(bsr_results[True], bsr_results[False], rtol=0, atol=1e-12), f'block size {b}'


def test_one_shot_borrowed_csr_preserves_ownership_report(monkeypatch):
    """The public wrapper forwards borrowing and reports it after native cleanup."""
    cp = pytest.importorskip('cupy')
    sparse = pytest.importorskip('cupyx.scipy.sparse')
    pytest.importorskip('pyamgx')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('no CUDA device')
    except Exception:
        pytest.skip('no CUDA device')
    from types import SimpleNamespace
    import hybridge.linalg.amgx.device_solver as runtime
    n = 16
    indptr, indices, data = _laplacian(n)
    matrix = sparse.csr_matrix((cp.asarray(data), cp.asarray(indices), cp.asarray(indptr)), shape=(n, n))
    rhs = matrix @ cp.ones(n)
    config = {'config_version': 2, 'solver': {'solver': 'PCG',
              'preconditioner': {'solver': 'JACOBI_L1'}, 'max_iters': 100,
              'monitor_residual': 1, 'tolerance': 1e-12, 'convergence': 'RELATIVE_INI_CORE'}}
    original = runtime._pyamgx_solve_csr_device
    reports = []

    def capture(*args, **kwargs):
        x, report = original(*args, **kwargs)
        reports.append(report)
        return x, report

    monkeypatch.setattr(runtime, '_pyamgx_solve_csr_device', capture)
    assembly = SimpleNamespace(matrix_format='csr', data=matrix.data, indices=matrix.indices,
                               indptr=matrix.indptr, rhs=rhs, shape=matrix.shape)
    result, x = runtime.solve_reduced_system_amgx_device(
        assembly, config=config, tolerance=1e-12, scale_system=False,
        materialize_host_solution=False, borrow_csr=True)
    assert reports[0]['amgx_csr_borrowed'] is True
    assert result.info == 0
    assert float(cp.linalg.norm(matrix @ x - rhs).get()) < 1e-10
