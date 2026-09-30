"""Independent host checks for the hierarchy storage experiment (no compilation)."""
import json

import numpy as np
import pytest
from scipy import sparse

from hdgfem.linalg.multigrid.hierarchy_bsr import (
    load_operator,
    padded_bsr,
    permute_operator,
    verify_reconstruction,
    storage_stats,
    level_permutation,
    deterministic_vectors,
    compare_product,
)


@pytest.mark.parametrize('shape', [(9, 5), (5, 9), (9, 9)])
@pytest.mark.parametrize('block', [2, 4, 7, 8])
def test_rectangular_roundtrip_and_products(shape, block):
    rng = np.random.default_rng(17)
    dense = rng.integers(-3, 4, size=shape).astype(float)
    dense[rng.random(shape) < .65] = 0
    a = sparse.csr_matrix(dense)
    a.data[0] = 0  # Explicit structural zero must survive.
    rp, cp = rng.permutation(shape[0]), rng.permutation(shape[1])
    permuted = permute_operator(a, rp, cp)
    b = padded_bsr(permuted, block)
    verify_reconstruction(a, b, rp, cp)
    expected = a.toarray()[np.ix_(rp, cp)]
    np.testing.assert_array_equal(b.toarray()[:shape[0], :shape[1]], expected)
    for x in deterministic_vectors(shape[1]):
        actual = (b @ np.pad(x, (0, b.shape[1]-len(x))))[:shape[0]]
        compare_product(permuted, x, actual)
    st = storage_stats(permuted, b)
    assert st['original_explicit_zeros'] == np.count_nonzero(a.data == 0)
    assert st['occupied_block_entries'] == (st['csr_entries'] + st['added_interior_zeros'] + st['padding_entries'])
    assert st['explicit_zeros'] == np.count_nonzero(b.data == 0)
    assert st['padding_entries'] >= 0
    b.data.flat[0] += 1
    with pytest.raises(AssertionError):
        verify_reconstruction(a, b, rp, cp)


def test_adjacent_transfer_permutations():
    rng = np.random.default_rng(27)
    a = sparse.diags([1, 3, 1], [-1, 0, 1], shape=(5, 5), format='csr', dtype=np.float64)
    p = sparse.csr_matrix(rng.normal(size=(9, 5)))
    fine = np.arange(9)
    coarse = level_permutation(a, 'rcm')
    pp = permute_operator(p, fine, coarse)
    rr = permute_operator(p.T.tocsr(), coarse, fine)
    np.testing.assert_array_equal(pp.toarray().T, rr.toarray())
    x = rng.normal(size=5)
    np.testing.assert_allclose(pp @ x[coarse], p @ x)


def test_load_raw_export_and_truncation(tmp_path):
    base = tmp_path/'hybrid.L0.P'
    meta = dict(version=1, level=0, source_level=0, role='P', block_rows=2,
        block_cols=3, nnzb=3, block_dimy=1, block_dimx=1, block_order='C',
        external_diagonal=False, index_dtype='<i4', value_dtype='<f8', values_count=4, diag_count=0)
    arrays = dict(indptr=np.array([0, 2, 3], dtype='<i4'), indices=np.array([0, 2, 1], dtype='<i4'),
                  values=np.array([2., 0., -3., 999.]), diag=np.empty(0, dtype='<i4'))
    for name, a in arrays.items():
        a.tofile(f'{base}.{name}.bin')
    path = tmp_path/'hybrid.L0.P.json'
    path.write_text(json.dumps(meta))
    actual, _ = load_operator(path)
    assert actual.shape == (2, 3) and actual.nnz == 3
    np.testing.assert_array_equal(actual.toarray(), [[2., 0., 0.], [0., -3., 0.]])
    (tmp_path/'hybrid.L0.P.values.bin').write_bytes(b'')
    with pytest.raises(ValueError, match='Truncated'):
        load_operator(path)
