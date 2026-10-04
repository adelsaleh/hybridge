"""Host-side lossless AMGX hierarchy loading and BSR feasibility diagnostics.

No device imports, assembly, builds, or time integration. Permutations map new
indices to old indices. Level zero is always the original fine ordering.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee


def load_operator(path):
    """Read v1 raw export; retain structural zeros and separate diagonals."""
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta['version'] != 1:
        raise ValueError('Unsupported hierarchy export version')
    base = path.with_suffix('')
    nr, nc, nnz = (int(meta[k]) for k in ('block_rows', 'block_cols', 'nnzb'))
    by, bx = int(meta['block_dimy']), int(meta['block_dimx'])
    if min(nr, nc, nnz) < 0 or min(by, bx) < 1:
        raise ValueError('Invalid operator dimensions')

    def read(name, dtype, count):
        """Read one raw export array, checking its byte size against ``count``."""
        file = Path(f'{base}.{name}.bin')
        dtype = np.dtype(dtype)
        if file.stat().st_size != count * dtype.itemsize:
            raise ValueError(f'Truncated or oversized export array: {file}')
        return np.fromfile(file, dtype=dtype, count=count)

    ip = read('indptr', meta['index_dtype'], nr + 1)
    ix = read('indices', meta['index_dtype'], nnz)
    values = read('values', meta['value_dtype'], meta['values_count'])
    diag = read('diag', meta['index_dtype'], meta['diag_count'])
    if ip[0] != 0 or ip[-1] != nnz or np.any(np.diff(ip) < 0):
        raise ValueError('Invalid exported row offsets')
    if np.any(ix < 0) or np.any(ix >= nc) or len(values) < nnz * bx * by:
        raise ValueError('Invalid exported indices/values')
    blocks = values[:nnz*bx*by].reshape(nnz, by, bx)
    if meta['block_order'] == 'F':
        blocks = values[:nnz*bx*by].reshape(nnz, bx, by).transpose(0, 2, 1)
    elif meta['block_order'] != 'C':
        raise ValueError('Unknown block order')
    result = sparse.bsr_matrix((blocks, ix, ip), shape=(nr*by, nc*bx)).tocsr()
    if meta['external_diagonal']:
        if nr != nc or len(diag) != nr or np.any(diag < 0) or np.any((diag+1)*bx*by > len(values)):
            raise ValueError('Invalid separate diagonal')
        # Construct disjoint entries; never add/reduce floating-point values.
        positions = diag[:, None]*bx*by + np.arange(bx*by)
        db = values[positions].reshape(nr, by, bx)
        if meta['block_order'] == 'F':
            db = db.reshape(nr, bx, by).transpose(0, 2, 1)
        d = sparse.bsr_matrix((db, np.arange(nr), np.arange(nr+1)), shape=result.shape).tocoo()
        a = result.tocoo()
        result = sparse.coo_matrix((np.concatenate((a.data, d.data)),
            (np.concatenate((a.row, d.row)), np.concatenate((a.col, d.col)))), shape=result.shape)
        # A separate diagonal must not overlap the off-diagonal structure.
        coords = np.lexsort((result.col, result.row))
        if np.any((np.diff(result.row[coords]) == 0) & (np.diff(result.col[coords]) == 0)):
            raise ValueError('Separate diagonal overlaps stored matrix entries')
        result = result.tocsr()
    result.sort_indices()
    if not result.has_canonical_format or not np.all(np.isfinite(result.data)):
        raise ValueError('Duplicate indices or nonfinite operator coefficients')
    if result.dtype != np.float64:
        raise ValueError('The saved-system feasibility study requires FP64 operators')
    meta['export_bytes'] = sum(Path(f'{base}.{key}.bin').stat().st_size
                               for key in ('indptr', 'indices', 'values', 'diag')) + path.stat().st_size
    return result, meta


def level_permutation(matrix, ordering):
    """Return the identity or reverse Cuthill-McKee ordering of a square level operator."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Coarse-level ordering requires a square A')
    if ordering == 'original':
        return np.arange(matrix.shape[0], dtype=np.int32)
    if ordering != 'rcm':
        raise ValueError(ordering)
    graph = matrix.copy()
    graph.data = np.ones(graph.nnz, dtype=np.int8)
    return reverse_cuthill_mckee(graph, symmetric_mode=False).astype(np.int32)


def _permutation(p, size):
    """Validate that ``p`` is a permutation of ``range(size)``."""
    p = np.asarray(p)
    if p.shape != (size,) or not np.array_equal(np.sort(p), np.arange(size)):
        raise ValueError('Invalid permutation')
    return p


def permute_operator(matrix, rows, cols):
    """Return ``matrix[rows][:, cols]`` as sorted CSR."""
    rows, cols = _permutation(rows, matrix.shape[0]), _permutation(cols, matrix.shape[1])
    out = matrix[rows, :][:, cols].tocsr()
    out.sort_indices()
    return out


def padded_bsr(matrix, block_size):
    """Preserve explicit CSR zeros, padding both rectangular dimensions."""
    b = int(block_size)
    if b < 1 or not matrix.has_canonical_format:
        raise ValueError('Require positive block size and canonical CSR')
    m, n = matrix.shape
    pm, pn = ((d+b-1)//b*b for d in (m, n))
    ip = np.pad(matrix.indptr, (0, pm-m), constant_values=matrix.nnz)
    padded = sparse.csr_matrix((matrix.data, matrix.indices, ip), shape=(pm, pn))
    return padded.tobsr(blocksize=(b, b))


def verify_reconstruction(original, candidate, rows, cols):
    """Undo padding/order; compare every stored coefficient exactly, no tolerance.

    BSR adds zeros. Check original structural entries including explicit zeros,
    then verify every additional entry is zero. Original arrays remain in the
    lossless export; BSR alone cannot distinguish a stored zero from a fill zero.
    """
    m, n = original.shape
    restored = candidate.tocsr()[:m, :n][np.argsort(rows), :][:, np.argsort(cols)]
    restored.sort_indices()
    coo = original.tocoo()
    actual = np.asarray(restored[coo.row, coo.col]).ravel()
    if not np.array_equal(actual, original.data):
        raise AssertionError('A stored coefficient changed after permutation/padding')
    left, right = original.copy(), restored.copy()
    left.eliminate_zeros(); right.eliminate_zeros()
    if not (np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.indices, right.indices)
            and np.array_equal(left.data, right.data)):
        raise AssertionError('Unexpected coefficients after BSR reconstruction')


def storage_stats(csr, bsr):
    """Compare CSR and BSR storage of one operator (entries, blocks, explicit zeros, bytes)."""
    b = bsr.blocksize[0]
    m, n = csr.shape
    br = np.repeat(np.arange(len(bsr.indptr)-1), np.diff(bsr.indptr))
    occupied_inside = int(np.sum(np.minimum(b, np.maximum(0, m-br*b)) *
        np.minimum(b, np.maximum(0, n-bsr.indices.astype(np.int64)*b)), dtype=np.int64))
    slots = int(bsr.data.size)
    stored_zeros = int(np.count_nonzero(csr.data == 0))
    cb = int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
    bb = int(bsr.data.nbytes + bsr.indices.nbytes + bsr.indptr.nbytes)
    return dict(csr_entries=int(csr.nnz), occupied_blocks=int(bsr.indices.size),
        occupied_block_entries=slots, nonzero_entries=int(np.count_nonzero(csr.data)),
        original_explicit_zeros=stored_zeros,
        added_interior_zeros=occupied_inside-int(csr.nnz),
        padding_entries=slots-occupied_inside, explicit_zeros=slots-int(np.count_nonzero(csr.data)),
        padded_rows=bsr.shape[0]-m, padded_cols=bsr.shape[1]-n,
        csr_bytes=cb, bsr_bytes=bb, storage_inflation=bb/cb,
        vector_padding_bytes=8*(sum(bsr.shape)-sum(csr.shape)))


def deterministic_vectors(size):
    """Return three fixed test vectors: ones, alternating signs and a pseudo-random pattern."""
    i = np.arange(size, dtype=np.int64)
    return (np.ones(size), np.where(i % 2, -1., 1.),
            ((i*17 % 101).astype(np.float64)-50.) / 51.)


def compare_product(matrix, x, actual):
    """Independent SciPy FP64 reference, scaled by absolute row sums."""
    reference = matrix @ x
    scale = np.abs(matrix) @ np.abs(x)
    error = float(np.max(np.abs(actual-reference), initial=0.))
    bound = 256*np.finfo(np.float64).eps*float(np.max(scale, initial=0.))
    if not np.all(np.isfinite(actual)) or error > max(bound, 1e-300):
        raise AssertionError(f'CSR/BSR product error {error} exceeds {bound}')
    return dict(max_abs_error=error, tolerance=bound)
