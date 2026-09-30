"""Parallel COO-to-CSR pattern and value kernels for repeated direct solves.

A fixed COO pattern ``(rows, cols)`` with duplicates is reduced once to a CSR
pattern of its unique entries (columns sorted within each row) and a gather
map: ``order`` lists the COO positions grouped by CSR entry, in increasing
position, and ``order[segments[u]:segments[u + 1]]`` are the duplicates summed
into entry ``u``. Each solve then only gathers the new values. The sums run in
increasing COO position, the order of ``np.bincount``, so the values are
bit-identical to the NumPy reduction.

:func:`build_coo_csr_pattern` and :func:`csr_values_from_coo` fall back to
NumPy (same outputs) without Numba.
"""

from __future__ import annotations

import math

import numpy as np

from hdgfem.runtime.optional import NUMBA_AVAILABLE, njit, prange

# Rows longer than this are sorted with a stable merge sort instead of an insertion sort.
_INSERTION_SORT_LIMIT = 64


def index_dtype(count: int):
    """``int32`` when every index and offset below ``count`` fits, else ``int64``."""
    return np.int32 if count < np.iinfo(np.int32).max else np.int64


@njit(cache=True, parallel=True)
def _chunk_row_counts(rows, cols, size, counts):
    """``counts[c, r]`` = entries of row ``r`` in COO chunk ``c``; returns the number of out-of-range indices."""
    chunks, count = counts.shape[0], rows.shape[0]
    bad = 0
    for chunk in prange(chunks):
        for k in range(count*chunk//chunks, count*(chunk + 1)//chunks):
            row, col = rows[k], cols[k]
            if row < 0 or row >= size or col < 0 or col >= size:
                bad += 1
            else:
                counts[chunk, row] += 1
    return bad


@njit(cache=True, parallel=True)
def _chunk_row_offsets(counts, row_start):
    """Turn per-chunk row counts into write offsets (chunk order, so placement stays stable)."""
    for row in prange(counts.shape[1]):
        position = row_start[row]
        for chunk in range(counts.shape[0]):
            count = counts[chunk, row]
            counts[chunk, row] = position
            position += count


@njit(cache=True, parallel=True)
def _place_by_row(rows, offsets, order):
    """Bucket COO positions by row, keeping increasing positions within each row."""
    chunks, count = offsets.shape[0], rows.shape[0]
    for chunk in prange(chunks):
        for k in range(count*chunk//chunks, count*(chunk + 1)//chunks):
            row = rows[k]
            order[offsets[chunk, row]] = k
            offsets[chunk, row] += 1


@njit(cache=True, parallel=True)
def _sort_rows_and_count(cols, row_start, order, unique_counts):
    """Stable-sort each row's positions by column and count its unique columns."""
    for row in prange(row_start.shape[0] - 1):
        start, stop = row_start[row], row_start[row + 1]
        if stop - start > _INSERTION_SORT_LIMIT:
            segment = order[start:stop].copy()
            keys = np.empty(stop - start, dtype=np.int64)
            for i in range(stop - start):
                keys[i] = cols[segment[i]]
            permutation = np.argsort(keys, kind="mergesort")
            for i in range(stop - start):
                order[start + i] = segment[permutation[i]]
        else:
            for i in range(start + 1, stop):
                position, key = order[i], cols[order[i]]
                j = i - 1
                while j >= start and cols[order[j]] > key:
                    order[j + 1] = order[j]
                    j -= 1
                order[j + 1] = position
        unique = 0
        for i in range(start, stop):
            if i == start or cols[order[i]] != cols[order[i - 1]]:
                unique += 1
        unique_counts[row] = unique


@njit(cache=True, parallel=True)
def _write_csr_pattern(cols, row_start, order, indptr, indices, segments):
    """Write the unique sorted columns of each row and the start of each entry's duplicates."""
    for row in prange(row_start.shape[0] - 1):
        entry = indptr[row] - 1
        for i in range(row_start[row], row_start[row + 1]):
            if i == row_start[row] or cols[order[i]] != cols[order[i - 1]]:
                entry += 1
                indices[entry] = cols[order[i]]
                segments[entry] = i
    segments[segments.shape[0] - 1] = order.shape[0]


def build_coo_csr_pattern(rows, cols, size: int, *, chunks: int | None = None):
    """CSR pattern and gather map of a COO pattern: ``(indptr, indices, order, segments)``.

    ``indptr`` has ``size + 1`` entries and ``indices`` the sorted unique
    columns of each row; ``order`` and ``segments`` form the gather map
    described in the module docstring. Index arrays are ``int32`` when every
    COO position fits, else ``int64``. Raises :class:`ValueError` for indices
    outside ``[0, size)``.
    """
    rows, cols, size = np.ascontiguousarray(rows), np.ascontiguousarray(cols), int(size)
    if rows.shape != cols.shape or rows.ndim != 1:
        raise ValueError("COO rows and cols must be 1-D arrays of the same length")
    dtype = index_dtype(max(rows.size, size) + 1)
    if not NUMBA_AVAILABLE:
        return _build_coo_csr_pattern_numpy(rows, cols, size, dtype)
    import numba
    chunks = max(1, min(int(chunks or numba.get_num_threads()), rows.size or 1))
    offsets = np.zeros((chunks, size), dtype=np.int64)
    if _chunk_row_counts(rows, cols, size, offsets):
        raise ValueError(f"COO indices must lie in [0, {size})")
    row_start = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(offsets.sum(axis=0), out=row_start[1:])
    _chunk_row_offsets(offsets, row_start)
    order = np.empty(rows.size, dtype=dtype)
    _place_by_row(rows, offsets, order)
    del offsets
    unique_counts = np.empty(size, dtype=np.int64)
    _sort_rows_and_count(cols, row_start, order, unique_counts)
    indptr = np.zeros(size + 1, dtype=dtype)
    np.cumsum(unique_counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices, segments = np.empty(nnz, dtype=dtype), np.empty(nnz + 1, dtype=dtype)
    _write_csr_pattern(cols, row_start, order, indptr, indices, segments)
    return indptr, indices, order, segments


def _build_coo_csr_pattern_numpy(rows, cols, size, dtype):
    """NumPy reference of :func:`build_coo_csr_pattern` (one stable sort of the COO keys)."""
    if rows.size and (min(rows.min(), cols.min()) < 0 or max(rows.max(), cols.max()) >= size):
        raise ValueError(f"COO indices must lie in [0, {size})")
    keys = rows.astype(np.int64)*size + cols.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    new = np.empty(sorted_keys.size, dtype=bool)
    new[:1] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=new[1:])
    unique = sorted_keys[new]
    indptr = np.zeros(size + 1, dtype=dtype)
    np.cumsum(np.bincount(unique // size, minlength=size), out=indptr[1:])
    segments = np.append(np.flatnonzero(new), sorted_keys.size).astype(dtype)
    return indptr, (unique % size).astype(dtype), order.astype(dtype), segments


@njit(cache=True, parallel=True)
def _gather_values(data, order, segments, out):
    """``out[u]`` = sum of ``data`` over entry ``u``'s duplicates; returns the count of nonfinite sums."""
    bad = 0
    for entry in prange(out.shape[0]):
        total = 0.
        for i in range(segments[entry], segments[entry + 1]):
            total += data[order[i]]
        out[entry] = total
        if not math.isfinite(total):
            bad += 1
    return bad


def csr_values_from_coo(data, order, segments, out=None):
    """CSR values of COO ``data`` through a gather map; returns ``(values, nonfinite_count)``.

    A nonfinite COO value makes its entry's sum nonfinite, so a zero count
    certifies finite input (barring overflow in a sum).
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    out = np.empty(segments.size - 1, dtype=np.float64) if out is None else out
    if not NUMBA_AVAILABLE:
        if out.size:
            out[:] = np.add.reduceat(data[order], segments[:-1])
        return out, int(np.count_nonzero(~np.isfinite(out)))
    return out, int(_gather_values(data, order, segments, out))


@njit(cache=True, parallel=True)
def coo_pattern_mismatches(rows, cols, stored_rows, stored_cols):
    """Number of positions where ``(rows, cols)`` differs from the stored pattern."""
    mismatches = 0
    for k in prange(rows.shape[0]):
        if rows[k] != stored_rows[k] or cols[k] != stored_cols[k]:
            mismatches += 1
    return mismatches


@njit(cache=True, parallel=True)
def csr_residual_norm(indptr, indices, values, x, rhs, base):
    """``||A x - rhs||_2`` for a CSR matrix whose index arrays start at ``base`` (0 or 1)."""
    total = 0.
    for row in prange(rhs.shape[0]):
        value = 0.
        for k in range(indptr[row] - base, indptr[row + 1] - base):
            value += values[k]*x[indices[k] - base]
        difference = value - rhs[row]
        total += difference*difference
    return math.sqrt(total)


__all__ = ["build_coo_csr_pattern", "coo_pattern_mismatches", "csr_residual_norm", "csr_values_from_coo",
           "index_dtype"]
