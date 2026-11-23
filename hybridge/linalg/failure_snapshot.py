"""Failure archives and matrix diagnostics for reduced trace systems.

A failed global solve can save its restored, unscaled system to an ``.npz``
archive for offline inspection. The archive and its matrix diagnostics are
equation-independent; transport-specific data and analysis are added by
:func:`hybridge.transport.diagnostics.save_transport_failure_snapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def trace_matrix_diagnostics(arrays):
    """Measure rows AND columns without expanding face BSR to scalar CSR.

    Zero columns prove singularity. Nonzero columns and row/column scales do
    not prove nonsingularity or estimate the condition number.
    """
    data = np.asarray(arrays["data"])
    n = int(np.asarray(arrays["rhs"]).size)
    fmt = str(np.asarray(arrays["matrix_format"]).item())
    if not np.isfinite(data).all():
        return {"matrix_size": n, "matrix_finite": False}
    stored_entries = int(data.size)
    if fmt == "coo":
        # Assembly COO contains repeated element contributions. Sum signed
        # duplicates first: abs() before summation can hide cancelled columns.
        from scipy.sparse import coo_matrix

        matrix = coo_matrix((data, (arrays["rows"], arrays["cols"])), shape=(n, n)).tocsr()
        data, indptr, indices = matrix.data, matrix.indptr, matrix.indices
        fmt = "csr"
        if not np.isfinite(data).all():
            return {"matrix_size": n, "matrix_finite": False}
    elif fmt in {"bsr", "csr"}:
        indptr, indices = arrays["indptr"], arrays["indices"]
    row_l1 = np.zeros(n, dtype=np.float64)
    col_l1 = np.zeros(n, dtype=np.float64)
    chunk = 8192
    if fmt in {"bsr", "csr"}:
        block = data.shape[1] if fmt == "bsr" else 1
        rows = row_l1.reshape(-1, block)
        cols = col_l1.reshape(-1, block)
        for start in range(0, len(data), chunk):
            end = min(start + chunk, len(data))
            row_ids = np.searchsorted(indptr, np.arange(start, end), side="right") - 1
            values = np.abs(data[start:end]).reshape(-1, block, block)
            np.add.at(rows, row_ids, values.sum(axis=2, dtype=np.float64))
            np.add.at(cols, indices[start:end], values.sum(axis=1, dtype=np.float64))
    else:
        raise ValueError(f"Unsupported matrix format {fmt!r}")
    zero_rows, zero_cols = np.flatnonzero(row_l1 == 0), np.flatnonzero(col_l1 == 0)
    return {
        "matrix_size": n,
        "matrix_scalar_stored_entries": stored_entries,
        "matrix_finite": True,
        "matrix_row_l1_min": float(row_l1.min()) if n else 0.0,
        "matrix_row_l1_max": float(row_l1.max()) if n else 0.0,
        "matrix_column_l1_min": float(col_l1.min()) if n else 0.0,
        "matrix_column_l1_max": float(col_l1.max()) if n else 0.0,
        "matrix_zero_rows": int(zero_rows.size),
        "matrix_zero_columns": int(zero_cols.size),
        "matrix_zero_row_samples": zero_rows[:28].tolist(),
        "matrix_zero_column_samples": zero_cols[:28].tolist(),
    }


def trace_face_column_diagnostics(data, indptr, indices, face_index):
    """SVD of all BSR blocks touching one face's columns, including neighbors.

    A null vector of this small panel extends by zero to a null vector of the
    complete matrix. A full-rank panel does not prove global nonsingularity.
    """
    positions = np.flatnonzero(np.asarray(indices) == face_index)
    block = data.shape[-1]
    panel = data[positions].reshape(-1, block)
    if len(positions):
        _, singular, vh = np.linalg.svd(panel, full_matrices=False)
        mode = vh[-1]
    else:
        singular = np.zeros(block)
        mode = np.eye(block)[0]
    scale = float(singular[0])
    tolerance = max(panel.shape) * np.finfo(data.dtype).eps
    residual = float(np.linalg.norm(panel @ mode))
    return {
        "reduced_face_index": int(face_index),
        "column_panel_shape": list(panel.shape),
        "touched_block_rows": (np.searchsorted(indptr, positions, side="right") - 1).tolist(),
        "column_panel_singular_values": singular.tolist(),
        "column_panel_rank": int(np.count_nonzero(singular > tolerance * scale)),
        "column_panel_rcond": float(singular[-1] / scale) if scale else 0.0,
        "column_panel_rank_relative_tolerance": float(tolerance),
        "weakest_trace_mode": mode.tolist(),
        "matrix_mode_residual_norm": residual,
        "matrix_mode_relative_residual": residual / scale if scale else 0.0,
    }


def _host(array):
    """Host NumPy view or copy of a NumPy or CuPy array."""
    return np.asarray(array.get() if hasattr(array, "get") else array)


def system_snapshot_arrays(assembly, *, initial_guess=None, best_solution=None) -> dict:
    """Host copies of a reduced system's matrix, RHS, boundary trace and guesses."""
    arrays = {"format_version": np.array(1), "matrix_format": np.array(assembly.matrix_format),
              "data": _host(assembly.data), "rhs": _host(assembly.rhs)}
    for name in ("indptr", "indices", "rows", "cols", "boundary_trace"):
        value = getattr(assembly, name, None)
        if value is not None:
            arrays[name] = _host(value)
    for name, value in (("initial_guess", initial_guess), ("best_solution", best_solution)):
        if value is not None:
            arrays[name] = _host(value)
    return arrays


@dataclass(frozen=True)
class SavedReducedSystem:
    """A reduced trace system A x = b read back from a failure archive.

    ``matrix_format`` is ``"bsr"`` (``data`` of shape (number of blocks, b, b) with
    ``indptr`` and ``indices``), ``"csr"`` or ``"coo"`` (``rows`` and ``cols``).
    ``rhs`` is the unscaled right-hand side b. ``initial_guess`` and ``best_solution``
    are the vectors the failed solve held, or None when the archive has none.
    The attribute names are those of the device assemblies, so a device copy can be
    passed to :func:`hybridge.linalg.amgx.device_solver.solve_reduced_system_amgx_device`.
    """

    matrix_format: str
    data: Any
    rhs: Any
    indptr: Any = None
    indices: Any = None
    rows: Any = None
    cols: Any = None
    boundary_trace: Any = None
    initial_guess: Any = None
    best_solution: Any = None

    def host_matrix(self):
        """Return A as a SciPy sparse matrix on the host, built from the stored arrays only."""
        from scipy.sparse import bsr_matrix, coo_matrix, csr_matrix

        data = _host(self.data)
        n = int(_host(self.rhs).size)
        if self.matrix_format == "bsr":
            return bsr_matrix((data, _host(self.indices), _host(self.indptr)), shape=(n, n)).tocsr()
        if self.matrix_format == "csr":
            return csr_matrix((data, _host(self.indices), _host(self.indptr)), shape=(n, n))
        if self.matrix_format == "coo":
            return coo_matrix((data, (_host(self.rows), _host(self.cols))), shape=(n, n)).tocsr()
        raise ValueError(f"Unsupported matrix format {self.matrix_format!r}")

    def relative_residual(self, x, matrix=None) -> float:
        """Return ||b - A x||_2 / ||b||_2 computed on the host in float64.

        This check does not use the solver that produced ``x``. Pass ``matrix``
        (from :meth:`host_matrix`) to avoid rebuilding A for several vectors.
        """
        matrix = self.host_matrix() if matrix is None else matrix
        b = _host(self.rhs).astype(np.float64, copy=False)
        r = b - matrix @ _host(x).astype(np.float64, copy=False).reshape(-1)
        norm_b = float(np.linalg.norm(b))
        return float(np.linalg.norm(r)) / norm_b if norm_b else float(np.linalg.norm(r))


def load_system_snapshot(path, *, device: bool = False) -> SavedReducedSystem:
    """Read an archive written by :func:`save_system_snapshot`.

    With ``device=True`` every array is copied to the current CUDA device with
    CuPy; otherwise the arrays stay NumPy arrays. Archives of another
    ``format_version`` are rejected.
    """
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    version = int(arrays.pop("format_version"))
    if version != 1:
        raise ValueError(f"Unsupported system snapshot version {version}")
    matrix_format = str(arrays.pop("matrix_format").item())
    if device:
        from hybridge.runtime.optional import require_cupy

        cp = require_cupy()
        arrays = {name: cp.asarray(value) for name, value in arrays.items()}
    known = {"data", "rhs", "indptr", "indices", "rows", "cols", "boundary_trace",
             "initial_guess", "best_solution"}
    return SavedReducedSystem(matrix_format=matrix_format,
                              **{name: value for name, value in arrays.items() if name in known})


def analyze_system_snapshot(arrays) -> dict:
    """Equation-independent analysis of a loaded failure archive."""
    return {"matrix_diagnostics": trace_matrix_diagnostics(arrays)}


def save_system_snapshot(path, assembly, *, initial_guess=None, best_solution=None,
                         extra_arrays=None, analyze=analyze_system_snapshot) -> dict:
    """Save a failed reduced system to ``path`` and return a report.

    Called only after failure. Arrays are copied to the host without running
    any GPU kernel, the archive contains no pickled objects, and it is written
    atomically. ``extra_arrays`` adds equation-specific data and ``analyze``
    replaces the default matrix analysis. The archive is saved before the
    analysis runs, so an analysis error never loses the numerical evidence.
    """
    arrays = system_snapshot_arrays(assembly, initial_guess=initial_guess, best_solution=best_solution)
    if extra_arrays:
        arrays.update(extra_arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report = {"system_snapshot": str(path), "system_snapshot_bytes": path.stat().st_size}
    try:
        report.update(analyze(arrays))
    except Exception as error:
        report["snapshot_analysis_error"] = f"{type(error).__name__}: {error}"
    return report
