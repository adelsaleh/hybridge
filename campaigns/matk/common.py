"""Load the exported MATLAB/HDF5 CSR system without importing HYBRIDGE."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix

DEFAULT_MATRIX = Path(__file__).resolve().with_name("matK.h5")


def amgx_library_provenance():
    """Identify the library actually mapped into this Linux worker."""
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name == "libamgxsh.so":
            paths.add(Path(fields[5]).resolve())
    if len(paths) != 1:
        raise RuntimeError(f"Expected one loaded libamgxsh.so, found {sorted(map(str, paths))}")
    path = paths.pop()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    p.add_argument("--group", default="mat")
    p.add_argument("--index-base", choices=("auto", "0", "1"), default="auto")
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--repeats", type=int, default=3, help="Solves reusing one setup, each from zero")
    p.add_argument("--inspect-only", action="store_true", help="Load and validate without a solver")
    p.add_argument("--output", type=Path, required=True, help="JSON report path")
    p.add_argument("--verbosity", "-v", type=int, choices=range(4), default=2,
                   help="0=result/errors, 1=stages, 2=timings/diagnostics, 3=native solver output (default 2)")
    p.add_argument("--quiet", action="store_const", const=0, dest="verbosity", help="Alias for --verbosity 0")
    return p


def log(verbosity, level, message):
    if verbosity >= level:
        print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def read_vector(group, name, dtype=None):
    ds = group[name]
    if len(ds.shape) > 2 or (len(ds.shape) == 2 and min(ds.shape) != 1):
        raise ValueError(f"{name} must be a vector, got {ds.shape}")
    # read_direct avoids a second full copy of the 5 GB values dataset.
    array = np.empty(ds.shape, dtype=dtype or ds.dtype)
    ds.read_direct(array)
    return array.reshape(-1)


def index_vector(group, name):
    if group[name].dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices")
    return read_vector(group, name)


def real_vector(group, name):
    if group[name].dtype.kind not in "fiu":
        raise ValueError(f"{name}: expected real numeric values; complex input is not discarded")
    a = read_vector(group, name, np.float64)
    for start in range(0, a.size, 4_000_000):
        if not np.isfinite(a[start:start + 4_000_000]).all():
            raise ValueError(f"{name} contains nonfinite values")
    return a


def load_system(path, group="mat", index_base="auto", verbosity=0, log_prefix=""):
    import h5py

    def progress(level, message):
        log(verbosity, level, log_prefix + message)

    started = time.perf_counter()
    progress(1, f"Opening {path}, HDF5 group /{group}")
    with h5py.File(path, "r") as f:
        g = f[group]
        rowptr = index_vector(g, "rowptr")
        n = rowptr.size - 1
        if n < 1 or n >= np.iinfo(np.int32).max:
            raise ValueError("Invalid/int32-incompatible matrix dimension")
        base = int(rowptr[0]) if index_base == "auto" else int(index_base)
        if base not in (0, 1) or int(rowptr[0]) != base:
            raise ValueError("CSR rowptr must begin at the chosen index base (0 or 1)")
        nnz = g["vals"].size
        if nnz >= np.iinfo(np.int32).max:
            raise ValueError("This benchmark requires int32 CSR indices")
        if int(rowptr[-1]) != nnz + base or np.any(rowptr[1:] < rowptr[:-1]):
            raise ValueError("Invalid CSR pointers / values length")
        for name, expected in (("n", n), ("nnz", nnz)):
            if name in g and (g[name].size != 1 or int(np.asarray(g[name])[()].item()) != expected):
                raise ValueError(f"Stored {name} disagrees with CSR data")
        cols = index_vector(g, "cols")
        if cols.size != nnz or (nnz and (int(cols.min()) < base or int(cols.max()) >= n + base)):
            raise ValueError("CSR column indices are out of range / wrong length")
        rowptr = rowptr.astype(np.int32, copy=False)
        cols = cols.astype(np.int32, copy=False)
        rowptr -= base
        cols -= base
        # Single-process global columns are assumed. Refuse partial/distributed exports.
        rowmap = index_vector(g, "loc2glob") if "loc2glob" in g else np.arange(n, dtype=np.int32) + base
        if rowmap.size != n or not np.array_equal(np.sort(rowmap), np.arange(n) + base):
            raise ValueError("loc2glob must be a full permutation; distributed/partial exports unsupported")
        progress(1, f"Reading FP64 CSR values: {n:,} rows, {nnz:,} entries, input index base {base}")
        vals = real_vector(g, "vals")
        matrix = csr_matrix((vals, cols, rowptr), shape=(n, n), copy=False)
        # Put local rows into global order to match the global column numbering.
        permuted = not np.array_equal(rowmap, np.arange(n) + base)
        if permuted:
            matrix = matrix[np.argsort(rowmap), :].tocsr()
        rhs_name = "rhs_vals" if "rhs_vals" in g else "rhs"
        if rhs_name not in g:
            raise ValueError("No rhs_vals or rhs dataset: refusing to invent a right-hand side")
        progress(1, f"Reading /{group}/{rhs_name} and aligning global row/RHS maps")
        rhs_values = real_vector(g, rhs_name)
        rhsmap = index_vector(g, "rhs_loc2glob") if "rhs_loc2glob" in g else rowmap
        if rhsmap.size != rhs_values.size or np.any(rhsmap < base) or np.any(rhsmap >= n + base):
            raise ValueError("Invalid RHS mapping")
        if np.unique(rhsmap).size != n:
            raise ValueError("RHS mapping must cover every global row")
        rhs = np.zeros(n, dtype=np.float64)
        np.add.at(rhs, rhsmap.astype(np.int64) - base, rhs_values)
        dataset_info = {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                        for k, v in g.items() if isinstance(v, h5py.Dataset)}
    progress(1, "Canonicalizing CSR (sorting columns and summing duplicate entries)")
    matrix.sum_duplicates()
    matrix.sort_indices()
    if np.any(np.diff(matrix.indptr) == 0):
        raise ValueError("Matrix contains empty rows")
    metadata = dict(path=str(Path(path).resolve()), file_bytes=Path(path).stat().st_size,
                    group=group, input_index_base=base, shape=list(matrix.shape),
                    stored_nnz=nnz, canonical_nnz=int(matrix.nnz),
                    csr_bytes=sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr)),
                    rows_reordered=permuted, rhs_dataset=rhs_name,
                    rhs_norm=float(np.linalg.norm(rhs)), rhs_nonzero=int(np.count_nonzero(rhs)),
                    datasets=dataset_info, load_seconds=time.perf_counter() - started)
    progress(1, f"Loaded {n:,} x {n:,} CSR, nnz={matrix.nnz:,}, ||b||={metadata['rhs_norm']:.6e}, {metadata['load_seconds']:.3f} s")
    progress(2, json.dumps(metadata, indent=2))
    return matrix, rhs, metadata


def residual(matrix, rhs, x, rtol):
    rn = float(np.linalg.norm(matrix @ x - rhs))
    bn = float(np.linalg.norm(rhs))
    rel = rn / bn if bn else rn
    finite = bool(np.isfinite(x).all() and np.isfinite(rel))
    return dict(residual_norm=rn, relative_residual=rel, finite=finite,
                passed=bool(finite and rel <= rtol))


def system_fingerprint(matrix, rhs):
    """Hash the original canonical CSR equations, independent of solver scaling."""
    digest = hashlib.sha256(b"matK canonical CSR FP64 system v1\0")
    for source, dtype in ((matrix.shape, "<i8"), (matrix.indptr, "<i4"),
                          (matrix.indices, "<i4"), (matrix.data, "<f8"), (rhs, "<f8")):
        array = np.asarray(source, dtype=dtype)
        for begin in range(0, array.size, 4_000_000):
            chunk = np.ascontiguousarray(array[begin:begin + 4_000_000])
            digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def save_iterate(path, solution, metadata):
    """Atomically save a finite FP64 vector in original global coordinates."""
    path = Path(path)
    x = np.ascontiguousarray(solution, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("A saved iterate must be a finite one-dimensional vector")
    info = dict(metadata, format_version=1, coordinates="original_global",
                vector_sha256=hashlib.sha256(memoryview(x).cast("B")).hexdigest(),
                size=int(x.size), dtype=str(x.dtype))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, x=x, metadata=np.array(json.dumps(info, allow_nan=False)))
    temporary.replace(path)
    return dict(path=str(path.resolve()), **info)


def load_iterate(path, fingerprint, size):
    """Validate an original-coordinate checkpoint before using it as x0."""
    with np.load(path, allow_pickle=False) as archive:
        x = archive["x"]
        metadata = json.loads(str(archive["metadata"].item()))
    if (metadata.get("format_version") != 1 or
            metadata.get("coordinates") != "original_global" or
            metadata.get("system_sha256") != fingerprint):
        raise ValueError("Initial-guess checkpoint does not match the original matrix/RHS")
    if x.dtype != np.float64 or x.shape != (size,) or not np.isfinite(x).all():
        raise ValueError("Initial-guess checkpoint has invalid dimensions, dtype or values")
    if hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast("B")).hexdigest() != metadata.get("vector_sha256"):
        raise ValueError("Initial-guess vector checksum does not match its metadata")
    return x, metadata


def provenance():
    return dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                affinity=sorted(os.sched_getaffinity(0)))


def write_report(path, report):
    # Strict JSON: numerical failures can contain NaN/Inf.
    def clean(obj):
        if isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate_args(args):
    if args.repeats < 1 or not np.isfinite(args.rtol) or args.rtol <= 0:
        raise ValueError("repeats and rtol must be positive and finite")
