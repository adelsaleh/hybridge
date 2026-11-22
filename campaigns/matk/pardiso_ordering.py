#!/usr/bin/env python3
"""Export PARDISO phase-11 ordering; permute CSR without a full column gather.

The archive stores a zero-based ``new_to_old`` permutation: C = A[p, :][:, p],
d = b[p], and the original-coordinate solution is recovered with x[p] = y.
No numerical LU factorization or solve is performed by this module.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import traceback


FORMAT = "matk_pardiso_permutation"
FORMAT_VERSION = 1
CHUNK_ENTRIES = 4_000_000


def _validate_permutation(p, n):
    import numpy as np
    if not isinstance(p, np.ndarray) or p.dtype != np.dtype("int32"):
        raise ValueError("Permutation must be an int32 NumPy array")
    if p.ndim != 1 or p.size != n or n < 1:
        raise ValueError(f"Permutation must have shape ({n},)")
    if p.min() < 0 or p.max() >= n:
        raise ValueError("Permutation indices must be zero-based and within the system size")
    if not np.array_equal(np.sort(p), np.arange(n, dtype=np.int32)):
        raise ValueError("Permutation must be a bijection")


def _validate_matrix(matrix):
    import numpy as np
    from scipy.sparse import issparse
    if not issparse(matrix) or matrix.format != "csr":
        raise TypeError("Matrix must use SciPy CSR storage")
    n, columns = matrix.shape
    limit = np.iinfo(np.int32).max
    if n != columns or not 0 < n < limit or not 0 < matrix.nnz < limit:
        raise ValueError("PARDISO requires a nonempty square matrix with n and nnz below INT32_MAX")
    if matrix.dtype != np.float64:
        raise TypeError("Matrix coefficients must be float64")
    for name in ("indices", "indptr"):
        array = getattr(matrix, name)
        if array.dtype != np.int32 or array.ndim != 1 or not array.flags.c_contiguous:
            raise TypeError(f"CSR {name} must be a contiguous int32 array")
    if matrix.data.ndim != 1 or not matrix.data.flags.c_contiguous:
        raise TypeError("CSR coefficients must be a contiguous vector")
    if (matrix.indptr.size != n + 1 or matrix.indptr[0] != 0
            or matrix.indptr[-1] != matrix.data.size
            or matrix.indices.size != matrix.data.size
            or np.any(matrix.indptr[1:] <= matrix.indptr[:-1])):
        raise ValueError("CSR row pointers are invalid or contain an empty row")
    for start in range(0, matrix.nnz, CHUNK_ENTRIES):
        stop = start + CHUNK_ENTRIES
        indices = matrix.indices[start:stop]
        if indices.min() < 0 or indices.max() >= n:
            raise ValueError("CSR column index is outside the matrix")
        if not np.isfinite(matrix.data[start:stop]).all():
            raise ValueError("Matrix coefficients must be finite")
    if not matrix.has_sorted_indices or not matrix.has_canonical_format:
        raise ValueError("CSR columns must be sorted and duplicate-free before export or reordering")


def _permutation_hash(p):
    import numpy as np
    # A fixed byte order makes the checksum independent of the host endianness.
    return hashlib.sha256(np.ascontiguousarray(p, dtype="<i4").tobytes()).hexdigest()


def _validate_sha256(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be a lowercase SHA256 hex digest")


def _log(verbosity, level, message):
    from common import log
    log(verbosity, level, "[matK ordering] " + message)


def _fields(verbosity, level, title, rows):
    if verbosity < level:
        return
    print(flush=True)
    _log(verbosity, level, title)
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"[matK ordering]   {label:<{width}} : {value}", flush=True)


def compute_permutation(matrix, threads, verbosity=2):
    """Run real nonsymmetric analysis only; return int32 new-to-old p and metadata.

    The caller supplies ``system_sha256`` before saving, because this function
    deliberately accepts no RHS. Input CSR must already be canonical FP64 with
    int32 indices. The input matrix is never reordered or factorized.
    """
    import numpy as np
    import pypardiso
    from importlib.metadata import version

    if (isinstance(threads, bool) or not isinstance(threads, (int, np.integer))
            or not 0 < threads <= np.iinfo(np.int32).max):
        raise ValueError("threads must be a positive int32 integer")
    if verbosity not in range(4):
        raise ValueError("verbosity must be between 0 and 3")
    _validate_matrix(matrix)
    if "ILP64" in os.environ.get("MKL_INTERFACE_LAYER", "").upper():
        raise ValueError("PyPardiso's int32 interface is incompatible with MKL_INTERFACE_LAYER=ILP64")

    n = matrix.shape[0]
    solver = pypardiso.PyPardisoSolver(mtype=11)
    local_threads = solver.libmkl.MKL_Set_Num_Threads_Local
    local_threads.argtypes = [ctypes.c_int]
    local_threads.restype = ctypes.c_int
    max_threads = solver.libmkl.MKL_Get_Max_Threads
    max_threads.argtypes = []
    max_threads.restype = ctypes.c_int
    get_dynamic = solver.libmkl.MKL_Get_Dynamic
    get_dynamic.argtypes = []
    get_dynamic.restype = ctypes.c_int
    set_dynamic = solver.libmkl.MKL_Set_Dynamic
    set_dynamic.argtypes = [ctypes.c_int]
    set_dynamic.restype = None
    previous_threads = local_threads(int(threads))
    previous_dynamic = get_dynamic()
    set_dynamic(0)
    try:
        init = solver.libmkl.pardisoinit
        init.argtypes = [ctypes.POINTER(solver._pt_type[0]),
                         ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        init.restype = None
        mtype = ctypes.c_int32(solver.mtype)
        init(solver.pt.ctypes.data_as(ctypes.POINTER(solver._pt_type[0])),
             ctypes.byref(mtype),
             solver.iparm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

        # PyPardiso starts with a length-zero perm; PARDISO writes n entries.
        solver.perm = np.empty(n, dtype=np.int32)
        # Parameter numbers here are ONE-based (PyPardiso's convention).
        # Its private native call converts CSR pointers and columns to one-based.
        for key, value in {1: 1, 5: 2, 31: 0, 35: 0, 36: 0}.items():
            solver.set_iparm(key, value)
        solver._check_A(matrix)
        solver.set_phase(11)
        if verbosity >= 3:
            solver.set_statistical_info_on()
            native_pardiso = solver._mkl_pardiso
            flush_native = ctypes.CDLL(None).fflush
            flush_native.argtypes = [ctypes.c_void_p]
            flush_native.restype = ctypes.c_int

            def labelled_native(*args):
                phase = solver.phase
                print(f"\n[PARDISO native | phase {phase}] BEGIN\n", flush=True)
                try:
                    return native_pardiso(*args)
                finally:
                    flush_native(None)
                    print(f"\n[PARDISO native | phase {phase}] END\n", flush=True)

            solver._mkl_pardiso = labelled_native
        else:
            solver.set_statistical_info_off()

        get_version = solver.libmkl.MKL_Get_Version_String
        get_version.argtypes = [ctypes.c_char_p, ctypes.c_int]
        get_version.restype = None
        version_buffer = ctypes.create_string_buffer(1024)
        get_version(version_buffer, len(version_buffer))
        info = dict(phase=11, mtype=11, threads_requested=int(threads),
                    threads_effective=int(max_threads()), mkl_dynamic=bool(get_dynamic()),
                    mkl_version=version_buffer.value.decode("utf-8", errors="replace"),
                    mkl_library=str(solver.libmkl._name), pypardiso_version=version("pypardiso"),
                    iparm_indexing="one_based",
                    iparm_before={str(k): int(v) for k, v in solver.get_iparms().items()},
                    shape=[n, n], nnz=int(matrix.nnz))
        if info["threads_effective"] != threads:
            raise RuntimeError(f"Requested {threads} MKL threads, got {info['threads_effective']}")
        _fields(verbosity, 2, "PARDISO analysis settings", [
            ("Matrix", f"{n:,} x {n:,}"), ("Stored entries", f"{matrix.nnz:,}"),
            ("MKL threads", info["threads_effective"]),
            ("Ordering iparm(2)", info["iparm_before"]["2"]),
            ("Phase", "11 (analysis / ordering)"), ("MKL version", info["mkl_version"])])
        _log(verbosity, 1, "Computing fill-reducing permutation (PARDISO phase 11)")
        rhs_placeholder = np.zeros(n, dtype=np.float64)
        started = time.perf_counter()
        solver._call_pardiso(matrix, rhs_placeholder)
        info["analysis_seconds"] = time.perf_counter() - started
        info["iparm_after"] = {str(k): int(v) for k, v in solver.get_iparms().items()}
        # Check the returned one-based range before subtracting in int32.
        if solver.perm.min() < 1 or solver.perm.max() > n:
            raise ValueError("PARDISO returned an invalid one-based permutation")
        p = np.ascontiguousarray(solver.perm - np.int32(1))
        _validate_permutation(p, n)
        info["moved_indices"] = int(np.count_nonzero(p != np.arange(n, dtype=np.int32)))
        _log(verbosity, 1, f"Analysis completed in {info['analysis_seconds']:.3f} s")
        return p, info
    finally:
        try:
            solver.free_memory(everything=True)
        finally:
            set_dynamic(previous_dynamic)
            local_threads(previous_threads)


def save_permutation(path, p, metadata):
    """Atomically create a validated NPZ, refusing to replace any existing path."""
    import numpy as np
    path = Path(path)
    _validate_permutation(p, p.size if isinstance(p, np.ndarray) else 0)
    info = dict(metadata)
    _validate_sha256(info.get("system_sha256"), "system_sha256")
    info.update(format=FORMAT, format_version=FORMAT_VERSION, direction="new_to_old",
                index_base=0, size=int(p.size), dtype="int32",
                permutation_sha256=_permutation_hash(p))
    serialized = json.dumps(info, sort_keys=True, allow_nan=False)
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite permutation file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            # Only numeric and Unicode arrays: consumers never need pickle.
            np.savez(stream, p=np.ascontiguousarray(p), metadata=np.asarray(serialized))
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link publishes the complete file atomically and
        # fails if another process has already created the destination.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return info


def load_permutation(path, system_sha256, n):
    """Read without pickle and validate coordinates, system identity and checksum."""
    import numpy as np
    _validate_sha256(system_sha256, "system_sha256")
    with np.load(Path(path), allow_pickle=False) as archive:
        if set(archive.files) != {"p", "metadata"}:
            raise ValueError("Permutation archive must contain exactly p and metadata")
        raw_metadata = archive["metadata"]
        if raw_metadata.shape != () or raw_metadata.dtype.kind != "U":
            raise ValueError("Permutation metadata must be a scalar Unicode JSON string")
        info = json.loads(raw_metadata.item())
        if not isinstance(info, dict):
            raise ValueError("Permutation metadata must be a JSON object")
        required = dict(format=FORMAT, format_version=FORMAT_VERSION, direction="new_to_old",
                        index_base=0, size=n, dtype="int32", system_sha256=system_sha256)
        for key, expected in required.items():
            if info.get(key) != expected:
                raise ValueError(f"Permutation metadata {key} mismatch: expected {expected!r}")
        p = archive["p"]
    _validate_permutation(p, n)
    _validate_sha256(info.get("permutation_sha256"), "permutation_sha256")
    if _permutation_hash(p) != info["permutation_sha256"]:
        raise ValueError("Permutation checksum mismatch")
    return np.ascontiguousarray(p), info


def reorder_csr(matrix, p):
    """Return A[p,:][:,p] as canonical FP64 CSR without a full column-gather copy."""
    import numpy as np
    _validate_matrix(matrix)
    n = matrix.shape[0]
    _validate_permutation(p, n)
    inverse = np.empty(n, dtype=np.int32)
    inverse[p] = np.arange(n, dtype=np.int32)
    # Fancy row indexing already allocates independent CSR buffers.
    reordered = matrix[p, :].tocsr(copy=False)
    for start in range(0, reordered.nnz, CHUNK_ENTRIES):
        stop = start + CHUNK_ENTRIES
        reordered.indices[start:stop] = inverse[reordered.indices[start:stop]]
    # SciPy can cache the original sorted flag through row gathering. Remapping
    # columns invalidates it; otherwise sort_indices() can silently do nothing.
    reordered.has_sorted_indices = False
    reordered.sort_indices()
    # Force SciPy to recompute its separate duplicate/sortedness cache too.
    if hasattr(reordered, "_has_canonical_format"):
        del reordered._has_canonical_format
    if reordered.nnz != matrix.nnz or not reordered.has_canonical_format:
        raise RuntimeError("CSR reordering changed the sparsity pattern unexpectedly")
    return reordered


def main():
    # Match the benchmark's early thread setup before NumPy/SciPy/MKL imports.
    import argparse
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--threads", type=int,
                             default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    early, _ = preliminary.parse_known_args()
    if early.threads < 1:
        preliminary.error("--threads must be positive")
    for key in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT"):
        os.environ[key] = str(early.threads)
    os.environ.update(MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
    os.environ.pop("MKL_DOMAIN_NUM_THREADS", None)

    from common import load_system, parser, provenance, system_fingerprint, write_report
    cli = parser(__doc__)
    cli.add_argument("--threads", type=int, default=early.threads)
    cli.add_argument("--permutation-output", type=Path,
                     help="NPZ path (default: --output with suffix .permutation.npz); never overwritten")
    cli.add_argument("--smoke-test", action="store_true", help="Use the benchmark's tiny synthetic system")
    cli.add_argument("--smoke-blocks", type=int, default=4,
                     help="Synthetic 20-DOF blocks; match the AMGX smoke size (512 exercises multigrid)")
    args = cli.parse_args()
    if args.smoke_blocks < 2:
        cli.error("--smoke-blocks must be at least 2")
    destination = args.permutation_output or args.output.with_suffix(".permutation.npz")
    if destination.resolve() == args.output.resolve():
        cli.error("--output and --permutation-output must be different files")
    report = dict(status="loading", machine=provenance(), smoke_test=args.smoke_test,
                  parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  permutation_file=str(destination))
    exit_code = 0
    try:
        if not args.inspect_only and os.path.lexists(destination):
            raise FileExistsError(f"Refusing to overwrite permutation file: {destination}")
        _log(args.verbosity, 1, "Loading and validating the original matrix and RHS")
        if args.smoke_test:
            from benchmark_amgx_strong import smoke_system
            matrix, rhs, matrix_info = smoke_system(args.smoke_blocks)
        else:
            matrix, rhs, matrix_info = load_system(args.matrix, args.group, args.index_base,
                                                  args.verbosity, "[matK ordering] ")
        _validate_matrix(matrix)
        report["matrix"] = matrix_info
        _log(args.verbosity, 1, "Fingerprinting the original canonical CSR matrix and RHS")
        started = time.perf_counter()
        report["system_sha256"] = system_fingerprint(matrix, rhs)
        report["fingerprint_seconds"] = time.perf_counter() - started
        if args.inspect_only:
            report["status"] = "inspected"
        else:
            report["status"] = "analysis"
            write_report(args.output, report)
            p, info = compute_permutation(matrix, args.threads, args.verbosity)
            info["system_sha256"] = report["system_sha256"]
            report["permutation"] = save_permutation(destination, p, info)
            report["status"] = "exported"
    except Exception as exc:
        exit_code = 1
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        _log(args.verbosity, 0, report["error"])
        if args.verbosity >= 3:
            print(report["traceback"], flush=True)
    finally:
        write_report(args.output, report)
    rows = [("Status", report["status"].upper())]
    if "permutation" in report:
        info = report["permutation"]
        rows += [("Analysis only", f"{info['analysis_seconds']:.6f} s"),
                 ("Moved indices", f"{info['moved_indices']:,} / {info['size']:,}"),
                 ("Permutation", str(destination)), ("Coordinates", "int32, zero-based, new_to_old")]
    rows.append(("JSON report", str(args.output)))
    _fields(args.verbosity, 0, "PARDISO ordering result", rows)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
