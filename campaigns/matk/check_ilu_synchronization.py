#!/usr/bin/env python3
"""Native C-API ILU synchronization, application and diagnostics oracles."""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.sparse import bsr_matrix, coo_matrix

from check_natural_iluk import API, apply_reference, fixture, reference_factors
from common import write_report


BASE_BLOCK = np.array([[1, .4, .1, -.2], [.2, 1.3, .3, .05],
                       [-.1, .1, .9, .2], [.15, -.2, .3, 1.1]])
COUNTERS = re.compile(
    r"Native multicolor ILU inverse diagnostics: scope=([^;]+); rows=(\d+); "
    r"blocks=(\d+); fallback=(\d+) \([^)]*\); zero_det=(\d+); nonfinite_det=(\d+)")


def bipartite_fixture(size, dtype):
    """Two independent colors with unequal 0/1/3/4/5/15/16/17 neighbor counts."""
    degrees = np.resize(np.array([0, 1, 3, 4, 5, 15, 16, 17]), size)
    left = np.repeat(np.arange(size), np.minimum(degrees, size))
    offset = np.concatenate([np.arange(min(int(d), size)) for d in degrees])
    right = (left + offset) % size + size
    n = 2 * size
    rows = np.concatenate((np.arange(n), left, right))
    cols = np.concatenate((np.arange(n), right, left))
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    graph.sort_indices()
    rng = np.random.default_rng(20261031)
    values = rng.normal(0, .018, (graph.nnz, 4, 4)).astype(dtype)
    owners = np.repeat(np.arange(n), np.diff(graph.indptr))
    diagonal = owners == graph.indices
    values[diagonal] += 4 * np.eye(4, dtype=dtype)
    matrix = bsr_matrix((values, graph.indices, graph.indptr), shape=(4*n, 4*n))
    colors = np.repeat(np.arange(2, dtype=np.int32), size)
    return matrix, colors


def bipartite_reference(matrix):
    """Independent two-color ILU0 block algebra, vectorized across graph edges."""
    n = matrix.shape[0] // 4
    size = n // 2
    owners = np.repeat(np.arange(n), np.diff(matrix.indptr))
    columns = matrix.indices.astype(np.int64)
    diagonal = owners == columns
    diag = matrix.data[diagonal].astype(np.float64).copy()
    upper = owners < columns
    lower = owners > columns
    ur, uc = owners[upper], columns[upper]
    lr, lc = owners[lower], columns[lower]
    u = matrix.data[upper].astype(np.float64)
    lower_values = matrix.data[lower].astype(np.float64)
    first_inverse = np.linalg.inv(diag[:size])
    l = lower_values @ first_inverse[lc]
    # Reverse edges have independent coefficients; locate their true partners.
    upper_keys = ur.astype(np.int64) * n + uc
    reverse_keys = lc * n + lr
    order = np.argsort(upper_keys)
    reverse = order[np.searchsorted(upper_keys[order], reverse_keys)]
    assert np.array_equal(upper_keys[reverse], reverse_keys)
    np.add.at(diag, lr, -(l @ u[reverse]))
    last_inverse = np.linalg.inv(diag[size:])

    def apply(rhs):
        y = np.asarray(rhs, dtype=np.float64).reshape(n, 4).copy()
        np.add.at(y, lr, -np.einsum("eij,ej->ei", l, y[lc]))
        y[size:] = np.einsum("eij,ej->ei", last_inverse, y[size:])
        np.add.at(y, ur, -np.einsum("eij,ej->ei", u, y[uc]))
        y[:size] = np.einsum("eij,ej->ei", first_inverse, y[:size])
        return y.ravel()
    return apply


def inverse_fixture(rows, dtype, nonfinite=False):
    scales = np.resize(np.array([1., 1e-2, 1e-4]), rows)
    blocks = (scales[:, None, None] * BASE_BLOCK).astype(dtype)
    if nonfinite:
        assert rows == 4
        blocks[:] = BASE_BLOCK
        blocks[0] = 0
        blocks[2, 0, 0] = np.nan
        blocks[3, 0, 0] = np.inf
    matrix = bsr_matrix((blocks, np.arange(rows, dtype=np.int32),
                         np.arange(rows+1, dtype=np.int32)), shape=(4*rows, 4*rows))
    return matrix, np.zeros(rows, dtype=np.int32)


def inverse_reference(matrix, dtype):
    """Dense solve where the retained cutoff allows it, stated diagonal policy otherwise."""
    blocks = matrix.data.astype(np.float64)
    cutoff = 1e-12 if dtype == np.float64 else 1e-7
    fallback = np.abs(np.linalg.det(blocks)) < cutoff
    inverse = np.empty_like(blocks)
    inverse[~fallback] = np.linalg.inv(blocks[~fallback])
    diagonal = np.diagonal(blocks[fallback], axis1=1, axis2=2)
    inverse[fallback] = np.eye(4)[None] / diagonal[:, None, :]
    def apply(rhs):
        return np.einsum("bij,bj->bi", inverse, np.asarray(rhs).reshape(-1, 4)).ravel()
    return apply, int(np.count_nonzero(fallback))


def upload_arrays(matrix, layout, external):
    data = matrix.data if layout == "ROW_MAJOR" else matrix.data.swapaxes(1, 2)
    if not external:
        return matrix.indptr, matrix.indices, np.ascontiguousarray(data), None
    rows = np.repeat(np.arange(matrix.shape[0] // 4), np.diff(matrix.indptr))
    diagonal = rows == matrix.indices
    counts = np.diff(matrix.indptr) - 1
    pointers = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return (pointers, np.ascontiguousarray(matrix.indices[~diagonal]),
            np.ascontiguousarray(data[~diagonal]), np.ascontiguousarray(data[diagonal]))


def run_case(api, case, mode, layout, diagnostics, repeat_setup, inverse_rows=8193):
    dtype, enum, tolerance = ((np.float64, 8193, 3e-10) if mode == "dDDI"
                              else (np.float32, 8465, 8e-5))
    external = case == "external"
    fill = int(external)
    if case in ("coupled", "reuse"):
        matrix, colors = bipartite_fixture(35 if case == "coupled" else 16385, dtype)
        reference = bipartite_reference(matrix)
        fallback = 0
    elif external:
        matrix = fixture(4, dtype, False)
        colors = np.arange(matrix.shape[0] // 4, dtype=np.int32)
        factors, _ = reference_factors(matrix, 1)
        reference = lambda rhs: apply_reference(factors, rhs)
        fallback = 0
    else:
        matrix, colors = inverse_fixture(4 if case == "nonfinite" else inverse_rows,
                                          dtype, case == "nonfinite")
        if case == "nonfinite":
            reference, fallback = None, 1
        else:
            reference, fallback = inverse_reference(matrix, dtype)
    n = matrix.shape[0] // 4
    expected_counts = dict(rows=n, blocks=n, fallback=fallback,
                           zero_det=int(case == "nonfinite"),
                           nonfinite_det=2 if case == "nonfinite" else 0)
    config = dict(config_version=2, exception_handling=0, determinism_flag=1,
        block_format=layout, solver=dict(solver="MULTICOLOR_ILU", scope="sync_oracle",
            max_iters=1, monitor_residual=0, store_res_history=0, print_solve_stats=0,
            ilu_sparsity_level=fill, block_ilu_backend="amgx", relaxation_factor=1.,
            coloring_level=fill+1, matrix_coloring_scheme="MIN_MAX",
            max_uncolored_percentage=0., reorder_cols_by_color=1,
            insert_diag_while_reordering=int(not external),
            print_coloring_info=int(diagnostics)))
    row = dict(case=case, mode=mode, layout=layout, diagnostics=diagnostics,
               block_rows=n, occupied_color_sizes=np.bincount(colors).tolist(),
               external_diagonal=external, configuration=config, expected_counters=expected_counts,
               setup_counters=[], applications=[], status="running")
    handles = []
    answers = []
    def make(kind, *args):
        handle = ct.c_void_p()
        api.call(kind+"_create", ct.byref(handle), *args)
        handles.append((kind, handle))
        return handle
    def ptr(array):
        return None if array is None else ct.c_void_p(array.ctypes.data)
    try:
        cfg = make("config", json.dumps(config).encode())
        resources = ct.c_void_p()
        api.call("resources_create_simple", ct.byref(resources), cfg)
        handles.append(("resources", resources))
        a = make("matrix", resources, enum)
        b = make("vector", resources, enum)
        x = make("vector", resources, enum)
        solver = make("solver", resources, enum, cfg)
        pointers, columns, values, diagonal = upload_arrays(matrix, layout, external)
        api.call("matrix_upload_all", a, n, len(columns), 4, 4,
                 ptr(pointers), ptr(columns), ptr(values), ptr(diagonal))
        api.call("matrix_attach_coloring", a, ptr(colors), n, int(colors.max())+1)
        rng = np.random.default_rng(963)
        for setup in range(2 if repeat_setup else 1):
            begin = len(api.messages)
            api.call("solver_setup", solver, a)
            matches = list(COUNTERS.finditer("".join(api.messages[begin:])))
            if diagnostics:
                if len(matches) != 1:
                    raise AssertionError(f"Expected one diagnostic per setup; got {len(matches)}")
                observed = dict(zip(("rows", "blocks", "fallback", "zero_det", "nonfinite_det"),
                                    map(int, matches[0].groups()[1:])))
                if observed != expected_counts:
                    raise AssertionError(f"Counter mismatch: {observed} != {expected_counts}")
                row["setup_counters"].append(observed)
            elif matches:
                raise AssertionError("Diagnostics-disabled setup printed inverse counters")
            if reference is None:
                continue
            for zero in (True, False):
                rhs = rng.normal(size=4*n).astype(dtype)
                initial = rng.normal(size=4*n).astype(dtype)
                expected = (reference(rhs) if zero else
                            initial + reference(rhs.astype(np.float64) - matrix.astype(np.float64) @ initial))
                api.call("vector_upload", b, n, 4, ptr(rhs))
                api.call("vector_upload", x, n, 4, ptr(initial))
                api.call("solver_solve_with_0_initial_guess" if zero else "solver_solve", solver, b, x)
                actual = np.empty_like(rhs)
                api.call("vector_download", x, ptr(actual))
                error_blocks = (actual - expected).reshape(n, 4)
                expected_blocks = expected.reshape(n, 4)
                block_error = np.linalg.norm(error_blocks, axis=1) / np.maximum(
                    np.linalg.norm(expected_blocks, axis=1), np.finfo(dtype).tiny)
                relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
                worst = float(np.max(block_error))
                if not np.isfinite(actual).all() or max(relative, worst) > tolerance:
                    raise AssertionError(f"Application oracle: relative={relative}, worst block={worst}, tolerance={tolerance}")
                true_residual = float(np.linalg.norm(matrix.astype(np.float64) @ actual - rhs) / np.linalg.norm(rhs))
                row["applications"].append(dict(setup=setup, zero_initial_guess=zero,
                    oracle_relative_error=relative, worst_block_oracle_error=worst,
                    true_relative_residual=true_residual))
                answers.append(actual)
        row["status"] = "passed"
        if case == "inverse":
            row["numerical_policy_limit"] = "Expected diagonal fallback preserves historical loss of dense coupling; oracle agreement is not an exact inverse claim."
        if case == "nonfinite":
            row["numerical_policy_limit"] = "Diagnostic classification only; no solve or numerical validity claimed."
        return row, answers
    finally:
        for kind, handle in reversed(handles):
            api.call(kind+"_destroy", handle)


def host_check():
    for dtype in (np.float32, np.float64):
        matrix, _ = bipartite_fixture(7, dtype)
        vectorized = bipartite_reference(matrix)
        factors, _ = reference_factors(matrix, 0)
        rhs = np.random.default_rng(5).normal(size=matrix.shape[0])
        np.testing.assert_allclose(vectorized(rhs), apply_reference(factors, rhs), rtol=2e-14, atol=2e-14)
        matrix, _ = inverse_fixture(3, dtype)
        _, fallbacks = inverse_reference(matrix, dtype)
        assert fallbacks == (2 if dtype == np.float32 else 1)
        for layout in ("ROW_MAJOR", "COL_MAJOR"):
            rr, cc, vv, dd = upload_arrays(matrix, layout, True)
            assert len(cc) == 0 and not np.any(rr) and len(dd) == 3
    print("[ILU synchronization] Host oracle/layout checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=("coupled", "external", "inverse", "nonfinite", "reuse"))
    parser.add_argument("--mode", choices=("dDDI", "dFFI"))
    parser.add_argument("--layout", choices=("ROW_MAJOR", "COL_MAJOR"))
    parser.add_argument("--diagnostics", type=int, choices=(0, 1))
    parser.add_argument("--single-setup", action="store_true")
    parser.add_argument("--inverse-rows", type=int, default=8193)
    parser.add_argument("--host-check", action="store_true")
    parser.add_argument("--verbosity", type=int, choices=range(4), default=2)
    args = parser.parse_args()
    if args.host_check:
        host_check()
        return
    if args.library is None or args.output is None:
        parser.error("GPU qualification requires --library and --output")
    if args.inverse_rows < 1:
        parser.error("--inverse-rows must be positive")
    api = API(args.library, args.verbosity)
    api.lib.AMGX_matrix_attach_coloring.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
    api.lib.AMGX_matrix_attach_coloring.restype = ct.c_int
    report = dict(test="native ILU synchronization/application oracle", status="running",
                  library=str(args.library.resolve()), library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  results=[], diagnostics_parity=[])
    try:
        for case in ((args.case,) if args.case else ("coupled", "external", "inverse", "nonfinite", "reuse")):
            for mode in ((args.mode,) if args.mode else ("dDDI", "dFFI")):
                for layout in ((args.layout,) if args.layout else ("ROW_MAJOR", "COL_MAJOR")):
                    previous = None
                    for diagnostics in ((args.diagnostics,) if args.diagnostics is not None else (0, 1)):
                        row, answers = run_case(api, case, mode, layout, diagnostics,
                                                not args.single_setup, args.inverse_rows)
                        report["results"].append(row)
                        if previous is not None and answers:
                            tolerance = 3e-10 if mode == "dDDI" else 8e-5
                            assert len(answers) == len(previous)
                            for left, right in zip(previous, answers):
                                np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)
                            report["diagnostics_parity"].append(dict(case=case, mode=mode, layout=layout,
                                bitwise_equal=all(np.array_equal(a, b) for a, b in zip(previous, answers)), status="passed"))
                        previous = answers
                        write_report(args.output, report)
                        print(f"[ILU synchronization] {case} {mode} {layout} diagnostics={diagnostics} passed", flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      native_messages="".join(api.messages[-25:]))
        raise
    finally:
        api.call("finalize")
        write_report(args.output, report)


if __name__ == "__main__":
    main()
