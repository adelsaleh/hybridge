#!/usr/bin/env python3
"""Benchmark batched local solvers on the real A_e / B_e of the FP64 tensor AR assembly.

For each order the guiding-center transport step of
``benchmark_advection_tsle_tensor`` is built on the showcase star mesh, the
FP64 tensor assembly (cuTENSOR + MAGMA) is run once, and the stage-2 inputs are
captured: ``A`` ``(E, NEL, NEL)`` and ``B`` ``(E, NCOLS, NEL)``, both
column-major per element. Every solver then solves ``A_e X_e = B_e`` from fresh
copies; rows report the CUDA-event median (copies excluded), the per-element
time, and the useful rate ``E (2/3 n^3 + 2 n^2 nrhs) / t``.

Accuracy per solver: the normwise residual ``||A_e X_e - B_e||_F /
(||A_e||_F ||X_e||_F)`` (max, p99, median over elements), the difference from
the MAGMA solution, and, on a random subsample, the forward error against a
host reference refined once with an extended-precision (``np.longdouble``)
residual. Pivot statistics of MAGMA's partial pivoting and diagonal-dominance
ratios are recorded once per order.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _parse_orders(text: str) -> list[int]:
    orders = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            orders.extend(range(int(low), int(high) + 1))
        else:
            orders.append(int(part))
    return orders


def _parse_configs(text: str) -> dict[int, list[tuple[int, int, int]]]:
    """``"55:8x16x3,16x16x2;45:8x16x4"`` -> ``{55: [(8,16,3), (16,16,2)], 45: [...]}``."""
    configs: dict[int, list[tuple[int, int, int]]] = {}
    for group in filter(None, text.split(";")):
        size, _, items = group.partition(":")
        configs[int(size)] = [tuple(int(v) for v in item.split("x")) for item in items.split(",") if item]
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", default="star:0.0095", help="star:H (showcase star mesh) or rect:N")
    parser.add_argument("--orders", default="6-9")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--methods", default="magma,magma_gesv,magma_nopiv,magma_gesv_nopiv,magma_recpanel,"
                        "magma_vbatched,cublas,coop,fused")
    parser.add_argument("--recpanel", default="8,16", help="MAGMA recursive-panel widths to try (32 is getrf)")
    parser.add_argument("--fused-configs", default="", help='extra fused configs, "N:TRxTCxMINB,...;N:..."')
    parser.add_argument("--sample", type=int, default=2048, help="elements in the forward-error subsample")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import cupy as cp

    import hybridge.linalg.gpu.magma_batched as magma_batched
    from hybridge.linalg.gpu.cublas_batched import BatchedLUWorkspace, lu_solve_batched_cublas
    from hybridge.linalg.gpu.fused_batched_solve import default_config, solve_batched_fused
    from hybridge.transport import tsle_tensor
    from hybridge.transport.tsle_tensor import (
        RawAdvectionTsleTensorWorkspace,
        assemble_projected_advection_trace_system_eliminated_tsle_tensor,
        default_tensor_engines,
    )
    from scripts.gpu.benchmark_advection_tsle_tensor import _split3_inputs, _transport_step, environment_metadata
    from scripts.gpu.transport_workload import transport_mesh as showcase_mesh

    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = environment_metadata(cp)
    try:
        import subprocess
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None

    def write(record):
        record = dict(record, commit=commit, gpu=environment["gpu"]["name"], mesh=args.mesh)
        with args.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        shown = {k: v for k, v in record.items() if k in (
            "order", "method", "config", "status", "ms", "us_per_element", "gflops", "residual_max",
            "residual_median", "diff_vs_magma", "forward_error_max", "forward_error_median", "error")}
        print(json.dumps(shown), flush=True)

    kind, _, size = args.mesh.partition(":")
    if kind == "star":
        mesh = showcase_mesh(float(size))
    else:
        from hybridge.core.mesh import rectangle_mesh
        mesh = rectangle_mesh(int(size), int(size))
    write(dict(kind="environment", environment=environment, elements=int(mesh.num_tri)))

    library = magma_batched.load_magma()
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    library.magma_dgetrf_nopiv_batched.argtypes = [integer, integer, pointer, integer, pointer, integer, pointer]
    library.magma_dgetrs_nopiv_batched.argtypes = [integer, integer, integer, pointer, integer, pointer, integer,
                                                   pointer, integer, pointer]
    library.magma_dgesv_nopiv_batched.argtypes = [integer, integer, pointer, integer, pointer, integer, pointer,
                                                  integer, pointer]
    library.magma_dgetrf_recpanel_batched.argtypes = [integer, integer, integer, pointer, integer, integer, integer,
                                                      pointer, pointer, pointer, integer, integer, pointer]
    library.magma_dgetrf_vbatched.argtypes = [pointer, pointer, pointer, pointer, pointer, pointer, integer, pointer]
    library.magma_dgetrs_batched.argtypes = [integer, integer, integer, pointer, integer, pointer, pointer, integer,
                                             integer, pointer]
    for name in ("magma_dgetrf_nopiv_batched", "magma_dgetrs_nopiv_batched", "magma_dgesv_nopiv_batched",
                 "magma_dgetrf_recpanel_batched", "magma_dgetrf_vbatched", "magma_dgetrs_batched"):
        getattr(library, name).restype = integer
    _NO_TRANS = 111

    def ptr(array):
        return ctypes.c_void_p(int(array.data.ptr))

    for order in _parse_orders(args.orders):
        started = time.perf_counter()
        space, source, beta = _transport_step(cp, mesh, order)
        inputs = _split3_inputs(space, source, beta)
        contraction, _ = default_tensor_engines()
        captured = {}
        real_magma = magma_batched.lu_solve_batched_magma

        def capture(matrices, rhs, **kwargs):
            captured["A"] = matrices.copy()
            captured["B"] = rhs.copy()
            return real_magma(matrices, rhs, **kwargs)

        magma_batched.lu_solve_batched_magma = capture
        try:
            workspace = RawAdvectionTsleTensorWorkspace()
            assemble_projected_advection_trace_system_eliminated_tsle_tensor(
                **dict(inputs, workspace=workspace, cache_local_response=False, precision="float64",
                       contraction=contraction, local_solver="magma"))
            # Compile the FP64 coop solve for the same signature (the tensor path's own kernel).
            assemble_projected_advection_trace_system_eliminated_tsle_tensor(
                **dict(inputs, workspace=RawAdvectionTsleTensorWorkspace(), cache_local_response=False,
                       precision="float64", contraction=contraction, local_solver="coop"))
        finally:
            magma_batched.lu_solve_batched_magma = real_magma
        del workspace
        A0, B0 = captured["A"], captured["B"]
        batch, n, nrhs = int(A0.shape[0]), int(A0.shape[1]), int(B0.shape[1])
        flops = batch * (2.0 / 3.0 * n ** 3 + 2.0 * n * n * nrhs)
        coop_kernel = next(kernels["solve"] for key, kernels in tsle_tensor._MODULE_CACHE.items()
                           if key[1] == n and key[2] == nrhs // 3 and key[-1] == "float64")
        cp.get_default_memory_pool().free_all_blocks()
        print(f"order {order}: E={batch} n={n} nrhs={nrhs} setup {time.perf_counter() - started:.1f} s", flush=True)

        # Problem statistics: diagonal dominance and (below) MAGMA's pivoting.
        absolute = cp.abs(A0)  # absolute[e, col, row] = |A[row, col]|
        diagonal = cp.diagonal(absolute, axis1=1, axis2=2)
        row_ratio = diagonal / (absolute.sum(axis=1) - diagonal)       # |a_ii| / sum_{j!=i} |a_ij|
        column_ratio = diagonal / (absolute.sum(axis=2) - diagonal)    # |a_jj| / sum_{i!=j} |a_ij|
        stats = dict(
            row_dominance_min=float(row_ratio.min()), row_dominance_median=float(cp.median(row_ratio)),
            elements_row_dominant=float(cp.mean(cp.all(row_ratio >= 1, axis=1))),
            column_dominance_min=float(column_ratio.min()),
            column_dominance_median=float(cp.median(column_ratio)),
            elements_column_dominant=float(cp.mean(cp.all(column_ratio >= 1, axis=1))))
        del absolute, diagonal, row_ratio, column_ratio

        rng = np.random.default_rng(order)
        sample = np.sort(rng.choice(batch, size=min(args.sample, batch), replace=False))
        A_s = cp.asnumpy(A0[sample]).transpose(0, 2, 1)   # logical A
        B_s = cp.asnumpy(B0[sample]).transpose(0, 2, 1)   # logical B (n x nrhs)
        X_s = np.linalg.solve(A_s, B_s)
        residual_ld = B_s.astype(np.longdouble) - np.matmul(A_s.astype(np.longdouble), X_s.astype(np.longdouble))
        X_ref = X_s + np.linalg.solve(A_s, residual_ld.astype(np.float64))
        conditions = np.linalg.cond(A_s[: min(512, len(sample))])
        stats.update(cond_median=float(np.median(conditions)), cond_max=float(conditions.max()),
                     host_forward_error_median=float(np.median(
                         np.linalg.norm(X_s - X_ref, axis=(1, 2)) / np.linalg.norm(X_ref, axis=(1, 2)))))

        A_work, B_work = cp.empty_like(A0), cp.empty_like(B0)
        lu = BatchedLUWorkspace()
        lu.ensure(cp, A_work, B_work)
        queue = magma_batched._queue(cp, library)
        info = cp.zeros(batch, dtype=cp.int32)
        reference = {}

        def accuracy(X):
            """Residual, MAGMA difference, and subsample forward error of a stored-form solution."""
            residual = cp.matmul(X, A0) - B0                    # (X_s A0) = (A X)^T in stored form
            res = cp.linalg.norm(residual.reshape(batch, -1), axis=1)
            res /= cp.linalg.norm(A0.reshape(batch, -1), axis=1) * cp.linalg.norm(X.reshape(batch, -1), axis=1)
            res_host = cp.asnumpy(res)
            out = dict(residual_max=float(res_host.max()), residual_p99=float(np.quantile(res_host, 0.99)),
                       residual_median=float(np.median(res_host)), residual_mean=float(res_host.mean()))
            if "X" in reference:
                diff = cp.linalg.norm((X - reference["X"]).reshape(batch, -1), axis=1)
                diff /= cp.linalg.norm(reference["X"].reshape(batch, -1), axis=1)
                out.update(diff_vs_magma=float(diff.max()), diff_vs_magma_median=float(cp.median(diff)))
            X_host = cp.asnumpy(X[sample]).transpose(0, 2, 1)
            forward = np.linalg.norm(X_host - X_ref, axis=(1, 2)) / np.linalg.norm(X_ref, axis=(1, 2))
            out.update(forward_error_max=float(forward.max()), forward_error_median=float(np.median(forward)),
                       finite=bool(cp.all(cp.isfinite(X))))
            return out

        def time_method(solve, *, prepare=None, result=None):
            """Median CUDA-event time of ``solve`` over fresh copies; returns (ms list, X stored form)."""
            samples = []
            for repeat in range(args.repeats + 1):
                if prepare is None:
                    A_work[...] = A0
                    B_work[...] = B0
                else:
                    prepare()
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                begin.record()
                solve()
                end.record()
                end.synchronize()
                if repeat:
                    samples.append(cp.cuda.get_elapsed_time(begin, end))
            return samples, (B_work if result is None else result())

        def record(method, samples, X, **extra):
            ms = statistics.median(samples)
            row = dict(kind="solve", order=order, n=n, nrhs=nrhs, elements=batch, method=method, status="ok",
                       ms=ms, samples_ms=samples, us_per_element=1e3 * ms / batch, gflops=flops / (ms * 1e6),
                       **extra)
            row.update(accuracy(X))
            write(row)
            return row

        # Memory reference: one copy of A and B (read + write each).
        samples = []
        for _ in range(args.repeats):
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
            A_work[...] = A0
            B_work[...] = B0
            end.record()
            end.synchronize()
            samples.append(cp.cuda.get_elapsed_time(begin, end))
        copy_ms = statistics.median(samples)
        write(dict(kind="stats", order=order, n=n, nrhs=nrhs, elements=batch, copy_AB_ms=copy_ms,
                   copy_AB_gbps=2 * (A0.nbytes + B0.nbytes) / (copy_ms * 1e6),
                   solve_bytes_floor_ms=(A0.nbytes + 2 * B0.nbytes) / (2 * (A0.nbytes + B0.nbytes)) * copy_ms,
                   flops_per_element=flops / batch, **stats))

        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        # MAGMA getrf + getrs is the reference and always runs first.
        for method in ["magma"] + [m for m in methods if m != "magma"]:
            try:
                if method == "magma":
                    samples, X = time_method(lambda: magma_batched.lu_solve_batched_magma(
                        A_work, B_work, workspace=lu, check_info=False))
                    nonzero = int(cp.count_nonzero(lu.info))
                    pivots = cp.asnumpy(lu.pivots)
                    natural = np.arange(1, n + 1, dtype=np.int32)
                    swapped = pivots != natural[None, :]
                    pivot_stats = dict(info_nonzero=nonzero, elements_with_row_exchange=float(swapped.any(axis=1).mean()),
                                       exchanges_per_element=float(swapped.sum(axis=1).mean()),
                                       exchange_rate_per_step=[round(float(v), 5) for v in swapped.mean(axis=0)])
                    reference["X"] = X.copy()
                    # getrf alone (getrs = total - getrf).
                    getrf_samples = []
                    for _ in range(args.repeats):
                        A_work[...] = A0
                        begin, end = cp.cuda.Event(), cp.cuda.Event()
                        begin.record()
                        library.magma_dgetrf_batched(n, n, ptr(lu.matrix_pointers), n, ptr(lu.pivot_pointers),
                                                     ptr(lu.info), batch, queue)
                        end.record()
                        end.synchronize()
                        getrf_samples.append(cp.cuda.get_elapsed_time(begin, end))
                    record("magma_getrf_getrs", samples, X, getrf_ms=statistics.median(getrf_samples), **pivot_stats)
                elif method == "magma_gesv":
                    samples, X = time_method(lambda: magma_batched.lu_solve_batched_magma(
                        A_work, B_work, workspace=lu, check_info=False, fused=True))
                    record(method, samples, X)
                elif method == "magma_nopiv":
                    def solve():
                        library.magma_dgetrf_nopiv_batched(n, n, ptr(lu.matrix_pointers), n, ptr(info), batch, queue)
                        library.magma_dgetrs_nopiv_batched(_NO_TRANS, n, nrhs, ptr(lu.matrix_pointers), n,
                                                           ptr(lu.rhs_pointers), n, ptr(info), batch, queue)
                    samples, X = time_method(solve)
                    record(method, samples, X, info_nonzero=int(cp.count_nonzero(info)))
                elif method == "magma_gesv_nopiv":
                    samples, X = time_method(lambda: library.magma_dgesv_nopiv_batched(
                        n, nrhs, ptr(lu.matrix_pointers), n, ptr(lu.rhs_pointers), n, ptr(info), batch, queue))
                    record(method, samples, X, info_nonzero=int(cp.count_nonzero(info)))
                elif method == "magma_recpanel":
                    pivinfo = cp.zeros((batch, n), dtype=cp.int32)
                    pivinfo_pointers = cp.arange(batch, dtype=cp.uintp) * cp.uintp(n * 4) + cp.uintp(pivinfo.data.ptr)
                    for width in [int(w) for w in args.recpanel.split(",") if w]:
                        def solve(width=width):
                            info.fill(0)
                            library.magma_dgetrf_recpanel_batched(
                                n, n, width, ptr(lu.matrix_pointers), 0, 0, n, ptr(lu.pivot_pointers),
                                ptr(pivinfo_pointers), ptr(info), 0, batch, queue)
                            library.magma_dgetrs_batched(_NO_TRANS, n, nrhs, ptr(lu.matrix_pointers), n,
                                                         ptr(lu.pivot_pointers), ptr(lu.rhs_pointers), n, batch, queue)
                        samples, X = time_method(solve)
                        record(f"magma_recpanel{width}", samples, X, info_nonzero=int(cp.count_nonzero(info)))
                elif method == "magma_vbatched":
                    sizes = cp.full(batch + 1, n, dtype=cp.int32)
                    def solve():
                        library.magma_dgetrf_vbatched(ptr(sizes), ptr(sizes), ptr(lu.matrix_pointers), ptr(sizes),
                                                      ptr(lu.pivot_pointers), ptr(info), batch, queue)
                        library.magma_dgetrs_batched(_NO_TRANS, n, nrhs, ptr(lu.matrix_pointers), n,
                                                     ptr(lu.pivot_pointers), ptr(lu.rhs_pointers), n, batch, queue)
                    samples, X = time_method(solve)
                    record(method, samples, X, info_nonzero=int(cp.count_nonzero(info)))
                elif method == "cublas":
                    samples, X = time_method(lambda: lu_solve_batched_cublas(A_work, B_work, workspace=lu,
                                                                            check_info=False))
                    record(method, samples, X, info_nonzero=int(cp.count_nonzero(lu.info)))
                elif method == "coop":
                    A_c = cp.ascontiguousarray(A0.transpose(0, 2, 1))
                    B_c = cp.ascontiguousarray(B0.transpose(0, 2, 1))
                    A_cw, B_cw = cp.empty_like(A_c), cp.empty_like(B_c)
                    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
                    limit = int(properties.get("sharedMemPerBlockOptin", 48 * 1024))
                    for block in tsle_tensor._COOP_BLOCKS:
                        shared = tsle_tensor._solve_shared_bytes(n, nrhs, block, 8)
                        if shared > limit:
                            continue

                        def prepare():
                            A_cw[...] = A_c
                            B_cw[...] = B_c

                        samples, X = time_method(
                            lambda: coop_kernel((batch,), (block,), (A_cw, B_cw, np.int64(batch)), shared_mem=shared),
                            prepare=prepare, result=lambda: cp.ascontiguousarray(B_cw.transpose(0, 2, 1)))
                        record("coop", samples, X, config=f"block{block}")
                    del A_c, B_c, A_cw, B_cw
                elif method == "fused":
                    configs = [default_config(n)] + [c for c in _parse_configs(args.fused_configs).get(n, [])
                                                     if c != default_config(n)]
                    for config in configs:
                        samples, X = time_method(lambda: solve_batched_fused(A_work, B_work, config=config, info=info))
                        record("fused", samples, X, config="x".join(map(str, config)),
                               info_nonzero=int(cp.count_nonzero(info)), matrix_untouched=bool(cp.all(A_work == A0)))
                else:
                    raise ValueError(f"unknown method {method}")
            except Exception as error:  # a broken method is a row, not the end of the run
                write(dict(kind="solve", order=order, n=n, method=method, status="error",
                           error=f"{type(error).__name__}: {error}"[:500]))
            cp.get_default_memory_pool().free_all_blocks()
        del A0, B0, A_work, B_work, reference, captured
        cp.get_default_memory_pool().free_all_blocks()
        print(f"order {order} done in {time.perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
