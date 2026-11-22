#!/usr/bin/env python3
"""Check one native block-ILU application against a known exact solution."""
from __future__ import annotations

import argparse
from pathlib import Path


def block_row_count(value):
    count = int(value)
    if not 1 <= count <= (2**31 - 1) // 4:
        raise argparse.ArgumentTypeError("block rows must be positive and fit 32-bit scalar dimensions")
    return count


def make_block_diagonal_arrays(matrix, rhs_block, block_rows, layout):
    """Repeat one dense block using BSR storage proportional to block_rows."""
    import numpy as np

    values = matrix if layout == "ROW_MAJOR" else matrix.T
    return (np.arange(block_rows + 1, dtype=np.int32),
            np.arange(block_rows, dtype=np.int32),
            np.ascontiguousarray(np.tile(values, (block_rows, 1, 1))),
            np.tile(rhs_block, block_rows))


def measure_block_errors(matrix, rhs, exact_block, result):
    """Keep aggregate metrics and reject an error isolated to any one block."""
    import numpy as np

    result_blocks = np.asarray(result, dtype=np.float64).reshape(-1, 4)
    rhs_blocks = np.asarray(rhs, dtype=np.float64).reshape(-1, 4)
    errors = result_blocks - exact_block
    residuals = result_blocks @ matrix.astype(np.float64).T - rhs_blocks
    exact_norm = np.linalg.norm(exact_block)
    block_errors = np.linalg.norm(errors, axis=1) / exact_norm
    block_residuals = np.linalg.norm(residuals, axis=1) / np.linalg.norm(rhs_blocks, axis=1)
    return dict(
        relative_solution_error=float(np.linalg.norm(errors) / (np.sqrt(len(result_blocks)) * exact_norm)),
        relative_residual=float(np.linalg.norm(residuals) / np.linalg.norm(rhs_blocks)),
        max_block_relative_solution_error=float(np.max(block_errors)),
        max_block_relative_residual=float(np.max(block_residuals)),
        worst_solution_block=int(np.argmax(block_errors)),
        worst_residual_block=int(np.argmax(block_residuals)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-library", type=Path, required=True)
    parser.add_argument("--verbosity", type=int, choices=range(4), default=2)
    parser.add_argument("--block-rows", type=block_row_count, default=1,
                        help="Repeat the oracle block on a diagonal graph; 8193 exercises paired inverse tiles")
    args = parser.parse_args()
    import cupy as cp
    import numpy as np
    import pyamgx
    from common import amgx_library_provenance, write_report

    library = amgx_library_provenance()
    if Path(library["path"]) != args.expected_library.resolve():
        raise RuntimeError("Loaded AMGX library differs from the requested build")
    block = np.array([[1, .4, .1, -.2], [.2, 1.3, .3, .05],
                      [-.1, .1, .9, .2], [.15, -.2, .3, 1.1]])
    exact = np.array([1., -2., 3., -4.])
    report = dict(test="one native block-ILU application", synthetic=True,
                  amgx_library=library, condition_number=float(np.linalg.cond(block)),
                  block_rows=args.block_rows, results=[], status="running")

    def native(message):
        if args.verbosity >= 3:
            print(message, end="", flush=True)

    pyamgx.register_print_callback(native)
    pyamgx.initialize()
    try:
        for mode, dtype, tolerance in (("dDDI", np.float64, 1e-10), ("dFFI", np.float32, 1e-5)):
            for backend in ("amgx", "cusparse_legacy"):
                for layout in (("ROW_MAJOR", "COL_MAJOR") if backend == "amgx" else ("ROW_MAJOR",)):
                    for scale in (1., 1e-2, 1e-4):
                        row = dict(mode=mode, backend=backend, layout=layout, scale=scale,
                                   block_rows=args.block_rows, status="failed")
                        objects = []
                        def keep(obj):
                            objects.append(obj)
                            return obj
                        try:
                            matrix = (scale * block).astype(dtype)
                            rhs_block = (matrix.astype(np.float64) @ exact).astype(dtype)
                            row_ptrs, columns, values, rhs = make_block_diagonal_arrays(
                                matrix, rhs_block, args.block_rows, layout)
                            configuration = dict(config_version=2, exception_handling=0,
                                determinism_flag=1, block_format=layout, solver=dict(
                                    scope="ilu_once", solver="MULTICOLOR_ILU", max_iters=1,
                                    ilu_sparsity_level=0, block_ilu_backend=backend,
                                    relaxation_factor=1., coloring_level=1,
                                    matrix_coloring_scheme="MIN_MAX", max_uncolored_percentage=0.,
                                    reorder_cols_by_color=int(backend == "amgx"),
                                    insert_diag_while_reordering=int(backend == "amgx"),
                                    monitor_residual=0, store_res_history=0, print_solve_stats=0,
                                    print_coloring_info=int(args.verbosity >= 3)))
                            cfg = keep(pyamgx.Config().create_from_dict(configuration))
                            resource = keep(pyamgx.Resources().create_simple(cfg))
                            a = keep(pyamgx.Matrix().create(resource, mode=mode))
                            b = keep(pyamgx.Vector().create(resource, mode=mode))
                            x = keep(pyamgx.Vector().create(resource, mode=mode))
                            solver = keep(pyamgx.Solver().create(resource, cfg, mode=mode))
                            a.upload(row_ptrs, columns, values, block_dims=[4, 4],
                                     shape=[args.block_rows, args.block_rows])
                            b.upload(rhs, block_dim=4)
                            x.upload(np.zeros(4 * args.block_rows, dtype=dtype), block_dim=4)
                            solver.setup(a)
                            solver.solve(b, x, zero_initial_guess=True)
                            cp.cuda.runtime.deviceSynchronize()
                            result = x.download().astype(np.float64)
                            metrics = measure_block_errors(matrix, rhs, exact, result)
                            row.update(metrics)
                            row.update(determinant=float(np.linalg.det(matrix.astype(np.float64))),
                                       solution=result[:4].tolist(), configuration=configuration,
                                       status="passed" if np.isfinite(result).all() and max(
                                           metrics["max_block_relative_solution_error"],
                                           metrics["max_block_relative_residual"]) <= tolerance else "failed")
                            if args.block_rows > 1:
                                row["last_block_solution"] = result[-4:].tolist()
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        finally:
                            for obj in reversed(objects):
                                obj.destroy()
                        report["results"].append(row)
                        write_report(args.output, report)
                        if args.verbosity >= 1:
                            block_summary = (f" blocks={args.block_rows} "
                                             f"max_block_residual={row.get('max_block_relative_residual', float('nan')):.3e}"
                                             if args.block_rows > 1 else "")
                            print(f"[ILU inverse] {mode:4s} {backend:16s} {layout:9s} scale={scale:.0e} "
                                  f"{row['status']} residual={row.get('relative_residual', float('nan')):.3e}"
                                  f"{block_summary}", flush=True)
    finally:
        pyamgx.finalize()
    failures = sum(row["status"] != "passed" for row in report["results"])
    report["status"] = "failed" if failures else "passed"
    write_report(args.output, report)
    print(f"[ILU inverse] {len(report['results']) - failures}/{len(report['results'])} passed; {args.output}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
