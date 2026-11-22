#!/usr/bin/env python3
"""GPU regression for native ILU row capacity; these are synthetic systems."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import traceback

CASES = ("star127", "star128", "star129", "star1023", "star1024", "star1025", "bipartite129")
# The tree is a separate coloring probe: MIN_MAX produces only 64 filled
# columns, so it is not a regression for the extended workspace.
PROBE_CASES = ("tree1024",)


def make_matrix(case, dtype):
    import numpy as np
    from scipy.sparse import bsr_matrix, coo_matrix

    n = int(re.search(r"\d+$", case)[0])
    rows, cols = list(range(n)), list(range(n))
    if case.startswith("star"):
        rows += [0] * (n - 1) + list(range(1, n))
        cols += list(range(1, n)) + [0] * (n - 1)
    elif case.startswith("bipartite"):
        for left in range(64):
            for right in range(64, n):
                rows += [left, right]
                cols += [right, left]
    else:
        # Fine rows have at most 34 blocks. Actual fill depends on coloring:
        # MIN_MAX reaches only 64, so this is a coloring probe, not a test of
        # the extended factor workspace.
        for branch in range(1, 32):
            rows += [0, branch]
            cols += [branch, 0]
            for leaf in range(32 + (branch - 1) * 32, 32 + branch * 32):
                rows += [branch, leaf]
                cols += [leaf, branch]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    graph.sort_indices()
    rng = np.random.default_rng(619)
    values = rng.normal(0, 0.02 / np.sqrt(n), (graph.nnz, 4, 4)).astype(dtype)
    for row in range(n):
        lo, hi = graph.indptr[row:row + 2]
        index = lo + np.searchsorted(graph.indices[lo:hi], row)
        values[index] += np.eye(4, dtype=dtype) * 8
    matrix = bsr_matrix((values, graph.indices, graph.indptr), shape=(4 * n, 4 * n))
    exact = rng.standard_normal(4 * n).astype(dtype)
    rhs = (matrix.astype(np.float64) @ exact.astype(np.float64)).astype(dtype)
    return matrix, rhs, exact


def worker(args):
    import cupy as cp
    import numpy as np
    import pyamgx
    from common import amgx_library_provenance, write_report

    report = dict(test="native ILU capacity regression", synthetic=True,
                  case=args.case, mode=args.mode, layout=args.layout,
                  solver=args.solver, preconditioner=args.preconditioner,
                  coloring=args.coloring,
                  known_initial_guess=args.known_initial_guess,
                  verify_after_rejection=args.verify_after_rejection,
                  without_preconditioner=args.without_preconditioner, status="failed")
    objects, messages = [], []
    initialized = False

    def keep(obj):
        objects.append(obj)
        return obj

    def native(message):
        messages.append(message)
        if args.verbosity >= 3:
            print(message, end="", flush=True)

    try:
        report["amgx_library"] = amgx_library_provenance()
        if Path(report["amgx_library"]["path"]) != args.expected_library.resolve():
            raise RuntimeError("The loaded AMGX library is not the requested test build")
        dtype = np.float64 if args.mode == "dDDI" else np.float32
        matrix, rhs, exact = make_matrix(args.case, dtype)
        level = int(args.case.startswith(("bipartite", "tree")))
        expected_rejection = (
            not args.without_preconditioner and args.preconditioner == "MULTICOLOR_ILU" and (args.case == "star1025" or
            (args.capacity == 128 and args.case not in ("star127", "star128", "tree1024")))
        )
        report["max_input_row_blocks"] = int(np.diff(matrix.indptr).max())
        report["expected_rejection"] = expected_rejection
        threshold = 1e-10 if dtype == np.float64 else 1e-5
        configuration = dict(config_version=2, exception_handling=0, determinism_flag=1,
            block_format=args.layout, solver=dict(solver=args.solver, max_iters=100,
                gmres_n_restart=40, tolerance=threshold * 0.01,
                convergence="RELATIVE_INI_CORE", norm="L2", use_scalar_norm=1,
                bsr_spmv_backend="cusparse_generic",
                monitor_residual=1, store_res_history=1, print_solve_stats=int(args.verbosity >= 3),
                preconditioner=dict(solver=args.preconditioner, block_ilu_backend="amgx",
                    ilu_sparsity_level=level, max_iters=1, relaxation_factor=1.0,
                    coloring_level=level + 1, matrix_coloring_scheme=args.coloring,
                    max_uncolored_percentage=0.0, reorder_cols_by_color=1,
                    insert_diag_while_reordering=1, print_coloring_info=1)))
        if args.without_preconditioner:
            configuration["solver"]["preconditioner"] = "NOSOLVER"
        report["configuration"] = configuration
        pyamgx.register_print_callback(native)
        pyamgx.initialize()
        initialized = True
        cfg = keep(pyamgx.Config().create_from_dict(configuration))
        resources = keep(pyamgx.Resources().create_simple(cfg))
        a = keep(pyamgx.Matrix().create(resources, mode=args.mode))
        b = keep(pyamgx.Vector().create(resources, mode=args.mode))
        x = keep(pyamgx.Vector().create(resources, mode=args.mode))
        solver = keep(pyamgx.Solver().create(resources, cfg, mode=args.mode))
        n = matrix.shape[0] // 4
        # block_format describes the uploaded coefficients; it does not
        # transpose each dense block on our behalf.
        values = matrix.data if args.layout == "ROW_MAJOR" else matrix.data.swapaxes(1, 2)
        a.upload(matrix.indptr, matrix.indices, np.ascontiguousarray(values),
                 block_dims=[4, 4], shape=[n, n])
        b.upload(rhs, block_dim=4)
        x.upload(exact if args.known_initial_guess else np.zeros_like(rhs), block_dim=4)
        setup_succeeded = False
        try:
            solver.setup(a)
            cp.cuda.runtime.deviceSynchronize()
            setup_succeeded = True
        except pyamgx.AMGXError as exc:
            if not (expected_rejection and exc.error_code == pyamgx.RC.NO_MEMORY):
                raise
            # The detailed FatalError text goes to stderr, not the print
            # callback. The parent checks that text after the worker exits.
            report.update(status="capacity_rejected", result="expected_capacity_rejection", error=str(exc))
            if args.verify_after_rejection:
                # Factorization rejects this graph after the fine matrix has
                # already been reordered. Inspect that same matrix with an
                # independent solver, without applying any rejected factors.
                import copy
                recovery_config = copy.deepcopy(configuration)
                recovery_config["solver"]["preconditioner"] = "NOSOLVER"
                recovery_cfg = keep(pyamgx.Config().create_from_dict(recovery_config))
                solver = keep(pyamgx.Solver().create(resources, recovery_cfg, mode=args.mode))
                solver.setup(a)
                cp.cuda.runtime.deviceSynchronize()
                report.update(recovered_after_capacity_rejection=True,
                              recovery_configuration=recovery_config)
        if setup_succeeded or args.verify_after_rejection:
            setup_rhs = b.download()
            setup_x = x.download()
            expected_x = exact if args.known_initial_guess else np.zeros_like(rhs)
            report.update(setup_rhs_max_change=float(np.max(np.abs(setup_rhs - rhs))),
                          setup_x_max_change=float(np.max(np.abs(setup_x - expected_x))))
            if not (np.array_equal(setup_rhs, rhs) and np.array_equal(setup_x, expected_x)):
                raise AssertionError("Solver setup changed an uploaded RHS or initial guess")
            if expected_rejection and setup_succeeded:
                raise AssertionError("An over-capacity matrix was unexpectedly accepted")
            if setup_succeeded and args.capacity == 1024 and not args.without_preconditioner and args.preconditioner == "MULTICOLOR_ILU":
                match = re.search(r"maximum LU row = (\d+) block columns; factor workspace = (\d+)", "".join(messages))
                if not match:
                    raise AssertionError("Missing capacity-dispatch diagnostic from the isolated library")
                row_size, selected = map(int, match.groups())
                report.update(max_factor_row_blocks=row_size, selected_capacity=selected)
                expected = 128 if args.case in ("star127", "star128", "tree1024") else 1024
                if selected != expected or (args.case.startswith("bipartite") and row_size <= 128):
                    raise AssertionError("The regression did not exercise the intended capacity path")
            solver.solve(b, x, zero_initial_guess=not args.known_initial_guess)
            cp.cuda.runtime.deviceSynchronize()
            downloaded = x.download()
            cp.cuda.runtime.deviceSynchronize()
            solution = downloaded.astype(np.float64)
            relative_residual = float(np.linalg.norm(matrix.astype(np.float64) @ solution - rhs) / np.linalg.norm(rhs))
            relative_error = float(np.linalg.norm(solution - exact) / np.linalg.norm(exact))
            report.update(relative_residual=relative_residual, relative_solution_error=relative_error,
                          iterations=int(solver.iterations_number),
                          native_initial_residual=float(solver.get_residual(0)),
                          solution_head=solution[:8].tolist(), exact_head=exact[:8].tolist(),
                          largest_solution_error_index=int(np.argmax(np.abs(solution - exact))))
            if not (np.isfinite(solution).all() and relative_residual <= threshold and relative_error <= threshold):
                raise AssertionError("Independent residual/known-solution check failed")
            report.update(status="passed", result="numerically_verified")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        for obj in reversed(objects):
            try:
                obj.destroy()
            except Exception:
                pass
        if initialized:
            pyamgx.finalize()
        write_report(args.output, report)
    print(f"[ILU capacity] {args.case} {args.mode} {args.layout}: {report['status']}", flush=True)
    return int(report["status"] not in ("passed", "capacity_rejected"))


def main():
    from benchmark_amgx_strong import run_trial
    from common import write_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-library", type=Path, required=True)
    parser.add_argument("--capacity", type=int, choices=(128, 1024), default=1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbosity", type=int, choices=range(4), default=2)
    parser.add_argument("--case", choices=CASES + PROBE_CASES)
    parser.add_argument("--mode", choices=("dDDI", "dFFI"))
    parser.add_argument("--layout", choices=("ROW_MAJOR", "COL_MAJOR"))
    parser.add_argument("--solver", choices=("FGMRES", "PBICGSTAB"), default="PBICGSTAB")
    parser.add_argument("--without-preconditioner", action="store_true")
    parser.add_argument("--preconditioner", choices=("MULTICOLOR_ILU", "MULTICOLOR_DILU"), default="MULTICOLOR_ILU")
    parser.add_argument("--coloring", choices=("MIN_MAX", "SERIAL_GREEDY_BFS"), default="MIN_MAX")
    parser.add_argument("--known-initial-guess", action="store_true")
    parser.add_argument("--verify-after-rejection", action="store_true",
                        help="Inspect the same reordered matrix after an expected ILU capacity rejection")
    args = parser.parse_args()
    if args.case:
        if not args.mode or not args.layout:
            parser.error("A worker requires --mode and --layout")
        return worker(args)
    results = []
    directory = args.output.with_suffix("").with_name(args.output.stem + "_trials")
    directory.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        for mode in ("dDDI", "dFFI"):
            for layout in ("ROW_MAJOR", "COL_MAJOR"):
                output = directory / f"{case}_{mode}_{layout}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--case", case,
                    "--mode", mode, "--layout", layout, "--capacity", str(args.capacity),
                    "--solver", args.solver,
                    "--preconditioner", args.preconditioner,
                    "--coloring", args.coloring,
                    "--expected-library", str(args.expected_library), "--output", str(output),
                    "--verbosity", str(args.verbosity)]
                if args.without_preconditioner:
                    command.append("--without-preconditioner")
                if args.known_initial_guess:
                    command.append("--known-initial-guess")
                if args.verify_after_rejection:
                    command.append("--verify-after-rejection")
                outcome = run_trial(command, output, timeout=180, verbosity=args.verbosity)
                observed = json.loads(output.read_text()) if output.exists() else dict(status="failed", error="Worker produced no report")
                observed.update(outcome)
                if observed.get("status") == "capacity_rejected":
                    text = output.with_suffix(".log").read_text()
                    expected_message = ("supported capacity is 1024" if args.capacity == 1024 else
                                        "Number of nonzeros per row exceeds allocated shared memory")
                    observed["status"] = "passed" if expected_message in text else "failed"
                    observed["capacity_error_text_verified"] = expected_message in text
                if observed.get("returncode", 0):
                    observed["status"] = "failed"
                results.append(observed)
                write_report(args.output, dict(status="running", synthetic=True, results=results))
    passed = sum(result.get("status") == "passed" for result in results)
    write_report(args.output, dict(status="passed" if passed == len(results) else "failed",
                                  synthetic=True, results=results))
    print(f"[ILU capacity] {passed}/{len(results)} checks passed; report {args.output}")
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
