#!/usr/bin/env python3
"""Isolated FP64 CSR AMGX trials for a poorly conditioned nonsymmetric system."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def configurations(rtol, max_iters, restart, verbosity=2):
    configs = {}
    for outer in ("FGMRES", "PBICGSTAB"):
        for kind in ("dilu", "ilu0", "classical_dilu", "classical_ilu0", "aggregation_dilu"):
            smooth = {"solver": "MULTICOLOR_ILU" if "ilu0" in kind else "MULTICOLOR_DILU",
                      "max_iters": 1, "relaxation_factor": 1.0}
            if "ilu0" in kind:
                smooth["ilu_sparsity_level"] = 0
                # Native multicolor ILU factorization supports 4x4 blocks in
                # this AMGX build. Use its explicit cuSPARSE backend for CSR.
                smooth.update(block_ilu_backend="cusparse_legacy", reorder_cols_by_color=0,
                              insert_diag_while_reordering=0)
            if kind.startswith(("classical", "aggregation")):
                classical = kind.startswith("classical")
                pre = dict(solver="AMG", algorithm="CLASSICAL" if classical else "AGGREGATION",
                           selector="PMIS" if classical else "SIZE_2", cycle="V", max_iters=1,
                           max_levels=30, presweeps=1, postsweeps=1, smoother=smooth,
                           coarse_solver="DENSE_LU_SOLVER", min_coarse_rows=128)
                if classical:
                    pre.update(interpolator="D2", strength_threshold=0.25)
            else:
                pre = smooth
            solver = dict(solver=outer, max_iters=max_iters, tolerance=rtol,
                          convergence="RELATIVE_INI_CORE", norm="L2", use_scalar_norm=1,
                          monitor_residual=1, store_res_history=1,
                          print_solve_stats=int(verbosity >= 3), obtain_timings=int(verbosity >= 2),
                          preconditioner=pre)
            if outer == "FGMRES":
                solver["gmres_n_restart"] = restart
            configs[f"{outer.lower()}_{kind}"] = dict(config_version=2, exception_handling=1,
                                                       determinism_flag=1, solver=solver)
    # Longer restart can help a badly conditioned nonnormal operator.
    configs["fgmres_classical_dilu_r200"] = copy.deepcopy(configs["fgmres_classical_dilu"])
    configs["fgmres_classical_dilu_r200"]["solver"]["gmres_n_restart"] = 200
    return configs


def worker(args, config):
    from common import load_system, log, provenance, residual, write_report
    import numpy as np
    report = dict(backend="pyamgx", candidate=args.worker, config=config, mode="dDDI",
                  machine=provenance(), status="loading", samples=[], rtol=args.rtol, verbosity=args.verbosity)
    write_report(args.output, report)
    objects = []
    initialized = False
    try:
        matrix, rhs, report["matrix"] = load_system(args.matrix, args.group, args.index_base, args.verbosity)
        if args.inspect_only:
            report["status"] = "inspected"
            return 0
        import cupy as cp
        import pyamgx
        cp.cuda.Device(0).use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        report["gpu"] = dict(name=props["name"].decode(), total_bytes=int(props["totalGlobalMem"]))
        if not args.allow_other_gpu and "H200" not in report["gpu"]["name"]:
            raise RuntimeError("Expected H200; use --allow-other-gpu for a small validation run")
        pyamgx.register_print_callback(lambda message: print(message, end="", flush=True) if args.verbosity >= 3 else None)
        pyamgx.initialize()
        initialized = True

        def keep(obj):
            objects.append(obj)
            return obj

        def sync():
            cp.cuda.runtime.deviceSynchronize()

        cfg = keep(pyamgx.Config().create_from_dict(config))
        resources = keep(pyamgx.Resources().create_simple(cfg))
        a = keep(pyamgx.Matrix().create(resources, mode="dDDI"))
        b = keep(pyamgx.Vector().create(resources, mode="dDDI"))
        x = keep(pyamgx.Vector().create(resources, mode="dDDI"))
        solver = keep(pyamgx.Solver().create(resources, cfg, mode="dDDI"))
        report["status"] = "uploading"
        log(args.verbosity, 1, f"{args.worker}: uploading CSR/RHS to {report['gpu']['name']}")
        log(args.verbosity, 2, f"AMGX configuration: {json.dumps(config)}")
        write_report(args.output, report)
        sync()
        started = time.perf_counter()
        # NumPy/SciPy CSR -> AMGX's owned CUDA CSR. No CuPy CSR duplicate is needed.
        a.upload_CSR(matrix)
        b.upload(rhs)
        x.upload(np.zeros_like(rhs))
        sync()
        report["upload_seconds"] = time.perf_counter() - started
        report["status"] = "setting_up"
        log(args.verbosity, 1, f"{args.worker}: setting up preconditioner")
        write_report(args.output, report)
        started = time.perf_counter()
        solver.setup(a)
        sync()
        report["setup_seconds"] = time.perf_counter() - started
        log(args.verbosity, 2, f"Upload {report['upload_seconds']:.6f} s, setup {report['setup_seconds']:.6f} s")
        report["status"] = "solving"
        write_report(args.output, report)
        for trial in range(args.repeats):
            log(args.verbosity, 1, f"{args.worker}: solve {trial + 1}/{args.repeats} from zero initial guess")
            b.upload(rhs)
            x.upload(np.zeros_like(rhs))
            sync()
            started = time.perf_counter()
            solver.solve(b, x, zero_initial_guess=True)
            sync()
            elapsed = time.perf_counter() - started
            # Validate against the original FP64 CPU CSR, outside GPU solve timing.
            solution = x.download()
            sample = dict(solve_seconds=elapsed, iterations=int(solver.iterations_number),
                          amgx_status=solver.status, **residual(matrix, rhs, solution, args.rtol))
            sample["native_residual_history"] = [float(solver.get_residual(i))
                                                  for i in range(sample["iterations"] + 1)]
            sample["passed"] = sample["passed"] and sample["amgx_status"] == "success"
            report["samples"].append(sample)
            log(args.verbosity, 2, f"Solve {trial + 1}: "
                f"{ {key: value for key, value in sample.items() if key != 'native_residual_history'} }")
            write_report(args.output, report)
        report["median_solve_seconds"] = float(np.median([s["solve_seconds"] for s in report["samples"]]))
        report["fresh_setup_solve_seconds"] = report["setup_seconds"] + report["samples"][0]["solve_seconds"]
        report["status"] = "passed" if all(s["passed"] for s in report["samples"]) else "numerical_failure"
        return 0 if report["status"] == "passed" else 1
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        print(report["error"], flush=True)
        return 1
    finally:
        for obj in reversed(objects):
            try:
                obj.destroy()
            except Exception:
                report.setdefault("cleanup_errors", []).append(traceback.format_exc())
        if initialized:
            try:
                pyamgx.finalize()
            except Exception:
                report.setdefault("cleanup_errors", []).append(traceback.format_exc())
        write_report(args.output, report)
        log(args.verbosity, 0, f"{args.worker}: {report['status']}; report: {args.output}")


def main():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from common import load_system, log, parser, provenance, validate_args, write_report
    p = parser(__doc__)
    p.add_argument("--max-iters", type=int, default=1000)
    p.add_argument("--restart", type=int, default=100)
    p.add_argument("--candidates", nargs="+", help="Subset of names from --list-configs")
    p.add_argument("--list-configs", action="store_true")
    p.add_argument("--timeout", type=float, default=1800, help="Seconds per isolated trial, including loading")
    p.add_argument("--allow-other-gpu", action="store_true")
    p.add_argument("--worker", help="Internal isolated worker", choices=list(configurations(1e-8, 1000, 100)))
    args = p.parse_args()
    validate_args(args)
    if args.max_iters < 1 or args.restart < 1 or args.timeout <= 0:
        p.error("max-iters, restart and timeout must be positive")
    configs = configurations(args.rtol, args.max_iters, args.restart, args.verbosity)
    if args.list_configs:
        print(json.dumps(configs, indent=2))
        return 0
    if args.worker:
        return worker(args, configs[args.worker])
    if args.inspect_only:
        _, _, metadata = load_system(args.matrix, args.group, args.index_base, args.verbosity)
        write_report(args.output, dict(status="inspected", matrix=metadata, machine=provenance()))
        log(args.verbosity, 0, f"Matrix inspected; report: {args.output}")
        return 0
    names = args.candidates or list(configs)
    if set(names) - configs.keys():
        p.error(f"Unknown candidates: {set(names) - configs.keys()}")
    trial_dir = args.output.parent / (args.output.stem + "_trials")
    trial_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", machine=provenance(), results=[])
    write_report(args.output, report)
    for name in names:
        output = trial_dir / f"{name}.json"
        # A new invocation must not mistake an old worker result for this attempt.
        output.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", name,
                   "--matrix", str(args.matrix.resolve()), "--group", args.group,
                   "--index-base", args.index_base, "--output", str(output.resolve()),
                   "--rtol", str(args.rtol), "--max-iters", str(args.max_iters),
                   "--restart", str(args.restart), "--repeats", str(args.repeats)]
        command.extend(["--verbosity", str(args.verbosity)])
        if args.allow_other_gpu:
            command.append("--allow-other-gpu")
        log(args.verbosity, 1, f"Starting {name}: {output}")
        with output.with_suffix(".log").open("w") as logstream:
            try:
                if args.verbosity:
                    # Tee the worker's flushed stage/iteration output while preserving its log.
                    # A thread copies output so subprocess.wait can enforce timeout even if
                    # a native solver stalls without producing another line.
                    import threading
                    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    def tee():
                        for line in proc.stdout:
                            logstream.write(line)
                            logstream.flush()
                            print(line, end="", flush=True)
                    reader = threading.Thread(target=tee, daemon=True)
                    reader.start()
                    try:
                        returncode = proc.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                        raise
                    finally:
                        reader.join()
                    status = dict(returncode=returncode)
                else:
                    result = subprocess.run(command, stdout=logstream, stderr=subprocess.STDOUT, timeout=args.timeout)
                    status = dict(returncode=result.returncode)
            except subprocess.TimeoutExpired:
                status = dict(status="timeout", timeout_seconds=args.timeout)
        observed = json.loads(output.read_text()) if output.exists() else {"status": "worker_failed"}
        observed.update(status)
        if observed.get("returncode", 0) and observed["status"] not in ("error", "numerical_failure"):
            observed["status"] = "worker_failed"
        report["results"].append(dict(observed, candidate=name, report=str(output)))
        log(args.verbosity, 1, f"{name}: {observed['status']}")
        write_report(args.output, report)
    valid = [r for r in report["results"] if r["status"] == "passed"]
    report["ranking"] = [r["candidate"] for r in sorted(valid, key=lambda r: r["fresh_setup_solve_seconds"])]
    report["status"] = "completed" if valid else "no_converged_candidate"
    write_report(args.output, report)
    log(args.verbosity, 0, f"Converged ranking (setup + first solve): {report['ranking']}; report: {args.output}")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
