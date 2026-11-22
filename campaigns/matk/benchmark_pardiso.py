#!/usr/bin/env python3
"""CPU-only in-core real nonsymmetric PARDISO LU benchmark."""
from __future__ import annotations

import os
import time
import traceback


def script_log(verbosity, level, message):
    from common import log
    log(verbosity, level, "[matK] " + message)


def print_fields(verbosity, level, title, rows):
    """One timestamp per block, with labels and values aligned beneath it."""
    if verbosity < level:
        return
    print(flush=True)
    script_log(verbosity, level, title)
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"[matK]   {label:<{width}} : {value}", flush=True)


def print_table(verbosity, level, title, headers, rows):
    if verbosity < level or not rows:
        return
    values = [[str(value) for value in row] for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in values)) for i, header in enumerate(headers)]

    def line(row):
        return "[matK]   " + "  ".join(f"{value:<{width}}" if i == 0 else f"{value:>{width}}"
                                for i, (value, width) in enumerate(zip(row, widths)))

    print(flush=True)
    script_log(verbosity, level, title)
    print(line(headers), flush=True)
    print("[matK]   " + "  ".join("-" * width for width in widths), flush=True)
    for row in values:
        print(line(row), flush=True)


def print_summary(args, report):
    rows = [("Status", report["status"].upper())]
    matrix = report.get("matrix")
    if matrix:
        rows.extend([("Matrix", f"{matrix['shape'][0]:,} x {matrix['shape'][1]:,}"),
                     ("Stored entries", f"{matrix['canonical_nnz']:,}"),
                     ("Load / validation", f"{matrix['load_seconds']:.3f} s")])
    for key, label in (("analysis_factorization_seconds", "Analysis + factorization"),
                       ("median_solve_seconds", "Median reused-factor solve"),
                       ("fresh_factor_solve_seconds", "Factorization + first solve")):
        if key in report:
            rows.append((label, f"{report[key]:.3f} s"))
    if report["samples"]:
        rows.extend([("Worst relative residual", f"{max(s['relative_residual'] for s in report['samples']):.3e}"),
                     ("Residual target", f"{args.rtol:.3e}")])
    rows.append(("JSON report", str(args.output)))
    print_fields(args.verbosity, 0, "PARDISO result", rows)


def main():
    # Set threading before importing NumPy, SciPy or MKL.
    import argparse
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    early, _ = preliminary.parse_known_args()
    if early.threads < 1:
        raise ValueError("threads must be positive")
    for key in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT"):
        os.environ[key] = str(early.threads)
    os.environ.update(MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
    os.environ.pop("MKL_DOMAIN_NUM_THREADS", None)
    from common import load_system, parser, provenance, residual, validate_args, write_report
    log = script_log
    import numpy as np

    p = parser(__doc__)
    p.add_argument("--threads", type=int, default=early.threads)
    p.add_argument("--save-solution", action="store_true")
    p.add_argument("--profile-cpu", action="store_true", help="Record per-phase and per-thread Linux CPU usage")
    p.add_argument("--cpu-sample-interval", type=float, default=0.2, help="CPU sampling interval in seconds (default 0.2)")
    args = p.parse_args()
    validate_args(args)
    if not np.isfinite(args.cpu_sample_interval) or args.cpu_sample_interval < 0.05:
        p.error("cpu-sample-interval must be finite and at least 0.05 seconds")
    report = dict(backend="pypardiso", machine=provenance(), status="loading", samples=[],
                  threads=args.threads, rtol=args.rtol, verbosity=args.verbosity)
    write_report(args.output, report)
    from cpu_monitor import CpuProfile
    profile = CpuProfile(args.output, enabled=args.profile_cpu, interval=args.cpu_sample_interval)
    solver = None
    try:
        print_fields(args.verbosity, 1, "PARDISO benchmark", [
            ("Matrix file", str(args.matrix)), ("Host", report["machine"]["host"]),
            ("MKL threads", args.threads), ("Repeated solves", args.repeats),
            ("Residual target", f"{args.rtol:.3e}"),
            ("CPU sampling", f"{args.cpu_sample_interval:g} s" if args.profile_cpu else "off")])
        with profile.stage("loading"):
            # Keep the loader's progress messages; present its metadata below
            # instead of repeating the full machine-readable JSON in the log.
            matrix, rhs, report["matrix"] = load_system(args.matrix, args.group, args.index_base,
                                                        min(args.verbosity, 1), log_prefix="[matK] ")
        metadata = report["matrix"]
        print_fields(args.verbosity, 2, "Matrix and right-hand side", [
            ("Dimensions", f"{matrix.shape[0]:,} x {matrix.shape[1]:,}"),
            ("CSR entries", f"{matrix.nnz:,}"),
            ("CSR storage", f"{metadata['csr_bytes'] / 1024**3:.3f} GiB"),
            ("Input index base", metadata["input_index_base"]),
            ("Rows reordered", "yes" if metadata["rows_reordered"] else "no"),
            ("RHS dataset", f"/{args.group}/{metadata['rhs_dataset']}"),
            ("RHS nonzero entries", f"{metadata['rhs_nonzero']:,}"),
            ("RHS norm", f"{metadata['rhs_norm']:.6e}")])
        if args.inspect_only:
            report["status"] = "inspected"
            return 0
        with profile.stage("backend_initialization"):
            import ctypes
            import pypardiso
            solver = pypardiso.PyPardisoSolver(mtype=11)
        if args.profile_cpu or args.verbosity >= 3:
            # Instrument the existing ctypes entry point to distinguish native
            # PARDISO from PyPardiso hashing/checks/index conversion. Arguments and
            # native return handling remain in the installed PyPardiso wrapper.
            native_pardiso = solver._mkl_pardiso
            from contextlib import nullcontext
            flush_native = ctypes.CDLL(None).fflush
            flush_native.argtypes, flush_native.restype = [ctypes.c_void_p], ctypes.c_int

            def measured_native(*native_args):
                phase = profile.stack[-1] if profile.stack else "other"
                native_phase = solver.phase
                if args.verbosity >= 3:
                    print(f"\n[PARDISO native | phase {native_phase}] BEGIN\n", flush=True)
                try:
                    with profile.stage(phase + ".native") if args.profile_cpu else nullcontext():
                        return native_pardiso(*native_args)
                finally:
                    if args.verbosity >= 3:
                        flush_native(None)
                        print(f"\n[PARDISO native | phase {native_phase}] END\n", flush=True)

            solver._mkl_pardiso = measured_native
        solver.set_statistical_info_on() if args.verbosity >= 3 else solver.set_statistical_info_off()
        # Same preset as the existing ADR campaign: MKL nonsymmetric defaults,
        # explicit CSR orientation and in-core factors. Threads are matrix-dependent.
        for key, value in {12: 0, 18: -1, 60: 0}.items():
            solver.set_iparm(key, value)
        getter = solver.libmkl.MKL_Get_Max_Threads
        getter.restype, getter.argtypes = ctypes.c_int, []
        report["mkl_max_threads"] = int(getter())
        if report["mkl_max_threads"] != args.threads:
            raise RuntimeError("MKL thread count does not match the request")
        report.update(status="factorizing", mtype=11,
                      preset="existing ADR in-core nonsymmetric LU (MKL defaults)",
                      iparm_indexing="one_based", pypardiso_version=pypardiso.__version__)
        write_report(args.output, report)
        log(args.verbosity, 1, f"Analysis + LU factorization  |  nonsymmetric  |  in-core  |  {args.threads} threads")
        started = time.perf_counter()
        with profile.stage("analysis_factorization"):
            solver.factorize(matrix)
        report["analysis_factorization_seconds"] = time.perf_counter() - started
        report["factor_iparm"] = {str(i): int(solver.get_iparm(i)) for i in (7, 14, 15, 16, 17, 18, 60)}
        if report["factor_iparm"]["60"] != 0:
            raise RuntimeError("Expected in-core PARDISO factors")
        log(args.verbosity, 1, f"Analysis + LU factorization  |  completed in {report['analysis_factorization_seconds']:.3f} s")
        print_table(args.verbosity, 2, "PARDISO factor diagnostics (1-based iparm)",
                    ["Parameter", "Value"],
                    [(f"iparm({key})", f"{value:,}") for key, value in report["factor_iparm"].items()])
        report["status"] = "solving"
        write_report(args.output, report)
        for trial in range(args.repeats):
            log(args.verbosity, 1, f"Solve {trial + 1:>2}/{args.repeats}  |  reusing LU factors")
            started = time.perf_counter()
            with profile.stage(f"solve_{trial + 1}"):
                x = solver.solve(matrix, rhs)
            elapsed = time.perf_counter() - started
            if solver.phase != 33:
                raise RuntimeError("Expected a solve reusing the LU factors (phase 33)")
            with profile.stage(f"validation_{trial + 1}"):
                sample = dict(solve_seconds=elapsed, **residual(matrix, rhs, x, args.rtol))
            sample["iterative_refinement_steps"] = int(solver.get_iparm(7))
            report["samples"].append(sample)
            print_table(args.verbosity, 2, f"Solve {trial + 1} validation", [
                "Solve", "Time [s]", "Residual norm", "Relative residual", "Refinements", "Result"], [[
                    str(trial + 1), f"{elapsed:.3f}", f"{sample['residual_norm']:.3e}",
                    f"{sample['relative_residual']:.3e}", sample["iterative_refinement_steps"],
                    "PASS" if sample["passed"] else "FAIL"]])
            write_report(args.output, report)
        report["median_solve_seconds"] = float(np.median([s["solve_seconds"] for s in report["samples"]]))
        report["fresh_factor_solve_seconds"] = report["analysis_factorization_seconds"] + report["samples"][0]["solve_seconds"]
        report["status"] = "passed" if all(s["passed"] for s in report["samples"]) else "numerical_failure"
        if args.save_solution:
            np.save(args.output.with_suffix(".solution.npy"), x)
        return 0 if report["status"] == "passed" else 1
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        print("\n".join("[matK] " + line for line in report["error"].splitlines()), flush=True)
        return 1
    finally:
        if solver is not None:
            try:
                with profile.stage("cleanup"):
                    solver.free_memory(everything=True)
            except Exception:
                report["cleanup_error"] = traceback.format_exc()
        try:
            if args.profile_cpu:
                report["cpu_profile"] = profile.finish()
                phases = report["cpu_profile"]["phases"]
                # Group nested native calls beneath their enclosing phase.
                phases = sorted(phases, key=lambda phase: next(
                    i for i, parent in enumerate(phases)
                    if parent["phase"] == phase["phase"].removesuffix(".native")) * 2
                    + int(phase["phase"].endswith(".native")))
                print_table(args.verbosity, 1, "CPU usage by phase", [
                    "Phase", "Wall [s]", "Avg CPUs", "Alloc. [%]", "Peak CPUs", "Active threads"], [[
                        ("  native PARDISO" if phase["phase"].endswith(".native")
                         else phase["phase"].replace("_", " ")),
                        f"{phase['wall_seconds']:.3f}", f"{phase['average_cpu_equivalent_cores']:.2f}",
                        f"{phase['allocated_cpu_percent']:.1f}",
                        f"{phase['peak_sample_cpu_equivalent_cores']:.2f}" if "peak_sample_cpu_equivalent_cores" in phase else "--",
                        phase.get("peak_active_threads", "--")] for phase in phases])
                log(args.verbosity, 1, "CPU time includes spin waits; peaks and active threads are sampled. '--' means no stable sample.")
                print_fields(args.verbosity, 2, "CPU profile files", [
                    (key.replace("_", " "), value) for key, value in report["cpu_profile"]["files"].items()])
        except Exception:
            report["cpu_profile_error"] = traceback.format_exc()
        write_report(args.output, report)
        print_summary(args, report)


if __name__ == "__main__":
    raise SystemExit(main())
