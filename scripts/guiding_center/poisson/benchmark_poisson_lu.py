"""Factor a captured Poisson matrix once and measure reused direct solves.

Multithreaded PyPardiso retains its factors on the CPU. An optional GPU
multigrid control uses the identical matrix and RHS. SciPy supplies sparse
containers and independent residual products; all CPU factorization and direct
solves use PyPardiso. No assembly, compilation or time integration is performed.
Workers are sequential and memory monitored.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback

import numpy as np
from scipy import sparse

from scripts.advection_diffusion_reaction.diagnostics.check_cached_adr_pardiso import monitor
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.poisson.benchmark_poisson_backends import digest


CANDIDATES = ("pardiso", "pmg-fast")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="directory containing data/indices/indptr/rhs/guess.npy and probe.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES,
                        default=["pardiso"],
                        help="PyPardiso CPU LU, optionally compared with native GPU multigrid")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-rss-gib", type=float, default=85.)
    parser.add_argument("--reserve-gib", type=float, default=16.)
    parser.add_argument("--timeout", type=float, default=1800.)
    parser.add_argument("--worker", choices=CANDIDATES, help=argparse.SUPPRESS)
    return parser


def worker(args):
    report = dict(status="running", candidate=args.worker, samples=[],
                  new_compilation_allowed=False, time_integration=False,
                  capture=str(args.capture.resolve()), python=sys.version,
                  diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  thread_environment={key: os.environ.get(key) for key in (
                      "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_DYNAMIC", "OMP_DYNAMIC")})
    destination = args.output / "result.json"

    def stage(name, **values):
        report.update(stage=name, **values)
        report["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(name, flush=True)

    factors = operator = None
    try:
        stage("load captured fixed operator")
        metadata = json.loads((args.capture / "probe.json").read_text())
        cpus = sorted(os.sched_getaffinity(0))
        if (args.worker == "pardiso" and max(metadata["shape"]) > 10_000
                and (args.threads < 2 or args.threads not in {16, len(cpus)})):
            raise RuntimeError(
                "Large CPU systems require multithreaded PyPardiso with 16 threads "
                "or all available cores."
            )
        os.sched_setaffinity(0, cpus[:args.threads])
        load = lambda name: np.load(args.capture / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        matrix = sparse.bsr_matrix((load("data"), load("indices"), load("indptr")), shape=tuple(metadata["shape"]))
        rhs, reference = load("rhs"), load("guess")
        norm_rhs = float(np.linalg.norm(rhs))
        target = max(metadata["atol"], metadata["rtol"]*norm_rhs)
        reference_residual = float(np.linalg.norm(rhs-matrix@reference))
        if not np.isfinite(reference_residual) or reference_residual > target:
            raise RuntimeError("Captured reference does not meet the original-system target")
        stage("convert original matrix", trace_dofs=int(matrix.shape[0]),
              matrix_sha256={key:digest(getattr(matrix,key)) for key in ("data","indices","indptr")},
              rhs_sha256=digest(rhs), target=target, rhs_norm=norm_rhs)
        started = time.perf_counter()
        csr = matrix.tocsr()
        csr.eliminate_zeros(); csr.sort_indices()
        if csr.nnz >= np.iinfo(np.int32).max:
            raise RuntimeError("Matrix exceeds the 32-bit sparse index interface")
        report["conversion_seconds"] = time.perf_counter()-started
        report["matrix_nnz"] = int(csr.nnz)

        with kernel_cache_only(True):
            if args.worker == "pardiso":
                import pypardiso
                from hdgfem.linalg.pardiso_diagnostics import pardiso_factor_statistics
                factors = pypardiso.PyPardisoSolver(mtype=11)
                getter = factors.libmkl.MKL_Get_Max_Threads
                getter.argtypes, getter.restype = [], ctypes.c_int
                actual_threads = int(getter())
                if actual_threads != args.threads:
                    raise RuntimeError(f"MKL thread limit {actual_threads} != requested {args.threads}")
                factors.set_iparm(18, -1)
                factors.set_iparm(60, 0)
                stage("CPU PyPardiso LU factorization", mkl_max_threads=actual_threads,
                      cpu_affinity=sorted(os.sched_getaffinity(0)))
                cpu_started = time.process_time()
                started = time.perf_counter()
                factors.factorize(csr)
                report["factorization_seconds"] = time.perf_counter()-started
                report["factorization_cpu_seconds"] = time.process_time()-cpu_started
                report["factorization_effective_cores"] = (
                    report["factorization_cpu_seconds"] / report["factorization_seconds"])
                if matrix.shape[0] > 10_000 and report["factorization_effective_cores"] < 1.5:
                    raise RuntimeError("Large PyPardiso factorization did not demonstrate parallel CPU use")
                report["factor_statistics"] = pardiso_factor_statistics(factors, matrix_nnz=csr.nnz)
                report["timing_scope"] = "PyPardiso solve with retained factors, including wrapper matrix-reuse checks; RHS and solution on CPU"
                solve = lambda: factors.solve(csr, rhs)
                sync = lambda: None
                to_host = np.asarray
            elif args.worker == "pmg-fast":
                import cupy as cp
                from hdgfem.backends.legendre_face_bsr import LegendreFaceBsrOperator, diagonal_block_positions
                from hdgfem.linalg.face_hp_multigrid import FaceBlockHpMgPcgSolver
                sync = cp.cuda.get_current_stream().synchronize
                stage("fast pMG control on identical captured matrix/RHS, zero initial guess")
                sync(); started = time.perf_counter()
                data, indices, indptr = (cp.asarray(matrix.data), cp.asarray(matrix.indices), cp.asarray(matrix.indptr))
                operator = LegendreFaceBsrOperator(indptr, indices, data)
                diagonal = cp.asarray(diagonal_block_positions(matrix.indptr, matrix.indices))
                factors = FaceBlockHpMgPcgSolver(indptr=indptr, indices=indices, data=data,
                            degree=matrix.blocksize[0]-1, diagonal_positions=diagonal,
                            preconditioner_policy="fast")
                device_rhs = cp.asarray(rhs)
                sync(); report["hierarchy_setup_seconds"] = time.perf_counter()-started
                report["timing_scope"] = "native fast PCGF solve on the identical matrix/RHS, zero initial guess, cached hierarchy; original-matrix residual checks included"
                report["iterative_details"] = []

                def solve():
                    result = factors.solve(device_rhs, initial_guess=None, assembly_matvec=operator.matvec,
                                rtol=metadata["rtol"], atol=metadata["atol"], maxiter=500)
                    report["iterative_details"].append(dict(iterations=result.iterations, converged=result.converged))
                    if not result.converged:
                        raise RuntimeError("Fast PCGF control did not converge")
                    return result.solution

                to_host = cp.asnumpy
            else:
                raise ValueError(f"Unsupported candidate: {args.worker}")
            stage("reused factor solves")
            for repeat in range(args.repeats+1):
                sync(); cpu_started = time.process_time(); started = time.perf_counter()
                x = solve()
                sync(); elapsed = time.perf_counter()-started
                cpu_seconds = time.process_time()-cpu_started
                x = to_host(x)
                residual = float(np.linalg.norm(rhs-matrix@x))
                difference = float(np.linalg.norm(x-reference)/np.linalg.norm(reference))
                row = dict(warmup=repeat==0, seconds=elapsed, residual_norm=residual,
                           relative_residual=residual/norm_rhs, relative_trace_difference=difference,
                           passed=bool(np.isfinite(residual) and residual<=target and difference<1e-8))
                if args.worker == "pardiso":
                    row["pardiso_phase"] = int(factors.phase)
                    row["cpu_seconds"] = cpu_seconds
                    row["effective_cores"] = cpu_seconds/elapsed
                    row["passed"] = row["passed"] and factors.phase == 33
                else:
                    row["gpu_free_bytes"] = cp.cuda.runtime.memGetInfo()[0]
                    row["cupy_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
                report["samples"].append(row)
                stage("reused solve completed")
                print(json.dumps(row), flush=True)
                if not row["passed"]:
                    raise RuntimeError("Solve failed the original-system residual/reference gate")
            report["mean_reused_seconds"] = statistics.mean(row["seconds"] for row in report["samples"] if not row["warmup"])
            stage("completed", status="passed")
    except Exception as exc:
        stage("failed", status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if args.worker == "pardiso" and factors is not None:
            factors.free_memory(everything=True)
        if args.worker == "pmg-fast":
            if factors is not None:
                factors.close()
            if operator is not None:
                operator.close()
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.repeats < 1 or args.threads < 1:
        raise ValueError("repeats and threads must be positive")
    if args.threads > len(os.sched_getaffinity(0)):
        raise ValueError("Requested more threads than affinity-visible CPU cores")
    if any(not np.isfinite(value) or value <= 0 for value in (
            args.timeout, args.max_rss_gib, args.reserve_gib)):
        raise ValueError("time and memory limits must be positive and finite")
    if args.worker:
        worker(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1", PYTHONDONTWRITEBYTECODE="1",
               MKL_NUM_THREADS=str(args.threads), OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE")
    results = {}
    for candidate in args.candidates:
        output = args.output / candidate
        output.mkdir()
        command = [sys.executable, "-u", "-B", "-m", __spec__.name,
                   "--worker", candidate, "--capture", str(args.capture.resolve()),
                   "--output", str(output.resolve()), "--repeats", str(args.repeats),
                   "--threads", str(args.threads)]
        args.monitor_label = candidate
        results[candidate] = monitor(command, env, output, args)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2)+"\n")
    return int(any(result.get("status") != "passed" for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
