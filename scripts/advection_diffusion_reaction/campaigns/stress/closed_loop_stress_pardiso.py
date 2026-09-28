"""Campaign adapter for the existing bounded, cached-system PyPardiso diagnostic.

Called only by explicit campaign execution. It never reassembles or changes a
cache, never replaces iterative timings, and preserves interrupted attempts.
"""
from pathlib import Path
import os
import sys
from time import monotonic

from scripts.advection_diffusion_reaction.diagnostics import check_cached_adr_pardiso as diagnostic
from scripts.advection_diffusion_reaction.campaigns.logging import event, read_json


def run_coarse_check(key, args, common):
    """Check one selected mesh with resource caps; the runner chooses coarse/all.

    The historical function name is retained for existing callers.
    """
    job = "pardiso_"+key
    result_path = args.output/"jobs"/f"{job}.json"
    previous = read_json(result_path)
    if args.resume and previous.get("status") not in (None, "running"):
        event(args.output, "pardiso_reused", job=job, details=previous)
        print(f"reuse {job}: {previous['status']} "
              f"relres={previous.get('face_relative_residual')}", flush=True)
        return previous
    root = args.output/"pardiso"/key
    attempt = 1
    while (root/f"attempt_{attempt}").exists():
        attempt += 1
    output = root/f"attempt_{attempt}"
    output.mkdir(parents=True)
    spec = args.output/"specs"/f"assemble_{key}.json"
    limits = diagnostic.parser().parse_args([
        "--spec", str(spec), "--output", str(output),
        "--threads", str(args.pardiso_threads),
        "--max-dofs", str(args.pardiso_max_dofs),
        "--rtol", "1e-10", "--timeout", str(args.timeout),
        "--max-rss-gib", str(args.pardiso_max_rss_gib),
        "--reserve-gib", str(args.pardiso_reserve_gib),
    ])
    command = [
        sys.executable, "-u", "-B", str(Path(diagnostic.__file__).resolve()),
        "--worker", "--spec", str(spec.resolve()), "--output", str(output.resolve()),
        "--threads", str(args.pardiso_threads), "--max-dofs", str(args.pardiso_max_dofs),
        "--rtol", "1e-10",
    ]
    environment = dict(os.environ, MKL_NUM_THREADS=str(args.pardiso_threads),
                       OMP_NUM_THREADS=str(args.pardiso_threads), MKL_DYNAMIC="FALSE",
                       OPENBLAS_NUM_THREADS="1", NUMBA_DISABLE_JIT="1",
                       HDGFEM_PRECISION="float64", PYTHONDONTWRITEBYTECODE="1")
    # Select master's direct-solver package even if a sourced environment puts
    # the companion worktree on PYTHONPATH.
    environment["PYTHONPATH"] = str(diagnostic.ROOT)+os.pathsep+environment.get("PYTHONPATH", "")
    common.atomic_json(result_path, dict(status="running", diagnostic_output=str(output)))
    event(args.output, "pardiso_started",
          message=f"start {job}: threads={args.pardiso_threads} "
                  f"max_dofs={args.pardiso_max_dofs} max_rss={args.pardiso_max_rss_gib:g}GiB "
                  f"log={output/'worker.log'}",
          job=job, command=command, diagnostic_output=str(output))
    started = monotonic()
    try:
        result = diagnostic.monitor(command, environment, output, limits)
        if result.get("status") in (None, "running"):
            result.update(status="error", error="PARDISO worker did not produce a terminal result")
    except (ValueError, RuntimeError, OSError) as exc:
        result = dict(status="error", error=str(exc))
    result.update(diagnostic_output=str(output), runner_wall_seconds=monotonic()-started)
    common.atomic_json(result_path, result)
    event(args.output, "pardiso_finished", job=job, details=result)
    print(f"result {job}: {result['status']} "
          f"physical_relres={result.get('physical_relative_residual')} "
          f"face_relres={result.get('face_relative_residual')} "
          f"probe_solution_error={result.get('probe_relative_solution_error')} "
          f"wall={result['runner_wall_seconds']:.3f}s", flush=True)
    if result.get("error"):
        print(f"  reason: {result['error']}", flush=True)
    return result
