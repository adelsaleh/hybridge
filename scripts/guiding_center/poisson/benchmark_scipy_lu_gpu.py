"""Temporary, explicitly requested SciPy LU -> GPU Poisson experiment.

This is an isolated exception to the workspace's PyPardiso-only CPU policy.
It never changes production solver options. SciPy uses the serial SuperLU
driver; its dense BLAS work may use multiple threads. Complete factors are saved before GPU
experiments, so a later run can reuse them with --factor-cache.
An optional smaller ITER capture assembles once from a cached mesh. No mesh
generation, time integration or new compilation is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

import numpy as np
from scipy import sparse

from scripts.advection_diffusion_reaction.diagnostics.check_cached_adr_pardiso import monitor, write_json
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
from scripts.guiding_center.poisson.benchmark_poisson_backends import digest


METHODS = ("spsv", "spsv-graph", "spsm", "spsm-graph", "cupyx")
CAPTURE = Path("run_outputs/solver_studies/iter_repeated_poisson_20260924")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--capture", type=Path, default=CAPTURE)
    source.add_argument("--iter-mesh-size", type=float, help="assemble a smaller p=6 ITER capture from an existing mesh cache")
    source.add_argument("--smoke", action="store_true", help="use a synthetic 48-by-48 matrix; omits pMG control")
    parser.add_argument("--max-capture-dofs", type=int, default=100_000, help="size limit for the optional smaller ITER capture")
    parser.add_argument("--output", type=Path, required=True, help="new empty result directory")
    parser.add_argument("--factor-cache", type=Path, help="new factor directory or a completed cache to reuse")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--threads", type=int, default=24, help="BLAS limit; SuperLU itself is serial")
    parser.add_argument("--factor-driver", choices=("auto", "splu", "spilu-nodrop"), default="auto")
    parser.add_argument("--ordering", choices=("MMD_AT_PLUS_A", "COLAMD", "MMD_ATA", "NATURAL"), default="MMD_AT_PLUS_A")
    parser.add_argument("--initial-fill", type=float, default=5., help="initial no-drop-driver allocation; not a fill cap")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--gpu-reserve-gib", type=float, default=4.)
    parser.add_argument("--max-rss-gib", type=float, default=85.)
    parser.add_argument("--reserve-gib", type=float, default=16.)
    parser.add_argument("--timeout", type=float, default=21600., help="per-worker seconds, including CPU factorization")
    parser.add_argument("--live-output", action=argparse.BooleanOptionalAction, default=True,
                        help="print worker stages and individual solve timings while retaining worker.log")
    parser.add_argument("--heartbeat-seconds", type=float, default=10.)
    parser.add_argument("--no-pmg", action="store_true", help="omit the matched native pMG-AMG control")
    parser.add_argument("--worker", choices=("capture", "factor", *METHODS), help=argparse.SUPPRESS)
    parser.add_argument("--capture-destination", type=Path, help=argparse.SUPPRESS)
    return parser


def load_capture(directory):
    metadata = json.loads((directory / "probe.json").read_text())
    load = lambda name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    matrix = sparse.bsr_matrix((load("data"), load("indices"), load("indptr")), shape=tuple(metadata["shape"]))
    rhs, reference = load("rhs"), load("guess")
    if (matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]
            or any(array.dtype != np.float64 for array in (matrix, rhs, reference))
            or rhs.shape != (matrix.shape[0],) or reference.shape != rhs.shape):
        raise ValueError("Require the captured square FP64 Poisson matrix and matching vectors")
    hashes = {key: digest(getattr(matrix, key)) for key in ("data", "indices", "indptr")}
    target = max(metadata["atol"], metadata["rtol"] * float(np.linalg.norm(rhs)))
    if not np.linalg.norm(rhs - matrix @ reference) <= target:
        raise ValueError("Captured reference fails the original-system residual target")
    return matrix, rhs, reference, metadata, hashes, target


def load_factor_cache(directory, shape):
    def factor(prefix):
        arrays = [np.load(directory / f"{prefix}_{name}.npy", mmap_mode="r", allow_pickle=False)
                  for name in ("data", "indices", "indptr")]
        result = sparse.csr_matrix(tuple(arrays), shape=shape, copy=False)
        if not result.has_canonical_format:
            raise ValueError("Factor cache is not canonical CSR")
        return result
    return (factor("L"), factor("U"),
            np.load(directory / "perm_r.npy", allow_pickle=False),
            np.load(directory / "perm_c.npy", allow_pickle=False))


def factor_cpu(args, matrix, matrix_hashes, stage):
    from scipy import __version__ as scipy_version
    from scipy.sparse.linalg import splu, spilu

    if any(args.factor_cache.iterdir()):
        raise ValueError("CPU factorization requires an empty factor-cache directory")
    csr = matrix.tocsr()
    csr.eliminate_zeros()
    csr.sort_indices()
    # SciPy's installed SuperLU uses int32 sparse factor indices. The standard
    # driver's initial 30*nnz estimate overflows on this 133.5-million-nnz matrix.
    maximum = np.iinfo(np.int32).max
    driver = args.factor_driver
    if driver == "auto":
        driver = "spilu-nodrop" if 30 * csr.nnz >= maximum else "splu"
    if driver == "splu" and 30 * csr.nnz >= maximum:
        raise ValueError("splu initial capacity exceeds int32; use --factor-driver auto or spilu-nodrop")
    if driver == "spilu-nodrop" and args.initial_fill * csr.nnz >= maximum:
        raise ValueError("Initial SuperLU allocation exceeds int32; reduce --initial-fill")
    if max(csr.shape[0], csr.nnz) >= maximum:
        raise ValueError("Input exceeds SciPy SuperLU's sparse-index limit")
    stage("SciPy CPU factorization: serial SuperLU driver with threaded BLAS",
          driver=driver, scipy=scipy_version, input_nnz=int(csr.nnz), ordering=args.ordering)
    wall, cpu = time.perf_counter(), time.process_time()
    if driver == "splu":
        factors = splu(csr.tocsc(), permc_spec=args.ordering, options={"Equil": False})
    else:
        # No basic/secondary/dynamic/interpolation dropping, fill quotas, MILU
        # modification, or artificial pivot fill. Initial storage may grow.
        # Full factor identity is verified below before this cache is accepted.
        factors = spilu(csr.tocsc(), permc_spec=args.ordering, drop_tol=0.,
                        fill_factor=args.initial_fill,
                        options={"Equil": False, "ILU_DropRule": 0,
                                 "ILU_FillTol": 0., "ILU_MILU": "SILU"})
    setup = time.perf_counter() - wall
    cpu_seconds = time.process_time() - cpu
    stage("factorization completed; saving explicit factors", factorization_seconds=setup,
          factorization_cpu_seconds=cpu_seconds, factorization_effective_cores=cpu_seconds/setup)
    expected_bytes = int(factors.nnz) * 12 + 2*(matrix.shape[0]+1)*4
    if expected_bytes + 1024**3 > shutil.disk_usage(args.factor_cache).free:
        raise OSError("Insufficient disk space to save the explicit LU factors")
    manifest = dict(status="writing", matrix_sha256=matrix_hashes, shape=list(matrix.shape),
                    driver=driver, scipy=scipy_version, ordering=args.ordering,
                    factorization_seconds=setup, factorization_cpu_seconds=cpu_seconds,
                    factorization_effective_cores=cpu_seconds/setup,
                    initial_fill=args.initial_fill, dropping_disabled=True, files={})
    for name in ("perm_r", "perm_c"):
        np.save(args.factor_cache / f"{name}.npy", getattr(factors, name), allow_pickle=False)
    for prefix in ("L", "U"):
        factor = getattr(factors, prefix).tocsr()
        factor.sum_duplicates()
        factor.sort_indices()
        if factor.nnz >= maximum:
            raise ValueError("Exported factor exceeds the current int32 upload path")
        manifest[f"{prefix}_nnz"] = int(factor.nnz)
        manifest[f"{prefix}_bytes"] = sum(getattr(factor, name).nbytes for name in ("data", "indices", "indptr"))
        for name in ("data", "indices", "indptr"):
            filename = f"{prefix}_{name}.npy"
            np.save(args.factor_cache / filename, getattr(factor, name), allow_pickle=False)
            manifest["files"][filename] = (args.factor_cache / filename).stat().st_size
        del factor
        stage(f"saved {prefix} factor", **{f"{prefix}_nnz": manifest[f"{prefix}_nnz"]})
    del factors, csr
    stage("validate complete Pr*A*Pc=L*U through three action probes")
    lower, upper, perm_r, perm_c = load_factor_cache(args.factor_cache, matrix.shape)
    from hdgfem.linalg.gpu.triangular import superlu_gather_indices
    inverse_rows, columns = superlu_gather_indices(perm_r, perm_c)
    rng = np.random.default_rng(928)
    errors = []
    for _ in range(3):
        vector = rng.standard_normal(matrix.shape[0])
        expected = (matrix @ vector[columns])[inverse_rows]
        actual = lower @ (upper @ vector)
        errors.append(float(np.linalg.norm(actual-expected) / np.linalg.norm(expected)))
    if not all(np.isfinite(error) and error < 1e-12 for error in errors):
        raise RuntimeError(f"Full LU factor-action validation failed: {errors}")
    manifest.update(status="passed", factor_action_relative_errors=errors)
    write_json(args.factor_cache / "metadata.json", manifest)
    stage("factor cache completed", factor_cache=str(args.factor_cache),
          factor_action_relative_errors=errors,
          factor_bytes=manifest["L_bytes"]+manifest["U_bytes"])


def gpu_benchmark(args, matrix, rhs, reference, matrix_hashes, target, stage):
    from hdgfem.runtime.optional import require_cupy
    from hdgfem.linalg.gpu.sparse import scipy_csr_to_cupy
    from hdgfem.linalg.gpu.triangular import ReusableCuPyLUSolve

    manifest = json.loads((args.factor_cache / "metadata.json").read_text())
    if manifest["status"] != "passed" or manifest["matrix_sha256"] != matrix_hashes:
        raise ValueError("Factor cache does not match the original matrix")
    for name, size in manifest["files"].items():
        if (args.factor_cache / name).stat().st_size != size:
            raise ValueError(f"Factor cache file is incomplete: {name}")
    lower, upper, perm_r, perm_c = load_factor_cache(args.factor_cache, matrix.shape)
    cp = require_cupy()
    sync = cp.cuda.get_current_stream().synchronize
    csr = matrix.tocsr()
    csr.eliminate_zeros()
    csr.sort_indices()
    factor_bytes = manifest["L_bytes"] + manifest["U_bytes"]
    original_bytes = sum(getattr(csr, name).nbytes for name in ("data", "indices", "indptr"))
    reserve = int(args.gpu_reserve_gib*1024**3)
    free, total = cp.cuda.runtime.memGetInfo()
    stage("GPU memory gate", gpu=str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
          cupy=cp.__version__, gpu_free_bytes=free, gpu_total_bytes=total, factor_bytes=factor_bytes,
          original_matrix_bytes=original_bytes)
    if factor_bytes + original_bytes + 16*rhs.nbytes + reserve > free:
        raise MemoryError("LU factors, original matrix, vector buffers and reserve do not fit GPU memory")
    sync(); started = time.perf_counter()
    l_gpu, u_gpu = scipy_csr_to_cupy(lower), scipy_csr_to_cupy(upper)
    a_gpu = scipy_csr_to_cupy(csr)
    b_gpu = cp.asarray(rhs)
    x_gpu, correction, residual = [cp.empty_like(b_gpu) for _ in range(3)]
    sync()
    stage("factors transferred; constructing reusable GPU solve",
          transfer_seconds=time.perf_counter()-started)
    method = args.worker.removesuffix("-graph")
    graph = args.worker.endswith("-graph")
    samples = []
    with ReusableCuPyLUSolve(l_gpu, u_gpu, perm_r, perm_c, method=method,
                            graph=graph, reserve_bytes=reserve) as solver:
        stage("GPU solve ready", analysis_seconds=solver.analysis_seconds,
              graph_setup_seconds=solver.graph_setup_seconds, workspace_bytes=solver.workspace_bytes,
              timing_scope="device RHS copy, permutations and triangular solves; checked time also includes original GPU residuals and any LU refinement; host validation excluded")
        start_event, stop_event = cp.cuda.Event(), cp.cuda.Event()
        for repeat in range(args.warmup+args.repeats):
            sync(); started = time.perf_counter()
            start_event.record()
            solver.solve(b_gpu, out=x_gpu)
            stop_event.record()
            sync()
            raw_seconds = time.perf_counter()-started
            gpu_seconds = cp.cuda.get_elapsed_time(start_event, stop_event)/1000
            for refinements in range(args.refinement_steps+1):
                cp.subtract(b_gpu, a_gpu @ x_gpu, out=residual)
                device_residual = float(cp.linalg.norm(residual))
                if np.isfinite(device_residual) and device_residual <= target:
                    break
                if refinements < args.refinement_steps and np.isfinite(device_residual):
                    solver.solve(residual, out=correction)
                    cp.add(x_gpu, correction, out=x_gpu)
                else:
                    raise RuntimeError(f"GPU residual {device_residual:.4g} exceeds target {target:.4g}")
            sync()
            checked_seconds = time.perf_counter()-started
            x = cp.asnumpy(x_gpu)
            host_residual = float(np.linalg.norm(rhs-matrix@x))
            difference = float(np.linalg.norm(x-reference)/np.linalg.norm(reference))
            row = dict(warmup=repeat<args.warmup, raw_solve_seconds=raw_seconds,
                       gpu_event_seconds=gpu_seconds, checked_solve_seconds=checked_seconds,
                       refinements=refinements, device_residual=device_residual,
                       original_host_residual=host_residual, relative_trace_difference=difference,
                       passed=bool(np.isfinite(host_residual) and host_residual<=target and difference<1e-8))
            samples.append(row)
            stage("GPU solve completed", samples=samples)
            label = f"warmup {repeat+1}/{args.warmup}" if row["warmup"] else f"solve {repeat-args.warmup+1}/{args.repeats}"
            print(f"[{args.worker}] {label}: raw={raw_seconds*1000:.3f} ms "
                  f"device={gpu_seconds*1000:.3f} ms checked={checked_seconds*1000:.3f} ms "
                  f"residual={host_residual:.3e} target={target:.3e} refinements={refinements}", flush=True)
            if not row["passed"]:
                raise RuntimeError("GPU LU solution failed the original host BSR residual/reference check")
        measured = [row for row in samples if not row["warmup"]]
        stage("GPU benchmark completed", samples=samples,
              mean_raw_solve_seconds=statistics.mean(row["raw_solve_seconds"] for row in measured),
              mean_gpu_event_seconds=statistics.mean(row["gpu_event_seconds"] for row in measured),
              mean_checked_solve_seconds=statistics.mean(row["checked_solve_seconds"] for row in measured))


def worker(args):
    from hdgfem.linalg.gpu import triangular as cupy_triangular

    report = dict(status="running", candidate=args.worker, time_integration=False,
                  new_compilation_allowed=False, thread_limit=args.threads,
                  scipy_factorization_exception=True,
                  triangular_helper_sha256=hashlib.sha256(Path(cupy_triangular.__file__).read_bytes()).hexdigest(),
                  thread_environment={key: os.environ.get(key) for key in (
                      "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                      "MKL_DYNAMIC", "OMP_DYNAMIC")},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def stage(name, **values):
        report.update(stage=name, **values)
        write_json(args.output / "result.json", report)
        if name == "GPU solve completed":
            return
        details = []
        for key, value in values.items():
            if key in {"samples", "matrix_sha256", "rhs_sha256", "traceback", "cpu_affinity"}:
                continue
            if key.endswith("_seconds"):
                details.append(f"{key}={value:.6f} s")
            elif key.endswith("_bytes"):
                details.append(f"{key}={value/1024**2:.2f} MiB")
            elif key.endswith("_nnz") or key in {"trace_dofs", "triangles"}:
                details.append(f"{key}={value:,}")
            else:
                details.append(f"{key}={value}")
        print(f"[{args.worker}] {name}" + ("; " + "; ".join(details) if details else ""), flush=True)

    try:
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.threads])
        if args.worker == "capture":
            if args.capture_destination is None or args.iter_mesh_size is None:
                raise ValueError("Capture worker needs an ITER mesh size and a destination")
            args.capture = args.capture_destination
            with kernel_cache_only(True):
                make_iter_capture(args, stage)
            stage("completed", status="passed")
            return
        stage("loading captured Poisson matrix", cpu_affinity=sorted(os.sched_getaffinity(0)))
        matrix, rhs, reference, metadata, hashes, target = load_capture(args.capture)
        stage("capture verified", trace_dofs=int(matrix.shape[0]), matrix_sha256=hashes,
              rhs_sha256=digest(rhs), target=target, capture=str(args.capture))
        if args.worker == "factor":
            factor_cpu(args, matrix, hashes, stage)
        else:
            with kernel_cache_only(True):
                gpu_benchmark(args, matrix, rhs, reference, hashes, target, stage)
        stage("completed", status="passed")
    except Exception as error:
        stage("failed", status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise


def make_smoke_capture(directory):
    rng = np.random.default_rng(916)
    n = 48
    matrix = (8*sparse.eye(n) + sparse.random(n, n, density=.14, random_state=rng)).tocsr()[rng.permutation(n)]
    matrix = matrix.tobsr(blocksize=(1, 1))
    reference = rng.standard_normal(n)
    rhs = matrix @ reference
    directory.mkdir()
    for name in ("data", "indices", "indptr"):
        np.save(directory / f"{name}.npy", getattr(matrix, name), allow_pickle=False)
    np.save(directory / "rhs.npy", rhs, allow_pickle=False)
    np.save(directory / "guess.npy", reference, allow_pickle=False)
    write_json(directory / "probe.json", dict(shape=list(matrix.shape), atol=1e-11, rtol=1e-11, synthetic=True))


def make_iter_capture(args, stage):
    """Use the production p=6 operator on a cached coarse ITER mesh, with f=1."""
    from dataclasses import replace
    from unittest.mock import patch

    from hdgfem import DGSpace, DiffusionReactionHDGSolver
    from hdgfem.runtime.optional import require_cupy
    from hdgfem.core.geometry import iter_geometry_path
    from hdgfem.core.mesh import gmsh_geo_mesh
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.configuration import _make_poisson_options

    config = replace(preset_by_key("positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr"),
                     mesh_size=args.iter_mesh_size, minimum_triangles=0, verbosity=0)
    args.capture.mkdir(parents=True)
    stage("loading smaller cached ITER mesh", mesh_size=args.iter_mesh_size, degree=config.order)
    with patch("gmsh.model.mesh.generate", side_effect=RuntimeError("An existing mesh cache is required")):
        mesh = gmsh_geo_mesh(args.iter_mesh_size, path=iter_geometry_path(),
                             algorithm=config.gmsh_algorithm, verbosity=0, log_cache=True)
    dofs = len(mesh.int_edges_inds)*(config.order+1)
    stage("cached mesh loaded", triangles=mesh.num_tri, trace_dofs=dofs)
    if dofs > args.max_capture_dofs:
        raise ValueError(f"Smaller capture has {dofs:,} DOFs, exceeds --max-capture-dofs={args.max_capture_dofs:,}")
    space = DGSpace(mesh, config.order, basis_type=config.basis,
                    volume_quadrature=config.volume_quadrature,
                    volume_quad_1d=config.volume_quad_1d, edge_quad_1d=config.edge_quad_1d)
    solver = DiffusionReactionHDGSolver(space, source=space.constant(1.), reaction=space.constant(0.),
                boundary_condition=lambda x, y: 0.*x, options=_make_poisson_options(config))
    cp = require_cupy()
    try:
        stage("assembling small stationary Poisson operator and native reference", source="constant 1; zero Dirichlet boundary")
        result = solver.solve()
        if solver._raw_cuda_fb_hp_mg_solver is None:
            raise RuntimeError("The native reference solver fell back")
        assembly = solver._raw_cuda_assembly_cache
        matrix = sparse.bsr_matrix(tuple(cp.asnumpy(getattr(assembly, key)) for key in ("data", "indices", "indptr")))
        rhs, reference = cp.asnumpy(assembly.rhs), cp.asnumpy(result.trace_reduced_device)
        target = max(config.poisson_solver_atol, config.poisson_solver_rtol*float(np.linalg.norm(rhs)))
        residual = float(np.linalg.norm(rhs-matrix@reference))
        if not np.isfinite(residual) or residual > target:
            raise RuntimeError(f"Small capture reference residual {residual:.3e} exceeds {target:.3e}")
        for name in ("data", "indices", "indptr"):
            np.save(args.capture / f"{name}.npy", getattr(matrix, name), allow_pickle=False)
        np.save(args.capture / "rhs.npy", rhs, allow_pickle=False)
        np.save(args.capture / "guess.npy", reference, allow_pickle=False)
        write_json(args.capture / "probe.json", dict(shape=list(matrix.shape), triangles=mesh.num_tri,
                   block_size=config.order+1, mesh_size=args.iter_mesh_size,
                   rtol=config.poisson_solver_rtol, atol=config.poisson_solver_atol,
                   initial_residual=residual, source="constant 1; zero Dirichlet boundary",
                   mesh_nodes_sha256=digest(mesh.node_coords), mesh_triangles_sha256=digest(mesh.triangles),
                   time_integration=False, new_compilation=False))
        stage("smaller Poisson capture saved", capture=str(args.capture), residual=residual, target=target)
    finally:
        solver.clear_cache()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.threads <= len(os.sched_getaffinity(0)) or args.repeats<1 or args.warmup<0 or args.refinement_steps<0:
        raise ValueError("Invalid thread/repetition/refinement counts")
    if any(not np.isfinite(value) or value<=0 for value in (
            args.timeout, args.initial_fill, args.max_rss_gib, args.reserve_gib, args.gpu_reserve_gib, args.heartbeat_seconds)):
        raise ValueError("Memory limits, timeout and initial fill must be positive and finite")
    if args.initial_fill < 1:
        raise ValueError("Initial fill must be at least 1")
    if args.max_capture_dofs < 1 or (args.iter_mesh_size is not None and (
            not np.isfinite(args.iter_mesh_size) or args.iter_mesh_size <= 0)):
        raise ValueError("Mesh size and capture DOF limit must be positive")
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("List each GPU method once")
    if args.worker:
        worker(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    if args.smoke:
        args.capture = args.output / "smoke_capture"
        make_smoke_capture(args.capture)
        args.no_pmg = True
    if args.iter_mesh_size is not None:
        args.capture = args.output / "poisson_capture"
    args.capture = args.capture.resolve()
    args.factor_cache = (args.factor_cache or args.output / "factors").resolve()
    args.factor_cache.mkdir(parents=True, exist_ok=True)
    if args.factor_cache == args.capture or args.factor_cache.is_relative_to(args.capture):
        raise ValueError("Factor cache must be separate from the input capture")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", NUMBA_DISABLE_JIT="1",
               MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               OMP_NUM_THREADS=str(args.threads), MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE")
    results = {}
    factor_metadata = None

    def run(candidate):
        output = args.output / candidate
        output.mkdir()
        command = [sys.executable, "-u", "-B", "-m", __spec__.name,
                   "--worker", candidate,
                   "--output", str(output.resolve()), "--factor-cache", str(args.factor_cache),
                   "--threads", str(args.threads), "--factor-driver", args.factor_driver,
                   "--ordering", args.ordering, "--initial-fill", str(args.initial_fill),
                   "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                   "--refinement-steps", str(args.refinement_steps),
                   "--gpu-reserve-gib", str(args.gpu_reserve_gib)]
        if candidate == "capture":
            command += ["--iter-mesh-size", str(args.iter_mesh_size),
                        "--max-capture-dofs", str(args.max_capture_dofs)]
            # The capture worker derives its destination from the parent output.
            command += ["--capture-destination", str(args.capture)]
        else:
            command += ["--capture", str(args.capture)]
        args.monitor_label = candidate
        try:
            results[candidate] = monitor(command, env, output, args)
        finally:
            # Retain a caught interruption or failure even when the monitor raises.
            if candidate not in results and (output / "result.json").exists():
                results[candidate] = json.loads((output / "result.json").read_text())
            write_json(args.output / "summary.json", dict(results=results, factor_cache=str(args.factor_cache),
                                                          factor_metadata=factor_metadata))

    if args.iter_mesh_size is not None:
        run("capture")
        if results["capture"]["status"] != "passed":
            return 1
    if not (args.factor_cache / "metadata.json").exists():
        run("factor")
        if results["factor"]["status"] != "passed":
            return 1
    else:
        print(f"Reusing completed factor cache: {args.factor_cache}", flush=True)
    factor_metadata = json.loads((args.factor_cache / "metadata.json").read_text())
    for method in args.methods:
        run(method)
    if not args.no_pmg:
        output = args.output / "pmg-fast"
        output.mkdir()
        command = [sys.executable, "-u", "-B", "-m", "scripts.guiding_center.poisson.benchmark_poisson_lu",
                   "--worker", "pmg-fast", "--capture", str(args.capture), "--output", str(output.resolve()),
                   "--threads", str(args.threads), "--repeats", str(args.repeats)]
        args.monitor_label = "pMG-AMG"
        results["pmg-fast"] = monitor(command, env, output, args)
    passed = {name: row for name, row in results.items() if name in METHODS and row.get("status") == "passed"}
    best = min(passed, key=lambda name: passed[name]["mean_checked_solve_seconds"]) if passed else None
    pmg = results.get("pmg-fast", {})
    summary = dict(results=results, factor_cache=str(args.factor_cache), factor_metadata=factor_metadata,
                   best_validated_gpu_lu=best,
                   comparison_scope="reused solve including original GPU residual checks and any refinement; setup, uploads and host validation excluded")
    if best and pmg.get("status") == "passed":
        if passed[best]["matrix_sha256"] != pmg["matrix_sha256"] or passed[best]["rhs_sha256"] != pmg["rhs_sha256"]:
            raise RuntimeError("pMG-AMG control matrix/RHS differs from the LU benchmark")
        summary["speedup_over_pmg_amg"] = pmg["mean_reused_seconds"] / passed[best]["mean_checked_solve_seconds"]
    write_json(args.output / "summary.json", summary)
    print("\nValidated steady-state GPU solve times:", flush=True)
    for name, row in passed.items():
        print(f"  {name}: raw {row['mean_raw_solve_seconds']:.6f} s; checked {row['mean_checked_solve_seconds']:.6f} s", flush=True)
    if "speedup_over_pmg_amg" in summary:
        print(f"  Best GPU LU / pMG-AMG speedup: {summary['speedup_over_pmg_amg']:.3f}x", flush=True)
    print(f"Results: {args.output / 'summary.json'}", flush=True)
    return int(any(row.get("status") != "passed" for row in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
