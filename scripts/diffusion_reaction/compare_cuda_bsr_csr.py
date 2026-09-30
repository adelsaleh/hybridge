#!/usr/bin/env python3
"""Compare direct raw-CUDA face-BSR and scalar-CSR HDG Poisson paths.

The default is a radius-5 unstructured Gmsh disk with approximately 150,000
triangles at p=6. The CSR and BSR configurations are independently selectable;
repeated runs can alternate their execution order and emit machine-readable
CSV timing and parity records.

The optional ``--preset poisson_300k_p6`` instead replays an existing radius-1
disk Poisson matrix (315,425 triangles, p=6, tau=1) through PyPardiso LU,
hybrid AMGX, pMG-AMG/PCGF, and element ASM+PP/GMRES. It performs no assembly,
time integration, or new compilation. A preset is read-only planning unless
``--execute`` is supplied. Use ``--output`` with a new directory; ``--threads``
screens LU thread counts before independent confirmation. ``--repeats`` counts
confirmation setups, each with a zero-start first and a reused-setup solve.

All solvers are checked against the original nodal matrix and archived modal
solution. pMG uses an exact basis congruence with its conversion included in
setup. Fresh timings include representation conversion/upload, solver setup,
and the first solve; reused timings exclude setup. File loading, initialization,
discarded warmups, independent host validation and cleanup are not ranked.
GPU solves keep RHS/solution on device; CPU solves keep them on host. The LU
backend timer includes PyPardiso's factor-reuse checks, as in the ADR campaign.
AMGX's best configuration means best of the two leading historical hybrid
Poisson configurations (0/2 and 0/3 Chebyshev-L1 sweeps), not a global search.
``--replay-suite spd-pmg`` screens SPD Cholesky thread counts and nine balanced
pMG cycles, retaining AMGX 0/2 as a control. SPD symmetry checking and upper-CSR
preparation are included in fresh setup; reused solves retain that storage and
the factorization. Both direct paths retain PyPardiso's factor-reuse checks.
``--spd-refinement 2`` enables MKL refinement of that symmetric interpretation.
``--spd-original-refinement 2`` permits up to two timed corrections against
the original matrix. ``--replay-suite spd`` isolates the CPU experiment, while
``spd-pmg-refined`` compares the two lightest pMG candidates with AMGX and SPD.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hdgfem import (
    DGSpace,
    DiffusionReactionHDGSolver,
    evaluate_scalar_error,
    gmsh_disc_mesh,
    rectangle_mesh,
)
from hdgfem.runtime.optional import require_cupy
from hdgfem.linalg.amgx.config import (
    describe_amgx_preconditioner,
    describe_amgx_solver,
    load_amgx_config,
)
from hdgfem.hdg.stabilization import GlobalLengthDiffusion
from scripts.diffusion_reaction.cases import trigonometric_poisson_case


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "amgx"
DEFAULT_CSR_CONFIG = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json"
DEFAULT_BSR_AGGREGATION_CONFIG = (
    CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_aggregation_block_jacobi_bsr.json"
)
DEFAULT_BSR_BLOCK_JACOBI_CONFIG = (
    CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_block_jacobi_bsr.json"
)

# Solver-only replay presets do not generate meshes or integrate trajectories.
REPLAY_PRESETS = {
    "poisson_300k_p6": ROOT / "artifacts/full_bsr_convergence_20260914/p6_300k_controls",
}
PMG_REPLAY_POLICIES = {
    "pmg": ("standard", {}),
    "pmg_cheb1": ("standard", {"chebyshev_order": 1}),
    "pmg_cheb1_coarse2": ("standard", {"chebyshev_order": 1, "coarse_sweeps": 2}),
    "pmg_cheb3": ("standard", {"chebyshev_order": 3}),
    "pmg_cheb4": ("standard", {"chebyshev_order": 4}),
    "pmg_coarse2": ("standard", {"coarse_sweeps": 2}),
    "pmg_coarse_w": ("standard", {"coarse_cycle": "W"}),
    "pmg_halve_light": ("robust", {"chebyshev_order": 2, "sweeps": 1, "coarse_sweeps": 1}),
    "pmg_robust": ("robust", {}),
}


def _default_bsr_config(order: int) -> Path:
    """Choose an AMGX config compatible with the face block size p+1."""
    if 1 <= order <= 4:
        return DEFAULT_BSR_AGGREGATION_CONFIG
    return DEFAULT_BSR_BLOCK_JACOBI_CONFIG


@dataclass(frozen=True)
class BenchmarkRow:
    repeat: int
    matrix_format: str
    config: str
    solver: str
    preconditioner: str
    matrix_bytes: int
    pattern_bytes: int
    assembly_seconds: float
    kernel_seconds: float
    amgx_setup_seconds: float
    amgx_solve_seconds: float
    reconstruction_seconds: float
    total_seconds: float
    iterations: int
    relative_residual: float
    l2_error: float | None
    coefficients: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(REPLAY_PRESETS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="execute a solver-only replay preset")
    parser.add_argument("--threads", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--max-rss-gib", type=float, default=72)
    parser.add_argument("--reserve-gib", type=float, default=16)
    parser.add_argument("--replay-worker", help=argparse.SUPPRESS)
    parser.add_argument("--replay-suite", choices=("baseline", "spd-pmg", "spd", "spd-pmg-refined"), default="baseline")
    parser.add_argument("--spd-refinement", type=int, default=0,
                        help="PARDISO one-based iparm(8), maximum SPD iterative refinements")
    parser.add_argument("--spd-original-refinement", type=int, default=0,
                        help="maximum timed Cholesky corrections using the original full-matrix residual")
    parser.add_argument("--pp-degree", type=int, default=24)
    parser.add_argument(
        "--domain",
        choices=("disk", "structured-rectangle"),
        default="disk",
    )
    parser.add_argument("--mesh-size", type=float, default=0.0345)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--nx", type=int, default=275)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--order", "-p", type=int, default=6)
    parser.add_argument("--only", choices=("both", "csr", "bsr"), default="both")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--format-order",
        choices=("csr-first", "bsr-first", "alternate"),
        default="csr-first",
        help="execution order within each repeat when --only=both",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--minimum-triangles", type=int, default=150_000)
    parser.add_argument("--basis", default="dub_orth")
    parser.add_argument(
        "--trace-basis",
        choices=("legacy-lagrange", "legendre-modal"),
        default="legacy-lagrange",
    )
    parser.add_argument(
        "--raw-block-size", choices=("auto", "32", "64", "128"), default="128"
    )
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--csr-amgx-config", type=Path, default=DEFAULT_CSR_CONFIG)
    parser.add_argument(
        "--bsr-amgx-config",
        type=Path,
        default=None,
        help="override the degree-dependent BSR AMGX config",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--maxiter", type=int, default=10_000)
    parser.add_argument("--evaluate-error", action="store_true")
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def _format_bytes(value: int) -> str:
    return f"{value / (1024.0 ** 2):.1f} MiB"


def _matrix_storage(solver: DiffusionReactionHDGSolver) -> tuple[int, int]:
    cached = solver._raw_cuda_assembly_cache
    if cached is None or cached.raw_assembly is None:
        raise RuntimeError("raw-CUDA matrix cache is unavailable")
    raw = cached.raw_assembly
    if raw.indptr is None or raw.indices is None or raw.data is None:
        raise RuntimeError("compressed raw-CUDA assembly is incomplete")
    return int(raw.data.nbytes), int(raw.indptr.nbytes + raw.indices.nbytes)


def _run_one(
    matrix_format: str,
    *,
    repeat: int,
    space,
    problem,
    tau,
    config,
    config_path,
    args,
):
    solver = DiffusionReactionHDGSolver(
        space,
        diffusion=problem.diffusion,
        stabilization=tau,
        solver="amgx",
        solver_rtol=args.rtol,
        maxiter=args.maxiter,
        scale_system=False,
        amgx_config=config,
        cache_device_matrix=True,
        assembly_backend="raw-cuda",
        trace_basis=args.trace_basis,
        raw_matrix_format=matrix_format,
        raw_block_size=args.raw_block_size,
        boundary_mode="eliminate",
        hdg_postprocess="none",
        verbose=args.verbosity,
    )
    solver.set_problem(problem.source, problem.reaction, problem.exact)
    started = time.perf_counter()
    try:
        result = solver.solve()
        require_cupy().cuda.get_current_stream().synchronize()
        wall_seconds = time.perf_counter() - started
        info = result.global_solve_result
        if info is None:
            raise RuntimeError("AMGX diagnostics are missing")
        details = result.timings.details or {}
        matrix_bytes, pattern_bytes = _matrix_storage(solver)
        coefficients = np.array(result.field.coeffs, copy=True)
        l2_error = None
        if args.evaluate_error:
            l2_error = evaluate_scalar_error(
                result.field,
                problem.exact,
                volume_quad_1d=args.error_volume_quad_1d,
                include_samples=False,
            ).metrics.l2
        return BenchmarkRow(
            repeat=repeat,
            matrix_format=matrix_format,
            config="embedded" if config_path is None else config_path.name,
            solver=describe_amgx_solver(config),
            preconditioner=describe_amgx_preconditioner(config),
            matrix_bytes=matrix_bytes,
            pattern_bytes=pattern_bytes,
            assembly_seconds=float(result.timings.trace_assembly),
            kernel_seconds=float(
                details.get(f"raw.assembly.raw.{matrix_format}_kernel", 0.0)
            ),
            amgx_setup_seconds=float(details.get("solve.amgx.setup", 0.0)),
            amgx_solve_seconds=float(details.get("solve.amgx.solve", 0.0)),
            reconstruction_seconds=float(result.timings.reconstruction),
            total_seconds=float(max(result.timings.total, wall_seconds)),
            iterations=int(info.iteration_count),
            relative_residual=float(info.physical_relative_residual_norm),
            l2_error=l2_error,
            coefficients=coefficients,
        )
    finally:
        solver.clear_cache()


def _coefficient_difference(csr: BenchmarkRow, bsr: BenchmarkRow) -> float:
    scale = max(np.linalg.norm(csr.coefficients.ravel()), np.finfo(float).tiny)
    return float(np.linalg.norm((bsr.coefficients - csr.coefficients).ravel()) / scale)


def _print_results(rows, *, domain, triangles, order, trace_dofs, tau):
    print("\nRaw-CUDA diffusion trace format comparison")
    print(f"  domain         : {domain}")
    print(f"  triangles      : {triangles:,}")
    print(f"  polynomial p   : {order}")
    print(f"  face block size: {order + 1}")
    print(f"  trace DOFs     : {trace_dofs:,}")
    print(f"  tau_d          : {tau:.8g}\n")
    print(
        f"{'rep':>3} {'fmt':<4} {'AMGX solver / preconditioner':<31} {'iter':>7} "
        f"{'residual':>10} {'assembly':>9} {'kernel':>9} {'setup':>9} "
        f"{'solve':>9} {'reconstruct':>11} {'total':>9} {'matrix':>11} {'pattern':>10}"
    )
    print("-" * 155)
    for row in rows:
        label = f"{row.solver}/{row.preconditioner}"
        print(
            f"{row.repeat:3d} {row.matrix_format:<4} {label:<31.31} {row.iterations:7d} "
            f"{row.relative_residual:10.3e} {row.assembly_seconds:9.3f} "
            f"{row.kernel_seconds:9.3f} {row.amgx_setup_seconds:9.3f} "
            f"{row.amgx_solve_seconds:9.3f} {row.reconstruction_seconds:11.3f} "
            f"{row.total_seconds:9.3f} {_format_bytes(row.matrix_bytes):>11} "
            f"{_format_bytes(row.pattern_bytes):>10}"
        )
        print(f"     config: {row.config}")
        if row.l2_error is not None:
            print(f"     manufactured L2 error: {row.l2_error:.6e}")
    for repeat in sorted({row.repeat for row in rows}):
        by_format = {
            row.matrix_format: row for row in rows if row.repeat == repeat
        }
        if {"csr", "bsr"} <= by_format.keys():
            csr, bsr = by_format["csr"], by_format["bsr"]
            difference = _coefficient_difference(csr, bsr)
            print(
                f"\n  repeat {repeat} BSR/CSR primal coefficient relative "
                f"difference: {difference:.3e}"
            )
            if bsr.amgx_solve_seconds > 0.0:
                print(
                    "  AMGX solve speed ratio (CSR / BSR): "
                    f"{csr.amgx_solve_seconds / bsr.amgx_solve_seconds:.3f}"
                )
            print(
                "  compressed-pattern byte ratio (BSR / CSR): "
                f"{bsr.pattern_bytes / csr.pattern_bytes:.4f}"
            )
    if len({row.repeat for row in rows}) > 1:
        print("\n  medians over repeats")
        for matrix_format in ("csr", "bsr"):
            selected = [row for row in rows if row.matrix_format == matrix_format]
            if not selected:
                continue
            print(
                f"    {matrix_format}: setup="
                f"{np.median([row.amgx_setup_seconds for row in selected]):.6f}s, "
                f"solve={np.median([row.amgx_solve_seconds for row in selected]):.6f}s, "
                f"total={np.median([row.total_seconds for row in selected]):.6f}s"
            )


def _write_csv(path, rows, *, domain, triangles, order, trace_dofs, tau):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "domain",
        "triangles",
        "order",
        "face_block_size",
        "trace_dofs",
        "tau",
        "repeat",
        "matrix_format",
        "config",
        "solver",
        "preconditioner",
        "matrix_bytes",
        "pattern_bytes",
        "assembly_seconds",
        "kernel_seconds",
        "amgx_setup_seconds",
        "amgx_solve_seconds",
        "reconstruction_seconds",
        "total_seconds",
        "iterations",
        "relative_residual",
        "l2_error",
        "coefficient_relative_difference",
    )
    differences = {}
    for repeat in sorted({row.repeat for row in rows}):
        by_format = {
            row.matrix_format: row for row in rows if row.repeat == repeat
        }
        if {"csr", "bsr"} <= by_format.keys():
            differences[repeat] = _coefficient_difference(
                by_format["csr"], by_format["bsr"]
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "domain": domain,
                    "triangles": triangles,
                    "order": order,
                    "face_block_size": order + 1,
                    "trace_dofs": trace_dofs,
                    "tau": tau,
                    "repeat": row.repeat,
                    "matrix_format": row.matrix_format,
                    "config": row.config,
                    "solver": row.solver,
                    "preconditioner": row.preconditioner,
                    "matrix_bytes": row.matrix_bytes,
                    "pattern_bytes": row.pattern_bytes,
                    "assembly_seconds": row.assembly_seconds,
                    "kernel_seconds": row.kernel_seconds,
                    "amgx_setup_seconds": row.amgx_setup_seconds,
                    "amgx_solve_seconds": row.amgx_solve_seconds,
                    "reconstruction_seconds": row.reconstruction_seconds,
                    "total_seconds": row.total_seconds,
                    "iterations": row.iterations,
                    "relative_residual": row.relative_residual,
                    "l2_error": row.l2_error,
                    "coefficient_relative_difference": differences.get(row.repeat),
                }
            )


def _replay_worker(args):
    """Use production package solvers on one hash-verified physical system."""
    from scipy import sparse
    from scripts.advection_diffusion_reaction.diagnostics.check_cached_adr_pardiso import write_json
    from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only
    source = REPLAY_PRESETS[args.preset]
    report = dict(status="running", candidate=args.replay_worker, warmups=[], samples=[])
    output = args.output

    def save(stage):
        report.update(stage=stage, peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output / "result.json", report)
        print(stage, flush=True)

    try:
        save("loading captured Poisson matrix")
        metadata = json.loads((source / "metadata.json").read_text())
        matrix = sparse.load_npz(source / "operator_bsr.npz")
        digest = lambda a: hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast("B")).hexdigest()
        for key, values in (("values", matrix.data), ("indices", matrix.indices), ("indptr", matrix.indptr)):
            if digest(values) != metadata[f"matrix_{key}_sha256"]:
                raise ValueError(f"Captured matrix {key} changed")
        with np.load(source / "system_step1.npz", allow_pickle=False) as capture:
            rhs = capture["rhs"].copy()
            reference = capture["reference_modal"].copy()
        evaluation = np.load(source / "trace_evaluation.npy", allow_pickle=False)
        inverse = np.linalg.inv(evaluation)
        q, size = matrix.blocksize[0], matrix.shape[0]
        if q != 7 or metadata["triangles"] != 315425 or size != 3307381:
            raise ValueError("Replay preset dimensions changed")
        report.update(triangles=metadata["triangles"], trace_dofs=size, degree=q-1,
                      matrix_sha256=metadata["matrix_values_sha256"], rhs_sha256=digest(rhs),
                      rtol=args.rtol, zero_initial_guess=True, new_compilation_allowed=False,
                      source=str(source), matrix_bytes=metadata["matrix_bytes"])
        rhs_norm = np.linalg.norm(rhs)

        def check(x):
            residual = rhs - matrix @ x
            relative = float(np.linalg.norm(residual) / rhs_norm)
            modal = x.reshape(-1, q) @ inverse
            error = float(np.linalg.norm(modal-reference) / np.linalg.norm(reference))
            return dict(relative_residual=relative, reference_modal_error=error,
                        passed=bool(np.isfinite(relative) and relative <= args.rtol
                                    and np.isfinite(error) and error <= 1e-6))

        cpu = args.replay_worker.startswith("pardiso")
        if cpu:
            import ctypes
            import pypardiso
            from hdgfem.linalg.direct import (
                            solve_pypardiso_system,
                            clear_pypardiso_cache,
                            prepare_pypardiso_spd_matrix,
                        )
            from hdgfem.linalg.results import refine_host_linear_solution
            from hdgfem.linalg.pardiso_diagnostics import pardiso_factor_statistics
            getter = pypardiso.ps.libmkl.MKL_Get_Max_Threads
            getter.restype = ctypes.c_int
            report["mkl_threads"] = int(getter())
            if report["mkl_threads"] != int(args.replay_worker.split("_t")[-1]):
                raise RuntimeError("MKL thread count does not match the requested count")
            spd = args.replay_worker.startswith("pardiso_spd_")
            report["factorization"] = "SPD Cholesky (mtype=2)" if spd else "real nonsymmetric LU (mtype=11), same as ADR campaign"
            if spd:
                report["maximum_refinements"] = args.spd_refinement
                report["maximum_original_matrix_corrections"] = args.spd_original_refinement
        else:
            from hdgfem.linalg.amgx.host import initialize_pyamgx_once
            cp = require_cupy()
            cp.cuda.Device(0).use()
            sync = cp.cuda.get_current_stream().synchronize
            amgx = initialize_pyamgx_once()
            amgx.register_print_callback(lambda message: print(message, end="", flush=True))
            props = cp.cuda.runtime.getDeviceProperties(0)
            report["gpu"] = str(props["name"])
            report["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
            report["cupy"] = cp.__version__

        with kernel_cache_only(True):
            for trial in range(args.warmup + args.repeats):
                phase = "warmups" if trial < args.warmup else "samples"
                save(f"{phase} {trial+1}: setup")
                solver = operator = preconditioner = base = device_rhs = csr = None
                direct_solver = None
                try:
                    if cpu:
                        clear_pypardiso_cache()
                        gc.collect()
                        started = time.perf_counter()
                        csr = matrix.tocsr()
                        csr.eliminate_zeros()
                        csr.sort_indices()
                        report["full_csr_nnz"] = int(csr.nnz)
                        if spd:
                            csr = prepare_pypardiso_spd_matrix(csr)
                            spd_diagonal = csr.diagonal()
                            direct_solver = pypardiso.PyPardisoSolver(mtype=2)
                            direct_solver.set_iparm(8, args.spd_refinement)
                        else:
                            direct_solver = pypardiso.ps
                        conversion = time.perf_counter()-started
                        direct_solver.set_iparm(18, -1)
                        direct_solver.set_iparm(60, 0)
                        started = time.perf_counter()
                        direct_solver.factorize(csr)
                        factor = time.perf_counter()-started
                        setup = conversion+factor
                        row = dict(conversion_seconds=conversion, factorization_seconds=factor,
                                   factor_statistics=pardiso_factor_statistics(direct_solver, matrix_nnz=csr.nnz))
                        report["factor_input_nnz"] = int(csr.nnz)
                        report["csr_bytes"] = sum(a.nbytes for a in (csr.data, csr.indices, csr.indptr))
                    else:
                        from scripts.guiding_center.poisson.replay_poisson_asm import BsrAction, captured_patch_map
                        sync(); started = time.perf_counter()
                        operator = BsrAction(matrix)
                        device_rhs = cp.asarray(rhs)
                        row = {}
                        if args.replay_worker.startswith("amgx"):
                            from hdgfem.linalg.amgx.device_solver import (
                                                            PyAMGXCsrDeviceSolver,
                                                        )
                            from hdgfem.linalg.gpu.sparse import _DeviceBsrMatrixView
                            from scripts.guiding_center.poisson.amgx_bsr_smoothing import smoothing_config
                            sweeps = int(args.replay_worker[-1])
                            config = smoothing_config(postsweeps=3, hierarchy="scalar_expand",
                                                      tolerance=0.1*args.rtol*rhs_norm, max_iters=args.maxiter)
                            config["solver"]["preconditioner"]["postsweeps"] = sweeps
                            solver = PyAMGXCsrDeviceSolver(config=config, maxiter=args.maxiter)
                            solver.setup(_DeviceBsrMatrixView(operator.data, operator.indices, operator.indptr, matrix.shape, q))
                            report["configuration"] = solver.config_dict
                        elif args.replay_worker in PMG_REPLAY_POLICIES:
                            from hdgfem.linalg.multigrid.face_hp import (
                                                            FaceBlockHpMgPcgSolver,
                                                        )
                            # E has modal basis rows: A_m=E A_n E^T, b_m=E b_n.
                            e_device = cp.asarray(evaluation)
                            modal_data = cp.ascontiguousarray(e_device @ operator.data @ e_device.T)
                            rows = np.repeat(np.arange(size//q), np.diff(matrix.indptr))
                            diagonal = cp.asarray(np.flatnonzero(matrix.indices == rows).astype(np.int32))
                            policy, tuning = PMG_REPLAY_POLICIES[args.replay_worker]
                            solver = FaceBlockHpMgPcgSolver(indptr=operator.indptr, indices=operator.indices,
                                data=modal_data, degree=q-1, diagonal_positions=diagonal,
                                preconditioner_policy=policy, preconditioner_tuning=tuning)
                            modal_rhs = cp.ascontiguousarray(device_rhs.reshape(-1,q) @ e_device.T).ravel()
                            modal_atol = 0.1*args.rtol*rhs_norm/np.linalg.norm(inverse, 2)
                            report["configuration"] = dict(policy=policy, tuning=tuning, outer="PCGF",
                                modal_atol=modal_atol, parameters=solver.preconditioner_parameters)
                            row.update(symmetry_defect=solver.symmetry_defect,
                                       positive_curvature=solver.positive_curvature)
                        elif args.replay_worker == "asm_pp":
                            from hdgfem.linalg.additive_schwarz import build_bsr_face_additive_schwarz_local_matrices
                            from hdgfem.linalg.gpu.cublas_batched import (
                                                            invert_batched_cublas,
                                                        )
                            from hdgfem.linalg.gpu.preconditioners import (
                                                            CuPyFaceAdditiveSchwarzPreconditioner,
                                                        )
                            from hdgfem.linalg.gpu.polynomial import (
                                                            CuPyPolynomialPreconditioner,
                                                        )
                            from hdgfem.linalg.gpu.production_gmres import (
                                                            CuPyProductionGMRESSolver,
                                                            CuPyProductionGMRESOptions,
                                                        )
                            patches, _ = captured_patch_map(metadata)
                            local = build_bsr_face_additive_schwarz_local_matrices(matrix, patches)
                            local_device = cp.asarray(local.local_matrices)
                            factors = invert_batched_cublas(local_device, label="Poisson ASM patches")
                            row["asm_inverse_residual"] = factors.maximum_inverse_residual
                            base = CuPyFaceAdditiveSchwarzPreconditioner(inverse_matrices=factors.inverse_matrices,
                                element_system_faces=cp.asarray(patches, dtype=cp.int32), block_size=q,
                                num_system_faces=size//q, device_id=0, application="fused")
                            operator.num_dofs, operator.device_id, operator.dtype = size, 0, operator.data.dtype
                            operator.matvec_into = lambda x, out: operator.matvec(x, out=out)
                            preconditioner = CuPyPolynomialPreconditioner.from_operator(operator,
                                degree=args.pp_degree, base_preconditioner=base, setup_orthogonalization="cgs2")
                            solver = CuPyProductionGMRESSolver(operator, preconditioner=preconditioner,
                                options=CuPyProductionGMRESOptions(restart=75, max_iterations=args.maxiter,
                                    rtol=0.1*args.rtol, orthogonalization="cgs2", cgs2_fallback_threshold=None))
                            report["configuration"] = dict(degree=args.pp_degree, restart=75, asm="element/fused",
                                operator="generic cuSPARSE BSR", outer="GMRES/CGS2")
                            del local, local_device, factors
                        else:
                            raise ValueError("Unknown replay solver")
                        sync(); setup = time.perf_counter()-started
                    row.update(setup_seconds=setup, solves=[])
                    report[phase].append(row)
                    save(f"{phase} {trial+1}: setup finished {setup:.3f}s")
                    for solve in range(2):
                        if cpu:
                            if spd:
                                # Same preparation as the package's SPD wrapper, retained
                                # once per setup. Keep both native matrix-reuse checks.
                                started = time.perf_counter()
                                x = pypardiso.spsolve(csr, rhs, solver=direct_solver)
                                uncorrected_x = x
                                corrections = 0
                                if args.spd_original_refinement:
                                    x, corrections = refine_host_linear_solution(matrix, rhs, x,
                                        solve_correction=lambda r: pypardiso.spsolve(csr, r, solver=direct_solver),
                                        rtol=args.rtol, max_corrections=args.spd_original_refinement)
                                elapsed, iterations = time.perf_counter()-started, None
                            else:
                                result = solve_pypardiso_system(csr, rhs, rtol=args.rtol)
                                elapsed, x, iterations = result.solve_elapsed_seconds, result.x, None
                            if direct_solver.phase != 33:
                                raise RuntimeError("Factor reuse failed")
                        else:
                            sync(); started = time.perf_counter()
                            if args.replay_worker.startswith("amgx"):
                                solution, info = solver.solve(device_rhs)
                                iterations = int(info["amgx_iterations"])
                            elif args.replay_worker in PMG_REPLAY_POLICIES:
                                result = solver.solve(modal_rhs, rtol=0., atol=modal_atol, maxiter=args.maxiter)
                                solution = cp.ascontiguousarray(result.solution.reshape(-1,q) @ e_device).ravel()
                                iterations = int(result.iterations)
                            else:
                                result = solver.solve(device_rhs, x0=None)
                                solution, iterations = result.solution, int(result.iterations)
                            sync(); elapsed = time.perf_counter()-started
                            x = cp.asnumpy(solution)
                            row["cupy_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
                            row["amgx_memory"] = amgx.get_device_memory_stats()
                        checked = check(x)
                        if cpu:
                            checked["native_refinement_steps"] = direct_solver.get_iparm(7)
                            if spd:
                                # Diagnostics only, outside solve timing. Never substitute
                                # this symmetric interpretation for the original check.
                                upper_residual = rhs - (csr @ uncorrected_x + csr.T @ uncorrected_x - spd_diagonal * uncorrected_x)
                                checked.update(original_matrix_corrections=corrections,
                                    uncorrected_relative_residual=float(np.linalg.norm(rhs-matrix@uncorrected_x)/rhs_norm),
                                    symmetric_interpretation_relative_residual=float(np.linalg.norm(upper_residual)/rhs_norm))
                        row["solves"].append(dict(seconds=elapsed, iterations=iterations, **checked))
                        save(f"{phase} {trial+1} solve {solve+1}: {elapsed:.3f}s residual={checked['relative_residual']:.3e}")
                        if not checked["passed"]:
                            raise RuntimeError("Original-matrix residual or reference parity failed")
                    row["fresh_seconds"] = setup + row["solves"][0]["seconds"]
                finally:
                    if cpu:
                        if direct_solver is not None and spd:
                            direct_solver.free_memory(everything=True)
                        clear_pypardiso_cache()
                    else:
                        if solver is not None and hasattr(solver, "close"):
                            solver.close()
                        if operator is not None:
                            operator.close()
                    solver = preconditioner = base = operator = csr = device_rhs = None
                    gc.collect()
            report.update(status="passed", fresh_median_seconds=statistics.median(r["fresh_seconds"] for r in report["samples"]),
                          reused_mean_seconds=statistics.mean(r["solves"][1]["seconds"] for r in report["samples"]))
            report["face_relative_residual"] = max(s["relative_residual"] for r in report["samples"] for s in r["solves"])
            report["timings_ms"] = dict(setup_solve=1000*report["fresh_median_seconds"])
            save("finished")
        return 0
    except Exception as exc:
        report.update(status="error", error=str(exc), traceback=traceback.format_exc())
        save("failed")
        return 1


def _run_replay_preset(args):
    """Bounded isolated pilots and independent confirmation, no assembly/JIT."""
    from scripts.advection_diffusion_reaction.diagnostics.check_cached_adr_pardiso import monitor, write_json, physical_threads
    if args.output is None or args.warmup < 1 or args.repeats < 1:
        raise ValueError("Replay requires --output and positive warmup/repeats")
    if not args.threads or min(args.threads) < 1 or max(args.threads) > physical_threads():
        raise ValueError("Thread counts must fit the affinity-visible physical cores")
    if args.spd_original_refinement < 0:
        raise ValueError("--spd-original-refinement must be nonnegative")
    if args.replay_worker:
        return _replay_worker(args)
    if args.replay_suite == "spd-pmg-refined":
        families = {"pypardiso_spd": [f"pardiso_spd_t{n}" for n in args.threads],
                    "amgx": ["amgx_hybrid_2"], "pmg": ["pmg_cheb1", "pmg_cheb1_coarse2"]}
    elif args.replay_suite == "spd":
        families = {"pypardiso_spd": [f"pardiso_spd_t{n}" for n in args.threads]}
    elif args.replay_suite == "spd-pmg":
        families = {"pypardiso_spd": [f"pardiso_spd_t{n}" for n in args.threads],
                    "amgx": ["amgx_hybrid_2"], "pmg": list(PMG_REPLAY_POLICIES)}
    else:
        families = {"pypardiso": [f"pardiso_t{n}" for n in args.threads],
                    "amgx": ["amgx_hybrid_2", "amgx_hybrid_3"], "pmg": ["pmg"], "asm_pp": ["asm_pp"]}
    candidates = [name for names in families.values() for name in names]
    print(f"{args.preset}: {candidates}; zero-start GPU solves, common rtol={args.rtol}", flush=True)
    if not args.execute:
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Choose a new empty output directory")
    summary = dict(status="running", preset=args.preset, suite=args.replay_suite, trials={}, selected={})
    write_json(args.output/"arguments.json", {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})

    def launch(name, phase, repeats):
        folder = args.output/f"{phase}_{name}"
        folder.mkdir()
        threads = int(name.split("_t")[-1]) if name.startswith("pardiso") else 1
        env = dict(os.environ, MKL_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                   MKL_DYNAMIC="FALSE", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1", NUMBA_DISABLE_JIT="1",
                   PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-u", "-B", "-m", "scripts.diffusion_reaction.compare_cuda_bsr_csr",
                   "--preset", args.preset, "--replay-worker", name, "--output", str(folder),
                   "--rtol", str(args.rtol), "--maxiter", str(args.maxiter), "--repeats", str(repeats),
                   "--warmup", str(args.warmup), "--pp-degree", str(args.pp_degree),
                   "--spd-refinement", str(args.spd_refinement),
                   "--spd-original-refinement", str(args.spd_original_refinement)]
        print(f"Starting {phase} {name}: {folder/'worker.log'}", flush=True)
        args.monitor_label = name
        result = monitor(command, env, folder, args)
        summary["trials"][folder.name] = result
        write_json(args.output/"summary.json", summary)
        return result

    pilots = {name: launch(name, "pilot", 2) for name in candidates}
    for family, names in families.items():
        passed = [n for n in names if pilots[n]["status"] == "passed"]
        if not passed:
            summary["selected"][family] = dict(status="no passing pilot")
            continue
        selected = {objective: min(passed, key=lambda n: pilots[n][metric]) for objective, metric in
                    (("fresh", "fresh_median_seconds"), ("reused", "reused_mean_seconds"))}
        summary["selected"][family] = selected
        for name in sorted(set(selected.values())):
            launch(name, "confirmation", args.repeats)
    summary["status"] = "completed"
    write_json(args.output/"summary.json", summary)
    print(json.dumps(summary["selected"], indent=2), flush=True)
    return int(any(r["status"] != "passed" for r in summary["trials"].values()))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.preset is not None:
        return _run_replay_preset(args)
    ny = args.nx if args.ny is None else args.ny
    if args.order < 1 or args.order > 6:
        raise ValueError("this raw-CUDA diffusion benchmark supports 1 <= p <= 6")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.domain == "disk":
        if args.mesh_size <= 0.0:
            raise ValueError("mesh-size must be positive")
        if args.radius <= 0.0:
            raise ValueError("radius must be positive")
        domain_label = (
            f"unstructured disk, radius={args.radius:g}, "
            f"mesh-size={args.mesh_size:g}"
        )
        print(f"building {domain_label} ...", flush=True)
        mesh = gmsh_disc_mesh(
            args.mesh_size,
            center=(0.0, 0.0),
            radius=args.radius,
            verbosity=0,
        )
    else:
        if args.nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive")
        domain_label = f"structured rectangle, {args.nx} x {ny}"
        print(f"building {domain_label} ...", flush=True)
        mesh = rectangle_mesh(
            args.nx,
            ny,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
        )
    if not args.allow_small and mesh.num_tri <= args.minimum_triangles:
        raise ValueError(
            f"mesh has {mesh.num_tri:,} triangles; expected more than "
            f"{args.minimum_triangles:,}. Increase --nx/--ny or pass --allow-small."
        )
    print(f"building p={args.order} DG space for {mesh.num_tri:,} triangles ...", flush=True)
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    problem = trigonometric_poisson_case()
    tau = float(
        args.tau
        if args.tau is not None
        else GlobalLengthDiffusion(gamma_d=1.0, domain_length="auto").resolve(
            problem.diffusion, space
        )
    )
    rows = []
    cp = require_cupy()
    for repeat in range(1, args.repeats + 1):
        if args.only == "both":
            bsr_first = args.format_order == "bsr-first" or (
                args.format_order == "alternate" and repeat % 2 == 0
            )
            requested = ("bsr", "csr") if bsr_first else ("csr", "bsr")
        else:
            requested = (args.only,)
        for matrix_format in requested:
            selected = args.csr_amgx_config
            if matrix_format == "bsr":
                selected = args.bsr_amgx_config or _default_bsr_config(args.order)
            config, config_path = load_amgx_config(
                selected,
                tolerance=args.rtol,
                maxiter=args.maxiter,
                verbose=args.verbosity,
            )
            print(
                f"\nrepeat {repeat}: running raw-CUDA {matrix_format.upper()} with "
                f"{describe_amgx_solver(config)}/"
                f"{describe_amgx_preconditioner(config)} ...",
                flush=True,
            )
            rows.append(
                _run_one(
                    matrix_format,
                    repeat=repeat,
                    space=space,
                    problem=problem,
                    tau=tau,
                    config=config,
                    config_path=config_path,
                    args=args,
                )
            )
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    trace_dofs = int(mesh.int_edges_inds.size * space.layout.edg_dof)
    _print_results(
        rows,
        domain=domain_label,
        triangles=mesh.num_tri,
        order=space.order,
        trace_dofs=trace_dofs,
        tau=tau,
    )
    if args.csv is not None:
        _write_csv(
            args.csv,
            rows,
            domain=domain_label,
            triangles=mesh.num_tri,
            order=space.order,
            trace_dofs=trace_dofs,
            tau=tau,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
