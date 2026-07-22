#!/usr/bin/env python3
"""Compare host upwind-SCC ordering for advection-reaction PyAMGX solves.

This is an experiment script.  It deliberately assembles the reduced trace
system on the host, applies the already-tested upwind-SCC permutation on the
host, and sends the resulting CSR matrices to PyAMGX.  The production raw-CUDA
path is not changed by this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.assembly import matrices_numpy as hdg_mats
from hdgfem.backends.cupy import require_cupy, require_cupyx_sparse, require_pyamgx, scipy_csr_to_cupy
from hdgfem.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.ordering import GraphOrderingResult, upwind_scc_trace_ordering
from hdgfem.linalg.system import assemble_global_matrix, diagonal_scale_system, residual_diagnostics, solve_global_system
from hdgfem.solvers.adv_rea import AdvectionReactionHDGSolver
from scripts.advection_reaction.adv_rea_cases import case_definition_by_key


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "amgx"
DEFAULT_AMGX_CONFIG_PATH = CONFIG_DIR / "adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json"


@dataclass
class DirectPyAMGXRun:
    variant: str
    route: str
    status: str
    iterations: int | None
    nnz: int
    scale_seconds: float
    csr_copy_seconds: float
    rhs_copy_seconds: float
    amgx_upload_seconds: float
    amgx_setup_seconds: float
    amgx_iterate_seconds: float
    solution_download_seconds: float
    residual_check_seconds: float
    total_seconds: float
    solver_residual_norm: float
    solver_relative_residual: float
    solver_residual_target: float
    physical_residual_norm: float
    physical_relative_residual: float
    physical_residual_target: float
    amgx_reported_residual: float | None


@dataclass
class SolveGlobalRun:
    variant: str
    route: str
    info: int | None
    nnz: int
    wall_seconds: float
    permutation_seconds: float | None
    scale_seconds: float | None
    csr_copy_seconds: float | None
    pyamgx_call_seconds: float | None
    total_seconds: float | None
    solver_residual_norm: float | None
    solver_relative_residual: float | None
    solver_residual_target: float | None
    physical_residual_norm: float | None
    physical_relative_residual: float | None
    physical_residual_target: float | None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="test2_legacy_gpu3")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.01)
    parser.add_argument("--mesh-type", "-mt", choices=("rectangle", "structured-rectangle"), default="rectangle")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default="legacy-lagrange")
    parser.add_argument("--assembly-backend", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--boundary-mode", choices=("eliminate", "penalty"), default="eliminate")
    parser.add_argument("--boundary-penalty", type=float, default=1.0e20)
    parser.add_argument("--trace-ordering-flux-tolerance", type=float, default=0.0)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--gmsh-num-threads", type=int, default=None)
    parser.add_argument("--amgx-config", default=None)
    parser.add_argument("--amgx-solver", default=None)
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--amgx-atol", type=float, default=0.0)
    parser.add_argument("--check-rtol", type=float, default=1.0e-10)
    parser.add_argument("--amgx-maxiter", type=int, default=1500)
    parser.add_argument("--amgx-monitor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scale-system", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--show-cupy-config", action="store_true")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2, 3), default=1)
    return parser


def build_mesh(args):
    if args.mesh_type == "structured-rectangle":
        return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    return gmsh_rectangle_mesh(
        args.mesh_size,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        verbosity=args.gmsh_verbosity,
        algorithm=args.gmsh_algorithm,
        num_threads=args.gmsh_num_threads,
    )


def load_amgx_config(args) -> tuple[dict[str, Any], Path | None]:
    config_path = Path(args.amgx_config).expanduser() if args.amgx_config else DEFAULT_AMGX_CONFIG_PATH
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    elif args.amgx_config:
        raise FileNotFoundError(f"AMGX config file not found: {config_path}")
    else:
        config = {
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "solver": "BICGSTAB",
                "monitor_residual": 0,
                "convergence": "RELATIVE_INI_CORE",
                "tolerance": float(args.amgx_tolerance),
                "max_iters": int(args.amgx_maxiter),
                "preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS", "cycle": "W"},
            },
        }
        config_path = None

    solver_config = config.setdefault("solver", {})
    if args.amgx_solver is not None:
        solver_config["solver"] = str(args.amgx_solver)
    solver_config["tolerance"] = float(args.amgx_tolerance)
    solver_config["max_iters"] = int(args.amgx_maxiter)
    solver_config["monitor_residual"] = int(bool(args.amgx_monitor))
    solver_config["print_solve_stats"] = int(bool(args.amgx_monitor))
    solver_config["store_res_history"] = int(bool(args.amgx_monitor))
    solver_config["obtain_timings"] = int(bool(args.amgx_monitor) or int(args.verbosity) >= 2)
    preconditioner = solver_config.get("preconditioner")
    if isinstance(preconditioner, dict):
        preconditioner["print_grid_stats"] = int(bool(args.amgx_monitor) and int(args.verbosity) >= 2)
    return config, config_path


def build_case(args):
    try:
        case = case_definition_by_key(args.case)
    except ValueError:
        if args.case != "test2_legacy_gpu3":
            raise
        case = case_definition_by_key("test2")
    return case.build()


def project_problem(space: DGSpace, beta_x, beta_y, reaction, source):
    start = time.perf_counter()
    beta_x_h = space.project_callable(beta_x, name="beta_x_h")
    beta_y_h = space.project_callable(beta_y, name="beta_y_h")
    beta_h = (space * space).field((beta_x_h, beta_y_h), name="beta_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    source_h = space.project_callable(source, name="source_h")
    return beta_h, reaction_h, source_h, time.perf_counter() - start


def assemble_host_trace_system(args, space: DGSpace, beta_h, reaction_h, source_h, exact):
    solver = AdvectionReactionHDGSolver(
        space,
        source=source_h,
        beta=beta_h,
        reaction=reaction_h,
        boundary_condition=exact,
        solver="direct",
        preconditioner=None,
        boundary_mode=args.boundary_mode,
        boundary_penalty=args.boundary_penalty,
        trace_ordering="none",
        assembly_backend=args.assembly_backend,
        trace_basis=args.trace_basis,
        verbose=max(0, int(args.verbosity) - 1),
    )
    start = time.perf_counter()
    result = solver.solve(matrix_pattern_only=True)
    elapsed = time.perf_counter() - start
    if result.solve_matrix_rows is None or result.solve_matrix_cols is None or result.solve_matrix_data is None:
        raise RuntimeError("matrix-only assembly did not return reduced trace triplets")
    if result.solve_rhs is None:
        raise RuntimeError("matrix-only assembly did not return a reduced trace RHS")
    matrix = assemble_global_matrix(
        result.solve_matrix_rows,
        result.solve_matrix_cols,
        result.solve_matrix_data,
        result.solve_rhs.size,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, np.ascontiguousarray(result.solve_rhs, dtype=np.float64), result, elapsed


def active_edges_for_ordering(space: DGSpace, boundary_mode: str):
    if boundary_mode != "eliminate":
        return None
    active_edge_mask = np.ones(space.mesh.num_edg, dtype=bool)
    active_edge_mask[space.mesh.bnd_edges_inds] = False
    return np.flatnonzero(active_edge_mask).astype(np.int64)


def build_ordering(args, space: DGSpace, beta_h) -> tuple[GraphOrderingResult, float]:
    beta_dot_normal = hdg_mats.advective_boundary_normal(beta_h, space)
    start = time.perf_counter()
    ordering = upwind_scc_trace_ordering(
        space.mesh,
        beta_dot_normal,
        space.quad_data.edg_dof,
        active_edges=active_edges_for_ordering(space, args.boundary_mode),
        flux_tolerance=args.trace_ordering_flux_tolerance,
    )
    return ordering, time.perf_counter() - start


def _sync(cp) -> None:
    cp.cuda.get_current_stream().synchronize()


def _norms(matrix, rhs, x, *, rtol: float, atol: float):
    residual = matrix @ x - rhs
    residual_norm = float(np.linalg.norm(residual))
    _, relative, target = residual_diagnostics(residual_norm, rhs, rtol=rtol, atol=atol)
    return residual_norm, relative, target


def solve_pyamgx_direct(
    variant: str,
    matrix: scipy.sparse.csr_matrix,
    rhs: np.ndarray,
    *,
    config: dict[str, Any],
    cp,
    pyamgx,
    scale_system: bool,
    rtol: float,
    atol: float,
) -> tuple[DirectPyAMGXRun, np.ndarray]:
    total_start = time.perf_counter()
    physical_matrix = matrix.tocsr()
    physical_rhs = np.ascontiguousarray(rhs, dtype=np.float64)

    scale_start = time.perf_counter()
    if scale_system:
        solve_matrix, solve_rhs = diagonal_scale_system(physical_matrix, physical_rhs, copy_matrix=True)
    else:
        solve_matrix = physical_matrix
        solve_rhs = physical_rhs
    scale_seconds = time.perf_counter() - scale_start

    csr_start = time.perf_counter()
    matrix_cp = scipy_csr_to_cupy(solve_matrix)
    _sync(cp)
    csr_copy_seconds = time.perf_counter() - csr_start

    rhs_start = time.perf_counter()
    rhs_cp = cp.asarray(solve_rhs, dtype=cp.float64)
    x_cp = cp.zeros_like(rhs_cp)
    _sync(cp)
    rhs_copy_seconds = time.perf_counter() - rhs_start

    cfg = rsrc = mat = vec_b = vec_x = solver = None
    try:
        cfg = pyamgx.Config().create_from_dict(copy.deepcopy(config))
        rsrc = pyamgx.Resources().create_simple(cfg)
        mat = pyamgx.Matrix().create(rsrc, mode="dDDI")
        vec_b = pyamgx.Vector().create(rsrc, mode="dDDI")
        vec_x = pyamgx.Vector().create(rsrc, mode="dDDI")

        upload_start = time.perf_counter()
        mat.upload_CSR(matrix_cp)
        vec_b.upload_raw(rhs_cp.data.ptr, rhs_cp.size)
        vec_x.upload_raw(x_cp.data.ptr, x_cp.size)
        _sync(cp)
        amgx_upload_seconds = time.perf_counter() - upload_start

        solver = pyamgx.Solver().create(rsrc, cfg)
        setup_start = time.perf_counter()
        solver.setup(mat)
        _sync(cp)
        amgx_setup_seconds = time.perf_counter() - setup_start

        solve_start = time.perf_counter()
        solver.solve(vec_b, vec_x)
        _sync(cp)
        amgx_iterate_seconds = time.perf_counter() - solve_start

        status = str(solver.status)
        iterations = int(solver.iterations_number)
        solver_config = config.get("solver", {})
        amgx_reported_residual = None
        if isinstance(solver_config, dict) and int(solver_config.get("store_res_history", 0)):
            try:
                amgx_reported_residual = float(solver.get_residual())
            except Exception:
                amgx_reported_residual = None

        download_start = time.perf_counter()
        vec_x.download_raw(x_cp.data.ptr)
        x = np.ascontiguousarray(cp.asnumpy(x_cp), dtype=np.float64)
        if not np.all(np.isfinite(x)):
            status = f"{status}-nonfinite"
        _sync(cp)
        solution_download_seconds = time.perf_counter() - download_start
    finally:
        for obj in (solver, mat, vec_x, vec_b, rsrc, cfg):
            if obj is not None:
                try:
                    obj.destroy()
                except AttributeError:
                    pass

    residual_start = time.perf_counter()
    solver_residual_norm, solver_relative_residual, solver_residual_target = _norms(
        solve_matrix,
        solve_rhs,
        x,
        rtol=rtol,
        atol=atol,
    )
    physical_residual_norm, physical_relative_residual, physical_residual_target = _norms(
        physical_matrix,
        physical_rhs,
        x,
        rtol=rtol,
        atol=atol,
    )
    residual_check_seconds = time.perf_counter() - residual_start

    run = DirectPyAMGXRun(
        variant=variant,
        route="direct",
        status=status,
        iterations=iterations,
        nnz=int(solve_matrix.nnz),
        scale_seconds=scale_seconds,
        csr_copy_seconds=csr_copy_seconds,
        rhs_copy_seconds=rhs_copy_seconds,
        amgx_upload_seconds=amgx_upload_seconds,
        amgx_setup_seconds=amgx_setup_seconds,
        amgx_iterate_seconds=amgx_iterate_seconds,
        solution_download_seconds=solution_download_seconds,
        residual_check_seconds=residual_check_seconds,
        total_seconds=time.perf_counter() - total_start,
        solver_residual_norm=solver_residual_norm,
        solver_relative_residual=solver_relative_residual,
        solver_residual_target=solver_residual_target,
        physical_residual_norm=physical_residual_norm,
        physical_relative_residual=physical_relative_residual,
        physical_residual_target=physical_residual_target,
        amgx_reported_residual=amgx_reported_residual,
    )
    return run, x


def solve_with_global_system(
    matrix: scipy.sparse.csr_matrix,
    rhs: np.ndarray,
    permutation: np.ndarray,
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    config: dict[str, Any],
    args,
) -> tuple[SolveGlobalRun, np.ndarray]:
    start = time.perf_counter()
    result = solve_global_system(
        rows,
        cols,
        data,
        rhs,
        rhs.size,
        solver="amgx",
        preconditioner=None,
        rtol=args.amgx_tolerance,
        atol=args.amgx_atol,
        maxiter=args.amgx_maxiter,
        amgx_config=copy.deepcopy(config),
        scale_system=bool(args.scale_system),
        assembled_matrix=matrix,
        permutation=permutation,
        raise_on_nonconvergence=False,
        verbose=max(0, int(args.verbosity) - 1),
    )
    wall_seconds = time.perf_counter() - start
    run = SolveGlobalRun(
        variant="ordered-solve-global",
        route="solve_global_system",
        info=result.info,
        nnz=int(matrix.nnz),
        wall_seconds=wall_seconds,
        permutation_seconds=result.permutation_elapsed_seconds,
        scale_seconds=result.scale_elapsed_seconds,
        csr_copy_seconds=result.preconditioner_elapsed_seconds,
        pyamgx_call_seconds=result.solve_elapsed_seconds,
        total_seconds=result.total_elapsed_seconds,
        solver_residual_norm=result.solver_residual_norm,
        solver_relative_residual=result.solver_relative_residual_norm,
        solver_residual_target=result.solver_residual_target,
        physical_residual_norm=result.physical_residual_norm,
        physical_relative_residual=result.physical_relative_residual_norm,
        physical_residual_target=result.physical_residual_target,
    )
    if result.x is None:
        raise RuntimeError("solve_global_system returned no solution")
    return run, np.ascontiguousarray(result.x, dtype=np.float64)


def unpermute_solution(x_permuted: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    x = np.empty_like(x_permuted)
    x[permutation] = x_permuted
    return x


def print_ordering_diagnostics(ordering: GraphOrderingResult, elapsed: float) -> None:
    diag = ordering.diagnostics
    levels = diag.level_widths
    timings = diag.timings
    print()
    print("Upwind-SCC ordering diagnostics")
    print("--------------------------------")
    print(
        f"nodes={diag.num_nodes:,}, directed_edges={diag.num_directed_edges:,}, "
        f"components={diag.num_components:,}, largest_component={diag.largest_component_size:,}, "
        f"cyclic_nodes={diag.cyclic_nodes:,}"
    )
    print(
        f"levels={levels.num_levels:,}, max_width={levels.max_width:,}, "
        f"median_width={levels.median_width:.1f}, mean_width={levels.mean_width:.1f}, "
        f"top10_fraction={levels.top10_width_fraction:.3f}"
    )
    print(
        "timings: "
        f"pairs={timings.graph_pairs:.5f}s, csr={timings.csr:.5f}s, scc={timings.scc:.5f}s, "
        f"dag={timings.dag:.5f}s, topo={timings.topological_order:.5f}s, "
        f"dof_perm={timings.dof_permutation:.5f}s, total={timings.total:.5f}s, wrapper={elapsed:.5f}s"
    )


def _fmt(value, precision: int = 5) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def _fmt_sci(value) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3e}"


def print_table(title: str, headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    text_rows = [tuple(str(item) for item in row) for row in rows]
    widths = [
        max(len(headers[i]), max((len(row[i]) for row in text_rows), default=0))
        for i in range(len(headers))
    ]
    sep = "  "
    print()
    print(title)
    print("-" * (sum(widths) + len(sep) * (len(widths) - 1)))
    print(sep.join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print(sep.join("-" * width for width in widths))
    for row in text_rows:
        print(sep.join(row[i].ljust(widths[i]) for i in range(len(widths))))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cp = require_cupy()
    require_cupyx_sparse()
    pyamgx = require_pyamgx()
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    if args.show_cupy_config:
        cp.show_config()

    run_start = time.perf_counter()
    config, config_path = load_amgx_config(args)
    mesh_start = time.perf_counter()
    mesh = build_mesh(args)
    mesh_seconds = time.perf_counter() - mesh_start
    space_start = time.perf_counter()
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    space_seconds = time.perf_counter() - space_start

    beta_x, beta_y, reaction, source, exact = build_case(args)
    beta_h, reaction_h, source_h, projection_seconds = project_problem(space, beta_x, beta_y, reaction, source)
    matrix, rhs, assembly_result, assembly_seconds = assemble_host_trace_system(args, space, beta_h, reaction_h, source_h, exact)
    ordering, ordering_seconds = build_ordering(args, space, beta_h)
    permutation = ordering.dof_permutation
    if permutation.shape != rhs.shape:
        raise RuntimeError(f"upwind permutation shape {permutation.shape} does not match RHS shape {rhs.shape}")

    permute_start = time.perf_counter()
    ordered_matrix = matrix[permutation][:, permutation].tocsr()
    ordered_matrix.sum_duplicates()
    ordered_matrix.sort_indices()
    ordered_rhs = rhs[permutation]
    explicit_permutation_seconds = time.perf_counter() - permute_start

    if int(args.verbosity) >= 1:
        print_ordering_diagnostics(ordering, ordering_seconds)

    pyamgx.initialize()
    try:
        natural_run, natural_x = solve_pyamgx_direct(
            "natural-direct",
            matrix,
            rhs,
            config=config,
            cp=cp,
            pyamgx=pyamgx,
            scale_system=bool(args.scale_system),
            rtol=args.amgx_tolerance,
            atol=args.amgx_atol,
        )
        ordered_run, ordered_x_permuted = solve_pyamgx_direct(
            "ordered-direct",
            ordered_matrix,
            ordered_rhs,
            config=config,
            cp=cp,
            pyamgx=pyamgx,
            scale_system=bool(args.scale_system),
            rtol=args.amgx_tolerance,
            atol=args.amgx_atol,
        )
    finally:
        pyamgx.finalize()

    ordered_x = unpermute_solution(ordered_x_permuted, permutation)
    solve_global_run, solve_global_x = solve_with_global_system(
        matrix,
        rhs,
        permutation,
        rows=assembly_result.solve_matrix_rows,
        cols=assembly_result.solve_matrix_cols,
        data=assembly_result.solve_matrix_data,
        config=config,
        args=args,
    )

    check_start = time.perf_counter()
    ordered_physical = _norms(matrix, rhs, ordered_x, rtol=args.check_rtol, atol=args.amgx_atol)
    solve_global_physical = _norms(matrix, rhs, solve_global_x, rtol=args.check_rtol, atol=args.amgx_atol)
    natural_ordered_diff = float(np.linalg.norm(natural_x - ordered_x) / max(np.linalg.norm(natural_x), 1.0))
    natural_global_diff = float(np.linalg.norm(natural_x - solve_global_x) / max(np.linalg.norm(natural_x), 1.0))
    ordered_global_diff = float(np.linalg.norm(ordered_x - solve_global_x) / max(np.linalg.norm(ordered_x), 1.0))
    comparison_seconds = time.perf_counter() - check_start

    solver_name = str(config.get("solver", {}).get("solver", "unknown"))
    print()
    print("HDGFEM host upwind-SCC PyAMGX ordering check")
    print("-------------------------------------------")
    print(
        f"case={args.case}, order={args.order}, mesh_size={args.mesh_size}, basis={args.basis}, "
        f"trace_basis={args.trace_basis}, backend={args.assembly_backend}"
    )
    print(
        f"triangles={mesh.num_tri:,}, edges={mesh.num_edg:,}, system_size={rhs.size:,}, "
        f"nnz={matrix.nnz:,}, solver={solver_name}, config={config_path.name if config_path else 'embedded'}"
    )

    print_table(
        "Setup timings",
        ("phase", "seconds"),
        [
            ("mesh", _fmt(mesh_seconds)),
            ("space", _fmt(space_seconds)),
            ("project coeffs", _fmt(projection_seconds)),
            ("host assembly", _fmt(assembly_seconds)),
            ("ordering", _fmt(ordering_seconds)),
            ("explicit CSR permutation", _fmt(explicit_permutation_seconds)),
            ("post comparisons", _fmt(comparison_seconds)),
            ("total script", _fmt(time.perf_counter() - run_start)),
        ],
    )

    print_table(
        "PyAMGX comparison",
        (
            "variant",
            "route",
            "iter",
            "status",
            "scale",
            "csr copy",
            "upload",
            "setup",
            "iterate",
            "total/wall",
            "solver rel",
            "physical rel",
        ),
        [
            (
                natural_run.variant,
                natural_run.route,
                natural_run.iterations,
                natural_run.status,
                _fmt(natural_run.scale_seconds),
                _fmt(natural_run.csr_copy_seconds),
                _fmt(natural_run.amgx_upload_seconds),
                _fmt(natural_run.amgx_setup_seconds),
                _fmt(natural_run.amgx_iterate_seconds),
                _fmt(natural_run.total_seconds),
                _fmt_sci(natural_run.solver_relative_residual),
                _fmt_sci(natural_run.physical_relative_residual),
            ),
            (
                ordered_run.variant,
                ordered_run.route,
                ordered_run.iterations,
                ordered_run.status,
                _fmt(ordered_run.scale_seconds),
                _fmt(ordered_run.csr_copy_seconds),
                _fmt(ordered_run.amgx_upload_seconds),
                _fmt(ordered_run.amgx_setup_seconds),
                _fmt(ordered_run.amgx_iterate_seconds),
                _fmt(ordered_run.total_seconds),
                _fmt_sci(ordered_run.solver_relative_residual),
                _fmt_sci(ordered_physical[1]),
            ),
            (
                solve_global_run.variant,
                solve_global_run.route,
                "-",
                solve_global_run.info,
                _fmt(solve_global_run.scale_seconds),
                _fmt(solve_global_run.csr_copy_seconds),
                "-",
                "-",
                _fmt(solve_global_run.pyamgx_call_seconds),
                _fmt(solve_global_run.wall_seconds),
                _fmt_sci(solve_global_run.solver_relative_residual),
                _fmt_sci(solve_global_physical[1]),
            ),
        ],
    )

    print_table(
        "Solution agreement",
        ("comparison", "relative L2"),
        [
            ("ordered-direct vs natural-direct", _fmt_sci(natural_ordered_diff)),
            ("solve_global permutation vs natural-direct", _fmt_sci(natural_global_diff)),
            ("solve_global permutation vs ordered-direct", _fmt_sci(ordered_global_diff)),
        ],
    )

    payload = {
        "inputs": {
            "case": args.case,
            "order": args.order,
            "mesh_size": args.mesh_size,
            "mesh_type": args.mesh_type,
            "basis": args.basis,
            "trace_basis": args.trace_basis,
            "assembly_backend": args.assembly_backend,
            "boundary_mode": args.boundary_mode,
            "scale_system": bool(args.scale_system),
            "amgx_config": None if config_path is None else str(config_path),
            "amgx_solver": solver_name,
            "amgx_tolerance": args.amgx_tolerance,
            "amgx_maxiter": args.amgx_maxiter,
        },
        "mesh": {
            "triangles": int(mesh.num_tri),
            "edges": int(mesh.num_edg),
            "system_size": int(rhs.size),
            "nnz": int(matrix.nnz),
        },
        "setup_timings": {
            "mesh": mesh_seconds,
            "space": space_seconds,
            "projection": projection_seconds,
            "host_assembly": assembly_seconds,
            "ordering": ordering_seconds,
            "explicit_csr_permutation": explicit_permutation_seconds,
            "post_comparisons": comparison_seconds,
            "total_script": time.perf_counter() - run_start,
        },
        "ordering": {
            "edge_order_size": int(ordering.edge_order.size),
            "dof_permutation_size": int(ordering.dof_permutation.size),
            "diagnostics": asdict(ordering.diagnostics),
        },
        "runs": {
            "natural_direct": asdict(natural_run),
            "ordered_direct": asdict(ordered_run),
            "ordered_solve_global": asdict(solve_global_run),
        },
        "ordered_physical_residual": {
            "norm": ordered_physical[0],
            "relative": ordered_physical[1],
            "target": ordered_physical[2],
        },
        "solve_global_physical_residual": {
            "norm": solve_global_physical[0],
            "relative": solve_global_physical[1],
            "target": solve_global_physical[2],
        },
        "solution_differences": {
            "ordered_vs_natural": natural_ordered_diff,
            "solve_global_vs_natural": natural_global_diff,
            "solve_global_vs_ordered": ordered_global_diff,
        },
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"wrote JSON results: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
