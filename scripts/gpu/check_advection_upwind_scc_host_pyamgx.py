#!/usr/bin/env python3
"""Compare host upwind-SCC ordering for advection-reaction CuPy solves.

This experiment is host-only assembly and ordering, then runs CuPy/Cupyx on
the already ordered reduced trace systems.  It is intended as a focused study
of ordering, diagonal scaling, and Cupyx iteration on the same preconditioned
problem.
"""

from __future__ import annotations

import argparse
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

from hybridge.hdg import matrices as hdg_mats
import hybridge.hdg.coefficients as hdg_coefficients
from hybridge.runtime.optional import require_cupy, require_cupyx_sparse
from hybridge.linalg.gpu.sparse import scipy_csr_to_cupy
from hybridge.transport.numba import reconstruct_projected_field_numba
from hybridge.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.linalg.ordering import GraphOrderingResult, upwind_scc_trace_ordering
from hybridge.linalg.system import assemble_global_matrix, solve_global_system
from hybridge.linalg.results import diagonal_scale_system, residual_diagnostics
from hybridge.linalg.reduction import expand_known_dofs
from hybridge.solvers.advection_reaction import AdvectionReactionHDGSolver
from scripts.advection_reaction.cases import case_definition_by_key


@dataclass
class SolveGlobalRun:
    backend: str
    variant: str
    route: str
    info: int | None
    nnz: int
    wall_seconds: float
    permutation_seconds: float | None
    scale_seconds: float | None
    matrix_copy_seconds: float | None
    preconditioner_seconds: float | None
    iterative_seconds: float | None
    total_seconds: float | None
    iterations: int | None
    preconditioner_apply_count: int | None
    preconditioner_apply_seconds: float | None
    solver_residual_norm: float | None
    solver_relative_residual: float | None
    solver_residual_target: float | None
    physical_residual_norm: float | None
    physical_relative_residual: float | None
    physical_residual_target: float | None


@dataclass
class FieldErrorRun:
    l2: float
    linf: float
    avg_max: float
    max_error_element: int
    reconstruct_seconds: float
    error_seconds: float
    total_seconds: float


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
    parser.add_argument("--check-rtol", type=float, default=1.0e-10)
    parser.add_argument("--cupyx-tolerance", "--amgx-tolerance", dest="cupyx_tolerance", type=float, default=1.0e-14)
    parser.add_argument("--cupyx-atol", "--amgx-atol", dest="cupyx_atol", type=float, default=0.0)
    parser.add_argument("--cupyx-maxiter", "--amgx-maxiter", dest="cupyx_maxiter", type=int, default=1500)
    parser.add_argument("--scale-system", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cupyx-solver", default="bicgstab")
    parser.add_argument(
        "--cupyx-preconditioner",
        choices=("none", "host-ilu-export", "cupyx-ilu1"),
        default="host-ilu-export",
        help=(
            "host-ilu-export builds SuperLU ILU on host, copies L/U/permutations to device, "
            "and applies the preconditioner during cupyx_bicgstab on device; "
            "cupyx-ilu1 builds the fill_factor=1 ILU directly with Cupyx"
        ),
    )
    parser.add_argument("--cupyx-ilu-fill-factor", type=float, default=35.0)
    parser.add_argument("--cupyx-ilu-drop-tol", type=float, default=1e-10)
    parser.add_argument(
        "--solve-variants",
        choices=("both", "natural", "ordered"),
        default="both",
        help="Choose whether to run both natural and upwind-SCC ordered systems or only one variant.",
    )
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
        raise RuntimeError("matrix-only assembly did not return reduced trace RHS")
    if result.reduction is None:
        raise RuntimeError("matrix-only assembly did not return boundary reduction metadata")
    matrix = assemble_global_matrix(
        result.solve_matrix_rows,
        result.solve_matrix_cols,
        result.solve_matrix_data,
        result.solve_rhs.size,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix, np.ascontiguousarray(result.solve_rhs, dtype=np.float64), result.reduction, elapsed


def active_edges_for_ordering(space: DGSpace, boundary_mode: str):
    if boundary_mode != "eliminate":
        return None
    active_edge_mask = np.ones(space.mesh.num_edg, dtype=bool)
    active_edge_mask[space.mesh.bnd_edges_inds] = False
    return np.flatnonzero(active_edge_mask).astype(np.int64)


def build_ordering(args, space: DGSpace, beta_h) -> tuple[GraphOrderingResult, float]:
    beta_dot_normal = hdg_coefficients.advective_boundary_normal(beta_h, space)
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


def cupy_matrix_from_host(cp, matrix: scipy.sparse.csr_matrix) -> tuple[Any, float]:
    start = time.perf_counter()
    matrix_cp = scipy_csr_to_cupy(matrix)
    _sync(cp)
    return matrix_cp, time.perf_counter() - start


def cupyx_preconditioner_settings(args) -> tuple[str | None, float, float]:
    mode = str(args.cupyx_preconditioner)
    if mode == "none":
        return None, float(args.cupyx_ilu_fill_factor), float(args.cupyx_ilu_drop_tol)
    if mode == "host-ilu-export":
        return "host_ilu_export", float(args.cupyx_ilu_fill_factor), float(args.cupyx_ilu_drop_tol)
    if mode == "cupyx-ilu1":
        return "cupyx_ilu1", 1.0, float(args.cupyx_ilu_drop_tol)
    raise ValueError(f"unknown Cupyx preconditioner mode: {mode}")


def _fmt_int(value) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _norms(matrix, rhs, x, *, rtol: float, atol: float):
    residual = matrix @ x - rhs
    residual_norm = float(np.linalg.norm(residual))
    _, relative, target = residual_diagnostics(residual_norm, rhs, rtol=rtol, atol=atol)
    return residual_norm, relative, target


def solve_with_global_system(
    matrix: scipy.sparse.csr_matrix,
    rhs: np.ndarray,
    *,
    args,
    preconditioner: str | None = None,
    variant: str | None = None,
    cupyx_prepared_matrix: Any | None = None,
    matrix_copy_seconds: float | None = None,
) -> tuple[SolveGlobalRun, np.ndarray]:
    start = time.perf_counter()
    cupyx_preconditioner, ilu_fill_factor, ilu_drop_tol = cupyx_preconditioner_settings(args)
    if preconditioner in {None, "none"}:
        cupyx_preconditioner = None
    result = solve_global_system(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        rhs,
        rhs.size,
        solver=f"cupyx_{str(args.cupyx_solver).lower()}",
        preconditioner=cupyx_preconditioner,
        rtol=args.cupyx_tolerance,
        atol=args.cupyx_atol,
        maxiter=args.cupyx_maxiter,
        scale_system=False,
        assembled_matrix=matrix,
        prepared_device_matrix=cupyx_prepared_matrix,
        permutation=None,
        ilu_fill_factor=ilu_fill_factor,
        ilu_drop_tol=ilu_drop_tol,
        ilu_permc_spec="NATURAL" if cupyx_preconditioner is not None else None,
        raise_on_nonconvergence=False,
        verbose=max(0, int(args.verbosity) - 1),
    )

    preconditioner_seconds = result.preconditioner_elapsed_seconds
    iterative_seconds = result.solve_elapsed_seconds

    run = SolveGlobalRun(
        backend="cupyx",
        variant=variant or "cupyx",
        route="solve_global_system",
        info=result.info,
        nnz=int(matrix.nnz),
        wall_seconds=time.perf_counter() - start,
        permutation_seconds=result.permutation_elapsed_seconds,
        scale_seconds=result.scale_elapsed_seconds,
        matrix_copy_seconds=None if matrix_copy_seconds is None else float(matrix_copy_seconds),
        preconditioner_seconds=None if preconditioner_seconds is None else float(preconditioner_seconds),
        iterative_seconds=None if iterative_seconds is None else float(iterative_seconds),
        total_seconds=result.total_elapsed_seconds,
        iterations=result.iteration_count,
        preconditioner_apply_count=result.preconditioner_apply_count,
        preconditioner_apply_seconds=result.preconditioner_apply_seconds,
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


def evaluate_field_error(
    reduced_solution: np.ndarray,
    reduction,
    source_h,
    beta_h,
    reaction_h,
    space: DGSpace,
    exact,
) -> FieldErrorRun:
    start = time.perf_counter()
    full_trace = expand_known_dofs(reduced_solution, reduction)
    recon_start = time.perf_counter()
    field = reconstruct_projected_field_numba(full_trace, source_h, beta_h, reaction_h, space)
    reconstruct_seconds = time.perf_counter() - recon_start

    error_start = time.perf_counter()
    l2_error = float(field.l2_error(exact))
    values = field.values()
    points = space.mapped_quads()
    exact_values = exact(points[:, :, 0], points[:, :, 1])
    abs_error = np.abs(values - exact_values)
    linf_error = float(np.max(abs_error))
    element_max_error = np.max(abs_error, axis=1)
    avg_error = float(np.average(element_max_error))
    max_error_element = int(np.argmax(element_max_error))
    error_seconds = time.perf_counter() - error_start
    return FieldErrorRun(
        l2=l2_error,
        linf=linf_error,
        avg_max=avg_error,
        max_error_element=max_error_element,
        reconstruct_seconds=reconstruct_seconds,
        error_seconds=error_seconds,
        total_seconds=time.perf_counter() - start,
    )


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
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    if args.show_cupy_config:
        cp.show_config()

    run_start = time.perf_counter()

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

    assemble_start = time.perf_counter()
    matrix, rhs, reduction, assembly_seconds = assemble_host_trace_system(args, space, beta_h, reaction_h, source_h, exact)
    matrix = matrix.tocsr()
    assembly_seconds = time.perf_counter() - assemble_start

    ordering, ordering_seconds = build_ordering(args, space, beta_h)
    permutation = ordering.dof_permutation
    if permutation.shape != rhs.shape:
        raise RuntimeError(f"upwind permutation shape {permutation.shape} does not match RHS shape {rhs.shape}")

    ordering_start = time.perf_counter()
    if bool(args.scale_system):
        matrix_for_solve, rhs_for_solve = diagonal_scale_system(matrix, rhs, copy_matrix=True)
        scale_seconds = time.perf_counter() - ordering_start
    else:
        matrix_for_solve = matrix.copy()
        rhs_for_solve = np.array(rhs, copy=True)
        scale_seconds = 0.0

    permutation_start = time.perf_counter()
    ordered_matrix = matrix_for_solve[permutation][:, permutation].tocsr()
    ordered_matrix.sum_duplicates()
    ordered_matrix.sort_indices()
    ordered_rhs = rhs_for_solve[permutation]
    explicit_permutation_seconds = time.perf_counter() - permutation_start

    run_natural = args.solve_variants in {"both", "natural"}
    run_ordered = args.solve_variants in {"both", "ordered"}

    natural_run = None
    natural_x = None
    natural_physical = None
    natural_errors = None
    matrix_copy_seconds = None
    if run_natural:
        natural_matrix_cp, matrix_copy_seconds = cupy_matrix_from_host(cp, matrix_for_solve)
        natural_run, natural_x = solve_with_global_system(
            matrix_for_solve,
            rhs_for_solve,
            args=args,
            preconditioner=args.cupyx_preconditioner,
            variant="cupyx-natural",
            cupyx_prepared_matrix=natural_matrix_cp,
            matrix_copy_seconds=matrix_copy_seconds,
        )
        natural_physical = _norms(matrix, rhs, natural_x, rtol=args.check_rtol, atol=args.cupyx_atol)
        natural_errors = evaluate_field_error(natural_x, reduction, source_h, beta_h, reaction_h, space, exact)

    ordered_run = None
    ordered_x = None
    ordered_physical = None
    ordered_errors = None
    ordered_matrix_copy_seconds = None
    if run_ordered:
        ordered_matrix_cp, ordered_matrix_copy_seconds = cupy_matrix_from_host(cp, ordered_matrix)
        ordered_run, ordered_x_permuted = solve_with_global_system(
            ordered_matrix,
            ordered_rhs,
            args=args,
            preconditioner=args.cupyx_preconditioner,
            variant="cupyx-ordered",
            cupyx_prepared_matrix=ordered_matrix_cp,
            matrix_copy_seconds=ordered_matrix_copy_seconds,
        )
        ordered_x = unpermute_solution(ordered_x_permuted, permutation)
        ordered_physical = _norms(matrix, rhs, ordered_x, rtol=args.check_rtol, atol=args.cupyx_atol)
        ordered_errors = evaluate_field_error(ordered_x, reduction, source_h, beta_h, reaction_h, space, exact)

    check_start = time.perf_counter()
    diff_ordered_vs_natural = None
    if natural_x is not None and ordered_x is not None:
        diff_ordered_vs_natural = float(np.linalg.norm(natural_x - ordered_x) / max(np.linalg.norm(natural_x), 1.0))
    check_seconds = time.perf_counter() - check_start

    print()
    print("HYBRIDGE host upwind-SCC CuPy/Cupyx ordering check")
    print("----------------------------------------------")
    print(
        f"case={args.case}, order={args.order}, mesh_size={args.mesh_size}, basis={args.basis}, "
        f"trace_basis={args.trace_basis}, backend={args.assembly_backend}, preconditioner={args.cupyx_preconditioner}"
    )
    print(
        f"triangles={mesh.num_tri:,}, edges={mesh.num_edg:,}, system_size={rhs.size:,}, "
        f"nnz={matrix.nnz:,}, permutation_width={permutation.size}, solver=cupyx_{args.cupyx_solver}"
    )

    print_ordering_diagnostics(ordering, ordering_seconds)

    print_table(
        "Setup timings",
        ("phase", "seconds"),
        [
            ("mesh", _fmt(mesh_seconds)),
            ("space", _fmt(space_seconds)),
            ("project coeffs", _fmt(projection_seconds)),
            ("host assembly", _fmt(assembly_seconds)),
            ("ordering", _fmt(ordering_seconds)),
            ("host diagonal scaling", _fmt(scale_seconds)),
            ("explicit CSR permutation", _fmt(explicit_permutation_seconds)),
            ("cupy copy natural", _fmt(matrix_copy_seconds)),
            ("cupy copy ordered", _fmt(ordered_matrix_copy_seconds)),
            ("post checks", _fmt(check_seconds)),
            ("total script", _fmt(time.perf_counter() - run_start)),
        ],
    )

    solve_rows = []
    for label, run in (("cupyx-natural", natural_run), ("cupyx-ordered", ordered_run)):
        if run is None:
            continue
        solve_rows.append(
            (
                label,
                args.cupyx_preconditioner,
                _fmt(run.matrix_copy_seconds),
                _fmt(run.preconditioner_seconds),
                _fmt(run.iterative_seconds),
                _fmt_int(run.iterations),
                _fmt_int(run.preconditioner_apply_count),
                _fmt(run.preconditioner_apply_seconds),
                _fmt(run.wall_seconds),
                _fmt(run.scale_seconds),
                _fmt_sci(run.solver_relative_residual),
                _fmt_sci(run.physical_relative_residual),
            )
        )

    print_table(
        "Cupy/Cupyx solve timings",
        (
            "variant",
            "preconditioner",
            "matrix copy",
            "preconditioner",
            "krylov",
            "krylov iter",
            "M calls",
            "M apply",
            "wall",
            "scale",
            "solver rel",
            "physical rel",
        ),
        solve_rows,
    )

    residual_rows = []
    if natural_physical is not None:
        residual_rows.append(
            ("natural", _fmt_sci(natural_physical[0]), _fmt_sci(natural_physical[1]), _fmt_sci(natural_physical[2]))
        )
    if ordered_physical is not None:
        residual_rows.append(
            ("ordered", _fmt_sci(ordered_physical[0]), _fmt_sci(ordered_physical[1]), _fmt_sci(ordered_physical[2]))
        )
    print_table(
        "Physical residual check",
        ("variant", "abs residual", "rel residual", "target"),
        residual_rows,
    )

    error_rows = []
    for label, errors in (("natural", natural_errors), ("ordered", ordered_errors)):
        if errors is None:
            continue
        error_rows.append(
            (
                label,
                _fmt_sci(errors.l2),
                _fmt_sci(errors.linf),
                _fmt_sci(errors.avg_max),
                _fmt_int(errors.max_error_element),
                _fmt(errors.reconstruct_seconds),
                _fmt(errors.error_seconds),
                _fmt(errors.total_seconds),
            )
        )
    print_table(
        "Field error check",
        ("variant", "L2", "Linf", "avg max", "max elem", "reconstruct", "error eval", "total"),
        error_rows,
    )

    if diff_ordered_vs_natural is not None:
        print_table(
            "Solution agreement",
            ("comparison", "relative L2"),
            [("ordered vs natural", _fmt_sci(diff_ordered_vs_natural))],
        )

    runs_payload = {}
    if natural_run is not None:
        runs_payload["cupyx_natural"] = asdict(natural_run)
    if ordered_run is not None:
        runs_payload["cupyx_ordered"] = asdict(ordered_run)

    physical_payload = {}
    if natural_physical is not None:
        physical_payload["natural"] = {
            "norm": natural_physical[0],
            "relative": natural_physical[1],
            "target": natural_physical[2],
        }
    if ordered_physical is not None:
        physical_payload["ordered"] = {
            "norm": ordered_physical[0],
            "relative": ordered_physical[1],
            "target": ordered_physical[2],
        }

    errors_payload = {}
    if natural_errors is not None:
        errors_payload["natural"] = asdict(natural_errors)
    if ordered_errors is not None:
        errors_payload["ordered"] = asdict(ordered_errors)

    payload = {
        "benchmark_scope": {
            "primary_reported_costs": "solver and preconditioner diagnostics",
            "excluded_from_solver_preconditioner_comparisons": [
                "host assembly",
                "mesh setup",
                "DG space setup",
                "coefficient projection",
                "host/device traffic",
                "COO-to-CSR conversion",
                "reconstruction",
                "field error evaluation",
            ],
        },
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
            "solve_variants": args.solve_variants,
            "cupyx_tolerance": args.cupyx_tolerance,
            "cupyx_atol": args.cupyx_atol,
            "cupyx_maxiter": args.cupyx_maxiter,
            "cupyx_solver": args.cupyx_solver,
            "cupyx_preconditioner": args.cupyx_preconditioner,
            "cupyx_ilu_fill_factor": args.cupyx_ilu_fill_factor,
            "cupyx_ilu_drop_tol": args.cupyx_ilu_drop_tol,
            "cupyx_ilu_permc_spec": "NATURAL" if args.cupyx_preconditioner != "none" else None,
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
            "host_diagonal_scaling": scale_seconds,
            "explicit_csr_permutation": explicit_permutation_seconds,
            "cupy_matrix_copy_natural": matrix_copy_seconds,
            "cupy_matrix_copy_ordered": ordered_matrix_copy_seconds,
            "post_checks": check_seconds,
            "total_script": time.perf_counter() - run_start,
        },
        "ordering": {
            "edge_order_size": int(ordering.edge_order.size),
            "dof_permutation_size": int(ordering.dof_permutation.size),
            "diagnostics": asdict(ordering.diagnostics),
        },
        "runs": runs_payload,
        "physical_residuals": physical_payload,
        "field_errors": errors_payload,
        "solution_differences": {
            "ordered_vs_natural": diff_ordered_vs_natural,
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
