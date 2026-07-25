#!/usr/bin/env python3
"""Package-backed GPU4 advection-reaction HDG runner.

The numerical GPU4 assembly/reconstruction path lives in ``hdgfem``. This file
is intentionally a thin CLI wrapper for benchmark/sweep compatibility.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.backends.cupy import as_cupy_space, require_cupy, require_cupyx_sparse, require_pyamgx
from hdgfem.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hdgfem.core.quadrature import ReferenceElementData
from hdgfem.core.space import DGSpace, VectorDGField
from hdgfem.io.output import pretty_print_sections
from hdgfem.solvers.adv_rea import AdvectionReactionHDGSolver
from scripts.advection_reaction.adv_rea_cases import case_definition_by_key


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "amgx"
DEFAULT_AMGX_CONFIG_PATH = CONFIG_DIR / "adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json"

AMGX_CONFIG = {
    "config_version": 2,
    "determinism_flag": 1,
    "exception_handling": 1,
    "solver": {
        "solver": "BICGSTAB",
        "monitor_residual": 1,
        "convergence": "RELATIVE_INI_CORE",
        "tolerance": 1.0e-14,
        "max_iters": 1500,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS", "cycle": "W"},
    },
}


def load_amgx_config(args):
    config_path = Path(args.amgx_config).expanduser() if args.amgx_config else DEFAULT_AMGX_CONFIG_PATH
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    elif args.amgx_config:
        raise FileNotFoundError(f"AMGX config file not found: {config_path}")
    else:
        config = copy.deepcopy(AMGX_CONFIG)
        config_path = None
    solver_config = config.setdefault("solver", {})
    if args.amgx_solver is not None:
        solver_config["solver"] = str(args.amgx_solver)
    solver_config["tolerance"] = float(args.amgx_tolerance)
    solver_config["max_iters"] = int(args.amgx_maxiter)
    return config, config_path


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
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default="legacy-lagrange")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="raw-cuda")
    parser.add_argument("--raw-local-assembly", choices=("precomputed", "fused"), default="precomputed")
    parser.add_argument("--raw-lu-mode", choices=("safe", "coop"), default="safe")
    parser.add_argument("--raw-block-size", type=int, choices=(1, 32, 64, 128), default=32)
    parser.add_argument("--raw-matrix-format", choices=("auto", "coo", "csr"), default="auto")
    parser.add_argument("--plot-resolution", "-pr", type=int, default=20)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--gmsh-num-threads", type=int, default=None)
    parser.add_argument("--trace-ordering", choices=("none", "upwind-scc"), default="none")
    parser.add_argument("--trace-ordering-flux-tolerance", type=float, default=0.0)
    parser.add_argument("--solver", choices=("amgx", "cupyx", "direct"), default="amgx")
    parser.add_argument("--cupyx-solver", default="bicgstab")
    parser.add_argument("--amgx-config", default=None)
    parser.add_argument("--amgx-solver", default=None, help="override the solver named in the AMGX config")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--check-rtol", type=float, default=1.0e-10, help="package post-solve residual check tolerance")
    parser.add_argument("--amgx-maxiter", type=int, default=1500)
    parser.add_argument("--materialize-host-system", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--materialize-host-solution", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluate-errors", action="store_true", default=True, help="materialize the host DGField and compute CPU-side error norms; always enabled")
    parser.add_argument("--show-cupy-config", action="store_true")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
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


def _clip_middle(value, max_chars: int = 42) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    if max_chars <= 6:
        return text[:max_chars]
    head = (max_chars - 3) // 2
    tail = max_chars - 3 - head
    return f"{text[:head]}...{text[-tail:]}"


def _display_config_path(path: Path | None) -> str:
    if path is None:
        return "embedded default"
    return _clip_middle(Path(path).name, 34)


def _display_optional_int(value) -> str:
    return "default" if value is None else f"{int(value):,d}"


def _display_width(value) -> int:
    text = str(value)
    try:
        from wcwidth import wcswidth

        width = wcswidth(text)
        return len(text) if width < 0 else width
    except Exception:
        from unicodedata import east_asian_width

        return sum(2 if east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad_right(value, width: int) -> str:
    text = str(value)
    return text + " " * max(0, width - _display_width(text))


def _pad_left(value, width: int) -> str:
    text = str(value)
    return " " * max(0, width - _display_width(text)) + text


def _rule_char(preferred: str, fallback: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        preferred.encode(encoding)
    except UnicodeEncodeError:
        return fallback
    return preferred


def _format_timing_cell(seconds: float, denominator: float) -> str:
    seconds = float(seconds)
    if denominator > 0.0:
        return f"{seconds:.5f}s ({100.0 * seconds / denominator:.1f}%)"
    return f"{seconds:.5f}s"


def _print_timing_rows(title: str, rows: list[tuple[str, float]], denominator: float, percent_label: str) -> None:
    if not rows:
        return
    formatted = []
    for label, seconds in rows:
        seconds = float(seconds)
        percent = 100.0 * seconds / denominator if denominator > 0.0 else float("nan")
        percent_text = "n/a" if not np.isfinite(percent) else f"{percent:.1f}%"
        formatted.append((label, f"{seconds:.5f}", percent_text))

    headers = ("Timing", "Seconds", percent_label)
    widths = [
        max(_display_width(headers[0]), max((_display_width(row[0]) for row in formatted), default=0)),
        max(_display_width(headers[1]), max((_display_width(row[1]) for row in formatted), default=0)),
        max(_display_width(headers[2]), max((_display_width(row[2]) for row in formatted), default=0)),
    ]
    sep = "    "
    heavy = _rule_char("═", "=")
    light = _rule_char("─", "-")
    rule_len = sum(widths) + len(sep) * 2
    print()
    print(title)
    print(heavy * rule_len)
    print(sep.join((_pad_right(headers[0], widths[0]), _pad_left(headers[1], widths[1]), _pad_left(headers[2], widths[2]))))
    print(sep.join((light * widths[0], light * widths[1], light * widths[2])))
    for label, seconds_text, percent_text in formatted:
        print(sep.join((_pad_right(label, widths[0]), _pad_left(seconds_text, widths[1]), _pad_left(percent_text, widths[2]))))


def evaluate_errors_device(coeffs, cspace, exact, plot_resolution: int, error_volume_quad_1d: int | None):
    cp = require_cupy()
    error_volume_quad_1d = None if error_volume_quad_1d is None else int(error_volume_quad_1d)
    if error_volume_quad_1d is None:
        q_points = cspace.quad_data.Krf_quads
        weights = cspace.quad_data.Krf_w
        basis_values = cspace.quad_data.bas_of_quads
        mapped = cspace.mapped_quads
    else:
        err_ref = ReferenceElementData.triangle(
            cspace.order,
            basis_type=cspace.host.quad_data.basis_type,
            volume_quad_1d=error_volume_quad_1d,
            edge_quad_1d=cspace.host.quad_data.edge_quad_1d,
        )
        q_points = cp.asarray(err_ref.Krf_quads, dtype=cp.float64)
        weights = cp.asarray(err_ref.Krf_w, dtype=cp.float64)
        basis_values = cp.asarray(err_ref.bas_of_quads, dtype=cp.float64)
        mapped = cp.einsum("Krc,qc->Krq", cspace.mesh.aff_mats, q_points) + cspace.mesh.aff_vecs[:, :, None]

    exact_q = exact(mapped[:, 0, :], mapped[:, 1, :])
    exact_q = cp.asarray(exact_q, dtype=cp.float64)
    uh_q = coeffs @ basis_values
    diff_q = uh_q - exact_q
    l2 = cp.sqrt(cp.sum(cspace.mesh.aff_jacs[:, None] * diff_q * diff_q * weights[None, :]))

    grid = cp.linspace(-1.0, 1.0, int(plot_resolution), endpoint=True, dtype=cp.float64)
    xx, yy = cp.meshgrid(grid, grid)
    mask = yy <= -xx
    ref_points = cp.stack((xx[mask], yy[mask]), axis=1)
    basis_plot = cp.asarray(cspace.host.basis_at(cp.asnumpy(ref_points)), dtype=cp.float64)
    mapped_plot = cp.einsum("Krc,qc->Krq", cspace.mesh.aff_mats, ref_points) + cspace.mesh.aff_vecs[:, :, None]
    exact_plot = cp.asarray(exact(mapped_plot[:, 0, :], mapped_plot[:, 1, :]), dtype=cp.float64)
    uh_plot = coeffs @ basis_plot.T
    abs_err = cp.abs(uh_plot - exact_plot)
    element_max = cp.max(abs_err, axis=-1)
    cp.cuda.get_current_stream().synchronize()
    return (
        float(l2.get()),
        float(cp.max(element_max).get()),
        float(cp.mean(element_max).get()),
        int(cp.argmax(element_max).get()),
    )


def evaluate_errors(field, exact, plot_resolution: int, error_volume_quad_1d: int | None):
    space = field.space
    if error_volume_quad_1d is None:
        l2 = field.l2_error(exact)
    else:
        err_ref = ReferenceElementData.triangle(
            space.order,
            basis_type=space.quad_data.basis_type,
            volume_quad_1d=int(error_volume_quad_1d),
            edge_quad_1d=space.quad_data.edge_quad_1d,
        )
        mapped = space.mesh.map_reference_points(err_ref.Krf_quads)
        exact_q = exact(mapped[:, :, 0], mapped[:, :, 1])
        uh_q = field.coeffs @ err_ref.bas_of_quads
        l2 = float(np.sqrt(np.einsum("K,Kq,q->", space.mesh.aff_jacs, (uh_q - exact_q) ** 2, err_ref.Krf_w)))

    grid = np.linspace(-1.0, 1.0, int(plot_resolution), endpoint=True)
    xx, yy = np.meshgrid(grid, grid)
    ref_points = np.column_stack((xx[yy <= -xx], yy[yy <= -xx]))
    basis_plot = space.basis_at(ref_points)
    mapped_plot = space.mesh.map_reference_points(ref_points)
    exact_plot = exact(mapped_plot[:, :, 0], mapped_plot[:, :, 1])
    uh_plot = field.coeffs @ basis_plot.T
    abs_err = np.abs(uh_plot - exact_plot)
    linf = float(np.max(abs_err))
    avg_max = float(np.average(np.max(abs_err, axis=-1)))
    max_element = int(np.argmax(np.max(abs_err, axis=-1)))
    return l2, linf, avg_max, max_element


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cp = require_cupy()
    require_cupyx_sparse()
    if args.solver in {"amgx", "pyamgx"}:
        require_pyamgx()
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    if args.show_cupy_config:
        cp.show_config()

    run_start = time.perf_counter()
    mesh_start = time.perf_counter()
    mesh = build_mesh(args)
    mesh_time = time.perf_counter() - mesh_start
    space_start = time.perf_counter()
    space = DGSpace(mesh, args.order, basis_type=args.basis, volume_quad_1d=args.volume_quad_1d, edge_quad_1d=args.edge_quad_1d)
    space_time = time.perf_counter() - space_start
    problem_start = time.perf_counter()
    try:
        case = case_definition_by_key(args.case)
    except ValueError:
        if args.case != "test2_legacy_gpu3":
            raise
        case = case_definition_by_key("test2")
    beta_x, beta_y, reaction, source, exact = case.build()
    solver_source = source
    solver_reaction = reaction
    solver_beta = (beta_x, beta_y)
    if args.assembly_backend == "raw-cuda":
        projection_start = time.perf_counter()
        solver_source = space.project_callable(source, name="source_h")
        solver_reaction = space.project_callable(reaction, name="reaction_h")
        solver_beta = VectorDGField((beta_x, beta_y), space, name="beta_h")
        raw_input_projection_time = time.perf_counter() - projection_start
    else:
        raw_input_projection_time = 0.0
    if args.solver in {"amgx", "pyamgx"}:
        amgx_config, amgx_config_path = load_amgx_config(args)
        effective_amgx_solver = str(amgx_config.get("solver", {}).get("solver", "unknown"))
    else:
        amgx_config, amgx_config_path = None, None
        effective_amgx_solver = args.cupyx_solver if args.solver == "cupyx" else args.solver
    problem_time = time.perf_counter() - problem_start
    device_error_candidate = args.assembly_backend == "raw-cuda"
    materialize_host_solution = args.materialize_host_solution
    if args.evaluate_errors and not device_error_candidate:
        materialize_host_solution = True

    solver = AdvectionReactionHDGSolver(
        space,
        source=solver_source,
        beta=solver_beta,
        reaction=solver_reaction,
        boundary_condition=exact,
        solver=args.solver,
        preconditioner=None,
        solver_rtol=args.check_rtol,
        maxiter=args.amgx_maxiter,
        cupyx_solver=args.cupyx_solver,
        amgx_config=amgx_config,
        scale_system=True,
        boundary_mode="eliminate",
        trace_ordering=args.trace_ordering,
        trace_ordering_flux_tolerance=args.trace_ordering_flux_tolerance,
        assembly_backend=args.assembly_backend,
        trace_basis=args.trace_basis,
        raw_local_assembly=args.raw_local_assembly,
        raw_lu_mode=args.raw_lu_mode,
        raw_block_size=args.raw_block_size,
        raw_matrix_format=args.raw_matrix_format,
        materialize_host_system=args.materialize_host_system,
        materialize_host_solution=materialize_host_solution,
        verbose=args.verbosity,
    )
    solve_call_start = time.perf_counter()
    result = solver.solve()
    solve_call_time = time.perf_counter() - solve_call_start
    error_time = 0.0
    error_eval_mode = "no"
    if args.evaluate_errors:
        error_start = time.perf_counter()
        if result.field_device is not None:
            cspace = as_cupy_space(space)
            l2, linf, avg_max, max_element = evaluate_errors_device(
                result.field_device,
                cspace,
                exact,
                args.plot_resolution,
                args.error_volume_quad_1d,
            )
            error_eval_mode = "device"
        else:
            if result.field is None:
                raise RuntimeError("error evaluation requires a device field or a host-materialized DGField")
            l2, linf, avg_max, max_element = evaluate_errors(result.field, exact, args.plot_resolution, args.error_volume_quad_1d)
            error_eval_mode = "host"
        error_time = time.perf_counter() - error_start
        error_items = [
            ("h^(p+1)", mesh.h ** (args.order + 1), ".3e"),
            ("L2 error", l2, ".3e"),
            ("Linf error", linf, ".3e"),
            ("avg max error", avg_max, ".3e"),
            ("max-error element", max_element, ",d"),
        ]
    else:
        error_items = [
            ("h^(p+1)", mesh.h ** (args.order + 1), ".3e"),
            ("L2 error", "not evaluated", "s"),
            ("Linf error", "not evaluated", "s"),
            ("avg max error", "not evaluated", "s"),
            ("max-error element", "not evaluated", "s"),
        ]
    total = time.perf_counter() - run_start
    solve = result.global_solve_result
    global_dof = result.solve_rhs.size if result.solve_rhs is not None else space.layout.reduced_trace_vector_size
    host_solution_state = "yes" if result.field is not None and result.trace is not None else "no"
    device_solution_state = "yes" if getattr(result, "field_device", None) is not None and getattr(result, "trace_device", None) is not None else "no"
    effective_matrix_format = args.raw_matrix_format
    if effective_matrix_format == "auto":
        direct_device_amgx = args.solver in {"amgx", "pyamgx"}
        effective_matrix_format = (
            "csr"
            if args.assembly_backend == "raw-cuda"
            and args.raw_local_assembly == "fused"
            and direct_device_amgx
            and not args.materialize_host_system
            else "coo"
        )

    detail = result.timings.details
    detail_order = [
        ("prepare.coefficients", "prepare coeffs (s)"),
        ("raw.cupy_space", "GPU space mirror (s)"),
        ("raw.trace_space.host", "trace host setup (s)"),
        ("raw.trace_space.device", "trace device copy (s)"),
        ("runner.raw_input_projection", "runner raw input projection (s)"),
        ("raw.beta_projection.x", "beta_x projection (s)"),
        ("raw.beta_projection.y", "beta_y projection (s)"),
        ("raw.beta_projection.stack", "beta stack (s)"),
        ("raw.beta_coeffs.to_device", "beta coeff copy (s)"),
        ("raw.beta_dot_normal", "beta dot normal (s)"),
        ("raw.assembly.projection.source", "source projection (s)"),
        ("raw.assembly.projection.reaction", "reaction projection (s)"),
        ("raw.assembly.reference_advection_tensor", "adv tensor setup (s)"),
        ("raw.assembly.raw.map_setup", "raw map setup (s)"),
        ("raw.assembly.raw.csr_pattern.incident", "CSR incident (s)"),
        ("raw.assembly.raw.csr_pattern.neighbors", "CSR neighbors (s)"),
        ("raw.assembly.raw.csr_pattern.cumsum", "CSR cumsum (s)"),
        ("raw.assembly.raw.csr_pattern.expand", "CSR expand (s)"),
        ("raw.assembly.raw.csr_zero", "CSR zero (s)"),
        ("raw.assembly.raw.kernel", "raw kernel (s)"),
        ("raw.assembly.raw.csr_kernel", "raw CSR kernel (s)"),
        ("raw.assembly.total", "backend assembly total (s)"),
        ("solve.amgx.csr", "AMGX CSR build (s)"),
        ("solve.scale", "row scaling (s)"),
        ("solve.amgx.setup", "AMGX setup (s)"),
        ("solve.amgx.solve", "AMGX iterate (s)"),
        ("solve.amgx.total", "AMGX call total (s)"),
        ("solve.amgx.overhead", "AMGX overhead (s)"),
        ("solve.validation.finite", "finite check (s)"),
        ("solve.validation.solver_residual", "scaled residual check (s)"),
        ("solve.validation.physical_residual", "physical residual check (s)"),
        ("solve.validation.total", "solve validation total (s)"),
        ("solve.global.overhead", "global solve overhead (s)"),
        ("raw.reconstruct.trace_device", "trace recon device (s)"),
        ("raw.reconstruct.field_device", "field recon device (s)"),
        ("raw.host_system_materialization", "host system copy (s)"),
        ("raw.host_solution_materialization", "host solution copy (s)"),
    ]
    detail_items = [(label, float(detail[key])) for key, label in detail_order if key in detail]
    if raw_input_projection_time:
        detail_items.insert(0, ("runner raw input projection (s)", raw_input_projection_time))

    run_options = [
        ("case", args.case, "s"),
        ("order", args.order, ",d"),
        ("basis", args.basis, "s"),
        ("trace basis", args.trace_basis, "s"),
        ("backend", args.assembly_backend, "s"),
        ("global solver", args.solver, "s"),
        ("raw local", args.raw_local_assembly, "s"),
        ("raw LU", args.raw_lu_mode, "s"),
        ("raw block", args.raw_block_size, ",d"),
        ("matrix", effective_matrix_format, "s"),
        ("host system", "yes" if result.solve_rhs is not None else "no", "s"),
        ("host solution", host_solution_state, "s"),
        ("device solution", device_solution_state, "s"),
        ("error eval", error_eval_mode, "s"),
    ]
    mesh_details = [
        ("mesh", args.mesh_type, "s"),
        ("mesh size", str(args.mesh_size) if args.mesh_type != "structured-rectangle" else f"{args.nx}x{args.ny or args.nx}", "s"),
        ("h", mesh.h, ".3e"),
        ("triangles", mesh.num_tri, ",d"),
        ("edges", mesh.num_edg, ",d"),
        ("global dof", global_dof, ",d"),
        ("element dof", space.el_dof, ",d"),
        ("edge dof", space.quad_data.edg_dof, ",d"),
        ("vol quad 1d", _display_optional_int(space.quad_data.volume_quad_1d), "s"),
        ("edge quad 1d", _display_optional_int(space.quad_data.edge_quad_1d), "s"),
    ]
    solver_details = [
        ("config", _display_config_path(amgx_config_path), "s"),
        ("solver", effective_amgx_solver, "s"),
        ("tol", args.amgx_tolerance, ".1e"),
        ("check rtol", args.check_rtol, ".1e"),
        ("maxiter", args.amgx_maxiter, ",d"),
        ("iterations", getattr(solve, "iteration_count", -1) or -1, ",d"),
        ("scaled rel residual", solve.solver_relative_residual_norm, ".3e"),
    ]
    errors = error_items
    main_timing_items = [
        ("mesh setup", mesh_time),
        ("space setup", space_time),
        ("problem setup", problem_time),
        ("solver call", solve_call_time),
        ("package solve total", result.timings.total),
        ("assembly total", result.timings.assembly),
        ("global solve phase", result.timings.solve),
        ("reconstruct total", result.timings.reconstruction),
        ("error eval", error_time),
        ("total measured", total),
    ]
    timings = [(label, _format_timing_cell(seconds, total), "s") for label, seconds in main_timing_items]

    _print_timing_rows(
        "HDGFEM GPU4 Detailed Solver Timings",
        detail_items,
        float(result.timings.total),
        "% solver",
    )

    sections = [
        ("Run / Options", run_options),
        ("Mesh / DOF", mesh_details),
        ("Solver", solver_details),
        ("Errors", errors),
        ("Timings", timings),
    ]
    pretty_print_sections(sections, title="HDGFEM GPU4 Advection-Reaction Solve Summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
