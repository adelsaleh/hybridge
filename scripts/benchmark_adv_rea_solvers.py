#!/usr/bin/env python3
"""Benchmark advection-reaction linear solvers after one trace assembly."""

from __future__ import annotations

import json
import sys
import time
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class LinearSolveConfig:
    """One global trace solver configuration."""

    name: str
    solver: str | None
    preconditioner: str | None = None
    native_preconditioner: str | None = None
    upwind_bgs_sweep: str = "forward"
    petsc_preset: str = "gmres_ilu"
    petsc_levels: int | None = None
    petsc_options: dict[str, Any] = field(default_factory=dict)
    ilu_drop_tol: float = 1.0e-10
    ilu_fill_factor: float = 35.0
    ilu_permc_spec: str = "NATURAL"
    maxiter: int | None = 2000


ITERATIVE_CONFIGS = [
    LinearSolveConfig(
        name="scipy_bicgstab_ilu",
        solver="BICGSTAB",
        preconditioner="ilu",
    ),
    LinearSolveConfig(
        name="scipy_bicgstab_upwind_bgs",
        solver="BICGSTAB",
        preconditioner=None,
        native_preconditioner="upwind_block_gs",
    ),
    LinearSolveConfig(
        name="scipy_gmres_upwind_bgs",
        solver="GMRES",
        preconditioner=None,
        native_preconditioner="upwind_block_gs",
    ),
    LinearSolveConfig(
        name="scipy_bicgstab_upwind_fbgs",
        solver="BICGSTAB",
        preconditioner=None,
        native_preconditioner="upwind_block_gs",
        upwind_bgs_sweep="forward_backward",
    ),
    LinearSolveConfig(
        name="scipy_gmres_upwind_fbgs",
        solver="GMRES",
        preconditioner=None,
        native_preconditioner="upwind_block_gs",
        upwind_bgs_sweep="forward_backward",
    ),
    LinearSolveConfig(
        name="petsc_bicgstab_ilu",
        solver="petsc",
        petsc_preset="bicgstab_ilu",
    ),
    LinearSolveConfig(
        name="petsc_gmres_ilu",
        solver="petsc",
        petsc_preset="gmres_ilu",
    ),
    LinearSolveConfig(
        name="petsc_bicgstab_asm_ilu",
        solver="petsc",
        petsc_preset="bicgstab_asm_ilu",
        petsc_options={"pc_asm_overlap": 1},
    ),
    LinearSolveConfig(
        name="petsc_gmres_asm_ilu",
        solver="petsc",
        petsc_preset="gmres_asm_ilu",
        petsc_options={"pc_asm_overlap": 1},
    ),
]

DIRECT_CONFIGS = [
    LinearSolveConfig(
        name="scipy_direct",
        solver="direct",
        preconditioner=None,
        maxiter=None,
    ),
    LinearSolveConfig(
        name="petsc_lu",
        solver="petsc",
        petsc_preset="lu",
        maxiter=None,
    ),
    LinearSolveConfig(
        name="petsc_mumps_lu",
        solver="petsc",
        petsc_preset="mumps_lu",
        maxiter=None,
    ),
]

CONFIGS_BY_NAME = {config.name: config for config in ITERATIVE_CONFIGS + DIRECT_CONFIGS}


def _format_seconds(seconds: float) -> str:
    if seconds >= 100.0:
        return f"{seconds:.1f}s"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds:.4f}s"


def _timed(label: str, function, *, verbose: bool = True):
    if verbose:
        print(f"{label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"done in {_format_seconds(elapsed)}", flush=True)
    return result, elapsed


def _build_mesh(mesh_size: float, gmsh_verbosity: int):
    from hdgfem.core.mesh import gmsh_rectangle_mesh

    return gmsh_rectangle_mesh(
        mesh_size,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        verbosity=gmsh_verbosity,
    )


def _assemble_trace_problem(args):
    from hdgfem.core.space import DGField, DGSpace, VectorDGField
    from hdgfem.solvers.adv_rea import AdvectionReactionHDGSolver
    from scripts.adv_rea_cases import test2

    mesh, mesh_time = _timed(
        "generating rectangle mesh",
        lambda: _build_mesh(args.lc, args.gmsh_verbosity),
        verbose=not args.quiet,
    )
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    beta_x, beta_y, reaction, source, exact = test2()
    source_h = DGField(source, space, name="source_h")
    beta_h = VectorDGField((beta_x, beta_y), space, name="beta_h")
    reaction_h = DGField(reaction, space, name="reaction_h")

    solver = AdvectionReactionHDGSolver(
        space,
        source=source_h,
        beta=beta_h,
        reaction=reaction_h,
        boundary_condition=exact,
        solver="BICGSTAB",
        preconditioner="ilu",
        boundary_mode="eliminate",
        trace_ordering="upwind-scc",
        assembly_backend="numba",
        cache_local_solvers=False,
        verbose=0 if args.quiet else args.verbosity,
    )
    result, assembly_time = _timed(
        "assembling reduced upwind-ordered trace system",
        solver.assemble_trace_system,
        verbose=not args.quiet,
    )
    return solver, result, mesh_time, assembly_time


def _selected_configs(args) -> list[LinearSolveConfig]:
    if args.config:
        return [CONFIGS_BY_NAME[name] for name in args.config]
    configs = list(ITERATIVE_CONFIGS)
    if args.include_direct:
        configs.extend(DIRECT_CONFIGS)
    return configs


def _inverse_diagonal(matrix) -> Any:
    import numpy as np

    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    diagonal[diagonal == 0.0] = 1.0
    return 1.0 / diagonal


def _upwind_bgs_sweep(config: LinearSolveConfig, args) -> str:
    return args.upwind_bgs_sweep or config.upwind_bgs_sweep


def _native_preconditioner_cache_key(config: LinearSolveConfig, solver, args) -> tuple[Any, ...]:
    return (
        config.native_preconditioner,
        solver.space.quad_data.edg_dof,
        args.upwind_bgs_diagonal_regularization,
        args.upwind_bgs_apply_mode,
        args.upwind_bgs_parallel_min_width,
        _upwind_bgs_sweep(config, args),
        not args.no_upwind_bgs_warmup,
    )


def _print_upwind_bgs_diagnostics(stats, *, setup_seconds: float, scaling_seconds: float) -> None:
    print(
        "  upwind block-GS preconditioner: "
        f"mode={stats.apply_mode}, "
        f"sweep={stats.sweep}, "
        f"levels={stats.num_levels:,}, max_width={stats.max_width:,}, "
        f"upstream={stats.retained_block_couplings:,}, "
        f"downstream={stats.downstream_block_couplings:,}, "
        f"dropped_same={stats.dropped_same_level_couplings:,}, "
        f"dropped_downstream={stats.dropped_downstream_couplings:,}, "
        f"dropped_fraction={stats.dropped_coupling_fraction:.3f}, "
        f"setup={_format_seconds(setup_seconds)}",
        flush=True,
    )
    print(
        "  upwind block-GS setup breakdown: "
        f"scaling={_format_seconds(scaling_seconds)}, "
        f"csr={_format_seconds(stats.csr_prepare_seconds)}, "
        f"count={_format_seconds(stats.coupling_count_seconds)}, "
        f"fill={_format_seconds(stats.block_fill_seconds)}, "
        f"invert={_format_seconds(stats.diagonal_inverse_seconds)}, "
        f"warmup={_format_seconds(stats.warmup_seconds)}",
        flush=True,
    )


def _build_native_preconditioner(config: LinearSolveConfig, matrix, rhs, trace_result, solver, args, native_cache):
    if config.native_preconditioner is None:
        return config.preconditioner, {}, 0.0
    if config.native_preconditioner != "upwind_block_gs":
        raise ValueError(f"unknown native preconditioner {config.native_preconditioner!r}")
    if config.solver is None or str(config.solver).lower() == "petsc":
        raise ValueError("upwind_block_gs is a SciPy LinearOperator preconditioner")
    if trace_result.ordering_result is None:
        raise RuntimeError("upwind_block_gs requires an upwind SCC ordering result")
    if trace_result.ordering_result.diagnostics.largest_component_size != 1:
        raise RuntimeError(
            "upwind_block_gs currently requires singleton SCCs; "
            f"largest SCC has size {trace_result.ordering_result.diagnostics.largest_component_size}"
        )

    import numpy as np
    from hdgfem.linalg.system import diagonal_scale_system
    from hdgfem.linalg.upwind_block_gs import build_upwind_block_gs_preconditioner

    cache_key = _native_preconditioner_cache_key(config, solver, args)
    cached = native_cache.get(cache_key)
    if cached is not None:
        preconditioner, prepared_scaling = cached
        preconditioner.reset_timing()
        if not args.quiet:
            print("  reusing cached upwind block-GS preconditioner", flush=True)
        return preconditioner, prepared_scaling, 0.0

    start = time.perf_counter()
    scaling_start = time.perf_counter()
    inverse_diagonal = _inverse_diagonal(matrix)
    scaled_matrix, _ = diagonal_scale_system(matrix, rhs, copy_matrix=True)
    scaling_seconds = time.perf_counter() - scaling_start
    preconditioner = build_upwind_block_gs_preconditioner(
        scaled_matrix,
        block_size=solver.space.quad_data.edg_dof,
        level_widths=trace_result.ordering_result.diagnostics.level_widths,
        diagonal_regularization=args.upwind_bgs_diagonal_regularization,
        apply_mode=args.upwind_bgs_apply_mode,
        parallel_min_width=args.upwind_bgs_parallel_min_width,
        sweep=_upwind_bgs_sweep(config, args),
        warm_start=not args.no_upwind_bgs_warmup,
    )
    elapsed = time.perf_counter() - start

    stats = preconditioner.stats
    if not args.quiet:
        _print_upwind_bgs_diagnostics(stats, setup_seconds=elapsed, scaling_seconds=scaling_seconds)

    prepared_scaling = {
        "prepared_scaled_matrix": scaled_matrix,
        "prepared_inverse_diagonal": np.ascontiguousarray(inverse_diagonal, dtype=np.float64),
    }
    native_cache[cache_key] = (preconditioner, prepared_scaling)
    return preconditioner, prepared_scaling, elapsed


def _solve_cached_system(config: LinearSolveConfig, matrix, rhs, trace_result, solver, args, native_cache):
    from hdgfem.linalg.system import solve_global_system

    solver_is_petsc = config.solver is not None and str(config.solver).lower() == "petsc"
    preconditioner, prepared_scaling, native_setup_seconds = _build_native_preconditioner(
        config,
        matrix,
        rhs,
        trace_result,
        solver,
        args,
        native_cache,
    )
    result = solve_global_system(
        row_indices=(),
        col_indices=(),
        matrix_values=(),
        rhs=rhs,
        system_size=rhs.size,
        solver=config.solver,
        preconditioner=preconditioner,
        rtol=args.rtol,
        atol=args.atol,
        maxiter=config.maxiter,
        ilu_drop_tol=config.ilu_drop_tol,
        ilu_fill_factor=config.ilu_fill_factor,
        ilu_failure="raise",
        ilu_permc_spec=config.ilu_permc_spec,
        petsc_preset=config.petsc_preset,
        petsc_levels=config.petsc_levels,
        petsc_options=config.petsc_options,
        petsc_divtol=args.petsc_divtol,
        petsc_monitor=args.petsc_monitor,
        scale_system=not solver_is_petsc,
        scale_matrix_in_place=False,
        raise_on_nonconvergence=True,
        verbose=0 if args.quiet else args.verbosity,
        assembled_matrix=matrix,
        **prepared_scaling,
    )
    if config.native_preconditioner is not None:
        result.preconditioner_elapsed_seconds = native_setup_seconds
        if result.total_elapsed_seconds is not None:
            result.total_elapsed_seconds += native_setup_seconds
    return result


def _result_row(config: LinearSolveConfig, result, elapsed: float) -> dict[str, Any]:
    preconditioner_stats = getattr(getattr(result, "preconditioner", None), "stats", None)
    return {
        "name": config.name,
        "solver": config.solver,
        "native_preconditioner": config.native_preconditioner,
        "petsc_preset": result.petsc_preset,
        "elapsed_seconds": elapsed,
        "total_seconds": result.total_elapsed_seconds,
        "preconditioner_seconds": result.preconditioner_elapsed_seconds,
        "solve_seconds": result.solve_elapsed_seconds,
        "iterations": result.iteration_count,
        "relative_residual": result.solver_relative_residual_norm,
        "preconditioner_apply_count": result.preconditioner_apply_count,
        "preconditioner_apply_seconds": result.preconditioner_apply_seconds,
        "upwind_block_gs": None if preconditioner_stats is None else asdict(preconditioner_stats),
        "petsc_reason": result.petsc_converged_reason,
        "info": result.info,
    }


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print("Solver Summary")
    print("==============")
    print(
        f"{'name':<30} {'elapsed':>9} {'setup':>9} {'solve':>9} "
        f"{'it':>6} {'applies':>8} {'relres':>12} {'reason':>7}"
    )
    for row in rows:
        elapsed = row["elapsed_seconds"]
        setup = row["preconditioner_seconds"]
        solve = row["solve_seconds"]
        relres = row["relative_residual"]
        applies = row["preconditioner_apply_count"]
        print(
            f"{row['name']:<30} "
            f"{elapsed if elapsed is not None else float('nan'):>9.3f} "
            f"{0.0 if setup is None else setup:>9.3f} "
            f"{0.0 if solve is None else solve:>9.3f} "
            f"{-1 if row['iterations'] is None else row['iterations']:>6d} "
            f"{-1 if applies is None else applies:>8d} "
            f"{float('nan') if relres is None else relres:>12.3e} "
            f"{'' if row['petsc_reason'] is None else row['petsc_reason']:>7}"
        )


def _warm_petsc_runtime():
    from petsc4py import PETSc

    return PETSc.Sys.getVersion()


def _main() -> None:
    parser = ArgumentParser(description="Assemble adv_rea test2 once and benchmark global solvers.")
    parser.add_argument("-p", "--order", type=int, default=6, help="DG polynomial order")
    parser.add_argument("--lc", type=float, default=0.01, help="Gmsh target mesh size")
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1.0e-13)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--petsc-divtol", type=float, default=1.0e4)
    parser.add_argument("--petsc-monitor", action="store_true")
    parser.add_argument("--include-direct", action="store_true", help="also run direct LU/MUMPS configs")
    parser.add_argument(
        "--upwind-bgs-diagonal-regularization",
        type=float,
        default=0.0,
        help="value added to each edge-block diagonal before block-GS inversion",
    )
    parser.add_argument(
        "--upwind-bgs-apply-mode",
        choices=("auto", "serial", "parallel"),
        default="auto",
        help="Numba apply kernel for upwind block-GS",
    )
    parser.add_argument(
        "--upwind-bgs-sweep",
        choices=("forward", "forward_backward"),
        default=None,
        help="override the block-GS sweep used by all upwind block-GS configs",
    )
    parser.add_argument(
        "--upwind-bgs-parallel-min-width",
        type=int,
        default=1024,
        help="max level width required before --upwind-bgs-apply-mode=auto chooses parallel",
    )
    parser.add_argument(
        "--no-upwind-bgs-warmup",
        action="store_true",
        help="do not warm the Numba block-GS apply kernel during preconditioner setup",
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=tuple(sorted(CONFIGS_BY_NAME)),
        help="run only this config; repeat for multiple configs",
    )
    parser.add_argument("--verbosity", "-v", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="suppress script phase messages")
    parser.add_argument("--json-out", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args()

    solver, trace_result, mesh_time, assembly_time = _assemble_trace_problem(args)
    if trace_result.solve_matrix_rows is None or trace_result.solve_rhs is None:
        raise RuntimeError("trace assembly did not produce a solve matrix")

    from hdgfem.linalg.system import assemble_global_matrix

    matrix, matrix_time = _timed(
        "building reusable SciPy CSR matrix",
        lambda: assemble_global_matrix(
            trace_result.solve_matrix_rows,
            trace_result.solve_matrix_cols,
            trace_result.solve_matrix_data,
            trace_result.solve_rhs.size,
        ),
        verbose=not args.quiet,
    )
    print(
        "assembled problem: "
        f"p={args.order}, lc={args.lc}, triangles={solver.space.mesh.num_tri:,}, "
        f"edges={solver.space.mesh.num_edg:,}, dofs={trace_result.solve_rhs.size:,}, "
        f"nnz={matrix.nnz:,}",
        flush=True,
    )

    selected_configs = _selected_configs(args)
    if any(config.solver is not None and str(config.solver).lower() == "petsc" for config in selected_configs):
        _timed(
            "warming PETSc runtime",
            _warm_petsc_runtime,
            verbose=not args.quiet,
        )

    rows = []
    native_cache = {}
    for config in selected_configs:
        print(f"\n===== {config.name} =====", flush=True)
        result, elapsed = _timed(
            f"solving with {config.name}",
            lambda config=config: _solve_cached_system(
                config,
                matrix,
                trace_result.solve_rhs,
                trace_result,
                solver,
                args,
                native_cache,
            ),
            verbose=not args.quiet,
        )
        rows.append(_result_row(config, result, elapsed))

    _print_summary(rows)

    if args.json_out is not None:
        payload = {
            "problem": {
                "case": "test2",
                "order": args.order,
                "lc": args.lc,
                "triangles": solver.space.mesh.num_tri,
                "edges": solver.space.mesh.num_edg,
                "dofs": int(trace_result.solve_rhs.size),
                "nnz": int(matrix.nnz),
                "mesh_seconds": mesh_time,
                "assembly_seconds": assembly_time,
                "csr_seconds": matrix_time,
            },
            "configs": [asdict(config) for config in selected_configs],
            "results": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote JSON: {args.json_out}", flush=True)


if __name__ == "__main__":
    _main()
