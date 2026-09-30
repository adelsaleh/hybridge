#!/usr/bin/env python3
"""Qualify mesh-independent versus inverse-h ADR diffusion stabilization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hdgfem.diagnostics.errors import evaluate_scalar_error, evaluate_vector_error
from scripts.advection_diffusion_reaction.studies.manufactured_disk import (
    run_manufactured_adr_disk,
)


STABILIZATION_MODES = ("global-length", "inverse-h")
FLUX_SPACES = ("l2_closest", "RT_projection")
ERROR_KEYS = (
    "primal_l2_error",
    "primal_linf_sampled",
    "post_primal_l2_error",
    "post_primal_linf_sampled",
    "total_flux_l2_error",
    "total_flux_linf_sampled",
    "post_total_flux_l2_error",
    "post_total_flux_linf_sampled",
)


def parse_float_list(raw: str, *, label: str) -> tuple[float, ...]:
    """Parse a nonempty comma-separated list of unique positive floats."""
    values = tuple(float(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{label} must be a comma-separated list of positive finite values")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def parse_int_list(raw: str, *, label: str) -> tuple[int, ...]:
    """Parse a nonempty comma-separated list of unique nonnegative integers."""
    values = tuple(int(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{label} must be a comma-separated list of nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def parse_choices(raw: str, *, label: str, choices: Iterable[str]) -> tuple[str, ...]:
    """Parse and validate a comma-separated ordered subset of string choices."""
    allowed = tuple(choices)
    values = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError(f"{label} must not be empty")
    unknown = tuple(value for value in values if value not in allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported values: {', '.join(unknown)}")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def stabilization_cases(
        modes: tuple[str, ...],
        gammas: tuple[float, ...],
) -> tuple[tuple[str, float | None], ...]:
    """Expand selected modes into inverse-h and per-gamma global-length cases."""
    cases: list[tuple[str, float | None]] = []
    for mode in modes:
        if mode == "global-length":
            cases.extend((mode, float(gamma)) for gamma in gammas)
        elif mode == "inverse-h":
            cases.append((mode, None))
        else:
            raise ValueError(f"unsupported stabilization mode {mode!r}")
    return tuple(cases)


def _host_array(value, *, dtype) -> np.ndarray:
    """Materialize a host diagnostic array through an explicit study boundary."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=dtype)


def dense_condition_number(result, *, max_dofs: int) -> tuple[float | None, str]:
    """Return exact dense 2-norm conditioning below a configured DOF limit."""
    num_dofs = int(result.rhs.size)
    if max_dofs <= 0 or num_dofs > int(max_dofs):
        return None, "skipped"
    import scipy.sparse

    rows = _host_array(result.matrix_rows, dtype=np.int64)
    cols = _host_array(result.matrix_cols, dtype=np.int64)
    data = _host_array(result.matrix_data, dtype=np.float64)
    matrix = scipy.sparse.coo_array(
        (data, (rows, cols)),
        shape=(num_dofs, num_dofs),
    ).tocsr()
    matrix.sum_duplicates()
    condition = float(np.linalg.cond(matrix.toarray()))
    return condition, "dense-2norm"


def _rate_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return the fields defining one independent h-convergence series."""
    return (
        int(row["order"]),
        str(row["stabilization_mode"]),
        row["gamma_d"],
        str(row["flux_postprocess_space"]),
    )


def add_pairwise_rates(rows: list[dict[str, Any]]) -> None:
    """Add actual-mesh-h pairwise rates for every recorded error metric."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_rate_group_key(row), []).append(row)
    for group in grouped.values():
        group.sort(key=lambda row: float(row["mesh_h"]), reverse=True)
        for index, row in enumerate(group):
            for error_key in ERROR_KEYS:
                rate_key = (
                    error_key.removesuffix("_error") + "_rate"
                    if error_key.endswith("_error")
                    else error_key + "_rate"
                )
                if index == 0:
                    row[rate_key] = None
                    continue
                coarse = group[index - 1]
                coarse_h = float(coarse["mesh_h"])
                fine_h = float(row["mesh_h"])
                coarse_error = float(coarse[error_key])
                fine_error = float(row[error_key])
                if (
                    coarse_h <= fine_h
                    or coarse_error <= 0.0
                    or fine_error <= 0.0
                ):
                    row[rate_key] = None
                    continue
                row[rate_key] = math.log(coarse_error / fine_error) / math.log(
                    coarse_h / fine_h
                )


def write_outputs(
        rows: list[dict[str, Any]],
        output_dir: Path,
        prefix: str,
) -> tuple[Path, Path, Path]:
    """Write complete CSV/JSON data and a compact Markdown rate summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"
    markdown_path = output_dir / f"{prefix}.md"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stationary ADR diffusion-stabilization study",
        "",
        "All Linf columns are maxima on the configured triangular sample grid, "
        "not certified continuum norm bounds.",
        "",
        "| p | mode | gamma | flux recovery | requested h | actual h | "
        "post u L2 | rate | post u sampled Linf | rate | post total flux L2 | rate |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["order"]),
            str(row["stabilization_mode"]),
            -1.0 if row["gamma_d"] is None else float(row["gamma_d"]),
            str(row["flux_postprocess_space"]),
            -float(row["mesh_h"]),
        ),
    )
    for row in ordered:
        gamma = "-" if row["gamma_d"] is None else f"{float(row['gamma_d']):.3g}"
        primal_rate = row.get("post_primal_l2_rate")
        primal_linf_rate = row.get("post_primal_linf_sampled_rate")
        flux_rate = row.get("post_total_flux_l2_rate")
        lines.append(
            f"| {int(row['order'])} | {row['stabilization_mode']} | {gamma} | "
            f"{row['flux_postprocess_space']} | {float(row['requested_mesh_size']):.4g} | "
            f"{float(row['mesh_h']):.4g} | {float(row['post_primal_l2_error']):.4e} | "
            f"{'-' if primal_rate is None else f'{float(primal_rate):.3f}'} | "
            f"{float(row['post_primal_linf_sampled']):.4e} | "
            f"{'-' if primal_linf_rate is None else f'{float(primal_linf_rate):.3f}'} | "
            f"{float(row['post_total_flux_l2_error']):.4e} | "
            f"{'-' if flux_rate is None else f'{float(flux_rate):.3f}'} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, markdown_path


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a concise postprocessed L2 convergence table."""
    print("\nStationary ADR diffusion-stabilization convergence")
    print(
        "p  mode           gamma  flux             h(actual)  "
        "post u L2    rate   post total q L2  rate"
    )
    print("-" * 100)
    for row in sorted(
        rows,
        key=lambda item: (
            int(item["order"]),
            str(item["stabilization_mode"]),
            -1.0 if item["gamma_d"] is None else float(item["gamma_d"]),
            str(item["flux_postprocess_space"]),
            -float(item["mesh_h"]),
        ),
    ):
        gamma = "-" if row["gamma_d"] is None else f"{float(row['gamma_d']):.3g}"
        primal_rate = row.get("post_primal_l2_rate")
        flux_rate = row.get("post_total_flux_l2_rate")
        print(
            f"{int(row['order']):<2d} {str(row['stabilization_mode']):<14} "
            f"{gamma:<6} {str(row['flux_postprocess_space']):<16} "
            f"{float(row['mesh_h']):<10.4g} "
            f"{float(row['post_primal_l2_error']):<12.4e} "
            f"{'-' if primal_rate is None else f'{float(primal_rate):.3f}':<6} "
            f"{float(row['post_total_flux_l2_error']):<16.4e} "
            f"{'-' if flux_rate is None else f'{float(flux_rate):.3f}':<6}"
        )


def plot_convergence(
        rows: list[dict[str, Any]],
        output: Path,
        *,
        show: bool = False,
) -> Path:
    """Write L2 and sampled-Linf panels for raw/postprocessed primal/flux."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("primal_l2_error", "raw primal L2"),
        ("post_primal_l2_error", "postprocessed primal L2"),
        ("total_flux_l2_error", "raw total-flux L2"),
        ("post_total_flux_l2_error", "postprocessed total-flux L2"),
        ("primal_linf_sampled", "raw primal sampled Linf"),
        ("post_primal_linf_sampled", "postprocessed primal sampled Linf"),
        ("total_flux_linf_sampled", "raw total-flux sampled Linf"),
        ("post_total_flux_linf_sampled", "postprocessed total-flux sampled Linf"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_rate_group_key(row), []).append(row)
    for key, group in groups.items():
        order, mode, gamma, flux_space = key
        selected = sorted(group, key=lambda row: float(row["mesh_h"]), reverse=True)
        h_values = np.asarray([row["mesh_h"] for row in selected], dtype=np.float64)
        gamma_label = "" if gamma is None else f", gamma={float(gamma):g}"
        label = f"p={order}, {mode}{gamma_label}, {flux_space}"
        for axis, (error_key, title) in zip(axes.flat, panels):
            errors = np.asarray([row[error_key] for row in selected], dtype=np.float64)
            axis.loglog(h_values, errors, "o-", label=label)
            axis.set_title(title)
            axis.set_xlabel("actual mesh h")
            axis.grid(True, which="both", alpha=0.25)
    for axis in axes.flat:
        axis.legend(fontsize=6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    return output


def run_stabilization_study(
        *,
        peclet: float = 10.0,
        mesh_sizes: tuple[float, ...] = (0.4, 0.3, 0.2, 0.15),
        orders: tuple[int, ...] = (2, 3),
        gammas: tuple[float, ...] = (0.5, 1.0, 2.0),
        modes: tuple[str, ...] = STABILIZATION_MODES,
        flux_spaces: tuple[str, ...] = FLUX_SPACES,
        domain_length: float = 1.0,
        assembly_backend: str = "numba",
        reconstruction_backend: str = "numba",
        postprocessing_backend: str = "numba",
        solver: str = "auto",
        solver_rtol: float = 1.0e-11,
        sample_resolution: int = 24,
        error_volume_quad_1d: int | None = 18,
        condition_max_dofs: int = 800,
        output_dir: str | Path = "run_outputs/advection_diffusion_reaction/stabilization",
        prefix: str = "stationary_adr_diffusion_stabilization",
        verbosity: int = 1,
        solver_verbosity: int = 0,
        plot: bool = False,
        plot_output: str | Path | None = None,
        show_plot: bool = False,
) -> tuple[list[dict[str, Any]], Path, Path, Path]:
    """Run the requested host/device stabilization and postprocessing sweep."""
    if not np.isfinite(peclet) or peclet <= 0.0:
        raise ValueError("peclet must be finite and positive")
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be finite and positive")
    if sample_resolution < 2:
        raise ValueError("sample_resolution must be at least 2")
    if error_volume_quad_1d is not None and error_volume_quad_1d < 1:
        raise ValueError("error_volume_quad_1d must be positive or None")
    if not mesh_sizes or any(
        not np.isfinite(value) or value <= 0.0 for value in mesh_sizes
    ):
        raise ValueError("mesh_sizes must contain positive finite values")
    if not orders or any(int(value) != value or value < 0 for value in orders):
        raise ValueError("orders must contain nonnegative integers")
    if not gammas or any(
        not np.isfinite(value) or value <= 0.0 for value in gammas
    ):
        raise ValueError("gammas must contain positive finite values")
    if not modes:
        raise ValueError("modes must not be empty")
    if not flux_spaces:
        raise ValueError("flux_spaces must not be empty")
    if any(mode not in STABILIZATION_MODES for mode in modes):
        raise ValueError(f"modes must be selected from {STABILIZATION_MODES}")
    if any(space not in FLUX_SPACES for space in flux_spaces):
        raise ValueError(f"flux_spaces must be selected from {FLUX_SPACES}")

    selected_solver = solver
    if str(solver).lower() == "auto":
        selected_solver = "amgx" if assembly_backend == "raw-cuda" else "pypardiso"
    rows: list[dict[str, Any]] = []
    condition_cache: dict[tuple[Any, ...], tuple[float | None, str]] = {}
    cases = stabilization_cases(modes, gammas)
    total_runs = len(orders) * len(mesh_sizes) * len(cases) * len(flux_spaces)
    run_index = 0
    for order in orders:
        for requested_mesh_size in sorted(mesh_sizes, reverse=True):
            for mode, gamma in cases:
                for flux_space in flux_spaces:
                    run_index += 1
                    if verbosity:
                        gamma_text = "-" if gamma is None else f"{gamma:g}"
                        print(
                            f"[{run_index}/{total_runs}] p={order} "
                            f"ms={requested_mesh_size:g} mode={mode} "
                            f"gamma={gamma_text} flux={flux_space}",
                            flush=True,
                        )
                    run = run_manufactured_adr_disk(
                        peclet=peclet,
                        mesh_size=requested_mesh_size,
                        order=order,
                        assembly_backend=assembly_backend,
                        reconstruction_backend=reconstruction_backend,
                        postprocessing_backend=postprocessing_backend,
                        hdg_postprocess="both",
                        flux_postprocess_space=flux_space,
                        solver=selected_solver,
                        solver_rtol=solver_rtol,
                        diffusion_stabilization_mode=mode,
                        diffusion_domain_length=domain_length,
                        diffusion_stabilization_gamma=1.0 if gamma is None else gamma,
                        verbosity=solver_verbosity,
                        plot=False,
                    )
                    result = run.result
                    raw_primal = evaluate_scalar_error(
                        result.field,
                        run.problem["exact"],
                        volume_quad_1d=error_volume_quad_1d,
                        sample_resolution=sample_resolution,
                    ).metrics
                    post_primal = evaluate_scalar_error(
                        result.postprocessed_field,
                        run.problem["exact"],
                        volume_quad_1d=error_volume_quad_1d,
                        sample_resolution=sample_resolution,
                    ).metrics
                    raw_flux = evaluate_vector_error(
                        result.total_flux,
                        run.problem["exact_total_flux"],
                        volume_quad_1d=error_volume_quad_1d,
                        sample_resolution=sample_resolution,
                    ).metrics
                    post_flux = evaluate_vector_error(
                        result.postprocessed_flux,
                        run.problem["exact_total_flux"],
                        volume_quad_1d=error_volume_quad_1d,
                        sample_resolution=sample_resolution,
                    ).metrics
                    mesh = result.field.space.mesh
                    tau_diffusion = _host_array(
                        result.tau_diffusion, dtype=np.float64
                    )
                    condition_key = (
                        int(order),
                        float(requested_mesh_size),
                        float(mesh.h),
                        int(mesh.num_tri),
                        str(mode),
                        None if gamma is None else float(gamma),
                    )
                    if condition_key not in condition_cache:
                        condition_cache[condition_key] = dense_condition_number(
                            result,
                            max_dofs=condition_max_dofs,
                        )
                    condition, condition_method = condition_cache[condition_key]
                    global_solve = result.global_solve_result
                    rows.append(
                        {
                            "peclet": float(peclet),
                            "kappa": 1.0 / float(peclet),
                            "order": int(order),
                            "requested_mesh_size": float(requested_mesh_size),
                            "mesh_h": float(mesh.h),
                            "triangles": int(mesh.num_tri),
                            "edges": int(mesh.num_edg),
                            "trace_dofs": int(result.rhs.size),
                            "stabilization_mode": str(mode),
                            "gamma_d": None if gamma is None else float(gamma),
                            "domain_length": (
                                None if mode == "inverse-h" else float(domain_length)
                            ),
                            "tau_diff_min": float(np.min(tau_diffusion)),
                            "tau_diff_max": float(np.max(tau_diffusion)),
                            "flux_postprocess_space": str(flux_space),
                            "assembly_backend": str(result.assembly_backend),
                            "reconstruction_backend": str(result.reconstruction_backend),
                            "postprocessing_backend": str(result.postprocessing_backend),
                            "solver": str(selected_solver),
                            "solver_iterations": (
                                -1
                                if global_solve is None
                                or global_solve.iteration_count is None
                                else int(global_solve.iteration_count)
                            ),
                            "physical_relative_residual": (
                                None
                                if global_solve is None
                                else float(global_solve.physical_relative_residual_norm)
                            ),
                            "condition_2": condition,
                            "condition_method": condition_method,
                            "sample_resolution": int(sample_resolution),
                            "linf_kind": "triangular-grid-sampled-euclidean",
                            "error_volume_quad_1d": error_volume_quad_1d,
                            "primal_l2_error": raw_primal.l2,
                            "primal_linf_sampled": raw_primal.linf,
                            "post_primal_l2_error": post_primal.l2,
                            "post_primal_linf_sampled": post_primal.linf,
                            "total_flux_l2_error": raw_flux.l2,
                            "total_flux_linf_sampled": raw_flux.linf,
                            "post_total_flux_l2_error": post_flux.l2,
                            "post_total_flux_linf_sampled": post_flux.linf,
                            "total_flux_component_linf_sampled": list(
                                raw_flux.component_linf
                            ),
                            "post_total_flux_component_linf_sampled": list(
                                post_flux.component_linf
                            ),
                            "mesh_seconds": float(run.mesh_seconds),
                            "space_seconds": float(run.space_seconds),
                            "assembly_seconds": float(result.timings.assembly),
                            "solve_seconds": float(result.timings.solve),
                            "reconstruction_seconds": float(
                                result.timings.reconstruction
                            ),
                            "postprocessing_seconds": float(
                                result.timings.postprocessing
                            ),
                            "solver_total_seconds": float(result.timings.total),
                        }
                    )
    add_pairwise_rates(rows)
    rows.sort(
        key=lambda row: (
            int(row["order"]),
            str(row["stabilization_mode"]),
            -1.0 if row["gamma_d"] is None else float(row["gamma_d"]),
            str(row["flux_postprocess_space"]),
            -float(row["mesh_h"]),
        )
    )
    csv_path, json_path, markdown_path = write_outputs(
        rows,
        Path(output_dir),
        prefix,
    )
    print_summary(rows)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"summary: {markdown_path}")
    if plot:
        figure_path = (
            Path(plot_output)
            if plot_output is not None
            else Path(output_dir) / f"{prefix}.png"
        )
        print(f"plot: {plot_convergence(rows, figure_path, show=show_plot)}")
    return rows, csv_path, json_path, markdown_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the stationary ADR diffusion-stabilization study CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare global-length and inverse-h ADR diffusion stabilization "
            "using L2 and sampled-Linf errors."
        )
    )
    parser.add_argument("--peclet", type=float, default=10.0)
    parser.add_argument("--mesh-sizes", default="0.4,0.3,0.2,0.15")
    parser.add_argument("--orders", default="2,3")
    parser.add_argument("--gammas", default="0.5,1,2")
    parser.add_argument(
        "--modes",
        default="global-length,inverse-h",
        help="comma-separated subset of global-length,inverse-h",
    )
    parser.add_argument(
        "--flux-spaces",
        default="l2_closest,RT_projection",
        help="comma-separated subset of l2_closest,RT_projection",
    )
    parser.add_argument("--domain-length", type=float, default=1.0)
    parser.add_argument(
        "--assembly-backend",
        choices=("numpy", "numba", "raw-cuda"),
        default="numba",
    )
    parser.add_argument(
        "--reconstruction-backend",
        choices=("auto", "numpy", "numba", "raw-cuda"),
        default="numba",
    )
    parser.add_argument(
        "--postprocessing-backend",
        choices=("auto", "numba", "cupy"),
        default="numba",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "pypardiso", "direct", "BICGSTAB", "amgx"),
        default="auto",
    )
    parser.add_argument("--solver-rtol", type=float, default=1.0e-11)
    parser.add_argument("--sample-resolution", type=int, default=24)
    parser.add_argument("--error-volume-quad-1d", type=int, default=18)
    parser.add_argument("--condition-max-dofs", type=int, default=800)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "run_outputs/advection_diffusion_reaction/stabilization"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="stationary_adr_diffusion_stabilization",
    )
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--solver-verbosity",
        type=int,
        choices=(0, 1, 2),
        default=0,
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--show-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the study from the command line."""
    args = build_arg_parser().parse_args(argv)
    run_stabilization_study(
        peclet=args.peclet,
        mesh_sizes=parse_float_list(args.mesh_sizes, label="mesh_sizes"),
        orders=parse_int_list(args.orders, label="orders"),
        gammas=parse_float_list(args.gammas, label="gammas"),
        modes=parse_choices(
            args.modes,
            label="modes",
            choices=STABILIZATION_MODES,
        ),
        flux_spaces=parse_choices(
            args.flux_spaces,
            label="flux_spaces",
            choices=FLUX_SPACES,
        ),
        domain_length=args.domain_length,
        assembly_backend=args.assembly_backend,
        reconstruction_backend=args.reconstruction_backend,
        postprocessing_backend=args.postprocessing_backend,
        solver=args.solver,
        solver_rtol=args.solver_rtol,
        sample_resolution=args.sample_resolution,
        error_volume_quad_1d=args.error_volume_quad_1d,
        condition_max_dofs=args.condition_max_dofs,
        output_dir=args.output_dir,
        prefix=args.prefix,
        verbosity=args.verbosity,
        solver_verbosity=args.solver_verbosity,
        plot=args.plot,
        plot_output=args.plot_output,
        show_plot=args.show_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
