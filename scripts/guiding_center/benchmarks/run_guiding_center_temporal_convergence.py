#!/usr/bin/env python3
"""Manufactured temporal convergence and fixed-space vortex-gas sensitivity."""

from __future__ import annotations

import csv
import json
import math
import sys
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hdgfem.diagnostics import evaluate_hdg_scalar_error
from hdgfem.core.field_ops import expand_interior_trace
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.runner import run_guiding_center_case

SCHEMES = ("si-euler", "predictor-corrector", "si-bdf2", "h1-bdf3", "h2-bdf3", "imex-ark3")
BASE_ERROR_KEYS = ("rho_l2_error", "rho_linf_error", "phi_l2_error", "phi_linf_error")
HDG_ERROR_KEYS = tuple(f"{field}_{norm}_error" for field in ("rho", "phi")
                       for norm in ("gradient_l2", "trace_mismatch", "hdg_h1"))
ERROR_KEYS = BASE_ERROR_KEYS + HDG_ERROR_KEYS


def _parse_dts(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("dts must be a comma-separated list of finite positive values")
    if len(set(values)) != len(values):
        raise ValueError("dts must not contain duplicates")
    return tuple(sorted(values, reverse=True))


def _selected_schemes(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return SCHEMES
    # Preserve the original two-scheme comparison for existing commands.
    return SCHEMES[:2] if selection == "both" else (selection,)


def _pairwise_rates(rows: list[dict[str, Any]]) -> None:
    by_scheme: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scheme.setdefault(str(row["scheme"]), []).append(row)
    for scheme_rows in by_scheme.values():
        scheme_rows.sort(key=lambda row: float(row["dt"]), reverse=True)
        for index, row in enumerate(scheme_rows):
            for error_key in ERROR_KEYS:
                rate_key = error_key.replace("_error", "_rate")
                if index == 0 or row.get(error_key) is None or scheme_rows[index - 1].get(error_key) is None:
                    row[rate_key] = None
                    continue
                coarse = scheme_rows[index - 1]
                coarse_error = float(coarse[error_key])
                fine_error = float(row[error_key])
                if (coarse_error <= 0.0 or fine_error <= 0.0
                        or not math.isfinite(coarse_error) or not math.isfinite(fine_error)):
                    row[rate_key] = None
                else:
                    row[rate_key] = math.log(coarse_error / fine_error) / math.log(
                        float(coarse["dt"]) / float(row["dt"])
                    )


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\nGuiding-center temporal convergence")
    print("scheme                 dt        steps   rho L2       rate    phi L2       rate")
    print("-" * 86)
    for row in rows:
        rho_rate = row.get("rho_l2_rate")
        phi_rate = row.get("phi_l2_rate")
        print(
            f"{row['scheme']:<22} {row['dt']:<9.3g} {row['num_steps']:<7d} "
            f"{row['rho_l2_error']:<12.4e} "
            f"{'-' if rho_rate is None else f'{rho_rate:.3f}':<7} "
            f"{row['phi_l2_error']:<12.4e} "
            f"{'-' if phi_rate is None else f'{phi_rate:.3f}':<7}"
        )

    if all("rho_hdg_h1_error" in row for row in rows):
        print("\nHDG H1 errors (L2 + physical gradient + 1/h_K face mismatch, in quadrature)")
        print("scheme                 dt        rho HDG H1   rate    phi HDG H1   rate")
        for row in rows:
            rates = [row.get(f"{field}_hdg_h1_rate") for field in ("rho", "phi")]
            rate_text = ["-" if value is None else f"{value:.3f}" for value in rates]
            print(f"{row['scheme']:<22} {row['dt']:<9.3g} "
                  f"{row['rho_hdg_h1_error']:<12.4e} {rate_text[0]:<7} "
                  f"{row['phi_hdg_h1_error']:<12.4e} {rate_text[1]:<7}")


def plot_convergence(rows: list[dict[str, Any]], output: Path, *, show: bool = False) -> Path:
    """Plot available L2/Linf and HDG error components, including older results."""
    import matplotlib.pyplot as plt

    norm_labels = {
        "l2": r"$L^2$ error", "linf": r"$L^\infty$ error",
        "gradient_l2": r"$\|\nabla_h(u_h-u)\|$",
        "trace_mismatch": r"$\sqrt{J_h}$ (error face term)",
        "hdg_h1": r"HDG $H^1$ error",
    }
    labels = {f"{field}_{norm}_error": f"${symbol}$: {label}"
              for norm, label in norm_labels.items() for field, symbol in (("rho", r"\rho"), ("phi", r"\phi"))}
    keys = [key for key in labels if any(row.get(key) is not None for row in rows)]
    if not keys:
        raise ValueError("no error metrics to plot")
    styles = {"si-euler": "o-", "predictor-corrector": "s-", "si-bdf2": "^-", "h1-bdf3": "d-", "h2-bdf3": "v-", "imex-ark3": "x-"}
    nrows = (len(keys) + 1) // 2
    figure, axes = plt.subplots(nrows, 2, figsize=(11, 3.5*nrows), squeeze=False, constrained_layout=True)
    for axis in axes.flat[len(keys):]:
        axis.set_visible(False)
    for axis, error_key in zip(axes.flat, keys):
        all_dts = []
        reference_scale = None
        for scheme in SCHEMES:
            selected = sorted(
                (row for row in rows if row["scheme"] == scheme and row.get(error_key) is not None
                 and math.isfinite(float(row[error_key])) and float(row[error_key]) > 0),
                key=lambda row: float(row["dt"]),
                reverse=True,
            )
            if not selected:
                continue
            dts = np.asarray([row["dt"] for row in selected], dtype=np.float64)
            errors = np.asarray([row[error_key] for row in selected], dtype=np.float64)
            axis.loglog(dts, errors, styles[scheme], label=scheme)
            all_dts.extend(dts.tolist())
            if reference_scale is None:
                reference_scale = float(errors[0])
        if all_dts and reference_scale is not None:
            refs = np.asarray(sorted(set(all_dts), reverse=True), dtype=np.float64)
            coarse_dt = float(refs[0])
            axis.loglog(refs, reference_scale * refs / coarse_dt, "k--", alpha=0.55, label=r"$O(\Delta t)$")
            axis.loglog(refs, reference_scale * (refs / coarse_dt) ** 2, "k:", alpha=0.65, label=r"$O(\Delta t^2)$")
            if any(row["scheme"] in {"h1-bdf3", "h2-bdf3", "imex-ark3"} for row in rows):
                axis.loglog(refs, reference_scale * (refs / coarse_dt) ** 3, "k-.", alpha=0.65, label=r"$O(\Delta t^3)$")
        axis.set_xlabel(r"$\Delta t$")
        axis.set_ylabel(labels[error_key])
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    return output


def _manufactured_hdg_errors(result) -> dict[str, float | str]:
    """Compare accepted fields/traces with the analytic solution at final time."""
    config, space = result.config, result.space
    case = case_definition_by_key(config.case).build(**config.case_params)
    time_value = float(result.diagnostics[-1]["time"])
    errors = {"hdg_jump_weight": "1/h_K; h_K=element diameter",
              "hdg_trace_faces": "all element sides, including prescribed boundaries"}
    for field_name, field, reduced, basis, exact, exact_gradient, boundary in (
        ("rho", result.final_density, result.final_density_trace_reduced,
         config.transport_trace_basis or config.trace_basis, case.exact_density_at(time_value),
         case.exact_density_gradient_at(time_value), case.density_boundary_at(time_value)),
        ("phi", result.final_potential, result.final_potential_trace_reduced,
         config.poisson_trace_basis or config.trace_basis, case.exact_potential_at(time_value),
         case.exact_potential_gradient_at(time_value), case.potential_boundary_at(time_value)),
    ):
        if exact is None or exact_gradient is None or boundary is None or reduced is None:
            raise ValueError("manufactured HDG error requires analytic gradients and accepted numerical traces")
        # Expand only prescribed boundaries, keeping accepted interior traces
        # (including the PC endpoint extrapolation) on their resident backend.
        full = expand_interior_trace(space, reduced, boundary, trace_basis=basis)
        metrics = evaluate_hdg_scalar_error(field, full.ravel(), exact, exact_gradient, trace_basis=basis)
        errors.update({f"{field_name}_gradient_l2_error": metrics.gradient_l2,
                       f"{field_name}_trace_mismatch_error": metrics.trace_mismatch,
                       f"{field_name}_hdg_h1_error": metrics.hdg_h1,
                       f"{field_name}_hdg_diagnostics_backend": metrics.backend})
    return errors


def run_temporal_convergence(
    *,
    scheme: str = "both",
    final_time: float = 0.2,
    dts: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005),
    mesh_size: float = 0.025,
    order: int = 6,
    output_dir: str | Path = "run_outputs/guiding_center/convergence",
    prefix: str = "rho_helm_wave_temporal_convergence",
    verbosity: int = 1,
    plot: bool = False,
    plot_output: str | Path | None = None,
    show_plot: bool = False,
) -> tuple[list[dict[str, Any]], Path, Path]:
    if scheme not in {*SCHEMES, "both", "all"}:
        raise ValueError("scheme must be 'si-euler', 'predictor-corrector', 'si-bdf2', 'h1-bdf3', 'h2-bdf3', 'imex-ark3', 'both', or 'all'")
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    output_path = Path(output_dir)
    rows: list[dict[str, Any]] = []
    base = preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy")
    for selected_scheme in _selected_schemes(scheme):
        for dt in sorted(dts, reverse=True):
            num_steps = int(round(final_time / dt))
            if num_steps < 1 or not math.isclose(num_steps * dt, final_time, rel_tol=1.0e-12, abs_tol=1.0e-14):
                raise ValueError(f"final_time={final_time} must be an integer multiple of dt={dt}")
            run_prefix = f"{prefix}_{selected_scheme.replace('-', '_')}_dt{dt:.8g}".replace(".", "p")
            config = replace(
                base,
                domain="rectangle",
                mesh_size=float(mesh_size),
                order=int(order),
                dt=float(dt),
                num_steps=num_steps,
                time_scheme=selected_scheme,
                poisson_assembly_backend="raw-cuda",
                # This option selects the CPU fallback only. Raw-CUDA uses
                # device assembly and reconstruction independently.
                poisson_local_backend="numpy",
                # The manufactured boundary is nonzero and time dependent;
                # the raw LU cache supports its eliminated trace contribution.
                poisson_cache_local_factors="schur-lu",
                poisson_solver="amgx",
                poisson_solver_rtol=1.0e-11,
                poisson_solver_atol=1.0e-12,
                poisson_raw_matrix_format="csr",
                transport_assembly_backend="raw-cuda",
                transport_solver="amgx",
                transport_solver_rtol=1.0e-11,
                transport_solver_atol=1.0e-12,
                transport_boundary_mode="eliminate",
                transport_raw_local_assembly="fused",
                transport_raw_lu_mode="coop",
                transport_raw_matrix_format="csr",
                transport_materialize_host_system=False,
                transport_materialize_host_solution=False,
                transport_initial_guess="initial-density-trace",
                transport_retry_policy="amgx-robust",
                plot_every=0,
                diagnostics_every=num_steps,
                diagnostics_dir=str(output_path / "runs"),
                diagnostics_prefix=run_prefix,
                verbosity=int(verbosity),
            )
            result = run_guiding_center_case(config, preset_key=run_prefix)
            final = result.diagnostics[-1]
            rows.append(
                {
                    "scheme": selected_scheme,
                    "dt": float(dt),
                    "final_time": float(final_time),
                    "num_steps": num_steps,
                    "mesh_size": float(mesh_size),
                    "order": int(order),
                    "triangles": int(result.mesh.num_tri),
                    **{key: float(final[key]) for key in BASE_ERROR_KEYS},
                    **_manufactured_hdg_errors(result),
                    "mass_relative_drift": float(final["mass_relative_drift"]),
                    "q_l2_relative_drift": float(final["q_l2_relative_drift"]),
                }
            )
    _pairwise_rates(rows)
    rows.sort(key=lambda row: (SCHEMES.index(row["scheme"]), -float(row["dt"])))
    csv_path, json_path = _write_outputs(rows, output_path, prefix)
    _print_table(rows)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if plot:
        figure_path = Path(plot_output) if plot_output is not None else output_path / f"{prefix}.png"
        print(f"plot: {plot_convergence(rows, figure_path, show=show_plot)}")
    return rows, csv_path, json_path


def _main() -> None:
    parser = ArgumentParser(description="Manufactured temporal convergence or matched-time vortex-gas comparison.")
    parser.add_argument("--study", choices=("manufactured", "vortex-gas"), default="manufactured")
    parser.add_argument("--scheme", choices=(*SCHEMES, "both", "all"), default=None)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--dts", default=None)
    parser.add_argument("--mesh-size", type=float, default=None)
    parser.add_argument("--order", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/guiding_center/convergence"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2, 3), default=None)
    parser.add_argument("--plot-convergence", action="store_true", help="Manufactured error panels.")
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--preset", default=None, help="Vortex study: inherit spatial settings, seed and tolerances.")
    parser.add_argument("--sample-interval", type=float, default=0.5, help="Vortex study: physical time between field samples.")
    parser.add_argument("--plot-comparison", action="store_true", help="Save vortex histories and matched field panels.")
    parser.add_argument("--resolution", type=int, default=1024, help="Vortex raster width/height; norms use DG quadrature.")
    parser.add_argument("--cached-kernels-only", action="store_true", help="Reject new Numba/NVRTC/NVCC compilation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print vortex run configurations without solving.")
    args = parser.parse_args()
    from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import (
        DEFAULT_VORTEX_PRESET, kernel_cache_only, prepare_vortex_comparison, run_vortex_comparison,
    )
    if args.study == "vortex-gas":
        if args.plot_convergence or args.plot_output or args.show_plot:
            parser.error("Use --plot-comparison for vortex-gas; two dts do not measure formal order")
        options = dict(
            preset=args.preset or DEFAULT_VORTEX_PRESET, scheme=args.scheme or "si-bdf2",
            final_time=5.0 if args.final_time is None else args.final_time,
            dts=_parse_dts(args.dts or "0.01,0.005"), sample_interval=args.sample_interval,
            mesh_size=args.mesh_size, order=args.order, output_dir=args.output_dir,
            prefix=args.prefix or "vortex_temporal_comparison",
            verbosity=0 if args.verbosity is None else args.verbosity,
        )
        if args.dry_run:
            from dataclasses import asdict
            print(json.dumps([asdict(c) for c in prepare_vortex_comparison(**options)], indent=2))
            return
        run_vortex_comparison(**options, resolution=args.resolution,
                              plot=args.plot_comparison, cached_kernels_only=args.cached_kernels_only)
        return
    if args.preset or args.plot_comparison or args.dry_run:
        parser.error("--preset, --plot-comparison and --dry-run require --study vortex-gas")
    with kernel_cache_only(args.cached_kernels_only):
        run_temporal_convergence(
            scheme=args.scheme or "both",
            final_time=0.2 if args.final_time is None else args.final_time,
            dts=_parse_dts(args.dts or "0.04,0.02,0.01,0.005"),
            mesh_size=0.025 if args.mesh_size is None else args.mesh_size,
            order=6 if args.order is None else args.order,
            output_dir=args.output_dir, prefix=args.prefix or "rho_helm_wave_temporal_convergence",
            verbosity=1 if args.verbosity is None else args.verbosity,
            plot=args.plot_convergence, plot_output=args.plot_output, show_plot=args.show_plot,
        )


if __name__ == "__main__":
    _main()
