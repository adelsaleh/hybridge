#!/usr/bin/env python3
"""Temporal convergence study for manufactured guiding-center solutions."""

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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.guiding_center.guiding_center_presets import preset_by_key
from scripts.guiding_center.run_guiding_center_cases import run_guiding_center_case

SCHEMES = ("si-euler", "predictor-corrector")
ERROR_KEYS = ("rho_l2_error", "rho_linf_error", "phi_l2_error", "phi_linf_error")


def _parse_dts(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("dts must be a comma-separated list of positive values")
    if len(set(values)) != len(values):
        raise ValueError("dts must not contain duplicates")
    return tuple(sorted(values, reverse=True))


def _selected_schemes(selection: str) -> tuple[str, ...]:
    return SCHEMES if selection == "both" else (selection,)


def _pairwise_rates(rows: list[dict[str, Any]]) -> None:
    by_scheme: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scheme.setdefault(str(row["scheme"]), []).append(row)
    for scheme_rows in by_scheme.values():
        scheme_rows.sort(key=lambda row: float(row["dt"]), reverse=True)
        for index, row in enumerate(scheme_rows):
            for error_key in ERROR_KEYS:
                rate_key = error_key.replace("_error", "_rate")
                if index == 0:
                    row[rate_key] = None
                    continue
                coarse = scheme_rows[index - 1]
                coarse_error = float(coarse[error_key])
                fine_error = float(row[error_key])
                if coarse_error <= 0.0 or fine_error <= 0.0:
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


def plot_convergence(rows: list[dict[str, Any]], output: Path, *, show: bool = False) -> Path:
    """Write four mpi2.log-mpi2.log error panels; matplotlib is imported only on request."""
    import matplotlib.pyplot as plt

    labels = {
        "rho_l2_error": r"$\rho$ $L^2$ error",
        "rho_linf_error": r"$\rho$ $L^\infty$ error",
        "phi_l2_error": r"$\phi$ $L^2$ error",
        "phi_linf_error": r"$\phi$ $L^\infty$ error",
    }
    styles = {"si-euler": "o-", "predictor-corrector": "s-"}
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, error_key in zip(axes.flat, ERROR_KEYS):
        all_dts = []
        reference_scale = None
        for scheme in SCHEMES:
            selected = sorted(
                (row for row in rows if row["scheme"] == scheme),
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
    if scheme not in {*SCHEMES, "both"}:
        raise ValueError("scheme must be 'si-euler', 'predictor-corrector', or 'both'")
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
                    **{key: float(final[key]) for key in ERROR_KEYS},
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
    parser = ArgumentParser(description="Run rho_helm_wave temporal convergence for one or both guiding-center schemes.")
    parser.add_argument("--scheme", choices=("si-euler", "predictor-corrector", "both"), default="both")
    parser.add_argument("--final-time", type=float, default=0.2)
    parser.add_argument("--dts", default="0.04,0.02,0.01,0.005")
    parser.add_argument("--mesh-size", type=float, default=0.025)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/guiding_center/convergence"))
    parser.add_argument("--prefix", default="rho_helm_wave_temporal_convergence")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--plot-convergence", action="store_true")
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()
    run_temporal_convergence(
        scheme=args.scheme,
        final_time=args.final_time,
        dts=_parse_dts(args.dts),
        mesh_size=args.mesh_size,
        order=args.order,
        output_dir=args.output_dir,
        prefix=args.prefix,
        verbosity=args.verbosity,
        plot=args.plot_convergence,
        plot_output=args.plot_output,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    _main()
