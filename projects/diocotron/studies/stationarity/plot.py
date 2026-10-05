#!/usr/bin/env python3
"""Plot MPI scaling and stationarity diagnostics from guiding-center runs."""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.diocotron.dolfinx.runtime.mpi_rank_policy import (  # noqa: E402
    smaller_rank_within_ten_percent,
)
from projects.diocotron.paths import rebase_archive_paths, resolve_archive_path


def _summary(run_dir: Path) -> dict:
    return rebase_archive_paths(json.loads((resolve_archive_path(run_dir) / "summary.json").read_text(encoding="utf-8")))


def _parameters(run_dir: Path) -> dict:
    return json.loads((resolve_archive_path(run_dir) / "parameters.json").read_text(encoding="utf-8"))


def _diagnostics(run_dir: Path) -> np.ndarray:
    data = np.genfromtxt(
        resolve_archive_path(run_dir) / "diagnostics.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    return np.atleast_1d(data)


def _checkpoint_dofs(summary: dict) -> int:
    with np.load(resolve_archive_path(summary["equilibrium"]), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    return int(metadata["num_dofs"])


def _rank_timings(root: Path) -> tuple[dict[int, list[float]], dict[int, float], int]:
    samples: dict[int, list[float]] = {}
    for path in sorted(root.glob("ranks*_rep*/summary.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        rank = int(document["ranks"])
        samples.setdefault(rank, []).append(float(document["median_step_seconds"]))
    if not samples:
        raise RuntimeError(f"no rank timing summaries found under {root}")
    medians = {rank: float(np.median(values)) for rank, values in samples.items()}
    selected = smaller_rank_within_ten_percent(medians)
    return samples, medians, selected


def _run_record(run_dir: Path) -> dict:
    summary = _summary(run_dir)
    params = _parameters(run_dir)
    data = _diagnostics(run_dir)
    ndof = _checkpoint_dofs(summary)
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "params": params,
        "data": data,
        "ndof": ndof,
        "dt": float(params["dt"]),
        "label": f"{ndof / 1000:.0f}k DOFs, dt={float(params['dt']):g}",
    }


def render(
    benchmark_root: Path,
    run_dirs: list[Path],
    outputs: list[Path],
    assessment_output: Path | None,
) -> None:
    samples, medians, selected_rank = _rank_timings(benchmark_root)
    records = sorted(
        (_run_record(path.expanduser().resolve()) for path in run_dirs),
        key=lambda item: (item["ndof"], -item["dt"]),
    )
    dof_levels = sorted({record["ndof"] for record in records})
    palette = plt.get_cmap("tab10")
    colors = {ndof: palette(index) for index, ndof in enumerate(dof_levels)}
    dt_levels = sorted({record["dt"] for record in records}, reverse=True)
    linestyles = {
        dt: ("-" if index == 0 else "--" if index == 1 else ":")
        for index, dt in enumerate(dt_levels)
    }

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    ranks = sorted(medians)
    center = np.asarray([medians[rank] for rank in ranks])
    lower = np.asarray([
        medians[rank] - np.percentile(samples[rank], 25) for rank in ranks
    ])
    upper = np.asarray([
        np.percentile(samples[rank], 75) - medians[rank] for rank in ranks
    ])
    axes[0, 0].errorbar(
        ranks, center, yerr=np.vstack((lower, upper)), marker="o",
        capsize=4, color="#1769aa", linewidth=1.6,
    )
    fastest = min(center)
    axes[0, 0].axhline(
        1.10 * fastest, color="#777777", linestyle=":", linewidth=1.0,
        label="10% above fastest",
    )
    axes[0, 0].scatter(
        [selected_rank], [medians[selected_rank]], marker="*", s=190,
        color="#e68613", edgecolor="black", linewidth=0.5,
        zorder=5, label=f"selected: {selected_rank} ranks",
    )
    axes[0, 0].set(
        xlabel="MPI ranks",
        ylabel="median step time [s]",
        title="Measured transient strong scaling",
        xticks=ranks,
    )
    axes[0, 0].legend(fontsize=8)

    def plot_metric(
        axis,
        field: str,
        title: str,
        ylabel: str,
        *,
        absolute: bool = False,
        logarithmic: bool = True,
    ) -> None:
        for record in records:
            values = np.asarray(record["data"][field], dtype=float)
            if absolute:
                values = np.abs(values)
            if logarithmic:
                values = np.maximum(values, 1.0e-18)
            axis.plot(
                record["data"]["time"],
                values,
                color=colors[record["ndof"]],
                linestyle=linestyles[record["dt"]],
                linewidth=1.6,
                label=record["label"],
            )
        if logarithmic:
            axis.set_yscale("log")
        axis.set(xlabel="time", ylabel=ylabel, title=title)
        axis.grid(alpha=0.22)

    plot_metric(
        axes[0, 1], "rho_change_l2_rel",
        "Density preservation", r"$\|\rho(t)-\rho(0)\|_{L^2}/\|\rho(0)\|_{L^2}$",
    )
    axes[0, 1].legend(fontsize=7)
    plot_metric(
        axes[0, 2], "rho_change_linf",
        r"Pointwise density change", r"$\|\rho(t)-\rho(0)\|_{L^\infty}$",
    )

    for record in records:
        time_values = record["data"]["time"]
        color = colors[record["ndof"]]
        style = linestyles[record["dt"]]
        axes[1, 0].plot(
            time_values,
            np.maximum(np.abs(record["data"]["mass_rel_drift"]), 1.0e-18),
            color=color, linestyle=style, linewidth=1.5,
            label=f"mass — {record['label']}",
        )
        axes[1, 0].plot(
            time_values,
            np.maximum(np.abs(record["data"]["energy_rel_drift"]), 1.0e-18),
            color=color, linestyle=style, linewidth=1.1, alpha=0.62,
            label=f"energy — {record['label']}",
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        xlabel="time", ylabel="absolute relative drift",
        title="Mass and field-energy conservation",
    )
    axes[1, 0].grid(alpha=0.22)
    axes[1, 0].legend(fontsize=6, ncol=2)

    plot_metric(
        axes[1, 1], "advective_defect_rel",
        "Discrete stationarity defect", r"$\|\mathbf{v}\cdot\nabla\rho\|_{L^2}/\|\rho\|_{L^2}$",
    )

    for record in records:
        if record["ndof"] != max(dof_levels):
            continue
        time_values = record["data"]["time"]
        style = linestyles[record["dt"]]
        axes[1, 2].plot(
            time_values, record["data"]["rho_min"],
            color="#1769aa", linestyle=style, linewidth=1.5,
            label=f"min, dt={record['dt']:g}",
        )
        axes[1, 2].plot(
            time_values, record["data"]["rho_max"],
            color="#e68613", linestyle=style, linewidth=1.5,
            label=f"max, dt={record['dt']:g}",
        )
    axes[1, 2].set(
        xlabel="time", ylabel="density extrema",
        title=f"High-resolution extrema ({max(dof_levels):,} DOFs)",
    )
    axes[1, 2].grid(alpha=0.22)
    axes[1, 2].legend(fontsize=7)

    fig.suptitle(
        "Horseshoe equilibrium preservation: P4, SUPG scale 0.1, no CIP",
        fontsize=14,
    )
    for output in outputs:
        output = output.expanduser().resolve()
        if output.suffix.lower() != ".png":
            raise ValueError(f"output must be PNG: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, facecolor="white")
        print(output)
    plt.close(fig)

    assessment = {
        "selected_mpi_ranks": selected_rank,
        "rank_median_step_seconds": {str(key): value for key, value in medians.items()},
        "rank_samples": {str(key): value for key, value in samples.items()},
        "runs": [],
    }
    for record in records:
        final = record["summary"]["final_diagnostics"]
        assessment["runs"].append({
            "run_dir": record["run_dir"],
            "num_dofs": record["ndof"],
            "dt": record["dt"],
            "steps": int(record["params"]["num_steps"]),
            "final_time": float(final["time"]),
            "rho_change_l2_rel": float(final["rho_change_l2_rel"]),
            "rho_change_linf": float(final["rho_change_linf"]),
            "mass_rel_drift": float(final["mass_rel_drift"]),
            "energy_rel_drift": float(final["energy_rel_drift"]),
            "advective_defect_rel": float(final["advective_defect_rel"]),
            "rho_min": float(final["rho_min"]),
            "rho_max": float(final["rho_max"]),
            "handoff_relative_l2": float(
                record["summary"]["equilibrium_handoff"]["relative_l2"]
            ),
            "handoff_relative_h1": float(
                record["summary"]["equilibrium_handoff"]["relative_h1"]
            ),
            "poisson_consistency_relative_l2": float(
                record["summary"]["equilibrium_poisson_consistency"]["relative_l2"]
            ),
            "poisson_consistency_relative_h1": float(
                record["summary"]["equilibrium_poisson_consistency"]["relative_h1"]
            ),
        })
    if assessment_output is not None:
        assessment_output = assessment_output.expanduser().resolve()
        assessment_output.parent.mkdir(parents=True, exist_ok=True)
        assessment_output.write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(assessment_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument("--assessment-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    render(args.benchmark_root, args.run, args.output, args.assessment_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
