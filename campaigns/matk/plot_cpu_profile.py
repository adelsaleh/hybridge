#!/usr/bin/env python3
"""Render a completed PARDISO CPU profile as a standalone PNG."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    profile = report["cpu_profile"]
    with open(profile["files"]["samples_csv"]) as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        p.error("No CPU samples in this profile")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [float(row["elapsed_seconds"]) for row in rows]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(13, 7), layout="constrained")
    axes[0].plot(x, [float(r["cpu_equivalent_cores"]) for r in rows], color="#1261a0", lw=1.3,
                 label="Scheduled CPU-seconds / wall-second")
    axes[0].axhline(profile["capacity_logical_cpus"], color="#777777", ls="--", lw=1,
                    label=f"Allocated CPUs ({profile['capacity_logical_cpus']})")
    axes[0].set_ylabel("CPU equivalents")
    axes[1].plot(x, [int(r["active_threads"]) for r in rows], color="#16804a", label="Threads consuming CPU ticks")
    axes[1].plot(x, [int(r["runnable_threads"]) for r in rows], color="#bc6d08", alpha=0.7,
                 label="Runnable threads (running or ready)")
    axes[1].set_ylabel("Thread count")
    axes[1].set_xlabel("Seconds since CPU sampler started")
    colors = {"loading": "#e1e7ed", "analysis_factorization": "#dceada",
              "solve_1": "#f7e5c7", "solve_2": "#eadff2", "solve_3": "#d9ecf0"}
    for name, color in colors.items():
        selected = [i for i, row in enumerate(rows) if row["phase"].split(".")[0] == name]
        if selected:
            start, end = x[selected[0]], x[selected[-1]]
            for ax in axes:
                ax.axvspan(start, end, color=color, alpha=0.5, zorder=0)
            axes[0].text((start + end) / 2, 0.98, name.replace("_", " "),
                         transform=axes[0].get_xaxis_transform(), rotation=90,
                         va="top", ha="center", fontsize=8)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"PARDISO CPU usage — job {report['machine'].get('slurm_job_id')}, "
                 f"{report['machine']['host']}, sampling {profile['interval_seconds']} s\n"
                 "CPU time includes spin waits; thread activity is sampled, not hardware occupancy", fontsize=12)
    output = args.output or args.report.with_suffix(".cpu.png")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
