"""Iterative-solver audit of guiding-center run records (JSONL); no time integration.

Per run and per time window: mean/max transport and Poisson iterations, the
worst final residual over its target (> 1 means a solve stopped short), AMGX
re-attempts, Poisson tau retries and residual restarts. When the records carry
the Poisson initial residual (``poisson_detail_solve_fb_hp_mg_initial_residual``),
also the decades the solve had to cover, log10(r0/target), and the mean
contraction per iteration (r_final/r0)^(1/iterations): together they separate a
harder start (worse initial guess) from slower convergence.

Transport (AMGX BiCGStab, ``RELATIVE_INI_CORE``) stops on a reduction relative to
its initial residual, so its iteration count already measures convergence.

    python -m scripts.guiding_center.diagnostics.audit_solver_convergence RUN.jsonl ... \\
        --windows 0:70,70:75,75:400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _values(rows, key, default=np.nan):
    return np.array([default if row.get(key) is None else float(row[key]) for row in rows], dtype=float)


def audit(path, windows=None) -> list[dict]:
    """One record per window (the whole run first) of a run's JSONL diagnostics."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("step", 0) > 0 and row.get("transport_step_iterations") is not None]
    t = _values(rows, "time")
    transport = _values(rows, "transport_step_iterations")
    poisson = _values(rows, "poisson_step_iterations")
    transport_ratio = _values(rows, "transport_solver_residual") / _values(rows, "transport_solver_residual_target")
    poisson_ratio = _values(rows, "poisson_physical_residual") / _values(rows, "poisson_physical_residual_target")
    attempts = _values(rows, "transport_amgx_attempt_count", 1.0)
    tau = _values(rows, "poisson_tau_retry_count", 0.0)
    restarts = _values(rows, "poisson_detail_solve_fb_hp_mg_residual_restarts", 0.0)
    initial = _values(rows, "poisson_detail_solve_fb_hp_mg_initial_residual")
    final = _values(rows, "poisson_detail_solve_fb_hp_mg_best_residual")
    target = _values(rows, "poisson_physical_residual_target")
    with np.errstate(divide="ignore", invalid="ignore"):
        contraction = (final / initial) ** (1.0 / poisson)
        decades = np.log10(initial / target)
    records = []
    for low, high in [(-np.inf, np.inf), *(windows or [])]:
        inside = (t > low) & (t <= high)
        if not inside.any():
            continue
        has_initial = np.isfinite(initial[inside]).any()
        records.append(dict(
            run=Path(path).stem, window=None if np.isinf(low) else [low, high], steps=int(inside.sum()),
            transport_iterations_mean=float(transport[inside].mean()),
            transport_iterations_max=int(transport[inside].max()),
            transport_residual_over_target_max=float(np.nanmax(transport_ratio[inside])),
            amgx_reattempts=int((attempts[inside] > 1).sum()),
            poisson_iterations_mean=float(poisson[inside].mean()), poisson_iterations_max=int(poisson[inside].max()),
            poisson_residual_over_target_max=float(np.nanmax(poisson_ratio[inside])),
            poisson_tau_retries=int(tau[inside].sum()), poisson_restarts=int(restarts[inside].sum()),
            poisson_decades_median=float(np.nanmedian(decades[inside])) if has_initial else None,
            poisson_contraction_median=float(np.nanmedian(contraction[inside])) if has_initial else None,
            poisson_contraction_max=float(np.nanmax(contraction[inside])) if has_initial else None,
        ))
    return records


def _format(record) -> str:
    window = "all" if record["window"] is None else f"({record['window'][0]:g},{record['window'][1]:g}]"
    text = (f"  {window:>12s} n={record['steps']:5d} | T it {record['transport_iterations_mean']:5.1f}/"
            f"{record['transport_iterations_max']:3d} res/tgt {record['transport_residual_over_target_max']:.1e} "
            f"re-att {record['amgx_reattempts']} | P it {record['poisson_iterations_mean']:5.1f}/"
            f"{record['poisson_iterations_max']:3d} res/tgt {record['poisson_residual_over_target_max']:.6f} "
            f"tau {record['poisson_tau_retries']} restarts {record['poisson_restarts']}")
    if record["poisson_contraction_median"] is not None:
        text += (f" | decades {record['poisson_decades_median']:4.2f} contraction "
                 f"{record['poisson_contraction_median']:.3f} (max {record['poisson_contraction_max']:.3f})")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--windows", default="", help="comma-separated low:high time windows")
    parser.add_argument("--json", type=Path, help="write the records as JSON")
    args = parser.parse_args()
    windows = [tuple(float(v) for v in item.split(":")) for item in args.windows.split(",") if item]
    records = []
    for path in args.runs:
        if path.name.endswith("_timings.jsonl"):
            continue
        run = audit(path, windows)
        records += run
        print(path.stem)
        for record in run:
            print(_format(record))
    if args.json is not None:
        args.json.write_text(json.dumps(records, indent=1) + "\n")


if __name__ == "__main__":
    main()
