#!/usr/bin/env python3
"""Postprocess Strategy A scheduled-adaptation Euclidean-residual logs.

Reads:
  ../logs/*_newton.csv
  ../logs/*_adapt.csv

Displays or saves:
  - Newton Euclidean weak residual histories
  - Newton timing
  - mass/maxRho/relRhoDesign histories
  - scheduled adaptation mesh sizes
  - adaptation preservation summaries: mass/max-rho/active-area changes

Usage:
  python postprocess_strategyA_scheduled_adapt_fastres.py --show
  python postprocess_strategyA_scheduled_adapt_fastres.py --save --figdir ../out/figures
  python postprocess_strategyA_scheduled_adapt_fastres.py --runTag strategyA_scheduled_adapt_euclid_fastres
"""
from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=Path, default=Path("../logs"))
    p.add_argument("--out", type=Path, default=Path("../out"))
    p.add_argument("--runTag", default="*")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--figdir", type=Path, default=None)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def load_csvs(logs: Path, run_tag: str, suffix: str) -> pd.DataFrame:
    pattern = f"*{suffix}.csv" if run_tag == "*" else f"{run_tag}{suffix}.csv"
    files = sorted(logs.glob(pattern))
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
            df["source"] = path.name
            frames.append(df)
        except Exception as exc:
            print(f"[warn] failed reading {path}: {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        if col not in {"record", "runTag", "status", "source"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_show(fig: plt.Figure, path: Path | None, show: bool, dpi: int) -> None:
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {path}")
    if show:
        plt.show()
    plt.close(fig)


def final_rows(df: pd.DataFrame, record: str) -> pd.DataFrame:
    rows = df[df["record"].eq(record)].copy()
    if rows.empty:
        return rows
    keys = [c for c in ["source", "ieps"] if c in rows]
    if keys:
        return rows.groupby(keys, dropna=False).tail(1)
    return rows.tail(1)


def plot_residuals(newton: pd.DataFrame, figdir: Path | None, show: bool, dpi: int) -> None:
    rows = newton[newton["record"].eq("NEWTON")].copy()
    if rows.empty or "resEuclid" not in rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (src, ieps), g in rows.groupby(["source", "ieps"], dropna=False):
        g = g.sort_values("k")
        tag = f"{Path(str(src)).stem} eps={int(ieps) if pd.notna(ieps) else ieps}"
        ax.semilogy(g["k"], g["resEuclid"].clip(lower=1e-300), marker="o", markersize=3, linewidth=1.2, label=tag)
    ax.set_xlabel("Newton iteration k")
    ax.set_ylabel("Euclidean weak residual resE")
    ax.set_title("Newton residual histories")
    ax.grid(True, which="both", linewidth=0.4)
    ax.legend(fontsize=7)
    save_show(fig, None if figdir is None else figdir / "newton_residuals_resE.png", show, dpi)


def plot_timing(newton: pd.DataFrame, figdir: Path | None, show: bool, dpi: int) -> None:
    rows = newton[newton["record"].eq("NEWTON")].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (src, ieps), g in rows.groupby(["source", "ieps"], dropna=False):
        g = g.sort_values("k")
        tag = f"{Path(str(src)).stem} eps={int(ieps) if pd.notna(ieps) else ieps}"
        for col, style in [("solveTime", "-"), ("metricTime", "--"), ("stepTime", ":")]:
            if col in g:
                ax.plot(g["k"], g[col], marker="o", markersize=3, linewidth=1.2, linestyle=style, label=f"{tag} {col}")
    ax.set_xlabel("Newton iteration k")
    ax.set_ylabel("seconds")
    ax.set_title("Per-step timing")
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=7)
    save_show(fig, None if figdir is None else figdir / "newton_timing.png", show, dpi)


def plot_state(newton: pd.DataFrame, figdir: Path | None, show: bool, dpi: int) -> None:
    rows = newton[newton["record"].eq("NEWTON")].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (src, ieps), g in rows.groupby(["source", "ieps"], dropna=False):
        g = g.sort_values("k")
        tag = f"{Path(str(src)).stem} eps={int(ieps) if pd.notna(ieps) else ieps}"
        for col in ["massRho", "maxRho", "relRhoDesign", "activeArea", "plateauArea"]:
            if col in g:
                ax.plot(g["k"], g[col], marker="o", markersize=3, linewidth=1.2, label=f"{tag} {col}")
    ax.set_xlabel("Newton iteration k")
    ax.set_ylabel("value")
    ax.set_title("State diagnostics")
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=7)
    save_show(fig, None if figdir is None else figdir / "newton_state_metrics.png", show, dpi)


def plot_adapt_size(adapt: pd.DataFrame, figdir: Path | None, show: bool, dpi: int) -> None:
    if adapt.empty:
        return
    rows = adapt[adapt["record"].isin(["PREADAPT", "ADAPT"])].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for src, g in rows.groupby("source", dropna=False):
        g = g.sort_values(["ieps", "k"])
        x = range(len(g))
        tag = Path(str(src)).stem
        ax.plot(x, g["ntOld"], marker="o", label=f"{tag} ntOld")
        ax.plot(x, g["ntNew"], marker="o", label=f"{tag} ntNew")
    ax.set_xlabel("adaptation event index")
    ax.set_ylabel("triangles")
    ax.set_title("Mesh size at pre-adapt / scheduled readapt")
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / "adapt_mesh_size.png", show, dpi)


def plot_adapt_changes(adapt: pd.DataFrame, figdir: Path | None, show: bool, dpi: int) -> None:
    if adapt.empty:
        return
    rows = adapt[adapt["record"].isin(["PREADAPT", "ADAPT"])].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for src, g in rows.groupby("source", dropna=False):
        g = g.sort_values(["ieps", "k"])
        x = range(len(g))
        tag = Path(str(src)).stem
        for col in ["massRelChange", "maxRhoRelChange", "activeAreaRelChange"]:
            if col in g:
                ax.semilogy(list(x), g[col].clip(lower=1e-16), marker="o", linewidth=1.2, label=f"{tag} {col}")
    ax.set_xlabel("adaptation event index")
    ax.set_ylabel("relative change")
    ax.set_title("Field changes caused by adaptation")
    ax.grid(True, which="both", linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / "adapt_relative_changes.png", show, dpi)


def plot_adapt_parameters(adapt: pd.DataFrame, figdir: Path | None, show: bool, dpi: int) -> None:
    if adapt.empty:
        return
    rows = adapt[adapt["record"].isin(["PREADAPT", "ADAPT"])].copy()
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for src, g in rows.groupby("source", dropna=False):
        g = g.sort_values(["ieps", "k"])
        x = range(len(g))
        tag = Path(str(src)).stem
        for col in ["hMinUsed", "hMax", "adaptGradWeight"]:
            if col in g:
                ax.plot(list(x), g[col], marker="o", linewidth=1.2, label=f"{tag} {col}")
    ax.set_xlabel("adaptation event index")
    ax.set_ylabel("value")
    ax.set_title("Adaptation parameters actually used")
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_show(fig, None if figdir is None else figdir / "adapt_parameters_used.png", show, dpi)


def print_summaries(newton: pd.DataFrame, adapt: pd.DataFrame) -> None:
    newton_rows = newton[newton["record"].eq("NEWTON")].copy()
    print("\nNewton log rows:", len(newton_rows))
    cols = ["source", "ieps", "k", "resEuclid", "massRho", "maxRho", "relRhoDesign", "solveTime", "metricTime", "stepTime", "status"]
    cols = [c for c in cols if c in newton_rows]
    if cols and not newton_rows.empty:
        print("\nFinal Newton row per eps stage:")
        print(final_rows(newton, "NEWTON")[cols].to_string(index=False))
    if not adapt.empty:
        rows = adapt[adapt["record"].isin(["PREADAPT", "ADAPT"])].copy()
        cols = ["source", "record", "ieps", "k", "ntOld", "ntNew", "hMinUsed", "hMax", "adaptGradWeight", "massRelChange", "activeAreaRelChange", "adaptTime", "status"]
        cols = [c for c in cols if c in rows]
        if cols and not rows.empty:
            print("\nAdaptation summary:")
            print(rows[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    show = args.show or not args.save
    figdir = (args.figdir or (args.out / "figures")) if args.save else None

    newton = load_csvs(args.logs, args.runTag, "_newton")
    adapt = load_csvs(args.logs, args.runTag, "_adapt")
    if newton.empty:
        raise SystemExit(f"No Newton CSV logs found in {args.logs} for runTag={args.runTag}")

    print_summaries(newton, adapt)

    if args.save:
        args.out.mkdir(parents=True, exist_ok=True)
        newton.to_csv(args.out / "parsed_strategyA_scheduled_adapt_newton.csv", index=False)
        if not adapt.empty:
            adapt.to_csv(args.out / "parsed_strategyA_scheduled_adapt_adapt.csv", index=False)

    plot_residuals(newton, figdir, show, args.dpi)
    plot_timing(newton, figdir, show, args.dpi)
    plot_state(newton, figdir, show, args.dpi)
    plot_adapt_size(adapt, figdir, show, args.dpi)
    plot_adapt_changes(adapt, figdir, show, args.dpi)
    plot_adapt_parameters(adapt, figdir, show, args.dpi)


if __name__ == "__main__":
    main()
