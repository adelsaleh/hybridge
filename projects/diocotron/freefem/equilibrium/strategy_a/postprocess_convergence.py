#!/usr/bin/env python3
"""
Postprocess FreeFEM convergence logs and optional VTU files.

Expected project layout:

    ../logs/*.csv or ../logs/*.log
    ../vtk/*.vtk
    ../out/*.txt

The recommended path is to make FreeFEM emit CSV logs using the logging patch.
This script also has a fallback parser for older whitespace logs, but CSV is
more reliable.

Examples:

    python postprocess_convergence.py
    python postprocess_convergence.py --logs ../logs --vtk ../vtk --out ../out --show
    python postprocess_convergence.py --save --figdir ../out/figures
    python postprocess_convergence.py --no-vtk --show

By default, figures are displayed with matplotlib and not saved.
Use --save to write PNG figures and CSV summary tables.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

CSV_COLUMNS = [
    "record", "tag", "N", "nt", "ndof", "k",
    "Tmax", "c1T", "c2T", "epsT", "phiDesignMax", "c1Phi", "c2Phi", "epsPhi",
    "resEuclid", "resHm1", "relHm1E", "relHm1Rho", "poissonRel", "nd",
    "alpha", "bt", "muShift", "minU", "maxU", "maxRho", "massRho",
    "activeArea", "plateauArea", "plateauFrac",
    "relL2PhiRef", "relH1PhiRef", "relL2RhoRef", "relL1RhoRef", "status",
]

SOLVE_COLUMNS_OLD = [
    "record", "tag", "N", "nt", "ndof", "k",
    "resEuclid", "resHm1", "relHm1E", "relHm1Rho", "poissonRel", "nd",
    "alpha", "bt", "muShift", "minU", "maxU", "maxRho", "massRho",
    "activeArea", "plateauArea", "status",
]

CONV_COLUMNS_OLD = [
    "record", "N", "nt", "ndof",
    "relL2PhiRef", "relH1PhiRef", "relL2RhoRef", "relL1RhoRef",
    "massRho", "maxRho", "activeArea", "plateauArea", "plateauFrac",
    "resHm1", "poissonRel",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", type=Path, default=Path("../logs"), help="Directory containing FreeFEM logs.")
    p.add_argument("--vtk", type=Path, default=Path("../vtk"), help="Directory containing VTU files.")
    p.add_argument("--out", type=Path, default=Path("../out"), help="Directory containing text outputs and optional CSV outputs.")
    p.add_argument("--figdir", type=Path, default=None, help="Directory for saved figures. Defaults to OUT/figures.")
    p.add_argument("--save", action="store_true", help="Save figures and parsed tables.")
    p.add_argument("--show", action="store_true", help="Display figures. Default behavior if --save is not given.")
    p.add_argument("--no-vtk", action="store_true", help="Skip VTU reading even if files exist.")
    p.add_argument("--pattern", default="*", help="Filename stem/glob filter, e.g. 'star_noadapt*'.")
    p.add_argument("--dpi", type=int, default=180, help="DPI for saved figures.")
    return p.parse_args()


def as_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return x
    s = str(x).strip()
    if s in {"NA", "", "nan", "NaN"}:
        return np.nan
    try:
        if re.search(r"[.eE+-]", s):
            return float(s)
        return int(s)
    except Exception:
        return s


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in {"record", "tag", "status", "source"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_csv_logs(log_dir: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(log_dir.glob(f"{pattern}.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] cannot read CSV log {path}: {e}")
            continue
        df["source"] = path.name
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS + ["source"])
    df = pd.concat(rows, ignore_index=True)
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return coerce_numeric(df)


def reconstruct_wrapped_records(text: str) -> list[str]:
    starts = (
        "REF ", "INIT ", "SOLVE ", "CONV ", "LEVEL_SETUP ",
        "TIME_TOTAL ", "LOG_FILE ", "EXPORT ", "==========", "----------",
    )
    records: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            records.append(buf)
            buf = ""

    for raw in text.splitlines():
        if not raw.strip():
            flush()
            continue
        s = raw.strip()
        starts_new = s.startswith(starts) or s.startswith("  --") or s.startswith("times:")
        if starts_new:
            flush()
            buf = s
        else:
            if not buf:
                buf = s
            elif raw[:1].isspace():
                buf += " " + s
            else:
                buf += s
    flush()
    return records


def normalize_old_conv(parts: list[str]) -> list[str]:
    if len(parts) == len(CONV_COLUMNS_OLD):
        return parts
    if len(parts) == len(CONV_COLUMNS_OLD) - 1:
        if len(parts) > 11 and parts[11] == "00":
            return parts[:11] + ["0", "0"] + parts[12:]
        if len(parts) > 12 and parts[11] == "0" and re.match(r"^0[0-9.]+e[-+][0-9]+$", parts[12], flags=re.I):
            return parts[:12] + ["0", parts[12][1:], parts[13]]
    return parts


def read_old_text_logs(log_dir: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(list(log_dir.glob(f"{pattern}.log")) + list(log_dir.glob(f"{pattern}.txt"))):
        try:
            text = path.read_text(errors="ignore")
        except Exception as e:
            print(f"[warn] cannot read text log {path}: {e}")
            continue
        for rec in reconstruct_wrapped_records(text):
            parts = rec.split()
            if not parts:
                continue
            if parts[0] == "SOLVE" and len(parts) >= len(SOLVE_COLUMNS_OLD):
                d = dict(zip(SOLVE_COLUMNS_OLD, [as_number(x) for x in parts[:len(SOLVE_COLUMNS_OLD)]]))
                d["source"] = path.name
                rows.append(d)
            elif parts[0] == "CONV":
                parts = normalize_old_conv(parts)
                if len(parts) >= len(CONV_COLUMNS_OLD):
                    d = dict(zip(CONV_COLUMNS_OLD, [as_number(x) for x in parts[:len(CONV_COLUMNS_OLD)]]))
                    d["record"] = "CONV"
                    d["tag"] = f"N{d.get('N')}"
                    d["source"] = path.name
                    rows.append(d)
            elif parts[0] == "INIT" and len(parts) >= 3:
                d = {"record": "INIT", "tag": parts[1], "N": as_number(parts[2]), "source": path.name}
                for token in parts[3:]:
                    if "=" in token:
                        k, v = token.split("=", 1)
                        d[k] = as_number(v)
                rows.append(d)
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS + ["source"])
    df = pd.DataFrame(rows)
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return coerce_numeric(df)


def load_logs(log_dir: Path, pattern: str) -> pd.DataFrame:
    csv_df = read_csv_logs(log_dir, pattern)
    txt_df = read_old_text_logs(log_dir, pattern)
    if len(csv_df) and len(txt_df):
        csv_stems = {Path(s).stem for s in csv_df["source"].dropna().unique()}
        txt_df = txt_df[~txt_df["source"].map(lambda s: Path(str(s)).stem in csv_stems)]
        df = pd.concat([csv_df, txt_df], ignore_index=True)
    elif len(csv_df):
        df = csv_df
    else:
        df = txt_df
    return coerce_numeric(df)


def get_solve_conv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    solve = df[df["record"].eq("SOLVE")].copy()
    conv = df[df["record"].eq("CONV")].copy()
    init = df[df["record"].eq("INIT")].copy()
    if len(solve):
        solve = solve.sort_values(["source", "N", "k"])
    if len(conv):
        conv = conv.sort_values(["source", "N"])
        conv["hProxy"] = 1.0 / np.sqrt(conv["nt"].astype(float))
    return solve, conv, init


def empirical_slopes(conv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(conv) < 2:
        return pd.DataFrame(columns=["source", "metric", "slopeVsNdof", "orderVsHProxy"])
    metrics = ["relL2PhiRef", "relH1PhiRef", "relL2RhoRef", "relL1RhoRef", "poissonRel", "resHm1"]
    for source, g in conv.groupby("source", dropna=False):
        g = g.sort_values("N")
        if "hProxy" not in g:
            g["hProxy"] = 1.0 / np.sqrt(g["nt"].astype(float))
        for m in metrics:
            if m not in g:
                continue
            vals = g[m].astype(float)
            mask = vals > 0
            if mask.sum() < 2:
                continue
            slope_ndof = np.polyfit(np.log(g.loc[mask, "ndof"].astype(float)), np.log(vals[mask]), 1)[0]
            order_h = np.polyfit(np.log(g.loc[mask, "hProxy"].astype(float)), np.log(vals[mask]), 1)[0]
            rows.append({"source": source, "metric": m, "slopeVsNdof": slope_ndof, "orderVsHProxy": order_h})
    return pd.DataFrame(rows)


def save_or_show(fig, path: Optional[Path], show: bool, dpi: int):
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_residual_histories(solve: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int):
    if solve.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for (source, tag, N), g in solve.groupby(["source", "tag", "N"], dropna=False):
        g = g.sort_values("k")
        label = f"{tag} N={int(N)}"
        if len(solve["source"].unique()) > 1:
            label = f"{Path(str(source)).stem}: {label}"
        ax.semilogy(g["k"], g["resHm1"], marker="o", linewidth=1.2, markersize=3, label=label)
    ax.set_xlabel("Newton iteration k")
    ax.set_ylabel(r"$H^{-1}_h$ residual")
    ax.set_title(r"Exact unshifted residual history for $-\Delta u=f(u)$")
    ax.grid(True, which="both", linewidth=0.4)
    ax.legend(fontsize=7)
    save_or_show(fig, None if figdir is None else figdir / "residual_history.png", show, dpi)


def plot_errors(conv: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int):
    if conv.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    metric_labels = [
        ("relL2PhiRef", r"$\phi$: rel $L^2$"),
        ("relH1PhiRef", r"$\phi$: rel $H^1$"),
        ("relL2RhoRef", r"$\rho$: rel $L^2$"),
        ("relL1RhoRef", r"$\rho$: rel $L^1$"),
    ]
    for source, g in conv.groupby("source", dropna=False):
        g = g.sort_values("ndof")
        prefix = "" if len(conv["source"].unique()) == 1 else f"{Path(str(source)).stem}: "
        for metric, label in metric_labels:
            if metric in g and g[metric].notna().any():
                ax.loglog(g["ndof"], g[metric], marker="o", linewidth=1.35, markersize=4, label=prefix + label)
    ax.set_xlabel("P2 degrees of freedom")
    ax.set_ylabel("relative error vs reference")
    ax.set_title("Mesh convergence against reference")
    ax.grid(True, which="both", linewidth=0.4)
    ax.legend(fontsize=8)
    save_or_show(fig, None if figdir is None else figdir / "errors_vs_ndof.png", show, dpi)


def plot_residual_metrics(conv: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int):
    if conv.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for source, g in conv.groupby("source", dropna=False):
        g = g.sort_values("ndof")
        prefix = "" if len(conv["source"].unique()) == 1 else f"{Path(str(source)).stem}: "
        if "poissonRel" in g:
            ax.loglog(g["ndof"], g["poissonRel"], marker="o", linewidth=1.35, markersize=4, label=prefix + "poissonRel")
        if "resHm1" in g:
            ax.loglog(g["ndof"], g["resHm1"], marker="o", linewidth=1.35, markersize=4, label=prefix + r"$H^{-1}_h$")
    ax.set_xlabel("P2 degrees of freedom")
    ax.set_ylabel("final residual metric")
    ax.set_title("Final residual metrics by mesh")
    ax.grid(True, which="both", linewidth=0.4)
    ax.legend(fontsize=8)
    save_or_show(fig, None if figdir is None else figdir / "final_residuals_vs_ndof.png", show, dpi)


def plot_band_metrics(conv: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int):
    if conv.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for source, g in conv.groupby("source", dropna=False):
        g = g.sort_values("ndof")
        prefix = "" if len(conv["source"].unique()) == 1 else f"{Path(str(source)).stem}: "
        for metric in ["activeArea", "plateauArea", "plateauFrac"]:
            if metric in g and g[metric].notna().any():
                ax.plot(g["ndof"], g[metric], marker="o", linewidth=1.35, markersize=4, label=prefix + metric)
    ax.set_xlabel("P2 degrees of freedom")
    ax.set_ylabel("area / fraction")
    ax.set_title("Band geometry metrics")
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_or_show(fig, None if figdir is None else figdir / "band_metrics.png", show, dpi)


def plot_mass_peak(conv: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int):
    if conv.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for source, g in conv.groupby("source", dropna=False):
        g = g.sort_values("ndof")
        prefix = "" if len(conv["source"].unique()) == 1 else f"{Path(str(source)).stem}: "
        ax.plot(g["ndof"], g["massRho"], marker="o", linewidth=1.35, markersize=4, label=prefix + "massRho")
        ax.plot(g["ndof"], g["maxRho"], marker="o", linewidth=1.35, markersize=4, label=prefix + "maxRho")
    ax.set_xlabel("P2 degrees of freedom")
    ax.set_ylabel("value")
    ax.set_title("Mass and peak-density stability")
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_or_show(fig, None if figdir is None else figdir / "mass_peak_stability.png", show, dpi)


def plot_newton_effort(solve: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int):
    if solve.empty:
        return
    effort = solve.groupby(["source", "tag", "N"], as_index=False).agg(
        ndof=("ndof", "last"),
        iterations=("k", "max"),
        finalResHm1=("resHm1", "last"),
        finalPoissonRel=("poissonRel", "last"),
    )
    effort["iterations"] = effort["iterations"] + 1
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for source, g in effort.groupby("source", dropna=False):
        g = g.sort_values("ndof")
        label = "iterations" if len(effort["source"].unique()) == 1 else Path(str(source)).stem
        ax.plot(g["ndof"], g["iterations"], marker="o", linewidth=1.35, markersize=4, label=label)
    ax.set_xlabel("P2 degrees of freedom")
    ax.set_ylabel("Newton iterations")
    ax.set_title("Newton effort by mesh")
    ax.grid(True, linewidth=0.4)
    ax.legend(fontsize=8)
    save_or_show(fig, None if figdir is None else figdir / "newton_effort.png", show, dpi)


def try_import_pyvista():
    try:
        import pyvista as pv  # type: ignore
        return pv
    except Exception:
        return None


def vtk_to_tri_and_fields(path: Path):
    pv = try_import_pyvista()
    if pv is None:
        print("[warn] pyvista is not installed; skipping VTU field plots.")
        return None
    try:
        mesh = pv.read(path)
    except Exception as e:
        print(f"[warn] cannot read VTU {path}: {e}")
        return None
    points = np.asarray(mesh.points)
    if points.shape[1] < 2:
        return None
    cells = np.asarray(mesh.cells)
    triangles = []
    i = 0
    while i < len(cells):
        n = int(cells[i])
        ids = cells[i + 1 : i + 1 + n]
        if n == 3:
            triangles.append(ids)
        i += n + 1
    if not triangles:
        print(f"[warn] no triangular cells found in {path}")
        return None
    tri = Triangulation(points[:, 0], points[:, 1], np.asarray(triangles))
    fields = {}
    for name in mesh.point_data.keys():
        arr = np.asarray(mesh.point_data[name])
        if arr.ndim == 1 and len(arr) == len(points):
            fields[name] = arr
    return tri, fields


def plot_latest_vtk_fields(vtk_dir: Path, pattern: str, figdir: Optional[Path], show: bool, dpi: int):
    files = sorted(vtk_dir.glob(f"{pattern}.vtk"))
    if not files:
        return
    preferred = [p for p in files if re.search(r"(REF|final|N320|strategy|equilibrium)", p.name, re.I)]
    path = preferred[-1] if preferred else files[-1]
    loaded = vtk_to_tri_and_fields(path)
    if loaded is None:
        return
    tri, fields = loaded
    interesting = [n for n in ["phi", "rho", "phiFinal", "rhoFinal", "rhoDesign", "phiDesign", "defectFinal"] if n in fields]
    if not interesting:
        interesting = list(fields.keys())[:4]
    for name in interesting:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        tpc = ax.tripcolor(tri, fields[name], shading="gouraud")
        ax.triplot(tri, linewidth=0.18, alpha=0.25)
        ax.set_aspect("equal")
        ax.set_title(f"{path.name}: {name}")
        fig.colorbar(tpc, ax=ax)
        save_or_show(fig, None if figdir is None else figdir / f"field_{path.stem}_{name}.png", show, dpi)


def main() -> None:
    args = parse_args()
    show = args.show or not args.save
    figdir = args.figdir or (args.out / "figures")
    figdir = figdir if args.save else None
    df = load_logs(args.logs, args.pattern)
    if df.empty:
        raise SystemExit(f"No parseable logs found in {args.logs} with pattern {args.pattern!r}")
    solve, conv, init = get_solve_conv(df)
    slopes = empirical_slopes(conv)
    print("\nParsed rows")
    print(f"  all:   {len(df)}")
    print(f"  solve: {len(solve)}")
    print(f"  conv:  {len(conv)}")
    print(f"  init:  {len(init)}")
    if len(conv):
        summary_cols = [
            "source", "N", "nt", "ndof",
            "relL2PhiRef", "relH1PhiRef", "relL2RhoRef", "relL1RhoRef",
            "massRho", "maxRho", "activeArea", "plateauFrac", "resHm1", "poissonRel",
        ]
        print("\nConvergence summary")
        print(conv[[c for c in summary_cols if c in conv]].to_string(index=False))
    if len(slopes):
        print("\nEmpirical slopes")
        print(slopes.to_string(index=False))
    if args.save:
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out / "parsed_all_records.csv", index=False)
        solve.to_csv(args.out / "parsed_solve_records.csv", index=False)
        conv.to_csv(args.out / "parsed_convergence_summary.csv", index=False)
        init.to_csv(args.out / "parsed_initial_records.csv", index=False)
        slopes.to_csv(args.out / "parsed_empirical_slopes.csv", index=False)
    plot_residual_histories(solve, figdir, show, args.dpi)
    plot_errors(conv, figdir, show, args.dpi)
    plot_residual_metrics(conv, figdir, show, args.dpi)
    plot_mass_peak(conv, figdir, show, args.dpi)
    plot_band_metrics(conv, figdir, show, args.dpi)
    plot_newton_effort(solve, figdir, show, args.dpi)
    if not args.no_vtk:
        plot_latest_vtk_fields(args.vtk, args.pattern, figdir, show, args.dpi)


if __name__ == "__main__":
    main()
