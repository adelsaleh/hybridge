#!/usr/bin/env python3
"""
Display and optionally save diagnostics from strategyA_adaptive_torsion_newton_logged.edp.

Expected layout, relative to the FreeFEM working directory:

	../logs/strategyA_adaptive_torsion_newton_runinfo.csv
	../logs/strategyA_adaptive_torsion_newton_newton.csv
	../logs/strategyA_adaptive_torsion_newton_continuation.csv
	../logs/strategyA_adaptive_torsion_newton_adapt.csv
	../logs/strategyA_adaptive_torsion_newton_final.csv
	../logs/strategyA_adaptive_torsion_newton_linesearch.csv   [optional]
	../vtu/strategyA_adaptive_torsion_newton.vtu               [optional]
	../vtu/adapt_stage_*.vtu                                  [optional]
	../out/rho_dof.txt                                        [optional]
	../out/phi_dof.txt                                        [optional]

Default behavior:
	- read all available logs;
	- print compact text tables;
	- display matplotlib figures;
	- do not write anything except when --save is passed.

Usage:
	python display_strategyA_adaptive_logs.py
	python display_strategyA_adaptive_logs.py --logs ../logs --vtu ../vtu --out ../out
	python display_strategyA_adaptive_logs.py --save
	python display_strategyA_adaptive_logs.py --save --figdir ../out/figures --tables-dir ../out/tables
	python display_strategyA_adaptive_logs.py --no-vtu
	python display_strategyA_adaptive_logs.py --prefix strategyA_adaptive_torsion_newton
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--logs", type=Path, default=Path("../logs"), help="Directory containing CSV logs.")
	parser.add_argument("--vtu", type=Path, default=Path("../vtu"), help="Directory containing VTU files.")
	parser.add_argument("--out", type=Path, default=Path("../out"), help="Directory for optional output tables.")
	parser.add_argument("--figdir", type=Path, default=None, help="Directory for saved figures. Default: OUT/figures.")
	parser.add_argument("--tables-dir", type=Path, default=None, help="Directory for saved parsed tables. Default: OUT/tables.")
	parser.add_argument("--prefix", default="strategyA_adaptive_torsion_newton", help="Run prefix used in log filenames.")
	parser.add_argument("--save", action="store_true", help="Save figures and parsed tables.")
	parser.add_argument("--show", action="store_true", help="Display figures. Default if --save is not passed.")
	parser.add_argument("--no-vtu", action="store_true", help="Skip VTU field preview plots.")
	parser.add_argument("--dpi", type=int, default=180, help="DPI for saved PNG figures.")
	parser.add_argument("--rolling", type=int, default=1, help="Rolling mean window for noisy iteration curves.")
	parser.add_argument("--max-vtu-files", type=int, default=4, help="Maximum VTU files to preview.")
	parser.add_argument("--field", action="append", default=None, help="Specific VTU point-data field(s) to plot. Can be repeated.")
	parser.add_argument("--print-all-columns", action="store_true", help="Print full tables without compact column selection.")
	return parser.parse_args()


# ---------------------------------------------------------------------
# Robust CSV loading
# ---------------------------------------------------------------------
def read_csv_log(path: Path) -> pd.DataFrame:
	if not path.exists():
		return pd.DataFrame()

	try:
		df = pd.read_csv(path, na_values=["NA", "NaN", "nan", "", " "], keep_default_na=True)
	except pd.errors.EmptyDataError:
		return pd.DataFrame()
	except Exception as exc:
		print(f"[warn] could not read {path}: {exc}")
		return pd.DataFrame()

	df["source_file"] = path.name

	for col in df.columns:
		if col not in {"record", "status", "source_file"}:
			df[col] = pd.to_numeric(df[col], errors="ignore")

	return df


def read_key_value_csv(path: Path) -> pd.DataFrame:
	"""Read runinfo/final logs that may use key,value or ordinary CSV."""
	if not path.exists():
		return pd.DataFrame()

	try:
		df = pd.read_csv(path, na_values=["NA", "NaN", "nan", "", " "], keep_default_na=True)
	except pd.errors.EmptyDataError:
		return pd.DataFrame()
	except Exception as exc:
		print(f"[warn] could not read {path}: {exc}")
		return pd.DataFrame()

	df["source_file"] = path.name

	for col in df.columns:
		if col not in {"record", "status", "key", "source_file"}:
			df[col] = pd.to_numeric(df[col], errors="ignore")

	return df


def load_logs(logs_dir: Path, prefix: str) -> dict[str, pd.DataFrame]:
	paths = {
		"runinfo": logs_dir / f"{prefix}_runinfo.csv",
		"newton": logs_dir / f"{prefix}_newton.csv",
		"continuation": logs_dir / f"{prefix}_continuation.csv",
		"adapt": logs_dir / f"{prefix}_adapt.csv",
		"final": logs_dir / f"{prefix}_final.csv",
		"linesearch": logs_dir / f"{prefix}_linesearch.csv",
	}

	logs = {
		"runinfo": read_key_value_csv(paths["runinfo"]),
		"newton": read_csv_log(paths["newton"]),
		"continuation": read_csv_log(paths["continuation"]),
		"adapt": read_csv_log(paths["adapt"]),
		"final": read_key_value_csv(paths["final"]),
		"linesearch": read_csv_log(paths["linesearch"]),
	}

	print("Input logs:")
	for name, path in paths.items():
		status = "ok" if not logs[name].empty else "missing/empty"
		print(f"  {name:13s} {status:13s} {path}")

	return logs


def require_any(logs: dict[str, pd.DataFrame]) -> None:
	if all(df.empty for df in logs.values()):
		raise SystemExit("No logs found. Check --logs and --prefix.")


# ---------------------------------------------------------------------
# Summary table construction
# ---------------------------------------------------------------------
def final_by_eps(newton: pd.DataFrame) -> pd.DataFrame:
	if newton.empty:
		return pd.DataFrame()

	df = newton.copy()
	if "ieps" not in df.columns or "k" not in df.columns:
		return pd.DataFrame()

	df = df.sort_values(["ieps", "k"])
	idx = df.groupby("ieps")["k"].idxmax()
	out = df.loc[idx].sort_values("ieps").reset_index(drop=True)

	keep = [
		"ieps", "epsPhiRatio", "epsPhi", "k", "status",
		"resAfter", "resHm1", "poissonRel", "nd",
		"alpha", "backtracks", "muAfter",
		"maxU", "maxRho", "massRho", "activeArea", "plateauArea", "plateauFraction",
		"relRhoDesign", "annularPhiMinusC2", "relDrop",
	]
	return out[[c for c in keep if c in out.columns]]


def newton_effort(newton: pd.DataFrame) -> pd.DataFrame:
	if newton.empty:
		return pd.DataFrame()

	df = newton.copy()
	if "ieps" not in df.columns:
		return pd.DataFrame()

	def count_status(s: pd.Series, value: str) -> int:
		return int((s.astype(str) == value).sum())

	agg = df.groupby("ieps").agg(
		epsPhiRatio=("epsPhiRatio", "last") if "epsPhiRatio" in df else ("ieps", "size"),
		epsPhi=("epsPhi", "last") if "epsPhi" in df else ("ieps", "size"),
		nRows=("ieps", "size"),
		maxK=("k", "max") if "k" in df else ("ieps", "size"),
		nAccept=("status", lambda s: count_status(s, "ACCEPT")) if "status" in df else ("ieps", "size"),
		nFail=("status", lambda s: count_status(s, "FAIL_LS")) if "status" in df else ("ieps", "size"),
		nConverged=("status", lambda s: count_status(s, "CONVERGED")) if "status" in df else ("ieps", "size"),
		meanBacktracks=("backtracks", "mean") if "backtracks" in df else ("ieps", "size"),
		maxBacktracks=("backtracks", "max") if "backtracks" in df else ("ieps", "size"),
		finalResHm1=("resHm1", "last") if "resHm1" in df else ("ieps", "size"),
		finalPoissonRel=("poissonRel", "last") if "poissonRel" in df else ("ieps", "size"),
		finalRelRhoDesign=("relRhoDesign", "last") if "relRhoDesign" in df else ("ieps", "size"),
		finalMass=("massRho", "last") if "massRho" in df else ("ieps", "size"),
	)
	agg = agg.reset_index()

	if "maxK" in agg:
		agg["iterations"] = agg["maxK"] + 1

	return agg


def adaptation_summary(adapt: pd.DataFrame) -> pd.DataFrame:
	if adapt.empty:
		return pd.DataFrame()

	keep = [
		"stage", "ieps", "event",
		"ntBefore", "ntAfter", "ndofBefore", "ndofAfter",
		"hmin", "hmax", "maxGradPhi", "targetH",
		"massBefore", "massAfter", "maxRhoBefore", "maxRhoAfter",
		"relRhoDesignBefore", "relRhoDesignAfter",
		"resHm1Before", "resHm1After", "poissonRelBefore", "poissonRelAfter",
		"adaptTime",
	]
	return adapt[[c for c in keep if c in adapt.columns]].copy()


def final_summary(final: pd.DataFrame) -> pd.DataFrame:
	if final.empty:
		return pd.DataFrame()

	# Supports either one-row wide table or key,value table.
	if {"key", "value"}.issubset(final.columns):
		return final[["key", "value"]].copy()

	cols = [
		"resEuclid", "resHm1", "relHm1E", "relHm1Rho", "poissonRel",
		"minPhi", "maxPhi", "maxRho", "massRho",
		"relRhoDesign", "annularPhiMinusC2", "normalizedDefect",
		"activeArea", "plateauArea", "plateauFraction",
	]
	return final[[c for c in cols if c in final.columns]].copy()


def print_table(title: str, df: pd.DataFrame, compact_cols: Optional[list[str]] = None, all_cols: bool = False) -> None:
	print("\n" + title)
	print("-" * len(title))
	if df.empty:
		print("(empty)")
		return

	if compact_cols is not None and not all_cols:
		cols = [c for c in compact_cols if c in df.columns]
		if cols:
			df = df[cols]

	with pd.option_context(
		"display.max_rows", 200,
		"display.max_columns", None,
		"display.width", 180,
		"display.float_format", lambda x: f"{x:.6g}",
	):
		print(df.to_string(index=False))


# ---------------------------------------------------------------------
# Plot utilities
# ---------------------------------------------------------------------
def smooth_series(y: pd.Series, window: int) -> pd.Series:
	if window <= 1:
		return y
	return y.rolling(window=window, min_periods=1).mean()


def save_or_show(fig: plt.Figure, path: Optional[Path], show: bool, dpi: int) -> None:
	fig.tight_layout()
	if path is not None:
		path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(path, dpi=dpi, bbox_inches="tight")
	if show:
		plt.show()
	plt.close(fig)


def plot_newton_residuals(newton: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int, rolling: int) -> None:
	if newton.empty:
		return

	fig, ax = plt.subplots(figsize=(8.5, 5.2))

	for ieps, g in newton.groupby("ieps", dropna=False):
		g = g.sort_values("k")
		x = g["k"]
		if "resHm1" in g:
			ax.semilogy(x, smooth_series(g["resHm1"], rolling), marker="o", markersize=3, linewidth=1.2, label=f"eps {int(ieps)}: H-1")
		elif "resAfter" in g:
			ax.semilogy(x, smooth_series(g["resAfter"], rolling), marker="o", markersize=3, linewidth=1.2, label=f"eps {int(ieps)}: resAfter")

	ax.set_xlabel("Newton iteration k")
	ax.set_ylabel("residual")
	ax.set_title("Newton residual history by continuation level")
	ax.grid(True, which="both", linewidth=0.4)
	ax.legend(fontsize=8)
	save_or_show(fig, None if figdir is None else figdir / "newton_residual_history.png", show, dpi)


def plot_newton_quality(newton: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int) -> None:
	if newton.empty:
		return

	fig, ax = plt.subplots(figsize=(8.5, 5.2))

	for ieps, g in newton.groupby("ieps", dropna=False):
		g = g.sort_values("k")
		label = f"eps {int(ieps)}"
		if "relRhoDesign" in g:
			ax.plot(g["k"], g["relRhoDesign"], marker="o", markersize=3, linewidth=1.2, label=label)

	ax.set_xlabel("Newton iteration k")
	ax.set_ylabel(r"relative distance to $\rho_{\mathrm{design}}$")
	ax.set_title("Shape drift during Newton")
	ax.grid(True, linewidth=0.4)
	ax.legend(fontsize=8)
	save_or_show(fig, None if figdir is None else figdir / "newton_rel_rho_design.png", show, dpi)

	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	for ieps, g in newton.groupby("ieps", dropna=False):
		g = g.sort_values("k")
		if "annularPhiMinusC2" in g:
			ax.plot(g["k"], g["annularPhiMinusC2"], marker="o", markersize=3, linewidth=1.2, label=f"eps {int(ieps)}")

	ax.axhline(0.0, linewidth=1.0, linestyle="--")
	ax.set_xlabel("Newton iteration k")
	ax.set_ylabel(r"$\max u-c_2$")
	ax.set_title("Annularity margin during Newton")
	ax.grid(True, linewidth=0.4)
	ax.legend(fontsize=8)
	save_or_show(fig, None if figdir is None else figdir / "newton_annularity_margin.png", show, dpi)


def plot_line_search(newton: pd.DataFrame, linesearch: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int) -> None:
	if not newton.empty and "backtracks" in newton:
		fig, ax = plt.subplots(figsize=(8.5, 5.2))
		for ieps, g in newton.groupby("ieps", dropna=False):
			g = g.sort_values("k")
			ax.plot(g["k"], g["backtracks"], marker="o", markersize=3, linewidth=1.2, label=f"eps {int(ieps)}")
		ax.set_xlabel("Newton iteration k")
		ax.set_ylabel("accepted-step backtracks")
		ax.set_title("Line-search backtracking burden")
		ax.grid(True, linewidth=0.4)
		ax.legend(fontsize=8)
		save_or_show(fig, None if figdir is None else figdir / "newton_backtracks.png", show, dpi)

	if not newton.empty and {"k", "alpha"}.issubset(newton.columns):
		fig, ax = plt.subplots(figsize=(8.5, 5.2))
		for ieps, g in newton.groupby("ieps", dropna=False):
			g = g.sort_values("k")
			ax.semilogy(g["k"], g["alpha"], marker="o", markersize=3, linewidth=1.2, label=f"eps {int(ieps)}")
		ax.set_xlabel("Newton iteration k")
		ax.set_ylabel("accepted alpha")
		ax.set_title("Accepted Armijo step size")
		ax.grid(True, which="both", linewidth=0.4)
		ax.legend(fontsize=8)
		save_or_show(fig, None if figdir is None else figdir / "newton_alpha.png", show, dpi)

	if not linesearch.empty and {"alpha", "resNew", "resOld"}.issubset(linesearch.columns):
		fig, ax = plt.subplots(figsize=(8.5, 5.2))
		ls = linesearch.copy()
		ls["resRatio"] = ls["resNew"] / np.maximum(ls["resOld"], 1e-300)
		for ieps, g in ls.groupby("ieps", dropna=False):
			ax.scatter(g["alpha"], g["resRatio"], s=12, alpha=0.65, label=f"eps {int(ieps)}")
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.axhline(1.0, linewidth=1.0, linestyle="--")
		ax.set_xlabel("trial alpha")
		ax.set_ylabel("trial residual ratio")
		ax.set_title("All line-search trial residual ratios")
		ax.grid(True, which="both", linewidth=0.4)
		ax.legend(fontsize=8)
		save_or_show(fig, None if figdir is None else figdir / "linesearch_trial_ratios.png", show, dpi)


def plot_mu_and_convergence(newton: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int) -> None:
	if newton.empty:
		return

	if {"muBefore", "muAfter"}.intersection(newton.columns):
		fig, ax = plt.subplots(figsize=(8.5, 5.2))
		for ieps, g in newton.groupby("ieps", dropna=False):
			g = g.sort_values("k")
			col = "muAfter" if "muAfter" in g else "muShift"
			if col in g:
				ax.semilogy(g["k"], g[col], marker="o", markersize=3, linewidth=1.2, label=f"eps {int(ieps)}")
		ax.set_xlabel("Newton iteration k")
		ax.set_ylabel(r"$\mu$ shift")
		ax.set_title("Shift stabilization history")
		ax.grid(True, which="both", linewidth=0.4)
		ax.legend(fontsize=8)
		save_or_show(fig, None if figdir is None else figdir / "mu_shift_history.png", show, dpi)

	if "relDrop" in newton:
		fig, ax = plt.subplots(figsize=(8.5, 5.2))
		for ieps, g in newton.groupby("ieps", dropna=False):
			g = g.sort_values("k")
			y = g["relDrop"].clip(lower=1e-18)
			ax.semilogy(g["k"], y, marker="o", markersize=3, linewidth=1.2, label=f"eps {int(ieps)}")
		ax.set_xlabel("Newton iteration k")
		ax.set_ylabel("relative residual drop")
		ax.set_title("Accepted convergence rate per step")
		ax.grid(True, which="both", linewidth=0.4)
		ax.legend(fontsize=8)
		save_or_show(fig, None if figdir is None else figdir / "newton_relative_drop.png", show, dpi)


def plot_band_geometry(newton: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int) -> None:
	if newton.empty:
		return

	cols = [c for c in ["massRho", "activeArea", "plateauArea", "plateauFraction", "maxRho"] if c in newton.columns]
	if not cols:
		return

	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	for ieps, g in newton.groupby("ieps", dropna=False):
		g = g.sort_values("k")
		if "massRho" in g:
			ax.plot(g["k"], g["massRho"], marker="o", markersize=3, linewidth=1.2, label=f"mass eps {int(ieps)}")
		if "activeArea" in g:
			ax.plot(g["k"], g["activeArea"], marker="x", markersize=4, linewidth=1.0, label=f"active eps {int(ieps)}")
		if "plateauArea" in g:
			ax.plot(g["k"], g["plateauArea"], marker="s", markersize=3, linewidth=1.0, label=f"plateau eps {int(ieps)}")
	ax.set_xlabel("Newton iteration k")
	ax.set_ylabel("mass / area")
	ax.set_title("Band geometry and mass during Newton")
	ax.grid(True, linewidth=0.4)
	ax.legend(fontsize=7, ncol=2)
	save_or_show(fig, None if figdir is None else figdir / "band_geometry_newton.png", show, dpi)

	if "plateauFraction" in newton:
		fig, ax = plt.subplots(figsize=(8.5, 5.2))
		for ieps, g in newton.groupby("ieps", dropna=False):
			g = g.sort_values("k")
			ax.plot(g["k"], g["plateauFraction"], marker="o", markersize=3, linewidth=1.2, label=f"eps {int(ieps)}")
		ax.set_xlabel("Newton iteration k")
		ax.set_ylabel("plateau fraction")
		ax.set_title("Band sharpness proxy")
		ax.grid(True, linewidth=0.4)
		ax.legend(fontsize=8)
		save_or_show(fig, None if figdir is None else figdir / "plateau_fraction.png", show, dpi)


def plot_continuation_summary(cont: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int) -> None:
	if cont.empty:
		return

	# Prefer EPS_END rows if event column is present.
	df = cont.copy()
	if "event" in df:
		end = df[df["event"].astype(str).str.contains("END", case=False, na=False)].copy()
		if not end.empty:
			df = end

	if "ieps" not in df:
		return

	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	if "resHm1" in df:
		ax.semilogy(df["ieps"], df["resHm1"], marker="o", linewidth=1.3, label=r"$H^{-1}_h$")
	if "poissonRel" in df:
		ax.semilogy(df["ieps"], df["poissonRel"], marker="o", linewidth=1.3, label="poissonRel")
	ax.set_xlabel("continuation level ieps")
	ax.set_ylabel("residual")
	ax.set_title("End-of-continuation residuals")
	ax.grid(True, which="both", linewidth=0.4)
	ax.legend(fontsize=8)
	save_or_show(fig, None if figdir is None else figdir / "continuation_residuals.png", show, dpi)

	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	if "epsPhi" in df:
		x = df["epsPhi"]
	else:
		x = df["ieps"]
	for col in ["relRhoDesign", "annularPhiMinusC2", "massRho", "maxRho"]:
		if col in df:
			ax.plot(x, df[col], marker="o", linewidth=1.3, label=col)
	ax.set_xlabel("epsPhi" if "epsPhi" in df else "ieps")
	ax.set_title("Continuation quality metrics")
	ax.grid(True, linewidth=0.4)
	ax.legend(fontsize=8)
	if "epsPhi" in df:
		ax.invert_xaxis()
	save_or_show(fig, None if figdir is None else figdir / "continuation_quality.png", show, dpi)


def plot_adaptation(adapt: pd.DataFrame, figdir: Optional[Path], show: bool, dpi: int) -> None:
	if adapt.empty:
		return

	x = adapt["stage"] if "stage" in adapt else np.arange(len(adapt))

	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	for col in ["ntBefore", "ntAfter", "ndofBefore", "ndofAfter"]:
		if col in adapt:
			ax.plot(x, adapt[col], marker="o", linewidth=1.3, label=col)
	ax.set_xlabel("adaptation stage")
	ax.set_ylabel("mesh size")
	ax.set_title("Mesh adaptation size history")
	ax.grid(True, linewidth=0.4)
	ax.legend(fontsize=8)
	save_or_show(fig, None if figdir is None else figdir / "adapt_mesh_size.png", show, dpi)

	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	for col in ["hmin", "hmax", "targetH"]:
		if col in adapt:
			ax.semilogy(x, adapt[col], marker="o", linewidth=1.3, label=col)
	ax.set_xlabel("adaptation stage")
	ax.set_ylabel("mesh length scale")
	ax.set_title("Mesh-size diagnostics")
	ax.grid(True, which="both", linewidth=0.4)
	ax.legend(fontsize=8)
	save_or_show(fig, None if figdir is None else figdir / "adapt_h_diagnostics.png", show, dpi)

	fig, ax = plt.subplots(figsize=(8.5, 5.2))
	for before, after, name in [
		("resHm1Before", "resHm1After", r"$H^{-1}_h$"),
		("poissonRelBefore", "poissonRelAfter", "poissonRel"),
		("relRhoDesignBefore", "relRhoDesignAfter", "relRhoDesign"),
	]:
		if before in adapt and after in adapt:
			ax.semilogy(x, adapt[before], marker="o", linewidth=1.0, linestyle="--", label=f"{name} before")
			ax.semilogy(x, adapt[after], marker="o", linewidth=1.3, label=f"{name} after")
	ax.set_xlabel("adaptation stage")
	ax.set_ylabel("metric")
	ax.set_title("Effect of adaptation on residual/shape metrics")
	ax.grid(True, which="both", linewidth=0.4)
	ax.legend(fontsize=8)
	save_or_show(fig, None if figdir is None else figdir / "adapt_metric_effect.png", show, dpi)


# ---------------------------------------------------------------------
# VTU reading and field plotting
# ---------------------------------------------------------------------
def import_pyvista():
	try:
		import pyvista as pv  # type: ignore
		return pv
	except Exception:
		return None


def extract_tri_and_fields_from_vtu(path: Path):
	pv = import_pyvista()
	if pv is None:
		print("[warn] pyvista is not installed; skipping VTU field previews.")
		return None

	try:
		mesh = pv.read(path)
	except Exception as exc:
		print(f"[warn] could not read VTU {path}: {exc}")
		return None

	points = np.asarray(mesh.points)
	if points.ndim != 2 or points.shape[1] < 2:
		print(f"[warn] VTU has invalid points: {path}")
		return None

	# PyVista/VTK unstructured grid cells: [n, id0, id1, ..., n, ...].
	cells = np.asarray(mesh.cells)
	triangles = []
	i = 0
	while i < len(cells):
		n = int(cells[i])
		ids = cells[i + 1 : i + 1 + n]
		if n == 3:
			triangles.append(ids)
		elif n == 6:
			# P2 triangle: use vertex corners only.
			triangles.append(ids[:3])
		i += n + 1

	if not triangles:
		print(f"[warn] no triangle cells found in {path}")
		return None

	tri = Triangulation(points[:, 0], points[:, 1], np.asarray(triangles))

	fields = {}
	for name in mesh.point_data.keys():
		arr = np.asarray(mesh.point_data[name])
		if arr.ndim == 1 and len(arr) == len(points):
			fields[name] = arr

	return tri, fields


def field_candidates(fields: dict[str, np.ndarray], requested: Optional[list[str]]) -> list[str]:
	if requested:
		return [f for f in requested if f in fields]

	preferred = [
		"rhoFinal", "rho", "rhoFromPhi",
		"phiFinal", "phi", "u",
		"rhoDesign", "phiDesign",
		"defectFinal", "defectDesign",
		"gradTNorm",
	]
	out = [f for f in preferred if f in fields]
	if out:
		return out[:6]

	return list(fields.keys())[:6]


def plot_vtu_fields(vtu_dir: Path, prefix: str, requested_fields: Optional[list[str]], max_files: int, figdir: Optional[Path], show: bool, dpi: int) -> None:
	files = sorted(vtu_dir.glob(f"{prefix}*.vtu"))
	files += sorted(vtu_dir.glob("adapt_stage_*.vtu"))
	# Deduplicate while preserving order.
	seen = set()
	unique = []
	for p in files:
		if p not in seen:
			unique.append(p)
			seen.add(p)

	if not unique:
		return

	# Prefer final file plus most recent adapt-stage files.
	finals = [p for p in unique if prefix in p.name]
	adapts = [p for p in unique if p.name.startswith("adapt_stage_")]
	selected = finals[:1] + adapts[-max(0, max_files - len(finals[:1])):]
	selected = selected[:max_files]

	for path in selected:
		loaded = extract_tri_and_fields_from_vtu(path)
		if loaded is None:
			continue

		tri, fields = loaded
		for name in field_candidates(fields, requested_fields):
			values = fields[name]
			fig, ax = plt.subplots(figsize=(7.0, 6.2))
			tpc = ax.tripcolor(tri, values, shading="gouraud")
			ax.triplot(tri, linewidth=0.15, alpha=0.22)
			ax.set_aspect("equal")
			ax.set_title(f"{path.name}: {name}")
			fig.colorbar(tpc, ax=ax)
			save_or_show(fig, None if figdir is None else figdir / f"field_{path.stem}_{name}.png", show, dpi)


# ---------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------
def save_tables(logs: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame], tables_dir: Path) -> None:
	tables_dir.mkdir(parents=True, exist_ok=True)

	for name, df in logs.items():
		if not df.empty:
			df.to_csv(tables_dir / f"raw_{name}.csv", index=False)

	for name, df in summaries.items():
		if not df.empty:
			df.to_csv(tables_dir / f"summary_{name}.csv", index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
	args = parse_args()
	show = args.show or not args.save

	figdir = args.figdir if args.figdir is not None else args.out / "figures"
	tables_dir = args.tables_dir if args.tables_dir is not None else args.out / "tables"

	if not args.save:
		figdir = None

	logs = load_logs(args.logs, args.prefix)
	require_any(logs)

	newton = logs["newton"]
	continuation = logs["continuation"]
	adapt = logs["adapt"]
	linesearch = logs["linesearch"]
	final = logs["final"]

	eps_final = final_by_eps(newton)
	effort = newton_effort(newton)
	adapt_sum = adaptation_summary(adapt)
	final_sum = final_summary(final)

	summaries = {
		"final_by_eps": eps_final,
		"newton_effort": effort,
		"adaptation": adapt_sum,
		"final": final_sum,
	}

	print_table(
		"Final Newton row by continuation level",
		eps_final,
		compact_cols=[
			"ieps", "epsPhiRatio", "epsPhi", "k", "status",
			"resHm1", "poissonRel", "maxRho", "massRho",
			"relRhoDesign", "annularPhiMinusC2", "plateauFraction",
		],
		all_cols=args.print_all_columns,
	)

	print_table(
		"Newton effort by continuation level",
		effort,
		compact_cols=[
			"ieps", "epsPhiRatio", "epsPhi", "iterations",
			"nAccept", "nFail", "nConverged",
			"meanBacktracks", "maxBacktracks",
			"finalResHm1", "finalPoissonRel", "finalRelRhoDesign", "finalMass",
		],
		all_cols=args.print_all_columns,
	)

	print_table(
		"Adaptation summary",
		adapt_sum,
		compact_cols=[
			"stage", "ieps", "event",
			"ntBefore", "ntAfter", "ndofBefore", "ndofAfter",
			"hmin", "hmax", "targetH",
			"massBefore", "massAfter",
			"resHm1Before", "resHm1After",
			"poissonRelBefore", "poissonRelAfter",
		],
		all_cols=args.print_all_columns,
	)

	print_table(
		"Final diagnostics",
		final_sum,
		compact_cols=None,
		all_cols=args.print_all_columns,
	)

	if args.save:
		save_tables(logs, summaries, tables_dir)
		print(f"\nSaved parsed tables to {tables_dir}")
		print(f"Saving figures to {figdir}")

	plot_newton_residuals(newton, figdir, show, args.dpi, args.rolling)
	plot_newton_quality(newton, figdir, show, args.dpi)
	plot_line_search(newton, linesearch, figdir, show, args.dpi)
	plot_mu_and_convergence(newton, figdir, show, args.dpi)
	plot_band_geometry(newton, figdir, show, args.dpi)
	plot_continuation_summary(continuation, figdir, show, args.dpi)
	plot_adaptation(adapt, figdir, show, args.dpi)

	if not args.no_vtu:
		plot_vtu_fields(args.vtu, args.prefix, args.field, args.max_vtu_files, figdir, show, args.dpi)

	if args.save:
		print("Done.")


if __name__ == "__main__":
	main()
