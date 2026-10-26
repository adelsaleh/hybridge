#!/usr/bin/env python3
"""Where is the density error of the aligned structured mesh (App. C comment TODO)?

Reads the snapshots of the campaign's boundary-layer tasks, paths given as arguments (copy of the article's
run_outputs/convergence_poisson_degree/boundary_layer/analyze_boundary_layer.py; aligned mesh, k_x = k_y = 8 pi,
p = 3, 320,000 triangles, raw drift with p_phi = 3 and 4; pixel rasters of rho_h and rho at t = T, row 0 at the
top) and splits the squared L2 error of
rho_h - rho over strips along the four sides of (-1, 1)^2 (inflow x = -1, outflow x = 1, y = -1, y = 1), the interior
square (-0.6, 0.6)^2 and the rest. A share well above the strip's area share locates a boundary layer. Also prints
the error profile (RMS over the other coordinate) against the distance to each side. No simulation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WIDTHS = (0.02, 0.05, 0.1)


def load(path: Path):
    data = np.load(path)
    x0, x1, y0, y1 = (float(v) for v in data["bounds"])
    error = np.asarray(data["rho_numerical_final"]) - np.asarray(data["rho_exact_final"])
    height, width = error.shape
    xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    ys = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height  # row 0 at the top
    x, y = np.meshgrid(xs, ys)
    valid = np.isfinite(error)
    return x[valid], y[valid], error[valid] ** 2, data


def shares(x, y, e2, width):
    total, count = e2.sum(), e2.size
    regions = {
        "x=-1 (inflow)": x < -1 + width, "x=+1 (outflow)": x > 1 - width,
        "y=-1": y < -1 + width, "y=+1": y > 1 - width,
        "interior (-0.6,0.6)^2": (np.abs(x) < 0.6) & (np.abs(y) < 0.6),
    }
    sides = regions["x=-1 (inflow)"] | regions["x=+1 (outflow)"] | regions["y=-1"] | regions["y=+1"]
    regions["rest"] = ~sides & ~regions["interior (-0.6,0.6)^2"]
    return {name: (e2[mask].sum() / total, mask.sum() / count) for name, mask in regions.items()}


def profile(x, y, e2, side, bins=np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4])):
    distance = {"x=-1": x + 1, "x=+1": 1 - x, "y=-1": y + 1, "y=+1": 1 - y}[side]
    index = np.digitize(distance, bins) - 1
    return [(bins[i], bins[i + 1], float(np.sqrt(e2[index == i].mean()))) for i in range(len(bins) - 1)]


def main() -> None:
    paths = sorted(Path(arg) for arg in sys.argv[1:]) or sorted(HERE.glob("aligned_p3_q*_h0.005.npz"))
    for path in paths:
        x, y, e2, data = load(path)
        print(f"\n{path.name}: p={int(data['order'])}, p_phi={int(data['poisson_order'])}, N_K={int(data['triangles'])}, "
              f"||rho_h - rho|| = {float(data['rho_l2_error']):.3e} (raster RMS {np.sqrt(e2.mean()):.3e})")
        for width in WIDTHS:
            row = "  ".join(f"{name} {err:.1%}/{area:.1%}" for name, (err, area) in shares(x, y, e2, width).items())
            print(f"  strip width {width:g}: share of squared error / of area: {row}")
        for side in ("x=-1", "x=+1", "y=-1", "y=+1"):
            print(f"  RMS error vs distance to {side}: " + "  ".join(
                f"[{a:g},{b:g}) {v:.2e}" for a, b, v in profile(x, y, e2, side)))


if __name__ == "__main__":
    main()
