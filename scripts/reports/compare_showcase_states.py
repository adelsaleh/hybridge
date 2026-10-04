"""Compare GPU showcase checkpoints for time-step, mesh and tau checks.

Each ``.restart.npz`` from ``record_gpu_showcase`` holds the final
full-precision density and potential. Same-mesh pairs use the physical DG
L2 norm; pairs on different meshes are sampled at shared area-uniform grid
points, without interpolation across DG faces. A ``--series`` of three or more
states at halved time steps also reports the observed order and the
Richardson estimate of each state's time-discretization error.
Run as ``python -m scripts.reports.compare_showcase_states``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from hdgfem.core.field_ops import field_linear_combination
from hdgfem.io.raster import RasterGeometry
from scripts.reports.gpu_showcase_setup import showcase_mesh, showcase_space

FIELDS = ("rho", "phi")


def load_state(path, meshes):
    """Rebuild the checkpoint's mesh and wrap its coefficients as host fields."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("node_coords", "triangles", *FIELDS)}
        h, dt, step = float(data["h"]), float(data["dt"]), int(data["step"])
    if h not in meshes:
        mesh = showcase_mesh(h)
        meshes[h] = (mesh, showcase_space(mesh))
    mesh, space = meshes[h]
    if not (np.array_equal(mesh.node_coords, arrays["node_coords"])
            and np.array_equal(mesh.triangles, arrays["triangles"])):
        raise ValueError(f"{path}: mesh differs from the regenerated showcase mesh h={h}")
    fields = {name: space.field(arrays[name], name=name) for name in FIELDS}
    return dict(path=str(path), name=Path(path).name.removesuffix(".restart.npz"),
                h=h, dt=dt, step=step, time=step*dt, space=space, fields=fields)


def same_mesh_difference(a, b):
    """Relative physical L2 difference ||a-b|| / ||b|| for each field."""
    space = b["space"]
    out = {}
    for name in FIELDS:
        difference = field_linear_combination(space, ((1., a["fields"][name]), (-1., b["fields"][name])))
        out[name] = difference.l2_norm() / b["fields"][name].l2_norm()
    return out


def sampled_difference(a, b, points):
    """Relative L2 difference at shared points owned by both meshes."""
    samples, valid = [], []
    for state in (a, b):
        space = state["space"]
        geometry = RasterGeometry.from_points(space.mesh, points, width=len(points), height=1)
        matrix = geometry.sampling_matrix(space, max_bytes=4 * 1024**3)
        valid.append(geometry.element_ids >= 0)
        samples.append({name: matrix @ np.ravel(state["fields"][name].coeffs) for name in FIELDS})
    common = valid[0] & valid[1]
    out = {name: float(np.linalg.norm(samples[0][name][common] - samples[1][name][common])
                       / np.linalg.norm(samples[1][name][common])) for name in FIELDS}
    out["common_points"] = int(common.sum())
    return out


def compare(a, b, points):
    """Choose the exact same-mesh norm when possible, else shared-point sampling."""
    if not math.isclose(a["time"], b["time"], rel_tol=0., abs_tol=1e-12):
        raise ValueError(f"{a['name']} ends at t={a['time']}, {b['name']} at t={b['time']}")
    if a["space"] is b["space"]:
        return dict(same_mesh_difference(a, b), method="dg-l2")
    return dict(sampled_difference(a, b, points), method="shared-points")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="+", default=[],
                        help="checkpoints at successively halved time steps, coarsest first")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="compare A with reference B (repeatable)")
    parser.add_argument("--points", type=int, default=1000, help="shared grid points per axis")
    parser.add_argument("--output", required=True, help="JSON file for the comparison table")
    args = parser.parse_args()
    if not args.series and not args.pair:
        parser.error("give --series and/or --pair")
    meshes, states = {}, {}

    def state(path):
        if path not in states:
            states[path] = load_state(path, meshes)
        return states[path]

    lo = np.array([np.inf, np.inf])
    hi = -lo
    for path in [*args.series, *(p for pair in args.pair for p in pair)]:
        nodes = state(path)["space"].mesh.node_coords
        lo, hi = np.minimum(lo, nodes.min(axis=0)), np.maximum(hi, nodes.max(axis=0))
    grid = [np.linspace(lo[axis], hi[axis], args.points) for axis in range(2)]
    points = np.column_stack([axis.ravel() for axis in np.meshgrid(*grid)])
    report = dict(pairs=[], series=None)
    for a, b in args.pair:
        row = dict(a=state(a)["name"], b=state(b)["name"], **compare(state(a), state(b), points))
        report["pairs"].append(row)
        print(json.dumps(row), flush=True)
    if args.series:
        chain = [state(path) for path in args.series]
        if len(chain) < 2 or any(not math.isclose(chain[i]["dt"], 2*chain[i+1]["dt"])
                                 for i in range(len(chain)-1)):
            parser.error("a series needs at least two states with successively halved dt")
        steps = [compare(chain[i], chain[i+1], points) for i in range(len(chain)-1)]
        series = dict(dt=[s["dt"] for s in chain], names=[s["name"] for s in chain],
                      differences=steps, observed_order={}, richardson_error={})
        for name in FIELDS:
            diffs = [step[name] for step in steps]
            orders = [math.log2(diffs[i]/diffs[i+1]) for i in range(len(diffs)-1)]
            order = orders[-1] if orders else 2.0
            ratio = 2**order
            series["observed_order"][name] = orders
            # Error of state i estimated from its difference with state i+1.
            series["richardson_error"][name] = [d * ratio / (ratio - 1) for d in diffs]
        report["series"] = series
        print(json.dumps(series), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
