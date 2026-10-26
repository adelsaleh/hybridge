"""D7 accuracy per unknown: magnetron runs of several degrees p against one reference run; no time integration.

Reads ``run_magnetron.py`` records (JSONL + snapshot NPZ) of runs that differ in the degree p, the
mesh or the time step, and one reference run (finest mesh, same physics). Each run gets

* its global system sizes (K triangles, E interior edges; wall traces are eliminated): the HDG
  trace system has E(p+1) unknowns and sum_K n_K^2 - E blocks of size (p+1)^2 (n_K interior edges of
  K); a non-hybridized DG(p) element system would have K(p+1)(p+2)/2 unknowns and K + 2E blocks of
  size ((p+1)(p+2)/2)^2;
* the density error at the snapshot times, an L2 distance to the reference over one common pixel
  grid (pixels covered by every mesh, as in ``magnetron_convergence``), relative to the reference;
* the anode-current and charge errors over the time levels shared with the reference: relative L2
  in time, and the maximum relative to the reference peak;
* its charge-balance defect, minimum density and median step wall time (shared GPU: indicative);
* the largest cell Courant number C = dt max_K |u|/h_K over the run (h_K the shortest edge, |u| the
  E x B speed logged by the run) and what an explicit DG step would need: the advective RK-DG limit
  dt <= h/((2p+1)|u|) gives C(2p+1) explicit steps per implicit one; the diffusive limit, estimated
  as dt <= h_min^2/(D (p+1)^4) without its constant, gives dt D (p+1)^4/h_min^2.

``<stem>.json`` holds the numbers; ``<stem>.pdf/.png`` plots the errors against the density
unknowns, one line per (p, dt).

    python -m scripts.guiding_center.diagnostics.magnetron_order_efficiency \\
        --reference REF.jsonl --runs A.jsonl B.jsonl ... --stem OUT
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hybridge.io.figures import publication_style, save_publication_figure  # noqa: E402
from scripts.guiding_center.diagnostics.magnetron_convergence import _snapshot_path  # noqa: E402
from scripts.guiding_center.diagnostics.run_records import records as _records, series as _series  # noqa: E402


def _run(path: Path) -> dict:
    rows = _records(path)
    configuration = next(row["run_configuration"] for row in rows if row.get("run_configuration"))
    steps = [row for row in rows if row.get("step", 0) > 0 and "mass" in row]
    triangles, order = int(configuration["triangles"]), int(configuration["order"])
    wall = [row["linear_step_wall_time"] for row in steps[2:] if row.get("linear_step_wall_time") is not None]
    dt = float(configuration["dt"])
    courant = dt * float(np.nanmax(_series(steps, "velocity_max_speed_over_min_edge")))
    diffusion = float((configuration["case_parameters"].get("collisional") or {}).get("diffusion", 0.0))
    npz = np.load(_snapshot_path(path))
    corners = npz["node_coords"][npz["triangles"]]  # (K, 3, 2)
    min_edge = float(np.min(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)))
    sizes = _system_sizes(np.asarray(npz["triangles"]), order)
    return dict(courant_max=courant, explicit_advective_step_ratio=courant * (2 * order + 1),
                explicit_diffusive_step_ratio_estimate=dt * diffusion * (order + 1) ** 4 / min_edge ** 2,
                min_edge=min_edge, diffusion=diffusion,
                path=path, order=order, dt=dt, triangles=triangles,
                density_unknowns=triangles * (order + 1) * (order + 2) // 2, **sizes,
                time=_series(steps, "time"), current=_series(steps, "wall_outflux_outer"),
                charge=_series(steps, "mass"),
                balance=float(np.nanmax(np.abs(_series(steps, "charge_balance_relative_defect")))),
                rho_min=float(np.nanmin(_series(steps, "rho_min"))),
                step_wall=statistics.median(wall) if wall else None)


def _system_sizes(triangles: np.ndarray, order: int) -> dict:
    """Global unknowns and nonzeros of the HDG trace system and of a DG(p) element system."""
    edges = np.sort(np.stack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=1), axis=2)
    _, index, counts = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True, return_counts=True)
    interior = (counts == 2)[index.reshape(-1)].reshape(-1, 3)  # (K, 3): edge shared by two triangles
    num_interior = int(np.count_nonzero(counts == 2))
    per_cell = interior.sum(axis=1)
    trace_block, element_block = order + 1, (order + 1) * (order + 2) // 2
    return dict(interior_edges=num_interior, trace_unknowns=num_interior * trace_block,
                trace_nonzeros=int((per_cell ** 2).sum() - num_interior) * trace_block ** 2,
                dg_element_unknowns=len(triangles) * element_block,
                dg_element_nonzeros=(len(triangles) + 2 * num_interior) * element_block ** 2)


def _pixel_densities(npz, pixels: int, bounds, times) -> tuple[dict, np.ndarray]:
    """Snapshot densities at the pixel centres (chunked, no sampling matrix), and the uncovered pixels."""
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.io.raster import RasterGeometry

    mesh = DGMesh(npz["node_coords"], npz["triangles"])
    space = DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]))
    geometry = RasterGeometry.from_mesh(mesh, pixels, pixels)
    if not np.allclose(geometry.bounds, bounds, atol=1e-9):
        raise ValueError("runs must share the domain bounding box")
    frames = {time: int(np.argmin(np.abs(npz["times"] - time))) for time in times}
    frames = {time: k for time, k in frames.items() if abs(npz["times"][k] - time) < 1e-6}
    outside = (geometry.element_ids < 0).reshape(geometry.height, geometry.width)
    return {time: geometry.evaluate(space, npz["density"][k], fill=0.0) for time, k in frames.items()}, outside


def _history_errors(run: dict, reference: dict, t_max: float | None = None) -> dict:
    keys_run, keys_ref = np.round(run["time"], 9), np.round(reference["time"], 9)
    common = np.intersect1d(keys_run, keys_ref)
    if t_max is not None:
        common = common[common <= t_max + 1e-9]
    i, j = np.searchsorted(keys_run, common), np.searchsorted(keys_ref, common)
    out = {}
    for name in ("current", "charge"):
        a, b = run[name][i], reference[name][j]
        out[f"{name}_l2"] = float(np.linalg.norm(a - b) / np.linalg.norm(b))
        out[f"{name}_max"] = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    return out


def analyse(reference_path, run_paths, stem, *, times=(10., 20., 30., 40.), pixels=1600, t_max=None):
    reference = _run(Path(reference_path))
    runs = [_run(Path(path)) for path in run_paths]
    snapshots = [np.load(_snapshot_path(item["path"])) for item in [reference] + runs]
    from hybridge.core.mesh import DGMesh
    from hybridge.io.raster import RasterGeometry

    first = snapshots[0]
    geometry = RasterGeometry.from_mesh(DGMesh(first["node_coords"], first["triangles"]), pixels, pixels)
    samples, masks = [], []
    for npz in snapshots:
        fields, outside = _pixel_densities(npz, pixels, geometry.bounds, times)
        samples.append(fields)
        masks.append(outside)
    common = ~np.logical_or.reduce(masks)
    x0, x1, y0, y1 = geometry.bounds
    area = (x1 - x0) * (y1 - y0) / common.size

    def norm(field):
        return float(np.sqrt(np.sum(field[common] ** 2) * area))

    rows = []
    for item, fields in zip(runs, samples[1:]):
        row = {key: (str(value) if key == "path" else value) for key, value in item.items()
               if key not in {"time", "current", "charge"}}
        row.update(_history_errors(item, reference, t_max))
        row["density_l2"] = {time: norm(fields[time] - samples[0][time]) / norm(samples[0][time])
                             for time in times if time in fields and time in samples[0]}
        rows.append(row)
    result = dict(reference={key: (str(value) if key == "path" else value) for key, value in reference.items()
                             if key not in {"time", "current", "charge"}},
                  times=list(times), pixels=pixels, runs=rows)
    Path(f"{stem}.json").write_text(json.dumps(result, indent=1, default=float) + "\n")

    groups: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["order"], row["dt"]), []).append(row)
    with publication_style():
        figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), layout="constrained")
        colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(groups)))
        for color, ((order, dt), members) in zip(colors, sorted(groups.items())):
            members = sorted(members, key=lambda row: row["density_unknowns"])
            if len(members) < 2:
                continue
            dofs = [row["density_unknowns"] for row in members]
            label = rf"$p={order}$" + ("" if dt == reference["dt"] else rf", $\Delta t={dt:g}$")
            final = max(times)
            axes[0].loglog(dofs, [row["density_l2"].get(final, np.nan) for row in members], "o-", color=color,
                           ms=3, lw=.9, label=label)
            axes[1].loglog(dofs, [row["current_l2"] for row in members], "o-", color=color, ms=3, lw=.9)
            axes[2].loglog(dofs, [row["charge_max"] for row in members], "o-", color=color, ms=3, lw=.9)
        axes[0].set(xlabel="density unknowns", ylabel=rf"$\rho$ error at $t={max(times):g}$ (rel. $L^2$)")
        axes[1].set(xlabel="density unknowns", ylabel=r"anode current error (rel. $L^2$ in $t$)")
        axes[2].set(xlabel="density unknowns", ylabel=r"charge error (max, rel.)")
        axes[0].legend(frameon=False, fontsize=6)
        save_publication_figure(figure, Path(stem))
        plt.close(figure)
    return result


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--stem", type=Path, required=True)
    parser.add_argument("--times", type=_floats, default=(10., 20., 30., 40.))
    parser.add_argument("--pixels", type=int, default=1600)
    parser.add_argument("--t-max", type=float, default=None, help="compare histories up to this time only")
    args = parser.parse_args()
    result = analyse(args.reference, args.runs, args.stem, times=args.times, pixels=args.pixels, t_max=args.t_max)
    for row in result["runs"]:
        print(f"p={row['order']} dt={row['dt']:g} K={row['triangles']:>7d} N={row['density_unknowns']:>9d} "
              f"rho@{max(result['times']):g} {row['density_l2'].get(max(result['times']), float('nan')):.3e} "
              f"I {row['current_l2']:.3e} M {row['charge_max']:.3e} balance {row['balance']:.1e} "
              f"C {row['courant_max']:.1f} explicit x{row['explicit_advective_step_ratio']:.0f} (adv) "
              f"x{row['explicit_diffusive_step_ratio_estimate']:.0f} (diff, est.)")


if __name__ == "__main__":
    main()
