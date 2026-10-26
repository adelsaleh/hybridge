#!/usr/bin/env python3
"""Density convergence of the travelling wave with the exact drift (diagnostic for the GC study).

Integrates ``rho_t + u . grad rho = 0`` with the exact drift ``u = -grad^perp phi`` of
``rho_helm_wave``, evaluated at the volume and face quadrature points of every solve: no
projection into a polynomial space and no Poisson solve. The exact density solves this
equation, so the error isolates the HDG transport discretization (mesh, degree, upwinding)
from the discrete electric field. With a continuous drift ``u.n`` is single-valued, so
the upwind flux needs no conflict averaging.

Time stepping: BDF3, one transport solve per step at ``t^{n+1}``, started from the L2
projections of the exact density at ``0, dt, 2 dt``; inflow traces are the exact density.
BDF3's phase error for this wave is about ``(3/22) (k_x U)^4 dt^3 T ||rho||``, so the time
steps (``--dt-table``) must be chosen small enough for the spatial error to dominate.

Assembly: CuPy, callables sampled on the host at the quadrature points. Global solve: AMGX
with the fastest recorded transport configuration (``fast_profiles.TRANSPORT_AMGX_CONFIG``:
scaled BiCGSTAB, no preconditioner), relative tolerance ``1e-11``, warm-started from the
previous trace.

Meshes: ``structured`` as ``--mesh-family structured`` of the convergence driver
(``nx = ny = round(2/h)`` squares split along their (-1,1) diagonal) or Gmsh. Errors: L2 in
Omega and in the interior square ``(-0.6, 0.6)^2``, as the driver's ``rho_l2_interior_error``, with the space's
volume rule (exact for degree 2p, which underestimates them) and with 2p + 2 collapsed-Gauss points per
direction (``*_accurate``). The final density (coefficients, trace, mesh) is saved next to the output CSV
(``<output stem>_fields/``).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hybridge import DGSpace, solve_advection_reaction_hdg
from hybridge.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hybridge.diagnostics.errors import evaluate_scalar_error
from scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence import (
    INTERIOR_HALF_WIDTH,
    _parse_case_params,
)
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.runtime.fast_profiles import TRANSPORT_AMGX_CONFIG

BDF3 = (11.0 / 6.0, (3.0, -1.5, 1.0 / 3.0))  # (leading coefficient, history coefficients)


def _mesh(family: str, mesh_size: float):
    if family == "structured":
        cells = int(round(2.0 / mesh_size))
        return rectangle_mesh(cells, cells, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    return gmsh_rectangle_mesh(mesh_size, log_cache=False)


def _inside(x, y):
    xp = np if isinstance(x, np.ndarray) else __import__("cupy")
    return ((xp.abs(x) <= INTERIOR_HALF_WIDTH) & (xp.abs(y) <= INTERIOR_HALF_WIDTH)).astype(x.dtype)


def run(case, *, order: int, mesh_size: float, family: str, dt: float, final_time: float,
        amgx_config: dict, rtol: float, log=print, log_every: int = 10, verbosity: int = 1,
        fields_dir: Path | None = None) -> dict:
    """One (p, h) run. ``verbosity``: 0 quiet; 1 step lines every ``log_every`` steps (t, dt, AMGX
    iterations, solve time, rho L2 error, ETA); 2 adds the solver's stage lines; 3 every timing
    (solver ``verbose=3``: backend timings and AMGX iteration tables)."""
    solver_verbose = {0: 0, 1: 0, 2: 1, 3: 3}[int(verbosity)]
    mesh = _mesh(family, mesh_size)
    space = DGSpace(mesh, order)
    steps = max(3, int(math.ceil(final_time / dt - 1.0e-12)))
    dt = final_time / steps
    leading, history_weights = BDF3
    reaction = leading / dt
    history = [space.project_callable(lambda x, y, t=n * dt: case.exact_density(x, y, t)) for n in range(3)]
    trace, solve_times, iterations = None, [], []
    log(f"  start p={order} triangles={mesh.num_tri} dt={dt:.4e} steps={steps}")
    started = time.perf_counter()
    for n in range(2, steps):
        t = (n + 1) * dt

        def drift_x(x, y, t=t):
            return -case.exact_flux(x, y, t)[1]  # u = q^perp = (-q_y, q_x), q = -grad(phi)

        def drift_y(x, y, t=t):
            return case.exact_flux(x, y, t)[0]

        # BDF3: (11/6 rho^{n+1} - 3 rho^n + 3/2 rho^{n-1} - 1/3 rho^{n-2}) / dt + u . grad rho^{n+1} = 0.
        source = (history_weights[0] / dt) * history[2]
        for weight, field in zip(history_weights[1:], (history[1], history[0])):
            source = source + (weight / dt) * field
        solve_start = time.perf_counter()
        result = solve_advection_reaction_hdg(
            source, (drift_x, drift_y), reaction, lambda x, y, t=t: case.exact_density(x, y, t), space,
            assembly_backend="cupy", solver="amgx", amgx_config=amgx_config, preconditioner=None,
            solver_rtol=rtol, scale_system=True, boundary_mode="eliminate", initial_guess=trace,
            verbose=solver_verbose,
        )
        solve_times.append(time.perf_counter() - solve_start)
        solve = result.global_solve_result
        iterations.append(-1 if solve is None or solve.iteration_count is None else int(solve.iteration_count))
        trace = result.trace if result.trace is not None else result.trace_device
        trace = trace.get().ravel() if hasattr(trace, "get") else np.asarray(trace).ravel()
        history = [history[1], history[2], result.field]
        done = n - 1  # BDF3 solves so far; levels 0, 1, 2 are projected
        if verbosity >= 1 and (done % log_every == 0 or n + 1 == steps):
            elapsed = time.perf_counter() - started
            error = float(result.field.l2_error(lambda x, y, t=t: case.exact_density(x, y, t)))
            log(f"  step {n + 1:5d}/{steps} t={t:.6f} dt={dt:.4e} amgx_its={iterations[-1]} "
                f"solve={solve_times[-1]:.3f}s rho_l2={error:.4e} elapsed={elapsed:.0f}s "
                f"eta={elapsed / done * (steps - n - 1):.0f}s")
    wall = time.perf_counter() - started
    exact = lambda x, y: case.exact_density(x, y, final_time)  # noqa: E731
    density = history[-1]
    points = 2 * int(order) + 2
    accurate = {
        "rho_l2_error_accurate": float(evaluate_scalar_error(density, exact, volume_quad_1d=points).metrics.l2),
        "rho_l2_interior_error_accurate": float(
            evaluate_scalar_error(density, exact, weight=_inside, volume_quad_1d=points).metrics.l2),
        "accurate_quad_1d": points,
    }
    if fields_dir is not None:
        try:
            fields_dir.mkdir(parents=True, exist_ok=True)
            np.savez(fields_dir / f"{family}_p{order}_h{mesh_size:g}_dt{dt:.6g}_final.npz",
                     density=np.asarray(density.coeffs), trace=np.asarray(trace), time=final_time, order=order,
                     node_coords=np.asarray(mesh.node_coords), triangles=np.asarray(mesh.triangles),
                     edges=np.asarray(mesh.edges), interior_edges=np.asarray(mesh.int_edges_inds), dt=dt,
                     mesh_family=family, mesh_size=mesh_size)
        except Exception as exc:  # noqa: BLE001 - never lose a finished run to its archive
            log(f"  final fields not saved: {exc!r}")
    return {
        "mesh_family": family, "order": order, "mesh_size": mesh_size, "triangles": mesh.num_tri,
        "dt": dt, "num_steps": steps, "final_time": final_time, "scheme": "bdf3-exact-drift",
        "rho_l2_error": float(density.l2_error(exact)),
        "rho_l2_interior_error": float(evaluate_scalar_error(density, exact, weight=_inside).metrics.l2),
        **accurate,
        "wall_time_s": wall, "step_time_median_s": float(np.median(solve_times)),
        "amgx_iterations_max": max(iterations) if iterations else -1,
    }


def _rates(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["mesh_family"], row["order"], row.get("case_params", "")), []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: r["triangles"])
        for previous, row in zip([None] + group[:-1], group):
            for key in ("rho_l2_error", "rho_l2_interior_error", "rho_l2_error_accurate",
                        "rho_l2_interior_error_accurate"):
                if key not in row or (previous is not None and key not in previous):
                    continue
                row[key.replace("_error", "_rate")] = None if previous is None else math.log(
                    previous[key] / row[key]) / math.log(math.sqrt(row["triangles"] / previous["triangles"]))


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", default="2,3,4")
    parser.add_argument("--mesh-sizes", default="0.04,0.02,0.01,0.005")
    parser.add_argument("--mesh-family", choices=("structured", "gmsh"), default="structured")
    parser.add_argument("--case-param", action="append", default=[],
                        help="rho_helm_wave parameters, e.g. kx=8*pi, ky=-8*pi, U=1, A=0.0281...")
    parser.add_argument("--final-time", type=float, default=0.0625)
    parser.add_argument("--dt", type=float, default=1.0e-3, help="Time step for (p,h) pairs not in --dt-table.")
    parser.add_argument("--dt-table", type=Path, default=None, help='JSON {"p,h": dt}.')
    parser.add_argument("--amgx-config", type=Path, default=TRANSPORT_AMGX_CONFIG)
    parser.add_argument("--solver-rtol", type=float, default=1.0e-11)
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV path (a JSON copy and a line-buffered step log <output>.log are written next to it).")
    parser.add_argument("--log-every", type=int, default=10, help="Log every this many time steps (and the last).")
    parser.add_argument("-v", "--verbosity", type=int, choices=(0, 1, 2, 3), default=1,
                        help="0 quiet, 1 step lines, 2 + solver stage lines, 3 every timing.")
    parser.add_argument("--combine", type=Path, nargs="*", default=[],
                        help="Only merge these per-run CSVs into --output and recompute the orders.")
    args = parser.parse_args()
    if args.combine:
        rows = []
        for path in args.combine:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append({key: (int(value) if key in {"order", "triangles", "num_steps", "amgx_iterations_max"}
                                       else float(value) if key.endswith(("_error", "_s", "dt", "time", "size"))
                                       else value) for key, value in row.items() if not key.endswith("_rate")})
        _write(rows, args.output)
        return
    case = case_definition_by_key("rho_helm_wave").build(**_parse_case_params(args.case_param))
    table = json.loads(args.dt_table.read_text()) if args.dt_table else {}
    amgx_config = json.loads(args.amgx_config.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.with_suffix(".log").open("w", buffering=1, encoding="utf-8") as log_file:

        def log(message: str) -> None:
            print(message, flush=True)
            log_file.write(message + "\n")

        for order in (int(v) for v in args.orders.split(",")):
            for mesh_size in (float(v) for v in args.mesh_sizes.split(",")):
                dt = float(table.get(f"{order},{mesh_size:g}", args.dt))
                row = run(case, order=order, mesh_size=mesh_size, family=args.mesh_family, dt=dt,
                          final_time=args.final_time, amgx_config=amgx_config, rtol=args.solver_rtol,
                          log=log, log_every=max(1, args.log_every), verbosity=args.verbosity,
                          fields_dir=args.output.parent / f"{args.output.stem}_fields")
                row["case_params"] = ";".join(args.case_param)
                rows.append(row)
                log(f"p={order} K={row['triangles']:7d} dt={row['dt']:.3e} steps={row['num_steps']:5d} "
                    f"rho={row['rho_l2_error']:.4e} interior={row['rho_l2_interior_error']:.4e} "
                    f"step={row['step_time_median_s']:.3f}s wall={row['wall_time_s']:.0f}s")
    _write(rows, args.output)


def _write(rows: list[dict], output: Path) -> None:
    _rates(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        rates = " ".join(f"{k[:-5]}={row[k]:.2f}" for k in ("rho_l2_rate", "rho_l2_interior_rate")
                         if row.get(k) is not None)
        print(f"p={row['order']} K={row['triangles']:7d} rho={row['rho_l2_error']:.3e} "
              f"interior={row['rho_l2_interior_error']:.3e} {rates}")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
