#!/usr/bin/env python3
r"""E9 inside D7: Poisson time per solve, HDG(p) against condensed CG(p+1), on the D7 vane meshes.

Every guiding-center step solves Poisson again with a new right-hand side for a constant operator.
For each mesh and transport degree p, after one untimed warm-up solve (operator, factorization or
AMG setup, reported as ``setup_s``), the median wall time of ``--repeats`` solves is recorded for

* ``hdg``: the production ``DiffusionReactionHDGSolver`` of the magnetron runs (fastest recorded
  device profile with zero Dirichlet data; otherwise, and above p = 6, raw-CUDA CSR + AMGX), whose
  raw flux q_h lies in DG(p)^2 (velocity order p+1 for tau = O(1); a large tau loses up to one order);
* ``cg-amgx`` / ``cg-pardiso``: :class:`hybridge.solvers.cg_poisson.CGPoissonSolver` of degree
  k = p + offset (``--cg-offsets``, default 1: equal velocity order), device AMGX or host PyPardiso
  SPD factorized once with 16 threads; q = -grad phi in DG(k-1)^2.

Sources: ``hub`` (the D7 initial density, zero Dirichlet data as in the runs, where the electrodes are
superposed separately) for timing, and ``manufactured`` (a smooth potential with exact Dirichlet
data) for the velocity L2 error at equal velocity order. Each timed solve alternates between two
right-hand sides, so no result can be reused. Rows go to ``<output>/poisson_cg_vs_hdg.jsonl``.

Performance study: run it alone on the GPU, under the campaign lock::

    flock ~/src/hybridge/.cache/gpu-campaign.lock .venv/bin/python -m \
        scripts.guiding_center.benchmarks.run_poisson_cg_vs_hdg --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def _phi(x, y):
    return np.cos(3.0 * x) * np.sin(2.0 * y) + 0.5 * x * y


def _rho(x, y):  # -lap phi
    return 13.0 * np.cos(3.0 * x) * np.sin(2.0 * y)


def _q(x, y):  # -grad phi
    return (3.0 * np.sin(3.0 * x) * np.sin(2.0 * y) - 0.5 * y,
            -2.0 * np.cos(3.0 * x) * np.cos(2.0 * y) - 0.5 * x)


def _case_and_mesh(args):
    from hybridge.core.mesh import gmsh_vane_anode_mesh
    from scripts.guiding_center.cases.guiding_center_cases import magnetron

    case = magnetron(anode_geometry="vanes", wave_mode=6, wave_amplitude=0.05, hub_edge=0.025, emission_top=0.6,
                     vane_width=args.vane_width, band_mesh_size=args.band_mesh_size,
                     refine_band=tuple(args.refine_band), corner_mesh_size=args.corner_mesh_size)
    g = case.parameters["geometry"]
    mesh = gmsh_vane_anode_mesh(
        args.mesh_size, cathode_radius=g["inner_radius"], anode_radius=g["anode_radius"],
        back_radius=g["back_radius"], vanes=g["vanes"], vane_fraction=g["vane_fraction"],
        vane_width=g.get("vane_width"), tip_radius=g.get("tip_radius"), band=g["band"],
        cathode_mesh_size=g.get("cathode_mesh_size"), cathode_grading=float(g.get("cathode_grading", 0.3)),
        band_mesh_size=g["band_mesh_size"],
        band_grading=g["band_grading"], corner_mesh_size=g["corner_mesh_size"],
        corner_grading=g["corner_grading"], log_cache=False)
    return case, mesh


def _hdg_options(order: int, triangles: int, zero_dirichlet: bool, tau: float | None = None):
    from scripts.guiding_center.benchmarks.run_magnetron import BASE_PRESET, _device_overrides
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.configuration import _make_poisson_options

    overrides = _device_overrides(order, triangles) if (zero_dirichlet and order <= 6) else _device_overrides(7, triangles)
    config = replace(preset_by_key(BASE_PRESET), **overrides, order=int(order))
    if tau is not None:  # LDG-H stabilization: 1 is the accuracy-optimal value, 1.28e5 the production one
        config = replace(config, poisson_tau=float(tau))
    return _make_poisson_options(config)


def _device_field(field, cp):
    from hybridge.core.device import field_from_cupy_coefficients

    return field_from_cupy_coefficients(field.space, cp.asarray(field.coeffs), device=0, name=field.name)


def _hdg_unknowns(mesh, order):
    """Free trace unknowns and nonzeros of the HDG Poisson trace system (Dirichlet traces eliminated)."""
    interior = np.zeros(mesh.num_edg, dtype=bool)
    interior[mesh.int_edges_inds] = True
    per_cell = interior[mesh.loc2glob_edge].sum(axis=1)
    block_nonzeros = int((per_cell ** 2).sum() - interior.sum())  # diagonal blocks shared by two cells
    return int(interior.sum()) * (order + 1), block_nonzeros * (order + 1) ** 2


def _timed(solve, repeats, sync):
    start = time.perf_counter()
    result = solve(0)
    sync()
    setup = time.perf_counter() - start
    times = []
    for index in range(repeats):
        start = time.perf_counter()
        result = solve(index + 1)
        sync()
        times.append(time.perf_counter() - start)
    return setup, statistics.median(times), result


def run_one(method, source_name, mesh, case, order, repeats, tau=None, cg_offset=1, cg_amgx_config=None):
    from hybridge import DGSpace
    from hybridge.diagnostics.errors import evaluate_scalar_error
    from hybridge.runtime.optional import require_cupy
    from hybridge.solvers.cg_poisson import CGPoissonOptions, CGPoissonSolver
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGSolver

    cp = require_cupy()
    space = DGSpace(mesh, order)
    if source_name == "hub":
        profile = case.initial_density_at()
        boundary = None
    else:
        profile, boundary = _rho, _phi
    first = space.project_callable(profile, name="rho_a")
    second = space.project_callable(lambda x, y: (1.0 + 0.1 * x) * profile(x, y), name="rho_b")
    device = method in {"hdg", "cg-amgx"}
    sources = [_device_field(first, cp), _device_field(second, cp)] if device else [first, second]
    sync = cp.cuda.Device().synchronize if device else (lambda: None)
    memory_before = cp.get_default_memory_pool().used_bytes()
    row = {"method": method, "source": source_name, "order": order, "triangles": int(mesh.num_tri)}
    if method == "hdg":
        options = _hdg_options(order, int(mesh.num_tri), boundary is None, tau)
        solver = DiffusionReactionHDGSolver(space, source=sources[0], reaction=space.zeros(name="zero"),
                                            boundary_condition=boundary or (lambda x, y: 0.0 * x), options=options)

        def solve(index):
            solver.set_source(sources[index % 2])
            return solver.solve()

        setup, median, result = _timed(solve, repeats, sync)
        flux = result.flux
        global_result = getattr(result, "global_solve_result", None)
        row.update(iterations=getattr(global_result, "iteration_count", None), poisson_solver=options.solver,
                   poisson_tau=float(options.stabilization))
        row["global_unknowns"], row["nonzeros"] = _hdg_unknowns(mesh, order)
    else:
        solver = CGPoissonSolver(space, boundary_condition=boundary, options=CGPoissonOptions(
            degree=order + cg_offset, solver="amgx" if method == "cg-amgx" else "pypardiso",
            amgx_config=cg_amgx_config))

        def solve(index):
            return solver.solve(sources[index % 2])

        setup, median, result = _timed(solve, repeats, sync)
        flux = result.flux
        row.update(cg_degree=order + cg_offset,
                   cg_amgx_config=None if method != "cg-amgx" else str(cg_amgx_config or "default"),
                   iterations=result.iterations, global_unknowns=int(solver.operator.matrix_free.shape[0]),
                   nonzeros=solver.operator.nonzeros(), stage_seconds=result.timings)
    row.update(setup_s=setup, solve_median_s=median,
               cupy_pool_bytes=int(cp.get_default_memory_pool().used_bytes() - memory_before))
    if source_name == "manufactured":  # the last timed source is index repeats; use the first one for the error
        result = solve(0)
        flux = result.flux
        errors = [evaluate_scalar_error(component, (lambda i: (lambda x, y: _q(x, y)[i]))(i)).metrics.l2
                  for i, component in enumerate(flux.components)]
        row["velocity_l2_error"] = math.hypot(*errors)
    close = getattr(solver, "close", None)
    if close is not None:
        close()
    cp.get_default_memory_pool().free_all_blocks()
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orders", default="6,7")
    parser.add_argument("--methods", default="hdg,cg-amgx,cg-pardiso")
    parser.add_argument("--sources", default="hub,manufactured")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--taus", default="1,128000", help="HDG Poisson stabilizations (1: accuracy-optimal)")
    parser.add_argument("--cg-offsets", default="1", help="CG degrees k = p + offset (1: equal velocity order)")
    parser.add_argument("--cg-amgx-configs", default="default",
                        help="cg-amgx: comma-separated AMGX JSON paths, or 'default' (CGPoissonSolver's default)")
    parser.add_argument("--vane-width", type=float, default=0.26)
    parser.add_argument("--mesh-size", type=float, default=0.03)
    parser.add_argument("--band-mesh-size", type=float, default=0.015)
    parser.add_argument("--refine-band", type=float, nargs=2, default=(0.55, 1.35))
    parser.add_argument("--corner-mesh-size", type=float, default=0.005)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    case, mesh = _case_and_mesh(args)
    print(f"[e9] vane mesh: {mesh.num_tri} triangles", flush=True)
    with (output / "poisson_cg_vs_hdg.jsonl").open("a") as stream:
        for order in (int(value) for value in args.orders.split(",")):
            for source_name in args.sources.split(","):
                configs = [None if c == "default" else c for c in args.cg_amgx_configs.split(",")]
                runs = [(method, tau, offset, config) for method in args.methods.split(",")
                        for tau in ([float(t) for t in args.taus.split(",")] if method == "hdg" else [None])
                        for offset in ([None] if method == "hdg" else [int(o) for o in args.cg_offsets.split(",")])
                        for config in (configs if method == "cg-amgx" else [None])]
                for method, tau, offset, config in runs:
                    row = run_one(method, source_name, mesh, case, order, args.repeats, tau,
                                  1 if offset is None else offset, config)
                    row.update(mesh_size=args.mesh_size, band_mesh_size=args.band_mesh_size,
                               corner_mesh_size=args.corner_mesh_size, vane_width=args.vane_width)
                    stream.write(json.dumps(row, default=float) + "\n")
                    stream.flush()
                    label = f"{method}(tau={tau:g})" if tau is not None else f"{method}(k={order + offset})"
                    label += "" if config is None else f"[{Path(config).stem}]"
                    print(f"[e9] p={order} {source_name:12s} {label:18s} setup {row['setup_s']:.3f}s "
                          f"solve {row['solve_median_s'] * 1e3:.1f} ms it={row['iterations']} "
                          f"unknowns={row['global_unknowns']} nnz={row['nonzeros']}"
                          + (f" |q-q_h|={row['velocity_l2_error']:.3e}" if "velocity_l2_error" in row else ""),
                          flush=True)
    (output / "all-done").write_text(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
