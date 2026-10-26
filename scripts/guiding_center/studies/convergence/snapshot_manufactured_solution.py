#!/usr/bin/env python3
"""Run one manufactured guiding-center configuration and save raster snapshots for figures.

Uses the configuration of ``run_guiding_center_temporal_convergence.py`` (fixed tau,
conflict-averaged upwind, device or host by ``--backend``) and writes an NPZ with the
exact density at t = 0, the numerical and exact density and potential at t = T on a
pixel grid (exact DG evaluation via ``hybridge.io.raster.RasterGeometry``), and the mesh.
Time integration: run only with the user's approval (it is a simulation).
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from hybridge.io.raster import RasterGeometry
from scripts.guiding_center.studies.convergence.run_guiding_center_temporal_convergence import (
    MESH_FAMILIES, _manufactured_config, _num_steps, _parse_case_params, _run_case, _resolve_backend,
)
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key


def _raster_values(field, geometry: RasterGeometry) -> np.ndarray:
    """Exact DG values at the owned pixels, NaN outside the mesh."""
    values = np.full(geometry.width * geometry.height, np.nan)
    sampled = geometry.sampling_matrix(field.space) @ np.asarray(field.coeffs).reshape(-1)
    values[geometry.valid_pixels] = np.asarray(sampled)[geometry.valid_pixels]
    return values.reshape(geometry.height, geometry.width)


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default="rho_helm_wave")
    parser.add_argument("--case-param", action="append", default=None)
    parser.add_argument("--scheme", default="imex-ark3")
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--mesh-size", type=float, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--final-time", type=float, required=True)
    parser.add_argument("--poisson-tau", type=float, default=1.0)
    parser.add_argument("--backend", choices=("auto", "device", "host"), default="auto")
    parser.add_argument("--device-profile", default="auto")
    parser.add_argument("--mesh-family", choices=MESH_FAMILIES, default="gmsh")
    parser.add_argument("--poisson-order-offset", type=int, default=0, help="Poisson degree p + offset (raw drift).")
    parser.add_argument("--resolution", type=int, default=1200, help="Raster width and height in pixels.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case_params = _parse_case_params(args.case_param)
    num_steps = _num_steps(args.final_time, args.dt)
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output.stem
    config = _manufactured_config(
        preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy"), scheme=args.scheme, dt=args.dt,
        num_steps=num_steps, mesh_size=args.mesh_size, order=args.order, case_params=case_params,
        poisson_tau=args.poisson_tau, output_path=out_dir / "snapshot_runs", run_prefix=prefix,
        verbosity=0, backend=args.backend, device_profile=args.device_profile, case=args.case,
        mesh_family=args.mesh_family, poisson_order_offset=args.poisson_order_offset,
    )
    result = _run_case(config, prefix, _resolve_backend(args.backend, args.mesh_size, args.scheme))
    case = case_definition_by_key(args.case).build(**case_params)
    time_value = float(result.diagnostics[-1]["time"])
    geometry = RasterGeometry.from_mesh(result.mesh, args.resolution, args.resolution)
    points = geometry.points if geometry.points is not None else None
    valid = geometry.valid_pixels

    def exact_image(function):
        values = np.full(geometry.width * geometry.height, np.nan)
        values[valid] = np.asarray(function(points[valid, 0], points[valid, 1]))
        return values.reshape(geometry.height, geometry.width)

    np.savez_compressed(
        args.output,
        bounds=np.asarray(geometry.bounds), time=time_value, order=args.order, mesh_size=args.mesh_size,
        dt=args.dt, scheme=args.scheme, case=args.case, case_params=str(case_params),
        mesh_family=args.mesh_family, poisson_order=args.order + args.poisson_order_offset,
        triangles=int(result.mesh.num_tri),
        rho_exact_initial=exact_image(case.exact_density_at(0.0)),
        rho_exact_final=exact_image(case.exact_density_at(time_value)),
        rho_numerical_final=_raster_values(result.final_density, geometry),
        phi_exact_final=exact_image(case.exact_potential_at(time_value)),
        phi_numerical_final=_raster_values(result.final_potential, geometry),
        nodes=np.asarray(result.mesh.node_coords), cells=np.asarray(result.mesh.triangles),
        rho_l2_error=float(result.diagnostics[-1]["rho_l2_error"]),
    )
    print(f"snapshot: {args.output} (T={time_value:g}, K={result.mesh.num_tri}, "
          f"rho L2 error {float(result.diagnostics[-1]['rho_l2_error']):.3e})")


if __name__ == "__main__":
    main()
