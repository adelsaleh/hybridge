#!/usr/bin/env python3
"""MPI geometry/target-field overview for the torsion optimizer study."""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import argparse
import csv
import json
import time
from pathlib import Path

import ufl
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import (
    assemble_scalar, boundary_bc, global_minmax, update_interpolated,
)
from projects.diocotron.dolfinx.torsion.optimization.homotopy import FixedStiffnessSolver, TrajectoryRecorder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--order", required=True, type=int)
    parser.add_argument("--alphaT1", required=True, type=float)
    parser.add_argument("--alphaT2", required=True, type=float)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-tag", default="geometry-overview")
    args = parser.parse_args(argv)
    if not (0.0 < args.alphaT1 < args.alphaT2 < 1.0):
        raise ValueError("require 0 < alphaT1 < alphaT2 < 1")

    comm = MPI.COMM_WORLD
    started = time.perf_counter()
    out_dir, logs_dir = args.run_dir.resolve() / "out", args.run_dir.resolve() / "logs"
    if comm.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    mesh_started = time.perf_counter()
    domain = gmshio.read_from_msh(args.mesh.resolve(), comm, rank=0, gdim=2).mesh
    mesh_seconds = time.perf_counter() - mesh_started
    V = fem.functionspace(domain, ("Lagrange", args.order))
    qdeg = max(2 * args.order + 8, 12)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": qdeg})
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    solver = FixedStiffnessSolver(
        ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx, V, [boundary_bc(V)],
        prefix="overview_stiffness_", solver="mumps", ksp_type="preonly",
        rtol=1.0e-12, atol=1.0e-14, max_it=200,
    )

    torsion_started = time.perf_counter()
    torsion = fem.Function(V, name="T")
    torsion_its, torsion_rel, _ = solver.solve_form(test * dx, torsion)
    _, torsion_max = global_minmax(comm, torsion)
    c1_t, c2_t = args.alphaT1 * torsion_max, args.alphaT2 * torsion_max
    eps_t = 0.08 * (c2_t - c1_t)
    mask = ufl.conditional(
        ufl.gt(torsion, c1_t), ufl.conditional(ufl.lt(torsion, c2_t), 1.0, 0.0), 0.0,
    )
    target_band, target_density = fem.Function(V), fem.Function(V)
    update_interpolated(target_band, mask)
    update_interpolated(target_density, mask)
    target_area = assemble_scalar(comm, mask * dx)
    torsion_seconds = time.perf_counter() - torsion_started

    target_started = time.perf_counter()
    target_potential = fem.Function(V, name="phiTarget")
    target_its, target_rel, _ = solver.solve_form(mask * test * dx, target_potential)
    _, target_potential_max = global_minmax(comm, target_potential)
    target_seconds = time.perf_counter() - target_started

    recorder = TrajectoryRecorder(
        enabled=True, every=1, output=out_dir / "overview.npz",
        mesh_path=args.mesh.resolve(), comm=comm,
    )
    recorder.set_fixed(
        torsion=torsion, target_band=target_band,
        target_density=target_density, target_potential=target_potential,
    )
    recorder.add(
        stage="design", phi=target_potential, rho=target_density,
        c1=c1_t, c2=c2_t, eps_phi=eps_t, homotopy_lambda=0.0, force=True,
    )
    recorder.write({
        "run_tag": args.run_tag, "order": args.order, "quadrature_degree": qdeg,
        "alpha_t1": args.alphaT1, "alpha_t2": args.alphaT2,
        "c1_t": c1_t, "c2_t": c2_t, "rho_amp": 1.0, "complete": True,
        "terminal_stage": "design", "terminal_status": "OVERVIEW_COMPLETE",
    })
    solver.close()
    total_seconds = time.perf_counter() - started

    if comm.rank == 0:
        summary = {
            "format": "hybridge_torsion_optimizer_geometry_overview_v1",
            "status": "OVERVIEW_COMPLETE", "mesh": str(args.mesh.resolve()),
            "order": args.order, "num_dofs": int(V.dofmap.index_map.size_global),
            "num_cells": int(domain.topology.index_map(domain.topology.dim).size_global),
            "alpha_t1": args.alphaT1, "alpha_t2": args.alphaT2,
            "torsion_max": torsion_max, "c1_t": c1_t, "c2_t": c2_t,
            "eps_t": eps_t, "target_area": target_area,
            "target_potential_max": target_potential_max,
            "torsion_iterations": int(torsion_its),
            "torsion_relative_residual": float(torsion_rel),
            "target_iterations": int(target_its),
            "target_relative_residual": float(target_rel),
            "trajectory": str((out_dir / "overview.npz").resolve()),
            "total_seconds": total_seconds,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        with (logs_dir / "phases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("phase", "elapsed", "calls", "detail"))
            writer.writeheader()
            writer.writerows((
                {"phase": "mesh", "elapsed": mesh_seconds, "calls": 1, "detail": "gmsh read"},
                {"phase": "torsion", "elapsed": torsion_seconds, "calls": 1, "detail": "Poisson"},
                {"phase": "target_solve", "elapsed": target_seconds, "calls": 1, "detail": "sharp indicator"},
                {"phase": "total", "elapsed": total_seconds, "calls": 1, "detail": "overview"},
            ))
    comm.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
