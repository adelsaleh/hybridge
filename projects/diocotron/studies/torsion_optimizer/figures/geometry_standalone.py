#!/usr/bin/env python3
"""Standalone MPI torsion/target overview with no optimizer-side imports."""

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

import meshio
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fem_petsc
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py import PETSc


def gather_fields(functions: list[fem.Function]) -> tuple[np.ndarray, np.ndarray] | None:
    V = functions[0].function_space
    comm = V.mesh.comm
    owned = int(V.dofmap.index_map.size_local)
    coordinates = np.asarray(V.tabulate_dof_coordinates()[:owned], dtype=float)
    values = np.vstack([np.asarray(function.x.array[:owned], dtype=float) for function in functions])
    gathered = comm.gather((coordinates, values), root=0)
    if comm.rank:
        return None
    coordinates = np.concatenate([part[0] for part in gathered])
    values = np.concatenate([part[1] for part in gathered], axis=1)
    order = np.lexsort(tuple(coordinates[:, axis] for axis in range(coordinates.shape[1] - 1, -1, -1)))
    return coordinates[order], values[:, order]


def interpolate(function: fem.Function, expression) -> None:
    points = function.function_space.element.interpolation_points
    function.interpolate(fem.Expression(expression, points))
    function.x.scatter_forward()


def solve_poisson(V, bc, dx, rhs, prefix: str) -> tuple[fem.Function, float]:
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    problem = fem_petsc.LinearProblem(
        ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx,
        rhs * test * dx,
        bcs=[bc],
        petsc_options_prefix=prefix,
        petsc_options={
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    started = time.perf_counter()
    solution = problem.solve()
    solution.x.scatter_forward()
    return solution, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--order", required=True, type=int)
    parser.add_argument("--alphaT1", required=True, type=float)
    parser.add_argument("--alphaT2", required=True, type=float)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-tag", default="geometry-overview")
    args = parser.parse_args(argv)
    comm = MPI.COMM_WORLD
    started = time.perf_counter()
    run_dir = args.run_dir.resolve()
    out_dir, logs_dir = run_dir / "out", run_dir / "logs"
    if comm.rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    mesh_started = time.perf_counter()
    domain = gmshio.read_from_msh(args.mesh.resolve(), comm, rank=0, gdim=2).mesh
    mesh_seconds = time.perf_counter() - mesh_started
    V = fem.functionspace(domain, ("Lagrange", args.order))
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, V)
    qdeg = max(2 * args.order + 8, 12)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": qdeg})

    torsion, torsion_seconds = solve_poisson(V, bc, dx, 1.0, "overview_torsion_")
    local_max = float(np.max(torsion.x.array[:V.dofmap.index_map.size_local]))
    torsion_max = comm.allreduce(local_max, op=MPI.MAX)
    c1_t, c2_t = args.alphaT1 * torsion_max, args.alphaT2 * torsion_max
    eps_t = 0.08 * (c2_t - c1_t)
    mask = ufl.conditional(
        ufl.gt(torsion, c1_t), ufl.conditional(ufl.lt(torsion, c2_t), 1.0, 0.0), 0.0,
    )
    target_band, target_density = fem.Function(V), fem.Function(V)
    interpolate(target_band, mask)
    interpolate(target_density, mask)
    target_potential, target_seconds = solve_poisson(V, bc, dx, mask, "overview_target_")
    local_phi_max = float(np.max(target_potential.x.array[:V.dofmap.index_map.size_local]))
    target_potential_max = comm.allreduce(local_phi_max, op=MPI.MAX)
    target_area = fem.assemble_scalar(fem.form(mask * dx))
    target_area = comm.allreduce(target_area, op=MPI.SUM)

    gathered = gather_fields([torsion, target_band, target_density, target_potential])
    total_seconds = time.perf_counter() - started
    if comm.rank == 0:
        coordinates, values = gathered
        source = meshio.read(args.mesh.resolve())
        triangles = np.vstack([block.data for block in source.cells if block.type == "triangle"])
        state = {
            "stage": "design", "c1": c1_t, "c2": c2_t,
            "width": c2_t - c1_t, "eps_phi": eps_t,
            "homotopy_lambda": 0.0, "outer_iteration": -1,
        }
        metadata = {
            "format": "hybridge_torsion_optimizer_trajectory_v2",
            "run_tag": args.run_tag, "mesh_path": str(args.mesh.resolve()),
            "order": args.order, "quadrature_degree": qdeg,
            "alpha_t1": args.alphaT1, "alpha_t2": args.alphaT2,
            "c1_t": c1_t, "c2_t": c2_t, "rho_amp": 1.0,
            "complete": True, "terminal_stage": "design",
            "terminal_status": "OVERVIEW_COMPLETE", "states": [state],
        }
        np.savez_compressed(
            out_dir / "overview.npz",
            mesh_points=np.asarray(source.points[:, :2], dtype=float), mesh_cells=triangles,
            dof_coordinates=coordinates, fixed_torsion=values[0], fixed_target_band=values[1],
            fixed_target_density=values[2], fixed_target_potential=values[3],
            states_phi=values[3][None, :], states_rho=values[2][None, :],
            states_mismatch=np.zeros((1, values.shape[1])),
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        summary = {
            "format": "hybridge_torsion_optimizer_geometry_overview_v1",
            "status": "OVERVIEW_COMPLETE", "mesh": str(args.mesh.resolve()),
            "order": args.order, "num_dofs": int(V.dofmap.index_map.size_global),
            "num_cells": int(domain.topology.index_map(domain.topology.dim).size_global),
            "alpha_t1": args.alphaT1, "alpha_t2": args.alphaT2,
            "torsion_max": torsion_max, "c1_t": c1_t, "c2_t": c2_t,
            "eps_t": eps_t, "target_area": target_area,
            "target_potential_max": target_potential_max,
            "trajectory": str((out_dir / "overview.npz").resolve()),
            "total_seconds": total_seconds,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with (logs_dir / "phases.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("phase", "elapsed", "calls", "detail"))
            writer.writeheader()
            writer.writerows((
                {"phase": "mesh", "elapsed": mesh_seconds, "calls": 1, "detail": "gmsh read"},
                {"phase": "torsion", "elapsed": torsion_seconds, "calls": 1, "detail": "MUMPS"},
                {"phase": "target_solve", "elapsed": target_seconds, "calls": 1, "detail": "sharp indicator"},
                {"phase": "total", "elapsed": total_seconds, "calls": 1, "detail": "overview"},
            ))
    comm.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
