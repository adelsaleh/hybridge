"""Correctness of HYBRIDGE's condensed CG Poisson solver against DOLFINx on the same mesh.

Run in the FEniCSx environment (it imports HYBRIDGE from the source tree)::

    cd ~/src/hybridge && ~/src/.envs/diocotron-fenicsx/bin/python -m projects.diocotron.comparisons.cg_poisson_check

Both codes solve the same Galerkin problem: continuous P_k, Dirichlet data g of degree 2 <= k
(interpolation is exact whatever the nodes) and a source of degree 2 <= p (its DG(p) projection is
exact). The discrete solutions are therefore the same function up to solver round-off, although
HYBRIDGE uses an equispaced lattice basis with static condensation and DOLFINx its own P_k basis with
a MUMPS LU solve. Compared: the potential at points inside every cell (DOLFINx cells mapped through
``original_cell_index``) and the energy (grad phi, grad phi), on the unit square and on a coarse
parallel-vane magnetron mesh (D7 geometry). Prints one JSON record per case; exits 1 on a mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

TOLERANCE = 1e-9  # relative, potential and energy


def _g(x, y):
    return 0.3 * x * x - x * y + 0.5 * y - 0.2


def _rho(x, y):
    return 1.0 + 2.0 * x - 0.5 * x * y


def _meshes(names):
    from hybridge import rectangle_mesh
    from hybridge.core.mesh import gmsh_vane_anode_mesh

    for name in names:
        if name == "square":
            yield name, rectangle_mesh(6, 6, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        elif name == "vanes":
            yield name, gmsh_vane_anode_mesh(0.12, vane_width=0.26, cache=False, log_cache=False)
        else:
            raise ValueError(f"unknown mesh {name!r}")


def _hybridge(mesh, degree):
    from hybridge import DGSpace
    from hybridge.solvers.cg_poisson import CGPoissonOptions, CGPoissonSolver

    space = DGSpace(mesh, max(degree - 1, 2))  # holds the degree-2 source exactly
    solver = CGPoissonSolver(space, boundary_condition=_g, options=CGPoissonOptions(degree=degree, solver="direct"),
                             flux_space=DGSpace(mesh, degree - 1))
    result = solver.solve(space.project_callable(_rho))
    qx, qy = result.flux.components
    energy = float(np.einsum("K,Kq,q->", mesh.aff_jacs, qx.values() ** 2 + qy.values() ** 2,
                             qx.space.quad_data.Krf_w))
    return DGSpace(mesh, degree), result, energy


def _dolfinx(points, triangles, degree):
    import basix.ufl
    import ufl
    from dolfinx import fem
    from dolfinx import mesh as dmesh
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI

    domain = dmesh.create_mesh(MPI.COMM_SELF, np.asarray(triangles, dtype=np.int64),
                               basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)),
                               np.asarray(points, dtype=np.float64))
    domain.topology.create_connectivity(1, 2)
    space = fem.functionspace(domain, ("Lagrange", degree))
    data = fem.Function(space)
    data.interpolate(lambda X: _g(X[0], X[1]))
    boundary = fem.locate_dofs_topological(space, 1, dmesh.exterior_facet_indices(domain.topology))
    x = ufl.SpatialCoordinate(domain)
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    problem = fem_petsc.LinearProblem(
        ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx,
        (1.0 + 2.0 * x[0] - 0.5 * x[0] * x[1]) * test * ufl.dx,
        bcs=[fem.dirichletbc(data, boundary)],
        petsc_options_prefix="cg_poisson_check_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    )
    solution = problem.solve()
    solution.x.scatter_forward()
    energy = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(solution), ufl.grad(solution)) * ufl.dx))
    return domain, solution, float(energy)


def check(mesh_name, mesh, degree) -> dict:
    """Solve with both codes and compare; return the record."""
    target, result, energy_hybridge = _hybridge(mesh, degree)
    reference = np.array([[-0.6, -0.6], [0.2, -0.7], [-0.7, 0.3], [-0.1, -0.2], [0.5, -0.9], [-0.95, 0.85]])
    values_hybridge = result.potential.to_dg_field(target).values_at_ref(reference)  # (K, n)
    physical = mesh.map_reference_points(reference)  # (K, n, 2)
    points = np.asarray(mesh.node_coords, dtype=np.float64)[:, :2]
    triangles = np.asarray(mesh.triangles)[:, :3]
    domain, solution, energy_dolfinx = _dolfinx(points, triangles, degree)
    local_of = np.empty(triangles.shape[0], dtype=np.int32)
    local_of[np.asarray(domain.topology.original_cell_index)] = np.arange(triangles.shape[0], dtype=np.int32)
    flat = physical.reshape(-1, 2)
    cells = np.repeat(local_of, reference.shape[0])
    values_dolfinx = solution.eval(np.column_stack((flat, np.zeros(len(flat)))), cells).reshape(values_hybridge.shape)
    scale = max(float(np.max(np.abs(values_dolfinx))), 1.0)
    potential_error = float(np.max(np.abs(values_dolfinx - values_hybridge))) / scale
    energy_error = abs(energy_hybridge - energy_dolfinx) / max(abs(energy_dolfinx), 1e-300)
    return {"mesh": mesh_name, "triangles": int(triangles.shape[0]), "degree": degree,
            "potential_relative_max_difference": potential_error, "energy_hybridge": energy_hybridge,
            "energy_dolfinx": energy_dolfinx, "energy_relative_difference": energy_error,
            "passed": bool(potential_error < TOLERANCE and energy_error < TOLERANCE)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrees", default="2,3,5,7", help="continuous degrees k")
    parser.add_argument("--meshes", default="square,vanes")
    args = parser.parse_args(argv)
    degrees = [int(value) for value in args.degrees.split(",") if value]
    passed = True
    for mesh_name, mesh in _meshes(args.meshes.split(",")):
        for degree in degrees:
            record = check(mesh_name, mesh, degree)
            passed &= record["passed"]
            print(json.dumps(record), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
