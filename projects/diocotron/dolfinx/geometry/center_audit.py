"""Independent h/p audit of torsion maxima and recovered-gradient zeros.

This diagnostic deliberately does not construct an equiband ray atlas or solve
an equilibrium. The production ray evaluator is P2/CG1; passing higher-degree
fields to it would silently discard polynomial terms. Here every field is
evaluated at a complete, degree-matched triangular lattice instead. Raw grad(T)
and L2-recovered CG1/CG(p-1) fields are inspected separately. Curved triangular
coordinate maps are evaluated at their actual geometry degree. For raw torsion
zeros, grad_ref(T) is used: it vanishes exactly where grad_x(T) vanishes for a
nonsingular coordinate map, and remains polynomial on curved cells. The
physical gradient on a curved cell is NOT fitted as a polynomial.

All cells are screened with componentwise Bernstein coefficient bounds. Cells
which cannot contain a vector zero are discarded; the remaining cells undergo
batched NumPy multistart Newton refinement. Counts are numerical observations,
not an interval-arithmetic proof of uniqueness. Unresolved candidate cells are
reported, never silently declared zero-free. Near-duplicate roots (including
facet duplicates) are clustered at a stated physical tolerance.

Run from the repository root, in the FEniCSx environment, for example::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -m projects.diocotron.dolfinx.geometry.center_audit \
      --mesh projects/diocotron/runs/equiband/meshes/horseshoe_h003.msh \
      --degrees 2 3 4 6 --output projects/diocotron/runs/equiband/horseshoe_center_hp \
      --save-terminal-log -v 2

MPI distributes the FE solve and cell screening; only candidate roots and
scalar summaries are gathered. Existing output-directory prompting/archiving
and native terminal capture are reused from equiband. Results are separate
JSON files per degree, with software versions, mesh hash, residual checks and
explicit recovery degrees. No production defaults or guards are modified.
"""
from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import math
from pathlib import Path
import sys


def lattice(degree):
    """Complete equispaced sites on the closed reference triangle."""
    import numpy as np
    if degree < 1:
        raise ValueError("degree must be positive")
    return np.array([(i / degree, j / degree) for i in range(degree + 1)
                     for j in range(degree + 1 - i)], dtype=float)


def powers(degree):
    return [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]


def monomials(points, degree, derivative=(0, 0)):
    """Tabulate monomials or reference derivatives in one vectorized batch."""
    import numpy as np
    a, b = derivative
    x, y = points[..., 0], points[..., 1]
    columns = []
    for i, j in powers(degree):  # small, degree-dependent loop, not a cell loop
        if i < a or j < b:
            columns.append(np.zeros_like(x))
        else:
            factor = math.factorial(i) / math.factorial(i-a) * math.factorial(j) / math.factorial(j-b)
            columns.append(factor * x**(i-a) * y**(j-b))
    return np.stack(columns, axis=-1)


def bernstein(points, degree):
    """Nonnegative partition-of-unity basis; coefficient extrema bound values."""
    import numpy as np
    x, y = points[..., 0], points[..., 1]
    return np.stack([math.factorial(degree) / (math.factorial(i)*math.factorial(j)*math.factorial(degree-i-j))
                     * x**i * y**j * (1-x-y)**(degree-i-j) for i, j in powers(degree)], axis=-1)


def excludes_vector_zero(coefficients, padding):
    """Separate the origin from the convex hull of vector Bernstein coefficients.

    Componentwise ranges alone leave many false-positive cells: the zero of
    one component need not coincide with the zero of the other. A vector
    polynomial also lies in the joint convex hull of its Bernstein vectors.
    In two dimensions, a strict separating half-plane is found from the
    largest angular gap. The final positive projection test includes a
    rounding margin; ambiguous cells are retained for root refinement.
    """
    import numpy as np
    angles = np.sort(np.arctan2(coefficients[..., 1], coefficients[..., 0]), axis=1)
    gaps = np.diff(np.concatenate((angles, angles[:, :1]+2*np.pi), axis=1), axis=1)
    index = np.argmax(gaps, axis=1)
    largest = gaps[np.arange(len(gaps)), index]
    start = angles[np.arange(len(angles)), (index+1) % angles.shape[1]]
    midpoint = start+(2*np.pi-largest)/2
    normal = np.column_stack((np.cos(midpoint), np.sin(midpoint)))
    projections = np.einsum("nij,nj->ni", coefficients, normal)
    return (largest > np.pi) & (np.min(projections, axis=1) > 2*padding)


def vector_roots(coefficients, degree, tolerance=1e-11, seed_degree=None):
    """Numerically find zeros of cellwise vector polynomials in reference cells.

    ``coefficients`` has shape (cells, monomials, physical vector components).
    All cell/seed pairs are refined together. A root must have a small actual
    vector residual and lie in its reference triangle. Returning a root does
    not certify that no additional polynomial root exists in that cell.
    """
    import numpy as np
    seeds = lattice(seed_degree or max(3, 2*degree))
    n = len(coefficients)
    if n == 0:
        return np.empty(0, dtype=int), np.empty((0, 2)), np.empty(0)
    ids = np.repeat(np.arange(n), len(seeds))
    c = coefficients[ids]
    r = np.tile(seeds, (n, 1))
    active = np.ones(len(r), dtype=bool)
    for _ in range(50):
        value = np.einsum("ni,nij->nj", monomials(r, degree), c)
        jac = np.stack([np.einsum("ni,nij->nj", monomials(r, degree, d), c)
                        for d in ((1, 0), (0, 1))], axis=-1)
        determinant = np.linalg.det(jac)
        regular = np.abs(determinant) > 1e-14*np.max(np.abs(jac), axis=(1, 2))**2
        working = active & regular & (np.linalg.norm(value, axis=1) > tolerance)
        if not np.any(working):
            break
        step = np.linalg.solve(jac[working], value[working, :, None])[..., 0]
        # Limit excursions so a failed seed cannot overflow a high-degree fit.
        step *= np.minimum(1., 0.5 / np.maximum(np.linalg.norm(step, axis=1), 1e-300))[:, None]
        r[working] -= step
        active &= regular & (np.max(np.abs(r), axis=1) < 2.)
    residual = np.linalg.norm(np.einsum("ni,nij->nj", monomials(r, degree), c), axis=1)
    inside = (r[:, 0] >= -1e-9) & (r[:, 1] >= -1e-9) & (r.sum(axis=1) <= 1+1e-9)
    valid = active & inside & (residual <= tolerance)
    return ids[valid], r[valid], residual[valid]


def cluster_roots(records, tolerance):
    """Merge numerically coincident roots, retaining the best-residual record."""
    import numpy as np
    result = []
    for record in sorted(records, key=lambda row: row["residual"]):
        if not result or np.min(np.linalg.norm(np.array([r["point"] for r in result])-record["point"], axis=1)) > tolerance:
            result.append(record)
    return sorted(result, key=lambda row: tuple(row["point"]))


class FieldAudit:
    """Bounded batches of exact FE polynomial evaluations on owned cells."""

    def __init__(self, domain):
        import numpy as np
        import ufl
        from dolfinx.mesh import CellType
        from mpi4py import MPI
        if domain.topology.cell_type != CellType.triangle or domain.geometry.dim != 2:
            raise ValueError("torsion-center audit requires two-dimensional triangular geometry")
        self.domain, self.comm = domain, domain.comm
        self.ncells = domain.topology.index_map(2).size_local
        self.cells = np.arange(self.ncells, dtype=np.int32)
        self.geometry_degree = domain.geometry.cmaps[0].degree
        self.geometry_nodes = domain.geometry.x[domain.geometry.dofmaps[0][:self.ncells], :2]
        self.triangles = self.geometry_nodes[:, :3]
        self.origins = self.triangles[:, 0]
        self.matrix = np.stack((self.triangles[:, 1]-self.origins, self.triangles[:, 2]-self.origins), axis=-1)
        self.inverse = np.linalg.inv(self.matrix)
        low = np.min(self.comm.allgather(np.min(self.geometry_nodes, axis=(0, 1))), axis=0)
        high = np.max(self.comm.allgather(np.max(self.geometry_nodes, axis=(0, 1))), axis=0)
        self.diameter = float(np.linalg.norm(high-low))
        self.geometry_power = np.empty((self.ncells, len(powers(self.geometry_degree)), 2))
        for cells, _, _, power in self.coefficients(ufl.SpatialCoordinate(domain), self.geometry_degree):
            self.geometry_power[cells] = power
        self.minimum_abs_jacobian = float("inf")
        invalid_mapping = False
        sites = lattice(max(4, 2*self.geometry_degree))
        for start in range(0, self.ncells, 8192):
            cells = self.cells[start:start+8192]
            ids = np.repeat(cells, len(sites))
            jac = self.mapping_jacobian(ids, np.tile(sites, (len(cells), 1)))
            determinant = np.linalg.det(jac).reshape(len(cells), -1)
            invalid_mapping |= bool(not np.all(np.isfinite(determinant)) or
                                    np.any(np.min(determinant, axis=1)*np.max(determinant, axis=1) <= 0))
            self.minimum_abs_jacobian = min(self.minimum_abs_jacobian, float(np.min(np.abs(determinant))))
        if any(self.comm.allgather(invalid_mapping)):
            raise ValueError("INVALID_CURVED_CELL: sampled Jacobian changes sign, vanishes or is nonfinite")
        self.minimum_abs_jacobian = self.comm.allreduce(self.minimum_abs_jacobian, op=MPI.MIN)

    def mapping(self, cells, refs):
        import numpy as np
        return np.einsum("ni,nij->nj", monomials(refs, self.geometry_degree), self.geometry_power[cells])

    def mapping_jacobian(self, cells, refs):
        import numpy as np
        return np.stack([np.einsum("ni,nij->nj", monomials(refs, self.geometry_degree, d), self.geometry_power[cells])
                         for d in ((1, 0), (0, 1))], axis=-1)

    def pull_back(self, cell, point):
        """Newton inversion of the actual polynomial geometry, not its chords."""
        import numpy as np
        r = self.inverse[cell] @ (np.asarray(point)-self.origins[cell])
        for _ in range(12):
            delta = self.mapping([cell], r[None])[0]-point
            if np.linalg.norm(delta) < 1e-13:
                break
            r -= np.linalg.solve(self.mapping_jacobian([cell], r[None])[0], delta)
        return r

    def coefficients(self, expression, degree):
        """Yield cell IDs, samples and Bernstein/power coefficients in batches."""
        import numpy as np
        from dolfinx import fem
        sites = lattice(degree)
        sample = fem.Expression(expression, sites)
        inv_b, inv_p = np.linalg.inv(bernstein(sites, degree)), np.linalg.inv(monomials(sites, degree))
        for start in range(0, self.ncells, 8192):
            cells = self.cells[start:start+8192]
            values = sample.eval(self.domain, cells).reshape(len(cells), len(sites), -1)
            yield cells, values, np.einsum("ij,njk->nik", inv_b, values), np.einsum("ij,njk->nik", inv_p, values)

    def zeros(self, expression, degree, label, *, raw_torsion=False):
        """Screen every owned cell, then refine all candidate cells in NumPy."""
        import numpy as np
        from mpi4py import MPI
        records, candidates, unresolved = [], 0, 0
        for cells, _, bounded, power in self.coefficients(expression, degree+1 if raw_torsion else degree):
            if raw_torsion:
                # Differentiate the reference T polynomial, then transform only
                # at its roots. grad_x(T)=J_geom^{-T} grad_ref(T); a physical
                # gradient is generally rational, not polynomial, on a curve.
                old = power[..., 0]
                power = np.zeros((len(cells), len(powers(degree)), 2))
                lookup = {pair: k for k, pair in enumerate(powers(degree))}
                for k, (i, j) in enumerate(powers(degree+1)):
                    if i:
                        power[:, lookup[(i-1, j)], 0] = i*old[:, k]
                    if j:
                        power[:, lookup[(i, j-1)], 1] = j*old[:, k]
                convert = np.linalg.solve(bernstein(lattice(degree), degree), monomials(lattice(degree), degree))
                bounded = np.einsum("ij,njk->nik", convert, power)
            pad = 1e-11*max(1., float(np.max(np.abs(bounded))))
            mask = np.all((np.min(bounded, axis=1) <= pad) & (np.max(bounded, axis=1) >= -pad), axis=1)
            mask &= ~excludes_vector_zero(bounded, pad)
            selected, c = cells[mask], power[mask]
            candidates += len(selected)
            root_tolerance = 1e-14 if raw_torsion else 1e-11
            ids, refs, residual = vector_roots(c, degree, tolerance=root_tolerance)
            unresolved += len(selected)-len(np.unique(ids))
            physical = self.mapping(selected[ids], refs)
            derivatives = np.stack([np.einsum("ni,nij->nj", monomials(refs, degree, d), c[ids])
                                    for d in ((1, 0), (0, 1))], axis=-1)
            inverse_geometry = np.linalg.inv(self.mapping_jacobian(selected[ids], refs))
            jacobian = derivatives @ inverse_geometry
            if raw_torsion:
                jacobian = np.swapaxes(inverse_geometry, 1, 2) @ jacobian
            for k in range(len(ids)):
                eig = np.linalg.eigvals(jacobian[k])
                records.append({"point": physical[k].tolist(), "residual": float(residual[k]),
                                "jacobian_eigenvalues_real": eig.real.tolist(),
                                "jacobian_eigenvalues_imag": eig.imag.tolist()})
        gathered = self.comm.allgather(records)
        roots = cluster_roots([r for part in gathered for r in part], 1e-8*self.diameter)
        return {"field": label, "polynomial_degree": degree, "detected_zero_count": len(roots), "zeros": roots,
                "candidate_cells": self.comm.allreduce(candidates, op=MPI.SUM),
                "unresolved_candidate_cells": self.comm.allreduce(unresolved, op=MPI.SUM),
                "root_residual_tolerance": 1e-14 if raw_torsion else 1e-11,
                "residual_coordinates": "reference" if raw_torsion else "physical",
                "clustering_tolerance": 1e-8*self.diameter,
                "search": "all-cell Bernstein convex-hull screening; multistart Newton; not a uniqueness proof"}

    def maximum(self, T, degree, raw_roots):
        """Maximize the actual degree-p field, including cell boundaries.

        Bernstein upper bounds select candidate cells globally. On this small
        set only, constrained local optimization supplements corner, sample,
        and raw-gradient-root values. This is not a largest-DOF approximation.
        """
        import numpy as np
        from mpi4py import MPI
        from scipy.optimize import minimize
        saved, lower = [], -np.inf
        for cells, values, bounded, power in self.coefficients(T, degree):
            lower = max(lower, float(np.max(values)))
            saved.append((cells, np.max(bounded[..., 0], axis=1), power[..., 0]))
        lower = self.comm.allreduce(lower, op=MPI.MAX)
        best = {"value": -1., "point": [0., 0.]}
        selected_count = 0
        for cells, upper, power in saved:
            selected = np.flatnonzero(upper >= lower-1e-12*max(1., abs(lower)))
            selected_count += len(selected)
            for index in selected:  # few extremum candidates; not a mesh-wide optimizer loop
                cell, c = cells[index], power[index]
                sites = lattice(degree)
                values = monomials(sites, degree) @ c
                scale = max(float(np.ptp(values)), 1e-12)
                def fun(r):
                    return -(monomials(r, degree) @ c)/scale
                def jac(r):
                    return -np.array([monomials(r, degree, d) @ c for d in ((1, 0), (0, 1))])/scale
                opt = minimize(fun, sites[np.argmax(values)], jac=jac, method="SLSQP", bounds=((0., 1.), (0., 1.)),
                               constraints={"type": "ineq", "fun": lambda r: 1-r.sum(), "jac": lambda r: -np.ones(2)},
                               options={"ftol": 1e-14, "maxiter": 100})
                trials = list(sites)
                if opt.success and opt.x.sum() <= 1+1e-10:
                    trials.append(opt.x)
                for root in raw_roots["zeros"]:
                    r = self.pull_back(cell, np.asarray(root["point"]))
                    if min(r[0], r[1], 1-r.sum()) >= -1e-9:
                        trials.append(r)
                trials = np.asarray(trials)
                values = monomials(trials, degree) @ c
                k = np.argmax(values)
                if values[k] > best["value"]:
                    best = {"value": float(values[k]), "point": self.mapping([cell], trials[k:k+1])[0].tolist()}
        best = max(self.comm.allgather(best), key=lambda row: row["value"])
        best["candidate_cells"] = self.comm.allreduce(selected_count, op=MPI.SUM)
        return best


def audit_degree(domain, degree, recovery_degrees, report):
    import numpy as np
    import time
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    if np.dtype(PETSc.ScalarType).kind != "f":
        raise ValueError("torsion-center audit requires a real PETSc build")
    started = time.perf_counter()
    V = fem.functionspace(domain, ("Lagrange", degree))
    domain.topology.create_connectivity(1, 2)
    facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, 1, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2*degree+2})
    options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "ksp_error_if_not_converged": True}
    problem = LinearProblem(ufl.inner(ufl.grad(u), ufl.grad(v))*dx, fem.Constant(domain, PETSc.ScalarType(1))*v*dx,
                            bcs=[bc], petsc_options_prefix=f"torsion_center_p{degree}_", petsc_options=options)
    report(f"TORSION_START degree={degree} dofs={V.dofmap.index_map.size_global}")
    T = problem.solve()
    T.x.scatter_forward()
    reason = problem.solver.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"TORSION_SOLVE_FAILED: {reason}")
    residual = problem.b.duplicate()
    problem.A.mult(problem.x, residual)
    residual.axpy(-1., problem.b)
    relative_residual = residual.norm()/max(problem.b.norm(), 1e-300)
    residual.destroy()
    if relative_residual > 1e-9:
        raise RuntimeError(f"UNRESOLVED_TORSION_RESIDUAL: {relative_residual}")
    audit = FieldAudit(domain)
    raw = audit.zeros(T, degree-1, "raw_grad_T", raw_torsion=True)
    center = audit.maximum(T, degree, raw)
    report(f"TORSION_MAX degree={degree} value={center['value']:.12g} point={center['point']} raw_zeros={raw['detected_zero_count']}")
    recovered = []
    for recovery_degree in sorted(set(recovery_degrees or (1, degree-1))):
        G = fem.functionspace(domain, ("Lagrange", recovery_degree, (2,)))
        a, b = ufl.TrialFunction(G), ufl.TestFunction(G)
        recovery_dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2*max(degree, recovery_degree)+2})
        recovery = LinearProblem(ufl.inner(a, b)*recovery_dx, ufl.inner(ufl.grad(T), b)*recovery_dx,
                                 petsc_options_prefix=f"torsion_center_p{degree}_g{recovery_degree}_", petsc_options=options)
        gradient = recovery.solve()
        gradient.x.scatter_forward()
        if recovery.solver.getConvergedReason() <= 0:
            raise RuntimeError("GRADIENT_RECOVERY_FAILED")
        result = audit.zeros(gradient, recovery_degree, f"recovered_CG{recovery_degree}")
        result["zero_distances_from_torsion_maximum"] = [float(np.linalg.norm(np.asarray(root["point"])-center["point"]))
                                                       for root in result["zeros"]]
        error2 = domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(gradient-ufl.grad(T), gradient-ufl.grad(T))*recovery_dx)), op=MPI.SUM)
        result["gradient_recovery_L2_error"] = float(np.sqrt(max(error2, 0.)))
        recovered.append(result)
        report(f"RECOVERY degree={degree} recovery_degree={recovery_degree} zeros={result['detected_zero_count']} "
               f"displacements={result['zero_distances_from_torsion_maximum']} unresolved_cells={result['unresolved_candidate_cells']}")
    return {"degree": degree, "dofs": V.dofmap.index_map.size_global, "cells": domain.topology.index_map(2).size_global,
            "torsion_ksp_reason": reason, "relative_algebraic_residual": relative_residual,
            "torsion_maximum": center, "raw_gradient": raw, "recovered_gradients": recovered,
            "diameter": audit.diameter, "production_center_radius": 1e-5*audit.diameter,
            "geometry_degree": audit.geometry_degree, "minimum_sampled_abs_geometry_jacobian": audit.minimum_abs_jacobian,
            "elapsed_seconds": time.perf_counter()-started}


def _run(argv, session):
    import json
    import shlex
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--degrees", nargs="+", type=int, default=[2, 3, 4, 6])
    parser.add_argument("--recovery-degrees", nargs="+", type=int, help="default: CG1 and CG(p-1)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite-output", action="store_true", help="archive previous directory, then start fresh")
    parser.add_argument("--save-terminal-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-v", "--verbosity", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args(argv)
    if any(p < 2 or p > 6 for p in args.degrees) or any(p < 1 or p > 6 for p in (args.recovery_degrees or [])):
        parser.error("torsion degrees must be 2..6; recovery degrees must be 1..6")
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    from projects.diocotron.dolfinx.geometry.canonical import sha256_file
    from projects.diocotron.dolfinx.equiband.output import software_versions
    from projects.diocotron.dolfinx.equiband.reporting import ProgressReporter
    from projects.diocotron.dolfinx.equiband.run_directory import prepare_run_directory
    comm = MPI.COMM_WORLD
    session.rank, session.size, session.active = comm.rank, comm.size, True
    session.record["command"] = shlex.join([sys.executable, "-m", "projects.diocotron.dolfinx.geometry.center_audit", *argv])
    report = ProgressReporter(comm, args.verbosity, started=session.started)
    directory = prepare_run_directory(args.output, comm, overwrite=args.overwrite_output, report=report)
    if directory.restart:
        raise ValueError("this torsion audit has no equilibrium restart; select fresh output or a new directory")
    session.bind(directory, comm)
    session.phase = "torsion hp audit"
    if not all(comm.allgather(args.mesh.is_file())):
        raise ValueError("MESH_FILE_NOT_FOUND_ON_ALL_RANKS")
    metadata = {"mesh_file": str(args.mesh.resolve()), "mesh_sha256": sha256_file(args.mesh), "degrees": args.degrees,
                "recovery_degrees": args.recovery_degrees, "software": software_versions(), "ranks": comm.size,
                "command": session.record["command"], "production_solver_modified": False,
                "audit_source_sha256": sha256_file(Path(__file__))}
    def write_result(name, value, *, release=False):
        error = None
        if comm.rank == 0:
            try:
                with (directory.path/name).open("x") as stream:
                    json.dump(value, stream, indent=2, allow_nan=False)
                if release:
                    directory.release_reservation()
            except Exception as exc:
                error = str(exc)
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(f"AUDIT_OUTPUT_FAILED: {error}")
    write_result("audit.json", metadata, release=True)
    domain = gmshio.read_from_msh(str(args.mesh), comm, rank=0, gdim=2).mesh
    report(f"GEOMETRY degree={domain.geometry.cmaps[0].degree} curved={int(domain.geometry.cmaps[0].degree > 1)}")
    for degree in dict.fromkeys(args.degrees):
        result = audit_degree(domain, degree, args.recovery_degrees, report)
        write_result(f"degree_{degree}.json", result)
        report(f"DEGREE_COMPLETE degree={degree} elapsed={result['elapsed_seconds']:.3f}s", level=0)
    session.outcome, session.phase = "AUDIT_COMPLETE", "complete"
    return 0


def main(argv=None, *, _process_entry=False):
    from projects.diocotron.dolfinx.equiband.terminal_logging import run_logged
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--save-terminal-log" not in argv and "--no-save-terminal-log" not in argv:
        argv.append("--save-terminal-log")
    return run_logged(_run, argv, process_entry=_process_entry)


if __name__ == "__main__":
    raise SystemExit(main(_process_entry=True))
