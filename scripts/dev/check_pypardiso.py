#!/usr/bin/env python3
"""Check pypardiso HDG parity and benchmark host sparse solves."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.linalg import clear_pypardiso_cache, solve_global_system
from hdgfem.solvers.advection_reaction import solve_advection_reaction_hdg
from hdgfem.solvers.diffusion_reaction import solve_diffusion_reaction_hdg
from scripts.advection_reaction.cases import test3 as advection_case
from scripts.diffusion_reaction.cases import quadratic_poisson_case


def _projected_advection_problem(space: DGSpace):
    beta_x, beta_y, reaction, source, exact = advection_case(r0=1.0, N=2, M=2.0)
    return (
        space.project_callable(source, name="source_h"),
        VectorDGField((space.project_callable(beta_x), space.project_callable(beta_y)), space, name="beta_h"),
        space.project_callable(reaction, name="reaction_h"),
        exact,
    )


def _check_hdg_parity(order: int) -> None:
    space = DGSpace(rectangle_mesh(2, 2), order, basis_type="dub_orth")
    source_h, beta_h, reaction_h, exact = _projected_advection_problem(space)
    advection_kwargs = {
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "assembly_backend": "numba",
        "trace_basis": "legendre-modal",
        "solver_rtol": 1.0e-11,
        "verbose": False,
    }
    advection_reference = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="direct",
        **advection_kwargs,
    )
    advection_pardiso = solve_advection_reaction_hdg(
        source_h,
        beta_h,
        reaction_h,
        exact,
        space,
        solver="pypardiso",
        **advection_kwargs,
    )
    np.testing.assert_allclose(advection_pardiso.trace, advection_reference.trace, rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(
        advection_pardiso.field.coeffs,
        advection_reference.field.coeffs,
        rtol=1.0e-10,
        atol=1.0e-11,
    )

    problem = quadratic_poisson_case()
    diffusion_source = space.project_callable(problem.source, name="poisson_source_h")
    diffusion_reaction = space.zeros(name="poisson_reaction_h")
    diffusion_kwargs = {
        "diffusion": problem.diffusion,
        "stabilization": 1.0,
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "assembly_backend": "numba",
        "trace_basis": "legendre-modal",
        "hdg_postprocess": "none",
        "solver_rtol": 1.0e-11,
        "verbose": False,
    }
    diffusion_reference = solve_diffusion_reaction_hdg(
        diffusion_source,
        diffusion_reaction,
        problem.exact,
        space,
        solver="direct",
        **diffusion_kwargs,
    )
    diffusion_pardiso = solve_diffusion_reaction_hdg(
        diffusion_source,
        diffusion_reaction,
        problem.exact,
        space,
        solver="pypardiso-spd",
        **diffusion_kwargs,
    )
    np.testing.assert_allclose(diffusion_pardiso.trace, diffusion_reference.trace, rtol=1.0e-10, atol=1.0e-11)
    np.testing.assert_allclose(
        diffusion_pardiso.field.coeffs,
        diffusion_reference.field.coeffs,
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    print(
        "HDG parity: PASS "
        f"(advection rel={advection_pardiso.global_solve_result.physical_relative_residual_norm:.3e}, "
        f"diffusion rel={diffusion_pardiso.global_solve_result.physical_relative_residual_norm:.3e})"
    )


def _benchmark_matrix(side: int, *, nonsymmetric: bool):
    one = np.ones(side)
    line = scipy.sparse.diags((-one[:-1], 2.0 * one, -one[:-1]), (-1, 0, 1), format="csr")
    identity = scipy.sparse.eye(side, format="csr")
    matrix = scipy.sparse.kron(identity, line, format="csr") + scipy.sparse.kron(line, identity, format="csr")
    matrix = matrix + 0.1 * scipy.sparse.eye(matrix.shape[0], format="csr")
    if nonsymmetric:
        upwind = scipy.sparse.diags((-0.35 * one[:-1], 0.35 * one[:-1]), (-1, 1), format="csr")
        matrix = matrix + scipy.sparse.kron(identity, upwind, format="csr")
    matrix.sum_duplicates()
    matrix.sort_indices()
    exact = np.sin(np.linspace(0.0, np.pi, matrix.shape[0])) + 1.0
    return matrix, matrix @ exact


def _run_solve(matrix, rhs, solver_name: str):
    direct_solver = solver_name == "direct" or solver_name.startswith(("pypardiso", "pardiso"))
    return solve_global_system(
        (),
        (),
        (),
        rhs,
        rhs.size,
        solver=solver_name,
        preconditioner=None if direct_solver else "ilu",
        rtol=1.0e-10,
        atol=0.0,
        maxiter=1000,
        scale_system=not direct_solver,
        assembled_matrix=matrix,
        raise_on_nonconvergence=True,
        verbose=0,
    )


def _benchmark(side: int, repeats: int) -> None:
    print(f"benchmark: side={side}, dofs={side * side}, repeats={repeats}")
    print(f"{'matrix':<14} {'solver':<16} {'run':>4} {'wall_s':>10} {'solve_s':>10} {'relres':>12}")
    for matrix_name, nonsymmetric in (("diffusion-spd", False), ("advection-ns", True)):
        configs = ("direct", "pypardiso", "BICGSTAB")
        if not nonsymmetric:
            configs = ("direct", "pypardiso-spd", "BICGSTAB")
        matrix, rhs = _benchmark_matrix(side, nonsymmetric=nonsymmetric)
        for solver_name in configs:
            if solver_name.startswith(("pypardiso", "pardiso")):
                clear_pypardiso_cache()
            for repeat in range(1, repeats + 1):
                start = time.perf_counter()
                result = _run_solve(matrix, rhs, solver_name)
                elapsed = time.perf_counter() - start
                print(
                    f"{matrix_name:<14} {solver_name:<16} {repeat:>4d} "
                    f"{elapsed:>10.6f} {result.solve_elapsed_seconds:>10.6f} "
                    f"{result.physical_relative_residual_norm:>12.3e}"
                )
    clear_pypardiso_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=2, help="HDG parity polynomial order")
    parser.add_argument("--side", type=int, default=40, help="synthetic matrix grid side")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.order < 1 or args.side < 2 or args.repeats < 1:
        parser.error("--order and --repeats must be positive; --side must be at least 2")

    print(
        f"pypardiso={importlib.metadata.version('pypardiso')} "
        f"mkl={importlib.metadata.version('mkl')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'runtime-default')} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'runtime-default')}"
    )
    _check_hdg_parity(args.order)
    _benchmark(args.side, args.repeats)


if __name__ == "__main__":
    main()
