"""Run CPU/GPU face-dense GMRES comparisons on a CUDA machine."""

from __future__ import annotations

import numpy as np

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceBlockJacobiPreconditioner,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct

if __package__:
    from .diff_rea_cases import quadratic_poisson_case
else:  # Support PyCharm's direct "Run file" action.
    from diff_rea_cases import quadratic_poisson_case


def run_case(boundary_mode: str, preconditioned: bool) -> None:
    cp = require_cupy_device()
    space = DGSpace(rectangle_mesh(4, 4), 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = quadratic_poisson_case()
    direct = solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode=boundary_mode,
        boundary_penalty=1.0e6,
    )
    system = direct.system
    operator = CuPyFaceDenseOperator.from_system(system)
    preconditioner = None
    label = "GMRES"
    if preconditioned:
        preconditioner = CuPyFaceBlockJacobiPreconditioner.from_system(
            system,
            device_id=operator.device_id,
        )
        label = "BJ-GMRES"

    rhs = operator.to_device(system.rhs)
    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=30,
        max_iterations=1000,
        rtol=1.0e-10,
        preconditioner=preconditioner,
        reorthogonalize=True,
    )
    operator.synchronize()

    solution = operator.to_host(result.solution).reshape(-1)
    difference = np.linalg.norm(solution - direct.system_solution) / max(
        np.linalg.norm(direct.system_solution),
        np.finfo(np.float64).eps,
    )
    print(f"{boundary_mode:9s} {label:8s}")
    print(f"  status               : {result.status}")
    print(f"  dofs                 : {system.num_dofs}")
    print(f"  Arnoldi iterations   : {result.iterations}")
    print(f"  restart cycles       : {result.restart_cycles}")
    print(f"  true relative resid. : {result.relative_residual:.3e}")
    print(f"  direct difference    : {difference:.3e}")
    print(f"  matvec / dot / axpy  : {result.matvec_count} / "
          f"{result.dot_count} / {result.axpy_count}")
    print(f"  solution on device   : {isinstance(result.solution, cp.ndarray)}")


def main() -> None:
    for boundary_mode in ("eliminate", "penalty"):
        for preconditioned in (False, True):
            run_case(boundary_mode, preconditioned)


if __name__ == "__main__":
    main()
