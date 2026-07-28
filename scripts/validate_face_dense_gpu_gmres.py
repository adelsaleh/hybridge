"""Run CPU/GPU face-dense GMRES comparisons on a CUDA machine."""

from __future__ import annotations

if __package__:
    from .validate_gpu_environment import check_gpu_validation_environment
else:  # Support PyCharm's direct "Run file" action.
    from validate_gpu_environment import check_gpu_validation_environment


if __name__ == "__main__":
    _environment = check_gpu_validation_environment(
        "scripts.validate_face_dense_gpu_gmres",
    )
    if not _environment.ready:
        raise SystemExit(_environment.exit_code)


import numpy as np

from hdgfem.assembly.face_dense import (
    face_dense_relative_residual,
    normalize_penalty_rows,
)
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


_BOUNDARY_PENALTY = 1.0e6


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
        boundary_penalty=_BOUNDARY_PENALTY,
    )
    physical_system = direct.system
    system = physical_system
    normalized_penalty_rows = boundary_mode == "penalty" and not preconditioned
    if normalized_penalty_rows:
        boundary_faces = np.flatnonzero(
            direct.assembly.topology.incidence_count == 1
        )
        system = normalize_penalty_rows(
            physical_system,
            boundary_faces,
            boundary_penalty=_BOUNDARY_PENALTY,
        )

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
    physical_residual = face_dense_relative_residual(
        physical_system,
        solution,
    )
    print(f"{boundary_mode:9s} {label:8s}")
    print(f"  status               : {result.status}")
    print(f"  dofs                 : {system.num_dofs}")
    print(
        "  normalized penalty   : "
        f"{'yes' if normalized_penalty_rows else 'no'}"
    )
    print(f"  Arnoldi iterations   : {result.iterations}")
    print(f"  restart cycles       : {result.restart_cycles}")
    print(f"  solve relative resid.: {result.relative_residual:.3e}")
    print(f"  physical rel. resid. : {physical_residual:.3e}")
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
