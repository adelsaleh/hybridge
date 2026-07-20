from __future__ import annotations

import numpy as np
import pytest

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import (
    exponential_bubble_poisson_case,
    quadratic_poisson_case,
    tensor_sine_diffusion_reaction_case,
)


def _solve_error(
    mesh_size: int,
    order: int,
    problem_factory,
    *,
    stabilization: float,
    boundary_mode: str = "eliminate",
) -> tuple[float, float]:
    space = DGSpace(
        rectangle_mesh(mesh_size, mesh_size),
        order,
        basis_type="dub_orth",
    )
    diffusion, reaction, source, exact = problem_factory()
    result = solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=stabilization,
        boundary_mode=boundary_mode,
        boundary_penalty=1.0e8,
    )
    return result.field.l2_error(exact), result.relative_residual


@pytest.mark.parametrize("boundary_mode", ["penalty", "eliminate"])
def test_face_dense_direct_solver_reproduces_quadratic_solution(
    boundary_mode: str,
) -> None:
    error, residual = _solve_error(
        2,
        2,
        quadratic_poisson_case,
        stabilization=1.3,
        boundary_mode=boundary_mode,
    )

    assert error < 1.0e-11
    assert residual < 1.0e-12


@pytest.mark.parametrize("order", [1, 2])
def test_face_dense_h_convergence_for_smooth_poisson_problem(order: int) -> None:
    errors = []
    residuals = []
    for mesh_size in (2, 4, 8):
        error, residual = _solve_error(
            mesh_size,
            order,
            exponential_bubble_poisson_case,
            stabilization=1.0,
        )
        errors.append(error)
        residuals.append(residual)

    errors = np.asarray(errors)
    rates = np.log2(errors[:-1] / errors[1:])

    assert np.all(np.diff(errors) < 0.0)
    assert rates[-1] > order + 0.55
    assert max(residuals) < 1.0e-12


def test_face_dense_h_convergence_for_tensor_diffusion() -> None:
    errors = []
    residuals = []
    for mesh_size in (2, 4, 8):
        error, residual = _solve_error(
            mesh_size,
            2,
            tensor_sine_diffusion_reaction_case,
            stabilization=4.0,
        )
        errors.append(error)
        residuals.append(residual)

    errors = np.asarray(errors)
    rates = np.log2(errors[:-1] / errors[1:])

    assert np.all(np.diff(errors) < 0.0)
    assert rates[-1] > 2.7
    assert max(residuals) < 5.0e-12