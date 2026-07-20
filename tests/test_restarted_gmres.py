from __future__ import annotations

import numpy as np
import pytest

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.gmres import restarted_gmres, solve_face_dense_gmres
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def _diagonally_dominant_nonsymmetric_matrix(
    size: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(size, size))
    diagonal = np.sum(np.abs(matrix), axis=1) + 1.0
    matrix[np.diag_indices(size)] += diagonal
    return matrix


def test_restarted_gmres_matches_dense_reference_with_multiple_cycles() -> None:
    size = 24
    matrix = _diagonally_dominant_nonsymmetric_matrix(size, seed=14)
    rng = np.random.default_rng(91)
    exact = rng.normal(size=size)
    rhs = matrix @ exact

    result = restarted_gmres(
        lambda vector: matrix @ vector,
        rhs,
        restart=4,
        max_iterations=200,
        rtol=1.0e-11,
        reorthogonalize=True,
    )

    assert result.converged
    assert result.status == "converged"
    assert result.restart_cycles > 1
    assert result.iterations > 4
    assert result.relative_residual < 1.0e-11
    np.testing.assert_allclose(result.solution, exact, rtol=2.0e-10, atol=2.0e-10)
    assert result.residual_history.shape == (result.iterations + 1,)
    assert result.preconditioned_residual_history.shape == (result.iterations + 1,)


def test_restarted_gmres_supports_left_preconditioning() -> None:
    size = 30
    matrix = _diagonally_dominant_nonsymmetric_matrix(size, seed=31)
    rng = np.random.default_rng(5)
    rhs = rng.normal(size=size)
    reference = np.linalg.solve(matrix, rhs)
    inverse_diagonal = 1.0 / np.diag(matrix)

    result = restarted_gmres(
        lambda vector: matrix @ vector,
        rhs,
        restart=6,
        max_iterations=150,
        rtol=1.0e-11,
        preconditioner=lambda vector: inverse_diagonal * vector,
    )

    assert result.converged
    assert result.preconditioner_count > 0
    assert result.relative_residual < 1.0e-11
    np.testing.assert_allclose(
        result.solution,
        reference,
        rtol=3.0e-10,
        atol=3.0e-10,
    )


def test_restarted_gmres_accepts_face_major_rhs_and_exact_initial_guess() -> None:
    matrix = np.array(
        [
            [4.0, 1.0, 0.0, 0.0],
            [-1.0, 3.0, 1.0, 0.0],
            [0.0, -2.0, 5.0, 1.0],
            [1.0, 0.0, -1.0, 4.0],
        ]
    )
    exact = np.array([[1.0, -2.0], [0.5, 3.0]])
    rhs = (matrix @ exact.reshape(-1)).reshape(2, 2)

    result = restarted_gmres(
        lambda vector: matrix @ vector,
        rhs,
        x0=exact,
        restart=3,
        max_iterations=20,
        rtol=1.0e-13,
    )

    assert result.converged
    assert result.iterations == 0
    assert result.solution.shape == rhs.shape
    np.testing.assert_allclose(result.solution, exact, rtol=0.0, atol=0.0)


def test_restarted_gmres_reports_iteration_limit() -> None:
    matrix = _diagonally_dominant_nonsymmetric_matrix(20, seed=77)
    rhs = np.arange(1.0, 21.0)

    result = restarted_gmres(
        lambda vector: matrix @ vector,
        rhs,
        restart=2,
        max_iterations=1,
        rtol=1.0e-15,
    )

    assert not result.converged
    assert result.status == "max_iterations"
    assert result.iterations == 1
    assert result.relative_residual > 1.0e-15


def _face_dense_direct_result(
    *,
    boundary_mode: str,
    boundary_penalty: float = 1.0e8,
):
    space = DGSpace(
        rectangle_mesh(3, 3) if boundary_mode == "eliminate" else rectangle_mesh(2, 2),
        2 if boundary_mode == "eliminate" else 1,
        basis_type="dub_orth",
    )
    diffusion, reaction, source, exact = quadratic_poisson_case()
    return solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode=boundary_mode,
        boundary_penalty=boundary_penalty,
    )


def test_face_dense_gmres_matches_eliminated_direct_solution() -> None:
    direct = _face_dense_direct_result(boundary_mode="eliminate")

    result = solve_face_dense_gmres(
        direct.system,
        restart=20,
        max_iterations=250,
        rtol=1.0e-11,
        reorthogonalize=True,
    )

    assert result.converged
    assert result.restart_cycles >= 2
    assert result.relative_residual < 1.0e-11
    np.testing.assert_allclose(
        result.solution.reshape(-1),
        direct.system_solution,
        rtol=1.0e-9,
        atol=1.0e-10,
    )


def test_face_dense_gmres_matches_small_penalty_system() -> None:
    direct = _face_dense_direct_result(
        boundary_mode="penalty",
        boundary_penalty=1.0e4,
    )

    result = solve_face_dense_gmres(
        direct.system,
        restart=20,
        max_iterations=100,
        rtol=1.0e-12,
        reorthogonalize=True,
    )

    assert result.converged
    assert result.relative_residual < 1.0e-12
    np.testing.assert_allclose(
        result.solution.reshape(-1),
        direct.system_solution,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"restart": 0},
        {"max_iterations": 0},
        {"rtol": -1.0},
        {"atol": -1.0},
    ],
)
def test_restarted_gmres_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        restarted_gmres(
            lambda vector: vector,
            np.ones(3),
            **kwargs,
        )