from __future__ import annotations

import numpy as np
import pytest

from hybridge.linalg.face_dense import FaceDenseSystem
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.linalg import (
    build_face_block_jacobi_preconditioner,
    solve_face_dense_gmres,
)
from hybridge.solvers.diffusion_face_dense import solve_diffusion_face_dense_direct
from scripts.diffusion_reaction.cases import quadratic_poisson_case


def _synthetic_system(diagonal_blocks: np.ndarray) -> FaceDenseSystem:
    diagonal_blocks = np.asarray(diagonal_blocks, dtype=np.float64)
    num_faces, block_size, block_size_2 = diagonal_blocks.shape
    assert block_size == block_size_2
    blocks = diagonal_blocks[:, None, :, :].copy()
    neighbors = np.arange(num_faces, dtype=np.int64)[:, None]
    return FaceDenseSystem(
        blocks=blocks,
        neighbors=neighbors,
        rhs=np.zeros((num_faces, block_size), dtype=np.float64),
        global_faces=np.arange(num_faces, dtype=np.int64),
        global_to_local=np.arange(num_faces, dtype=np.int64),
        boundary_trace=np.zeros((num_faces, block_size), dtype=np.float64),
        mode="penalty",
    )


def test_block_jacobi_matches_independent_dense_solves() -> None:
    diagonal_blocks = np.array(
        [
            [[4.0, 1.0], [-1.0, 3.0]],
            [[2.0, -0.5], [0.75, 5.0]],
            [[6.0, 2.0], [1.0, 4.0]],
        ]
    )
    system = _synthetic_system(diagonal_blocks)
    preconditioner = build_face_block_jacobi_preconditioner(system)
    vector = np.array([[1.0, 2.0], [-3.0, 0.5], [4.0, -1.0]])

    expected = np.stack(
        [
            np.linalg.solve(diagonal_blocks[face], vector[face])
            for face in range(system.num_rows)
        ]
    )

    np.testing.assert_allclose(preconditioner(vector), expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        preconditioner(vector.reshape(-1)),
        expected.reshape(-1),
        rtol=1e-14,
        atol=1e-14,
    )
    assert preconditioner.maximum_inverse_residual < 1.0e-14


def test_block_jacobi_is_exact_for_block_diagonal_operator() -> None:
    diagonal_blocks = np.array(
        [
            [[3.0, 0.5], [0.25, 2.0]],
            [[5.0, -1.0], [1.5, 4.0]],
        ]
    )
    system = _synthetic_system(diagonal_blocks)
    preconditioner = build_face_block_jacobi_preconditioner(system)
    vector = np.array([[1.0, -2.0], [0.5, 3.0]])
    applied_operator = np.einsum("fij,fj->fi", diagonal_blocks, vector)

    np.testing.assert_allclose(
        preconditioner(applied_operator),
        vector,
        rtol=1e-14,
        atol=1e-14,
    )


def test_block_jacobi_rejects_missing_diagonal_slot() -> None:
    system = _synthetic_system(np.eye(2)[None, :, :])
    bad = FaceDenseSystem(
        blocks=system.blocks,
        neighbors=np.array([[-1]], dtype=np.int64),
        rhs=system.rhs,
        global_faces=system.global_faces,
        global_to_local=system.global_to_local,
        boundary_trace=system.boundary_trace,
        mode=system.mode,
    )

    with pytest.raises(ValueError, match="slot zero"):
        build_face_block_jacobi_preconditioner(bad)


def test_block_jacobi_reports_singular_face_block() -> None:
    system = _synthetic_system(
        np.array(
            [
                [[2.0, 0.0], [0.0, 3.0]],
                [[1.0, 2.0], [2.0, 4.0]],
            ]
        )
    )

    with pytest.raises(np.linalg.LinAlgError, match="face block 1"):
        build_face_block_jacobi_preconditioner(system)


def _direct_face_system(boundary_mode: str):
    mesh_size = 4 if boundary_mode == "eliminate" else 2
    order = 2 if boundary_mode == "eliminate" else 1
    space = DGSpace(
        rectangle_mesh(mesh_size, mesh_size),
        order,
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
        boundary_penalty=1.0e4,
    )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_block_jacobi_preconditioned_gmres_matches_direct_solution(
    boundary_mode: str,
) -> None:
    direct = _direct_face_system(boundary_mode)
    preconditioner = build_face_block_jacobi_preconditioner(direct.system)

    result = solve_face_dense_gmres(
        direct.system,
        restart=25,
        max_iterations=400,
        rtol=1.0e-11,
        preconditioner=preconditioner,
        reorthogonalize=True,
    )

    assert result.converged
    assert result.preconditioner_count > 0
    assert result.relative_residual < 1.0e-11
    np.testing.assert_allclose(
        result.solution.reshape(-1),
        direct.system_solution,
        rtol=2.0e-9,
        atol=2.0e-10,
    )
