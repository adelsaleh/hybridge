from __future__ import annotations

import numpy as np
import pytest

from hybridge.linalg.face_dense import FaceDenseSystem, materialize_face_dense_matrix
from hybridge.core.mesh import rectangle_mesh
from hybridge.core.space import DGSpace
from hybridge.linalg import (
    build_face_additive_schwarz_preconditioner,
    solve_face_dense_gmres,
)
from hybridge.solvers.diffusion_face_dense import solve_diffusion_face_dense_direct
from scripts.diffusion_reaction.cases import quadratic_poisson_case


def _direct_face_problem(boundary_mode: str, *, mesh_size: int = 3, order: int = 2):
    space = DGSpace(
        rectangle_mesh(mesh_size, mesh_size),
        order,
        basis_type="dub_orth",
    )
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
    preconditioner = build_face_additive_schwarz_preconditioner(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    return space, direct, preconditioner


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_asm_local_active_blocks_equal_global_principal_submatrices(
    boundary_mode: str,
) -> None:
    _, direct, preconditioner = _direct_face_problem(
        boundary_mode,
        mesh_size=2,
        order=2,
    )
    matrix = materialize_face_dense_matrix(
        direct.system.blocks,
        direct.system.neighbors,
    )
    block_size = direct.system.block_size

    for element in range(preconditioner.num_elements):
        system_faces = preconditioner.element_system_faces[element]
        active_local_faces = np.flatnonzero(system_faces >= 0)
        if not active_local_faces.size:
            continue

        local_dofs = np.concatenate(
            [
                np.arange(local_face * block_size, (local_face + 1) * block_size)
                for local_face in active_local_faces
            ]
        )
        global_dofs = np.concatenate(
            [
                np.arange(
                    system_faces[local_face] * block_size,
                    (system_faces[local_face] + 1) * block_size,
                )
                for local_face in active_local_faces
            ]
        )

        np.testing.assert_allclose(
            preconditioner.local_matrices[element][np.ix_(local_dofs, local_dofs)],
            matrix[np.ix_(global_dofs, global_dofs)],
            rtol=0.0,
            atol=0.0,
        )

    assert preconditioner.maximum_inverse_residual < 5.0e-14


def test_asm_restriction_and_prolongation_handle_overlap_and_elimination() -> None:
    _, direct, preconditioner = _direct_face_problem(
        "eliminate",
        mesh_size=3,
        order=1,
    )
    rng = np.random.default_rng(3412)
    vector = rng.standard_normal(
        (direct.system.num_rows, direct.system.block_size)
    )

    restricted = preconditioner.restrict(vector)
    expected_restricted = np.zeros_like(restricted)
    active = preconditioner.element_system_faces >= 0
    expected_restricted[active] = vector[
        preconditioner.element_system_faces[active]
    ]
    np.testing.assert_allclose(restricted, expected_restricted, rtol=0.0, atol=0.0)

    prolonged = preconditioner.prolong(restricted)
    expected_prolonged = np.zeros_like(vector)
    np.add.at(
        expected_prolonged,
        preconditioner.element_system_faces[active],
        restricted[active],
    )
    np.testing.assert_allclose(prolonged, expected_prolonged, rtol=0.0, atol=0.0)

    # Each interior system face belongs to two element subdomains.
    multiplicity = np.zeros(direct.system.num_rows, dtype=np.int64)
    np.add.at(
        multiplicity,
        preconditioner.element_system_faces[active],
        1,
    )
    np.testing.assert_array_equal(multiplicity, 2)


def test_asm_application_matches_explicit_local_solve_sum() -> None:
    _, direct, preconditioner = _direct_face_problem(
        "eliminate",
        mesh_size=3,
        order=2,
    )
    rng = np.random.default_rng(8821)
    vector = rng.standard_normal(
        (direct.system.num_rows, direct.system.block_size)
    )

    expected = np.zeros_like(vector)
    for element in range(preconditioner.num_elements):
        system_faces = preconditioner.element_system_faces[element]
        local_rhs = np.zeros(
            (preconditioner.num_local_faces, preconditioner.block_size),
            dtype=np.float64,
        )
        active = system_faces >= 0
        local_rhs[active] = vector[system_faces[active]]
        local_solution = np.linalg.solve(
            preconditioner.local_matrices[element],
            local_rhs.reshape(-1),
        ).reshape(preconditioner.num_local_faces, preconditioner.block_size)
        np.add.at(expected, system_faces[active], local_solution[active])

    np.testing.assert_allclose(
        preconditioner(vector),
        expected,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        preconditioner(vector.reshape(-1)),
        expected.reshape(-1),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


@pytest.mark.parametrize("boundary_mode", ["eliminate", "penalty"])
def test_asm_preconditioned_gmres_matches_direct_solution(
    boundary_mode: str,
) -> None:
    _, direct, preconditioner = _direct_face_problem(
        boundary_mode,
        mesh_size=3 if boundary_mode == "eliminate" else 2,
        order=2 if boundary_mode == "eliminate" else 1,
    )

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
        rtol=3.0e-9,
        atol=3.0e-10,
    )


def test_asm_reports_singular_element_matrix() -> None:
    block_size = 1
    num_faces = 3
    system = FaceDenseSystem(
        blocks=np.zeros((num_faces, 1, block_size, block_size)),
        neighbors=np.arange(num_faces, dtype=np.int64)[:, None],
        rhs=np.zeros((num_faces, block_size)),
        global_faces=np.arange(num_faces, dtype=np.int64),
        global_to_local=np.arange(num_faces, dtype=np.int64),
        boundary_trace=np.zeros((num_faces, block_size)),
        mode="eliminate",
    )
    element_blocks = np.zeros((1, 3, 3, block_size, block_size))
    loc2glob_face = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(np.linalg.LinAlgError, match="element=0"):
        build_face_additive_schwarz_preconditioner(
            system,
            element_blocks,
            loc2glob_face,
        )
