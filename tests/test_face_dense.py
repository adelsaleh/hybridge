from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from scipy.sparse import coo_array

from hdgfem.mixed.face_dense import build_face_topology
from hdgfem.linalg.face_dense import (
    expand_eliminated_solution,
    face_dense_matvec,
    face_dense_to_dense,
)
from hdgfem.hdg.condensation import block_source_moments, free_trace_dofs
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.reduction import eliminate_known_dofs
from hdgfem.mixed.local_numpy import (
    assemble_diffusion_trace_system,
    diffusion_element_boundary_mats,
    local_solvers,
)
from hdgfem.solvers.diffusion_face_dense import (
    DiffusionFaceDenseAssembly,
    assemble_diffusion_face_dense_components,
)
from scripts.diffusion_reaction.cases import (
    quadratic_poisson_case,
    quadratic_variable_reaction_case,
    tensor_sine_diffusion_reaction_case,
)


@dataclass(frozen=True)
class ValidationCase:
    space: DGSpace
    stabilization: object
    reference: object
    face_dense: DiffusionFaceDenseAssembly


def _build_validation_case(
    *,
    nx: int,
    ny: int,
    order: int,
    basis_type: str = "dub_orth",
    problem_factory: Callable = quadratic_poisson_case,
    stabilization_kind: str = "scalar",
    boundary_penalty: float = 1.0e8,
) -> ValidationCase:
    space = DGSpace(rectangle_mesh(nx, ny), order, basis_type=basis_type)
    diffusion, reaction, source, exact = problem_factory()

    if stabilization_kind == "scalar":
        stabilization = 1.3
    elif stabilization_kind == "element":
        stabilization = np.linspace(0.8, 1.6, space.mesh.num_tri)
    elif stabilization_kind == "element_face":
        stabilization = np.linspace(
            0.7,
            1.9,
            3 * space.mesh.num_tri,
        ).reshape(space.mesh.num_tri, 3)
    else:
        raise ValueError(f"unknown stabilization kind: {stabilization_kind}")

    local_solver = local_solvers(
        reaction,
        stabilization,
        space,
        backend="numpy",
        diffusion=diffusion,
    )
    element_boundary_mats = diffusion_element_boundary_mats(stabilization, space)
    source_rhs = block_source_moments(source, space, num_blocks=3, source_block=0)

    reference = assemble_diffusion_trace_system(
        local_solver,
        element_boundary_mats,
        source_rhs,
        exact,
        stabilization,
        space,
        boundary_penalty=boundary_penalty,
    )
    face_dense = assemble_diffusion_face_dense_components(
        local_solver,
        element_boundary_mats,
        source_rhs,
        exact,
        stabilization,
        space,
        boundary_penalty=boundary_penalty,
    )
    return ValidationCase(
        space=space,
        stabilization=stabilization,
        reference=reference,
        face_dense=face_dense,
    )


def _reference_penalty_dense(case: ValidationCase) -> np.ndarray:
    num_dofs = case.space.mesh.num_edg * case.space.quad_data.edg_dof
    return coo_array(
        (
            case.reference.data,
            (case.reference.rows, case.reference.cols),
        ),
        shape=(num_dofs, num_dofs),
    ).toarray()


def _reference_eliminated(case: ValidationCase):
    known_mask = ~free_trace_dofs(case.space)
    return eliminate_known_dofs(
        case.reference.rows,
        case.reference.cols,
        case.reference.data,
        case.reference.rhs,
        known_mask,
        case.reference.boundary_trace.ravel(),
    )


def _normalized_tau(case: ValidationCase) -> np.ndarray:
    tau = case.stabilization
    num_elements = case.space.mesh.num_tri
    if np.isscalar(tau):
        return np.full((num_elements, 3), float(tau))
    tau = np.asarray(tau, dtype=np.float64)
    if tau.shape == (num_elements,):
        return np.broadcast_to(tau[:, None], (num_elements, 3))
    return tau


def test_face_topology_invariants_and_inverse_lookups() -> None:
    mesh = rectangle_mesh(3, 2)
    topology = build_face_topology(mesh.loc2glob_edge)

    assert topology.neighbors.shape == (mesh.num_edg, 5)
    assert topology.adjacent_elements.shape == (mesh.num_edg, 2)
    assert topology.adjacent_local_faces.shape == (mesh.num_edg, 2)
    assert topology.element_face_slots.shape == (mesh.num_tri, 3, 3)
    np.testing.assert_array_equal(topology.neighbors[:, 0], np.arange(mesh.num_edg))

    for face in range(mesh.num_edg):
        valid_neighbors = topology.neighbors[face][topology.neighbors[face] >= 0]
        assert np.unique(valid_neighbors).size == valid_neighbors.size
        expected_incidence = 1 if face in set(mesh.bnd_edges_inds.tolist()) else 2
        assert topology.incidence_count[face] == expected_incidence

        for side in range(expected_incidence):
            element = int(topology.adjacent_elements[face, side])
            local_face = int(topology.adjacent_local_faces[face, side])
            assert mesh.loc2glob_edge[element, local_face] == face

    for element in range(mesh.num_tri):
        for row_local in range(3):
            row_global = int(mesh.loc2glob_edge[element, row_local])
            for column_local in range(3):
                column_global = int(mesh.loc2glob_edge[element, column_local])
                slot = int(topology.element_face_slots[element, row_local, column_local])
                assert topology.neighbors[row_global, slot] == column_global


@pytest.mark.parametrize("stabilization_kind", ["scalar", "element", "element_face"])
def test_complete_element_blocks_match_the_exact_local_formula(
    stabilization_kind: str,
) -> None:
    case = _build_validation_case(
        nx=2,
        ny=1,
        order=2,
        stabilization_kind=stabilization_kind,
    )
    assembly = case.face_dense
    q = case.space.quad_data
    mesh = case.space.mesh
    tau = _normalized_tau(case)

    expected = -assembly.trace_blocks.copy()
    stabilization_mass = (
        (tau * mesh.jacs_el_fc)[..., None, None]
        * q.M_rf_fc[None, None, :, :]
    )
    local_faces = np.arange(3)
    expected[:, local_faces, local_faces] += stabilization_mass

    np.testing.assert_allclose(assembly.element_blocks, expected, rtol=0.0, atol=0.0)
    assert assembly.element_blocks.flags.c_contiguous


@pytest.mark.parametrize(
    "nx,ny,order,basis_type,problem_factory,stabilization_kind",
    [
        (1, 1, 0, "dub_orth", quadratic_poisson_case, "scalar"),
        (2, 1, 1, "hier_C0", quadratic_variable_reaction_case, "element"),
        (2, 2, 2, "bernstein", quadratic_poisson_case, "element_face"),
        (3, 2, 3, "dub_orth", tensor_sine_diffusion_reaction_case, "element_face"),
    ],
)
def test_penalty_face_dense_matrix_and_rhs_match_current_coo_assembly(
    nx: int,
    ny: int,
    order: int,
    basis_type: str,
    problem_factory: Callable,
    stabilization_kind: str,
) -> None:
    case = _build_validation_case(
        nx=nx,
        ny=ny,
        order=order,
        basis_type=basis_type,
        problem_factory=problem_factory,
        stabilization_kind=stabilization_kind,
    )

    reference_matrix = _reference_penalty_dense(case)
    face_matrix = face_dense_to_dense(
        case.face_dense.penalty_system.blocks,
        case.face_dense.penalty_system.neighbors,
    )

    np.testing.assert_allclose(face_matrix, reference_matrix, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        case.face_dense.penalty_system.rhs.ravel(),
        case.reference.rhs,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize(
    "nx,ny,order,problem_factory,stabilization_kind",
    [
        (1, 1, 0, quadratic_poisson_case, "scalar"),
        (2, 1, 1, quadratic_variable_reaction_case, "element"),
        (2, 2, 2, quadratic_poisson_case, "element_face"),
        (3, 2, 3, tensor_sine_diffusion_reaction_case, "element_face"),
    ],
)
def test_direct_dirichlet_elimination_matches_scalar_coo_elimination(
    nx: int,
    ny: int,
    order: int,
    problem_factory: Callable,
    stabilization_kind: str,
) -> None:
    case = _build_validation_case(
        nx=nx,
        ny=ny,
        order=order,
        problem_factory=problem_factory,
        stabilization_kind=stabilization_kind,
    )
    reference = _reference_eliminated(case)
    num_free_dofs = reference.rhs.size
    reference_matrix = coo_array(
        (reference.data, (reference.rows, reference.cols)),
        shape=(num_free_dofs, num_free_dofs),
    ).toarray()
    reduced = case.face_dense.eliminated_system
    face_matrix = face_dense_to_dense(reduced.blocks, reduced.neighbors)

    np.testing.assert_allclose(face_matrix, reference_matrix, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(reduced.rhs.ravel(), reference.rhs, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_array_equal(reduced.global_faces, case.space.mesh.int_edges_inds)
    np.testing.assert_array_equal(
        reduced.global_to_local[reduced.global_faces],
        np.arange(reduced.num_rows),
    )
    if reduced.neighbors.size:
        valid = reduced.neighbors >= 0
        assert np.all(reduced.neighbors[valid] < reduced.num_rows)
        np.testing.assert_allclose(reduced.blocks[~valid], 0.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("mode", ["penalty", "eliminate"])
def test_face_dense_matvec_matches_explicit_matrix_for_flat_and_face_major_vectors(
    mode: str,
) -> None:
    case = _build_validation_case(
        nx=3,
        ny=2,
        order=3,
        problem_factory=tensor_sine_diffusion_reaction_case,
        stabilization_kind="element_face",
    )
    system = (
        case.face_dense.penalty_system
        if mode == "penalty"
        else case.face_dense.eliminated_system
    )
    explicit = face_dense_to_dense(system.blocks, system.neighbors)
    rng = np.random.default_rng(47821)
    x_flat = rng.standard_normal(system.num_dofs)
    x_faces = x_flat.reshape(system.num_rows, system.block_size)

    expected = explicit @ x_flat
    actual_flat = face_dense_matvec(system.blocks, system.neighbors, x_flat)
    actual_faces = face_dense_matvec(system.blocks, system.neighbors, x_faces)

    np.testing.assert_allclose(actual_flat, expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(actual_faces.ravel(), expected, rtol=1.0e-13, atol=1.0e-13)
    assert actual_flat.shape == x_flat.shape
    assert actual_faces.shape == x_faces.shape


def test_penalty_and_eliminated_systems_produce_the_same_full_trace_solution() -> None:
    case = _build_validation_case(
        nx=2,
        ny=2,
        order=2,
        problem_factory=quadratic_variable_reaction_case,
        stabilization_kind="element_face",
        boundary_penalty=1.0e6,
    )
    penalty = case.face_dense.penalty_system
    eliminated = case.face_dense.eliminated_system

    penalty_matrix = face_dense_to_dense(penalty.blocks, penalty.neighbors)
    eliminated_matrix = face_dense_to_dense(eliminated.blocks, eliminated.neighbors)
    penalty_trace = np.linalg.solve(penalty_matrix, penalty.rhs.ravel())
    reduced_trace = np.linalg.solve(eliminated_matrix, eliminated.rhs.ravel())
    expanded_trace = expand_eliminated_solution(reduced_trace, eliminated)

    np.testing.assert_allclose(expanded_trace, penalty_trace, rtol=1.0e-10, atol=1.0e-10)

    penalty_relative_residual = np.linalg.norm(
        face_dense_matvec(penalty.blocks, penalty.neighbors, penalty_trace)
        - penalty.rhs.ravel()
    ) / np.linalg.norm(penalty.rhs.ravel())
    eliminated_relative_residual = np.linalg.norm(
        face_dense_matvec(eliminated.blocks, eliminated.neighbors, reduced_trace)
        - eliminated.rhs.ravel()
    ) / np.linalg.norm(eliminated.rhs.ravel())

    assert penalty_relative_residual < 1.0e-12
    assert eliminated_relative_residual < 1.0e-12
    np.testing.assert_allclose(
        expanded_trace.reshape(case.space.mesh.num_edg, case.space.quad_data.edg_dof)[
            case.space.mesh.bnd_edges_inds
        ],
        case.face_dense.boundary_trace[case.space.mesh.bnd_edges_inds],
        rtol=1.0e-12,
        atol=1.0e-12,
    )
