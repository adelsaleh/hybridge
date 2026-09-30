"""Run a compact numerical validation of the face-dense HDG assembly path.

This script is intended for development use.  It compares the new face-dense
matrix and right-hand side with the established COO implementation, checks both
Dirichlet treatments, and verifies the face-dense matrix-vector product on
random vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import coo_array

from hdgfem.linalg.face_dense import face_dense_matvec, face_dense_to_dense
from hdgfem.hdg.condensation import block_source_moments, free_trace_dofs
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.reduction import eliminate_known_dofs
from hdgfem.mixed.local_numpy import (
    assemble_diffusion_trace_system,
    diffusion_element_boundary_mats,
    local_solvers,
)
from hdgfem.solvers.diffusion_face_dense import assemble_diffusion_face_dense_components
from scripts.diffusion_reaction.cases import (
    quadratic_poisson_case,
    quadratic_variable_reaction_case,
    tensor_sine_diffusion_reaction_case,
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    nx: int
    ny: int
    order: int
    basis_type: str
    problem_factory: Callable
    stabilization_kind: str


def stabilization_for(spec: CaseSpec, num_elements: int):
    if spec.stabilization_kind == "scalar":
        return 1.3
    if spec.stabilization_kind == "element":
        return np.linspace(0.8, 1.6, num_elements)
    if spec.stabilization_kind == "element_face":
        return np.linspace(0.7, 1.9, 3 * num_elements).reshape(num_elements, 3)
    raise ValueError(spec.stabilization_kind)


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(np.linalg.norm(expected), np.finfo(float).tiny)
    return float(np.linalg.norm(actual - expected) / denominator)


def run_case(spec: CaseSpec, *, boundary_penalty: float = 1.0e8) -> dict[str, float]:
    space = DGSpace(
        rectangle_mesh(spec.nx, spec.ny),
        spec.order,
        basis_type=spec.basis_type,
    )
    diffusion, reaction, source, exact = spec.problem_factory()
    stabilization = stabilization_for(spec, space.mesh.num_tri)

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
    face = assemble_diffusion_face_dense_components(
        local_solver,
        element_boundary_mats,
        source_rhs,
        exact,
        stabilization,
        space,
        boundary_penalty=boundary_penalty,
    )

    num_dofs = space.mesh.num_edg * space.quad_data.edg_dof
    reference_penalty = coo_array(
        (reference.data, (reference.rows, reference.cols)),
        shape=(num_dofs, num_dofs),
    ).toarray()
    face_penalty = face_dense_to_dense(
        face.penalty_system.blocks,
        face.penalty_system.neighbors,
    )

    reduction = eliminate_known_dofs(
        reference.rows,
        reference.cols,
        reference.data,
        reference.rhs,
        ~free_trace_dofs(space),
        reference.boundary_trace.ravel(),
    )
    reference_eliminated = coo_array(
        (reduction.data, (reduction.rows, reduction.cols)),
        shape=(reduction.rhs.size, reduction.rhs.size),
    ).toarray()
    face_eliminated = face_dense_to_dense(
        face.eliminated_system.blocks,
        face.eliminated_system.neighbors,
    )

    rng = np.random.default_rng(20260718 + spec.order)
    x_penalty = rng.standard_normal(num_dofs)
    x_eliminated = rng.standard_normal(reduction.rhs.size)

    return {
        "penalty_matrix": relative_error(face_penalty, reference_penalty),
        "penalty_rhs": relative_error(face.penalty_system.rhs.ravel(), reference.rhs),
        "penalty_matvec": relative_error(
            face_dense_matvec(
                face.penalty_system.blocks,
                face.penalty_system.neighbors,
                x_penalty,
            ),
            reference_penalty @ x_penalty,
        ),
        "eliminated_matrix": relative_error(face_eliminated, reference_eliminated),
        "eliminated_rhs": relative_error(face.eliminated_system.rhs.ravel(), reduction.rhs),
        "eliminated_matvec": relative_error(
            face_dense_matvec(
                face.eliminated_system.blocks,
                face.eliminated_system.neighbors,
                x_eliminated,
            ),
            reference_eliminated @ x_eliminated,
        ),
    }


def main() -> None:
    cases = [
        CaseSpec(
            "p0 scalar Poisson",
            1,
            1,
            0,
            "dub_orth",
            quadratic_poisson_case,
            "scalar",
        ),
        CaseSpec(
            "p1 variable reaction",
            2,
            1,
            1,
            "hier_C0",
            quadratic_variable_reaction_case,
            "element",
        ),
        CaseSpec(
            "p2 face-varying tau",
            2,
            2,
            2,
            "bernstein",
            quadratic_poisson_case,
            "element_face",
        ),
        CaseSpec(
            "p3 tensor diffusion",
            3,
            2,
            3,
            "dub_orth",
            tensor_sine_diffusion_reaction_case,
            "element_face",
        ),
    ]

    threshold = 5.0e-13
    failed = False
    print("Face-dense validation against current COO implementation")
    print("=" * 72)
    for spec in cases:
        metrics = run_case(spec)
        worst = max(metrics.values())
        status = "PASS" if worst <= threshold else "FAIL"
        failed |= status == "FAIL"
        print(f"{status:4s}  {spec.name:28s}  worst relative error = {worst:.3e}")
        for name, value in metrics.items():
            print(f"      {name:20s}: {value:.3e}")

    if failed:
        raise SystemExit(1)
    print("=" * 72)
    print(f"All cases passed with threshold {threshold:.1e}.")


if __name__ == "__main__":
    main()
