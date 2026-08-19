"""Face-dense reference assembly for the diffusion-reaction HDG solver.

This module is deliberately separate from :mod:`hdgfem.solvers.diffusion_reaction` so
that the existing COO/CSR solver path remains untouched while the new data
layout is developed and tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np

from ..assembly import hdg as hdg_assembly
from ..assembly.face_dense import (
    FaceDenseSystem,
    FaceTopology,
    assemble_global_face_blocks,
    build_face_topology,
    eliminate_dirichlet_faces,
    expand_eliminated_solution,
    face_dense_relative_residual,
    make_penalty_system,
    materialize_face_dense_matrix,
)
from ..core.space import DGField, DGSpace, VectorDGField
from .diffusion_reaction import (
    diffusion_element_boundary_mats,
    diffusion_trace_lift,
    local_solvers,
    split_diffusion_unknowns,
)


@dataclass(frozen=True)
class DiffusionFaceDenseAssembly:
    """All intermediate arrays requested for the initial implementation stage."""

    topology: FaceTopology
    trace_blocks: np.ndarray
    element_blocks: np.ndarray
    interior_row_blocks: np.ndarray
    interior_rhs: np.ndarray
    boundary_trace: np.ndarray
    penalty_system: FaceDenseSystem
    eliminated_system: FaceDenseSystem


@dataclass(frozen=True)
class DiffusionFaceDenseDirectResult:
    """End-to-end reference solution obtained from the face-dense path.

    This result is for correctness and convergence validation only.  It
    materializes the small face-dense matrix and calls ``numpy.linalg.solve``;
    it is not a scalable production solver.
    """

    field: DGField
    flux: VectorDGField
    trace: np.ndarray
    local_unknowns: np.ndarray
    system_solution: np.ndarray
    relative_residual: float
    assembly: DiffusionFaceDenseAssembly
    system: FaceDenseSystem
    local_solver: np.ndarray
    element_boundary_mats: np.ndarray
    source_rhs: np.ndarray


def _normalize_stabilization(stabilization, space: DGSpace) -> np.ndarray:
    """Return element-face stabilization values with shape ``(NE, 3)``."""

    num_elements = space.mesh.num_tri
    if np.isscalar(stabilization):
        return np.full((num_elements, 3), float(stabilization), dtype=np.float64)
    tau = np.asarray(stabilization, dtype=np.float64)
    if tau.shape == (num_elements,):
        return np.ascontiguousarray(np.broadcast_to(tau[:, None], (num_elements, 3)))
    if tau.shape != (num_elements, 3):
        raise ValueError(
            f"stabilization must be scalar or have shape ({num_elements}, 3); "
            f"got {tau.shape}"
        )
    return np.ascontiguousarray(tau)


def build_complete_diffusion_element_blocks(
    trace_blocks: np.ndarray,
    stabilization,
    space: DGSpace,
) -> np.ndarray:
    r"""Form complete condensed elemental face matrices.

    The existing ``trace_blocks`` contains the local Schur term produced by
    ``element_to_trace_matrix_from_lift``.  The matrix assembled by the current
    COO path is

    ``A_e[r,c] = -trace_blocks[e,r,c] + delta_rc * tau[e,r] * J[e,r] * M_face``.

    The output has shape ``(NE, 3, 3, PF, PF)`` and uses the already-corrected
    global face orientation of ``trace_blocks``.
    """

    mesh = space.mesh
    q = space.quad_data
    trace_blocks = np.asarray(trace_blocks, dtype=np.float64)
    expected_shape = (mesh.num_tri, 3, 3, q.edg_dof, q.edg_dof)
    if trace_blocks.shape != expected_shape:
        raise ValueError(
            f"trace_blocks must have shape {expected_shape}; got {trace_blocks.shape}"
        )

    tau = _normalize_stabilization(stabilization, space)
    element_blocks = -trace_blocks.copy()
    stabilization_mass = (
        (tau * mesh.jacs_el_fc)[..., None, None]
        * q.M_rf_fc[None, None, :, :]
    )

    diagonal_faces = np.arange(3)
    element_blocks[:, diagonal_faces, diagonal_faces] += stabilization_mass
    return np.ascontiguousarray(element_blocks)


def build_diffusion_interior_rhs(
    trace_lift: np.ndarray,
    source_rhs: np.ndarray,
    local_solver: np.ndarray,
    boundary_condition: Callable,
    space: DGSpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the unpenalized global RHS on interior trace rows.

    Boundary rows remain zero.  ``boundary_trace`` contains the projected
    prescribed trace values and is used by both penalty row replacement and
    direct elimination.
    """

    mesh = space.mesh
    q = space.quad_data
    trace_lift = np.asarray(trace_lift, dtype=np.float64)
    source_rhs = np.asarray(source_rhs, dtype=np.float64)
    local_solver = np.asarray(local_solver, dtype=np.float64)

    expected_lift = (mesh.num_tri, 3, q.edg_dof, local_solver.shape[-1])
    if trace_lift.shape != expected_lift:
        raise ValueError(f"trace_lift must have shape {expected_lift}; got {trace_lift.shape}")
    if source_rhs.shape != (mesh.num_tri, local_solver.shape[-1]):
        raise ValueError(
            "source_rhs must have shape "
            f"({mesh.num_tri}, {local_solver.shape[-1]}); got {source_rhs.shape}"
        )

    local_solution_from_source = local_solver @ source_rhs[..., None]
    local_face_rhs = (trace_lift @ local_solution_from_source[:, None, :, :]).squeeze(-1)

    interior_rhs = np.zeros((mesh.num_edg, q.edg_dof), dtype=np.float64)
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    if valid_elements.size:
        np.add.at(
            interior_rhs,
            mesh.loc2glob_edge[valid_elements, valid_faces],
            local_face_rhs[valid_elements, valid_faces],
        )

    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space)
    return np.ascontiguousarray(interior_rhs), np.ascontiguousarray(boundary_trace)


def assemble_diffusion_face_dense_components(
    local_solver: np.ndarray,
    element_boundary_mats: np.ndarray,
    source_rhs: np.ndarray,
    boundary_condition: Callable,
    stabilization,
    space: DGSpace,
    *,
    boundary_penalty: float = 1.0e20,
) -> DiffusionFaceDenseAssembly:
    """Build the requested face-dense diffusion structures without solving."""

    trace_lift = diffusion_trace_lift(stabilization, space)
    trace_blocks = hdg_assembly.element_to_trace_matrix_from_lift(
        trace_lift,
        local_solver,
        element_boundary_mats,
        space,
    )
    element_blocks = build_complete_diffusion_element_blocks(
        trace_blocks,
        stabilization,
        space,
    )

    topology = build_face_topology(space.mesh.loc2glob_edge)
    interior_row_blocks = assemble_global_face_blocks(
        element_blocks,
        space.mesh.loc2glob_edge,
        topology,
        active_row_faces=space.mesh.interior_face_mask,
    )
    interior_rhs, boundary_trace = build_diffusion_interior_rhs(
        trace_lift,
        source_rhs,
        local_solver,
        boundary_condition,
        space,
    )

    penalty_system = make_penalty_system(
        interior_row_blocks,
        topology,
        interior_rhs,
        boundary_trace,
        space.mesh.bnd_edges_inds,
        boundary_penalty=boundary_penalty,
    )
    eliminated_system = eliminate_dirichlet_faces(
        interior_row_blocks,
        topology,
        interior_rhs,
        boundary_trace,
        space.mesh.int_edges_inds,
    )

    return DiffusionFaceDenseAssembly(
        topology=topology,
        trace_blocks=np.ascontiguousarray(trace_blocks),
        element_blocks=np.ascontiguousarray(element_blocks),
        interior_row_blocks=np.ascontiguousarray(interior_row_blocks),
        interior_rhs=np.ascontiguousarray(interior_rhs),
        boundary_trace=np.ascontiguousarray(boundary_trace),
        penalty_system=penalty_system,
        eliminated_system=eliminated_system,
    )


def solve_diffusion_face_dense_direct(
    source,
    reaction,
    boundary_condition: Callable,
    space: DGSpace,
    *,
    diffusion=1.0,
    stabilization=1.0,
    boundary_mode: str = "eliminate",
    boundary_penalty: float = 1.0e20,
) -> DiffusionFaceDenseDirectResult:
    """Solve a small diffusion problem entirely through face-dense assembly.

    The function is an independent end-to-end validation path:

    1. build the local mixed inverses and elemental boundary matrices;
    2. assemble the complete elemental and global face-dense blocks;
    3. choose penalty rows or direct Dirichlet elimination;
    4. materialize the small scalar matrix and solve it directly;
    5. reconstruct the volume field and conservative flux.

    It intentionally does not call the existing COO trace assembler.  The
    scalar dense materialization makes it suitable only for tests and modest
    diagnostic meshes.
    """

    if boundary_mode not in {"penalty", "eliminate"}:
        raise ValueError("boundary_mode must be 'penalty' or 'eliminate'")

    local_solver = local_solvers(
        reaction,
        stabilization,
        space,
        backend="numpy",
        diffusion=diffusion,
    )
    element_boundary_mats = diffusion_element_boundary_mats(stabilization, space)
    source_rhs = hdg_assembly.block_source_moments(
        source,
        space,
        num_blocks=3,
        source_block=0,
    )
    assembly = assemble_diffusion_face_dense_components(
        local_solver,
        element_boundary_mats,
        source_rhs,
        boundary_condition,
        stabilization,
        space,
        boundary_penalty=boundary_penalty,
    )
    system = (
        assembly.penalty_system
        if boundary_mode == "penalty"
        else assembly.eliminated_system
    )

    matrix = materialize_face_dense_matrix(system.blocks, system.neighbors)
    system_solution = np.linalg.solve(matrix, system.rhs.reshape(-1))
    relative_residual = face_dense_relative_residual(system, system_solution)
    trace = (
        system_solution
        if boundary_mode == "penalty"
        else expand_eliminated_solution(system_solution, system)
    )
    local_unknowns = hdg_assembly.reconstruct_local_unknowns(
        trace,
        source_rhs,
        local_solver,
        element_boundary_mats,
        space,
    )
    field, flux = split_diffusion_unknowns(local_unknowns, space)

    return DiffusionFaceDenseDirectResult(
        field=field,
        flux=flux,
        trace=np.ascontiguousarray(trace),
        local_unknowns=np.ascontiguousarray(local_unknowns),
        system_solution=np.ascontiguousarray(system_solution),
        relative_residual=relative_residual,
        assembly=assembly,
        system=system,
        local_solver=np.ascontiguousarray(local_solver),
        element_boundary_mats=np.ascontiguousarray(element_boundary_mats),
        source_rhs=np.ascontiguousarray(source_rhs),
    )

__all__ = [
    "DiffusionFaceDenseAssembly",
    "DiffusionFaceDenseDirectResult",
    "assemble_diffusion_face_dense_components",
    "build_complete_diffusion_element_blocks",
    "build_diffusion_interior_rhs",
    "solve_diffusion_face_dense_direct",
]