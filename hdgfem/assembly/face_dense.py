"""Face-dense storage for condensed HDG trace systems.

The routines in this module are intentionally CPU/NumPy reference
implementations.  They define the data structures and exact algebra that can
later be ported to CUDA/HIP without changing the mathematical layout.

For a scalar problem with 'b' trace dofs per face and 'S' face-neighbour
slots, the matrix is stored as

    blocks[row_face, slot, row_dof, col_dof]

with shape '(num_rows, S, b, b)'.  'neighbors[row_face, slot]' identifies
the column face represented by a block.  Slot zero is always the diagonal
face block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FaceTopology:
    """Fixed-width face connectivity for an HDG skeleton matrix.

    Attributes
    ----------
    neighbors
        '(NF, S)' global face ids.  'neighbors[f, 0] == f' and unused
        slots are '-1'.
    adjacent_elements
        '(NF, 2)' element ids incident to each global face.  Boundary faces
        use only column zero and pad the second column with '-1'.
    adjacent_local_faces
        '(NF, 2)' local-face ids corresponding to 'adjacent_elements'.
    incidence_count
        Number of incident elements for each face (one or two on a manifold
        triangular mesh).
    element_face_slots
        '(NE, Nlfe, Nlfe)' lookup.  Entry '[e, r, c]' is the slot in the
        global row of face 'loc2glob_face[e, r]' corresponding to column
        face 'loc2glob_face[e, c]'.
    """

    neighbors: np.ndarray
    adjacent_elements: np.ndarray
    adjacent_local_faces: np.ndarray
    incidence_count: np.ndarray
    element_face_slots: np.ndarray

    @property
    def num_faces(self) -> int:
        return int(self.neighbors.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.neighbors.shape[1])


@dataclass(frozen=True)
class FaceDenseSystem:
    """A face-dense linear system.

    'neighbors' always contains indices into the rows of this system.  Thus
    it contains global face ids for a full/penalty system and compact reduced
    face ids for an eliminated system.
    """

    blocks: np.ndarray
    neighbors: np.ndarray
    rhs: np.ndarray
    global_faces: np.ndarray
    global_to_local: np.ndarray
    boundary_trace: np.ndarray
    mode: Literal["penalty", "eliminate"]

    @property
    def num_rows(self) -> int:
        return int(self.blocks.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.blocks.shape[1])

    @property
    def block_size(self) -> int:
        return int(self.blocks.shape[2])

    @property
    def num_dofs(self) -> int:
        return self.num_rows * self.block_size


def build_face_topology(loc2glob_face: np.ndarray) -> FaceTopology:
    """Build all face-connectivity tables needed by direct block assembly.

    Parameters
    ----------
    loc2glob_face
        Element-to-global-face table of shape '(NE, Nlfe)'.
    """

    loc2glob_face = np.ascontiguousarray(loc2glob_face, dtype=np.int64)
    if loc2glob_face.ndim != 2:
        raise ValueError("loc2glob_face must have shape (NE, Nlfe)")
    if loc2glob_face.size == 0:
        raise ValueError("loc2glob_face must be non-empty")
    if np.min(loc2glob_face) < 0:
        raise ValueError("loc2glob_face cannot contain negative face ids")

    num_elements, num_local_faces = loc2glob_face.shape
    num_faces = int(np.max(loc2glob_face)) + 1
    num_slots = 2 * num_local_faces - 1

    adjacent_elements = np.full((num_faces, 2), -1, dtype=np.int64)
    adjacent_local_faces = np.full((num_faces, 2), -1, dtype=np.int64)
    incidence_count = np.zeros(num_faces, dtype=np.int64)

    for element in range(num_elements):
        for local_face in range(num_local_faces):
            face = int(loc2glob_face[element, local_face])
            side = int(incidence_count[face])
            if side >= 2:
                raise ValueError(
                    f"global face {face} belongs to more than two elements; "
                    "the fixed manifold-face layout is not applicable"
                )
            adjacent_elements[face, side] = element
            adjacent_local_faces[face, side] = local_face
            incidence_count[face] += 1

    if np.any(incidence_count == 0):
        missing = np.flatnonzero(incidence_count == 0)[:10]
        raise ValueError(f"unused global face ids detected: {missing.tolist()}")

    neighbors = np.full((num_faces, num_slots), -1, dtype=np.int64)

    for face in range(num_faces):
        # Slot zero is always the diagonal block.
        row_neighbors = [face]

        for side in range(int(incidence_count[face])):
            element = int(adjacent_elements[face, side])
            for column_face in loc2glob_face[element]:
                column_face = int(column_face)
                if column_face not in row_neighbors:
                    row_neighbors.append(column_face)

        if len(row_neighbors) > num_slots:
            raise ValueError(
                f"face {face} needs {len(row_neighbors)} slots, but the fixed "
                f"layout provides only {num_slots}"
            )
        neighbors[face, : len(row_neighbors)] = row_neighbors

    element_face_slots = np.full(
        (num_elements, num_local_faces, num_local_faces),
        -1,
        dtype=np.int64,
    )

    for element in range(num_elements):
        for row_local_face in range(num_local_faces):
            row_global_face = int(loc2glob_face[element, row_local_face])
            slot_lookup = {
                int(column_global_face): slot
                for slot, column_global_face in enumerate(neighbors[row_global_face])
                if column_global_face >= 0
            }
            for column_local_face in range(num_local_faces):
                column_global_face = int(loc2glob_face[element, column_local_face])
                element_face_slots[element, row_local_face, column_local_face] = slot_lookup[
                    column_global_face
                ]

    return FaceTopology(
        neighbors=np.ascontiguousarray(neighbors),
        adjacent_elements=np.ascontiguousarray(adjacent_elements),
        adjacent_local_faces=np.ascontiguousarray(adjacent_local_faces),
        incidence_count=np.ascontiguousarray(incidence_count),
        element_face_slots=np.ascontiguousarray(element_face_slots),
    )


def assemble_global_face_blocks(
    element_blocks: np.ndarray,
    loc2glob_face: np.ndarray,
    topology: FaceTopology,
    *,
    active_row_faces: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate complete elemental face blocks into global face rows.

    Parameters
    ----------
    element_blocks
        Shape '(NE, Nlfe, Nlfe, b, b)'.  These blocks must already include
        every local contribution, including the stabilization face mass.
    loc2glob_face
        Shape '(NE, Nlfe)'.
    topology
        Result of :func:'build_face_topology'.
    active_row_faces
        Optional boolean mask of shape '(NE, Nlfe)'. Only elemental block
        rows for which the mask is true are assembled. For strong Dirichlet
        treatment in the current code, use 'mesh.interior_face_mask' so
        boundary equations are left empty and can then be replaced or removed.
    """

    loc2glob_face = np.asarray(loc2glob_face, dtype=np.int64)
    element_blocks = np.asarray(element_blocks, dtype=np.float64)

    if loc2glob_face.ndim != 2:
        raise ValueError("loc2glob_face must have shape (NE, Nlfe)")
    num_elements, num_local_faces = loc2glob_face.shape

    if element_blocks.ndim != 5:
        raise ValueError("element_blocks must have shape (NE, Nlfe, Nlfe, b, b)")
    block_size = int(element_blocks.shape[-1])
    expected_shape = (
        num_elements,
        num_local_faces,
        num_local_faces,
        block_size,
        block_size,
    )
    if element_blocks.shape != expected_shape:
        raise ValueError(
            f"element_blocks must have shape {expected_shape}; got {element_blocks.shape}"
        )
    if topology.element_face_slots.shape != (
        num_elements,
        num_local_faces,
        num_local_faces,
    ):
        raise ValueError("topology is incompatible with loc2glob_face")

    if active_row_faces is None:
        active_row_faces = np.ones((num_elements, num_local_faces), dtype=bool)
    else:
        active_row_faces = np.asarray(active_row_faces, dtype=bool)
        if active_row_faces.shape != (num_elements, num_local_faces):
            raise ValueError(
                "active_row_faces must have shape "
                f"({num_elements}, {num_local_faces}); got {active_row_faces.shape}"
            )

    blocks = np.zeros(
        (topology.num_faces, topology.num_slots, block_size, block_size),
        dtype=np.float64,
    )

    # Expand the row mask and global row-face ids over all local column faces.
    active = np.broadcast_to(
        active_row_faces[:, :, None],
        (num_elements, num_local_faces, num_local_faces),
    )
    row_faces = np.broadcast_to(
        loc2glob_face[:, :, None],
        (num_elements, num_local_faces, num_local_faces),
    )
    slots = topology.element_face_slots

    # np.add.at is required because two adjacent elements contribute to the same
    # global interior-face row (especially to its diagonal block).
    np.add.at(
        blocks,
        (row_faces[active], slots[active]),
        element_blocks[active],
    )

    return np.ascontiguousarray(blocks)


def make_penalty_system(
    interior_row_blocks: np.ndarray,
    topology: FaceTopology,
    interior_rhs: np.ndarray,
    boundary_trace: np.ndarray,
    boundary_faces: np.ndarray,
    *,
    boundary_penalty: float = 1.0e20,
) -> FaceDenseSystem:
    """Replace Dirichlet boundary rows by 'gamma I' equations.

    The resulting system exactly represents the penalty-row convention used by
    the current COO assembly: interior rows keep all their couplings (including
    columns on boundary faces), while every boundary row becomes
    'gamma * uhat_f = gamma * g_f'.
    """

    blocks = np.asarray(interior_row_blocks, dtype=np.float64).copy()
    neighbors = np.asarray(topology.neighbors, dtype=np.int64)
    interior_rhs = np.asarray(interior_rhs, dtype=np.float64)
    boundary_trace = np.asarray(boundary_trace, dtype=np.float64)
    boundary_faces = np.asarray(boundary_faces, dtype=np.int64)

    if blocks.ndim != 4 or blocks.shape[0] != topology.num_faces:
        raise ValueError("interior_row_blocks has an incompatible shape")
    block_size = int(blocks.shape[2])
    if blocks.shape != (
        topology.num_faces,
        topology.num_slots,
        block_size,
        block_size,
    ):
        raise ValueError("interior_row_blocks must have shape (NF, S, b, b)")
    if interior_rhs.shape != (topology.num_faces, block_size):
        raise ValueError(
            f"interior_rhs must have shape ({topology.num_faces}, {block_size})"
        )
    if boundary_trace.shape != (topology.num_faces, block_size):
        raise ValueError(
            f"boundary_trace must have shape ({topology.num_faces}, {block_size})"
        )
    if not np.isfinite(boundary_penalty) or boundary_penalty <= 0.0:
        raise ValueError("boundary_penalty must be a positive finite number")

    rhs = interior_rhs.copy()
    identity = np.eye(block_size, dtype=np.float64)

    blocks[boundary_faces] = 0.0
    blocks[boundary_faces, 0] = boundary_penalty * identity[None, :, :]
    rhs[boundary_faces] = boundary_penalty * boundary_trace[boundary_faces]

    global_faces = np.arange(topology.num_faces, dtype=np.int64)
    return FaceDenseSystem(
        blocks=np.ascontiguousarray(blocks),
        neighbors=np.ascontiguousarray(neighbors),
        rhs=np.ascontiguousarray(rhs),
        global_faces=global_faces,
        global_to_local=global_faces.copy(),
        boundary_trace=np.ascontiguousarray(boundary_trace),
        mode="penalty",
    )


def eliminate_dirichlet_faces(
    interior_row_blocks: np.ndarray,
    topology: FaceTopology,
    interior_rhs: np.ndarray,
    boundary_trace: np.ndarray,
    free_faces: np.ndarray,
    ) -> FaceDenseSystem:
    r"""Eliminate prescribed boundary faces blockwise.

    This forms

    'K_ff * u_free = rhs_free - K_fb * g_boundary'

    où

    K = [ K_ff  K_fb
          K_bf  K_bb ]

    while retaining the same fixed number of stencil slots per row.  Slots
    formerly associated with boundary columns are set to '-1' and their
    blocks are zeroed after their action has been transferred to the RHS.
    """

    blocks = np.asarray(interior_row_blocks, dtype=np.float64)
    interior_rhs = np.asarray(interior_rhs, dtype=np.float64)
    boundary_trace = np.asarray(boundary_trace, dtype=np.float64)
    free_faces = np.ascontiguousarray(free_faces, dtype=np.int64)

    if blocks.ndim != 4 or blocks.shape[0] != topology.num_faces:
        raise ValueError("interior_row_blocks has an incompatible shape")
    block_size = int(blocks.shape[2])
    if blocks.shape != (
        topology.num_faces,
        topology.num_slots,
        block_size,
        block_size,
    ):
        raise ValueError("interior_row_blocks must have shape (NF, S, b, b)")
    if interior_rhs.shape != (topology.num_faces, block_size):
        raise ValueError(
            f"interior_rhs must have shape ({topology.num_faces}, {block_size})"
        )
    if boundary_trace.shape != (topology.num_faces, block_size):
        raise ValueError(
            f"boundary_trace must have shape ({topology.num_faces}, {block_size})"
        )
    if free_faces.ndim != 1:
        raise ValueError("free_faces must be one-dimensional")
    if free_faces.size and (free_faces.min() < 0 or free_faces.max() >= topology.num_faces):
        raise ValueError("free_faces contains a face id outside the mesh")
    if np.unique(free_faces).size != free_faces.size:
        raise ValueError("free_faces must not contain duplicates")

    num_free = int(free_faces.size)
    global_to_free = np.full(topology.num_faces, -1, dtype=np.int64)
    global_to_free[free_faces] = np.arange(num_free, dtype=np.int64)

    reduced_blocks = np.zeros(
        (num_free, topology.num_slots, block_size, block_size),
        dtype=np.float64,
    )
    reduced_neighbors = np.full(
        (num_free, topology.num_slots),
        -1,
        dtype=np.int64,
    )
    reduced_rhs = interior_rhs[free_faces].copy()

    global_neighbors = topology.neighbors[free_faces]
    row_blocks = blocks[free_faces]

    # Only S iterations are needed (S=5 for triangles).  Each slot is handled
    # vectorially over all free row faces.
    for slot in range(topology.num_slots):
        column_global = global_neighbors[:, slot]
        valid = column_global >= 0
        if not np.any(valid):
            continue

        column_local = np.full(num_free, -1, dtype=np.int64)
        column_local[valid] = global_to_free[column_global[valid]]

        free_column = valid & (column_local >= 0)
        if np.any(free_column):
            reduced_blocks[free_column, slot] = row_blocks[free_column, slot]
            reduced_neighbors[free_column, slot] = column_local[free_column]

        known_column = valid & (column_local < 0)
        if np.any(known_column):
            reduced_rhs[known_column] -= np.einsum(
                "fij,fj->fi",
                row_blocks[known_column, slot],
                boundary_trace[column_global[known_column]],
                optimize=True,
            )

    return FaceDenseSystem(
        blocks=np.ascontiguousarray(reduced_blocks),
        neighbors=np.ascontiguousarray(reduced_neighbors),
        rhs=np.ascontiguousarray(reduced_rhs),
        global_faces=free_faces,
        global_to_local=np.ascontiguousarray(global_to_free),
        boundary_trace=np.ascontiguousarray(boundary_trace),
        mode="eliminate",
    )


def face_dense_matvec(
    blocks: np.ndarray,
    neighbors: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Apply a face-dense matrix to a face-major vector.

    Parameters
    ----------
    blocks
        Shape '(Nrow, S, b, b)'.
    neighbors
        Shape '(Nrow, S)' with indices in '[0, Nrow)' and '-1' padding.
    x
        Either '(Nrow, b)' or flat '(Nrow*b,)'.  The return shape matches
        the input shape.
    """

    blocks = np.asarray(blocks)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    x = np.asarray(x)

    if blocks.ndim != 4:
        raise ValueError("blocks must have shape (Nrow, S, b, b)")
    num_rows, num_slots, block_size, block_size_2 = blocks.shape
    if block_size != block_size_2:
        raise ValueError("face blocks must be square")
    if neighbors.shape != (num_rows, num_slots):
        raise ValueError(
            f"neighbors must have shape ({num_rows}, {num_slots}); got {neighbors.shape}"
        )
    valid_neighbors = neighbors[neighbors >= 0]
    if valid_neighbors.size and valid_neighbors.max() >= num_rows:
        raise ValueError("neighbors contains a row index outside the system")

    flat_input = x.ndim == 1
    if flat_input:
        if x.shape != (num_rows * block_size,):
            raise ValueError(
                f"flat x must have shape ({num_rows * block_size},); got {x.shape}"
            )
        x_faces = x.reshape(num_rows, block_size)
    else:
        if x.shape != (num_rows, block_size):
            raise ValueError(
                f"x must have shape ({num_rows}, {block_size}); got {x.shape}"
            )
        x_faces = x

    # Gather the central and neighbouring face vectors contiguously.  This is
    # the exact operation that will later become the GPU gather kernel.
    x_extended = np.zeros(
        (num_rows, num_slots, block_size),
        dtype=np.result_type(blocks.dtype, x_faces.dtype),
    )
    valid = neighbors >= 0
    x_extended[valid] = x_faces[neighbors[valid]]

    y_faces = np.einsum(
        "fsij,fsj->fi",
        blocks,
        x_extended,
        optimize=True,
    )
    y_faces = np.ascontiguousarray(y_faces)
    return y_faces.reshape(-1) if flat_input else y_faces



def materialize_face_dense_matrix(
    blocks: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    """Materialize a face-dense operator as a scalar dense matrix.

    This helper is intended only for small validation problems and reference
    direct solves.  Production CPU/GPU paths should call
    :func:'face_dense_matvec' and must not form this matrix.
    """

    blocks = np.asarray(blocks)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if blocks.ndim != 4:
        raise ValueError("blocks must have shape (Nrow, S, b, b)")
    num_rows, num_slots, block_size, block_size_2 = blocks.shape
    if block_size != block_size_2:
        raise ValueError("face blocks must be square")
    if neighbors.shape != (num_rows, num_slots):
        raise ValueError(
            f"neighbors must have shape ({num_rows}, {num_slots}); got {neighbors.shape}"
        )
    valid_neighbors = neighbors[neighbors >= 0]
    if valid_neighbors.size and valid_neighbors.max() >= num_rows:
        raise ValueError("neighbors contains a row index outside the system")

    matrix = np.zeros(
        (num_rows * block_size, num_rows * block_size),
        dtype=blocks.dtype,
    )
    for row_face in range(num_rows):
        row = slice(row_face * block_size, (row_face + 1) * block_size)
        for slot in range(num_slots):
            column_face = int(neighbors[row_face, slot])
            if column_face < 0:
                continue
            column = slice(
                column_face * block_size,
                (column_face + 1) * block_size,
            )
            matrix[row, column] += blocks[row_face, slot]
    return np.ascontiguousarray(matrix)


def face_dense_relative_residual(
    system: FaceDenseSystem,
    solution: np.ndarray,
) -> float:
    """Return '||A x - b||_2 / max(||b||_2, eps)' for a face system."""

    solution = np.asarray(solution, dtype=np.float64)
    if solution.shape not in {
        (system.num_dofs,),
        (system.num_rows, system.block_size),
    }:
        raise ValueError("solution has an incompatible shape")
    residual = face_dense_matvec(system.blocks, system.neighbors, solution)
    residual = residual.reshape(-1) - system.rhs.reshape(-1)
    rhs_norm = float(np.linalg.norm(system.rhs.reshape(-1)))
    denominator = max(rhs_norm, np.finfo(np.float64).eps)
    return float(np.linalg.norm(residual) / denominator)

def expand_eliminated_solution(
    reduced_trace: np.ndarray,
    system: FaceDenseSystem,
) -> np.ndarray:
    """Reinsert prescribed boundary traces into an eliminated solution."""

    if system.mode != "eliminate":
        raise ValueError("expand_eliminated_solution requires an eliminated system")
    reduced_trace = np.asarray(reduced_trace, dtype=np.float64)
    block_size = system.block_size
    flat_input = reduced_trace.ndim == 1
    if flat_input:
        if reduced_trace.shape != (system.num_rows * block_size,):
            raise ValueError("reduced_trace has an incompatible flat shape")
        reduced_faces = reduced_trace.reshape(system.num_rows, block_size)
    else:
        if reduced_trace.shape != (system.num_rows, block_size):
            raise ValueError("reduced_trace has an incompatible face-major shape")
        reduced_faces = reduced_trace

    full = system.boundary_trace.copy()
    full[system.global_faces] = reduced_faces
    return np.ascontiguousarray(full.reshape(-1) if flat_input else full)


__all__ = [
    "FaceDenseSystem",
    "FaceTopology",
    "assemble_global_face_blocks",
    "build_face_topology",
    "eliminate_dirichlet_faces",
    "expand_eliminated_solution",
    "face_dense_matvec",
    "face_dense_relative_residual",
    "materialize_face_dense_matrix",
    "make_penalty_system",
]