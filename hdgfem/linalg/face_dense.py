"""hdgfem.linalg.face_dense."""

from __future__ import annotations

import numpy as np
from typing import Literal
from dataclasses import dataclass


@dataclass(frozen=True)
class FaceDenseSystem:
    """A face-dense linear system.

    ``neighbors`` always contains indices into the rows of this system.  Thus
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
        """Return the number of block rows in the face system."""
        return int(self.blocks.shape[0])

    @property
    def num_slots(self) -> int:
        """Return the fixed number of block slots per row."""
        return int(self.blocks.shape[1])

    @property
    def block_size(self) -> int:
        """Return the scalar dimension of each dense face block."""
        return int(self.blocks.shape[2])

    @property
    def num_dofs(self) -> int:
        """Return the total number of scalar degrees of freedom."""
        return self.num_rows * self.block_size


def face_dense_matvec(
    blocks: np.ndarray,
    neighbors: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Apply a face-dense matrix to a face-major vector.

    Parameters
    ----------
    blocks
        Shape ``(Nrow, S, b, b)``.
    neighbors
        Shape ``(Nrow, S)`` with indices in ``[0, Nrow)`` and ``-1`` padding.
    x
        Either ``(Nrow, b)`` or flat ``(Nrow*b,)``.  The return shape matches
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


def face_dense_to_dense(blocks: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Materialize a face-dense block matrix as a scalar dense matrix."""
    blocks = np.asarray(blocks)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if blocks.ndim != 4:
        raise ValueError("blocks must have shape (Nrow, S, b, b)")
    num_rows, num_slots, block_size, block_size_2 = blocks.shape
    if block_size != block_size_2 or neighbors.shape != (num_rows, num_slots):
        raise ValueError("face block or neighbor dimensions are inconsistent")
    dense = np.zeros((num_rows * block_size, num_rows * block_size), dtype=blocks.dtype)
    for row in range(num_rows):
        row_slice = slice(row * block_size, (row + 1) * block_size)
        for slot in range(num_slots):
            column = int(neighbors[row, slot])
            if column < 0:
                continue
            if column >= num_rows:
                raise ValueError("neighbors contains a row index outside the system")
            column_slice = slice(column * block_size, (column + 1) * block_size)
            dense[row_slice, column_slice] += blocks[row, slot]
    return np.ascontiguousarray(dense)


def materialize_face_dense_matrix(
    blocks: np.ndarray,
    neighbors: np.ndarray,
) -> np.ndarray:
    """Materialize a face-dense operator as a scalar dense matrix.

    This helper is intended only for small validation problems and reference
    direct solves.  Production CPU/GPU paths should call
    :func:`face_dense_matvec` and must not form this matrix.
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
    """Return ``||A x - b||_2 / max(||b||_2, eps)`` for a face system."""

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
