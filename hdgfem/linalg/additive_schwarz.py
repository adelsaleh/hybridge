"""CPU reference additive-Schwarz preconditioner for face-dense HDG systems.

The natural HDG Schwarz subdomain is one mesh element.  For an element with
``Nlfe`` local faces and ``b`` trace unknowns per face, the local matrix has
size ``(Nlfe*b) x (Nlfe*b)``.  Its off-diagonal face blocks come directly from
the complete condensed elemental matrix.  Every active diagonal face block is
replaced by the corresponding *global* diagonal block, which adds the
contribution of the element on the other side of an interior face.  This is the
one-element overlapping ASM construction described in the GPU HDG paper.

Application follows the standard restriction--solve--prolongation cycle::

    r_e = R_e r
    z_e = P_e^{-1} r_e
    z   = sum_e R_e^T z_e

The implementation explicitly stores inverse local matrices.  This is a
correctness-oriented NumPy reference for the later GPU version, where setup can
use batched LU/inversion and application can use gather, batched GEMV, and
scatter-add kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assembly.face_dense import FaceDenseSystem


@dataclass(frozen=True)
class FaceAdditiveSchwarzLocalMatrices:
    """Uniform one-element Schwarz matrices before inversion.

    This object separates the algebraic subdomain construction from the
    factorization backend.  CPU validation can invert the matrices with NumPy,
    while CUDA/ROCm backends can transfer the same matrices and factor or invert
    them on the accelerator.
    """

    local_matrices: np.ndarray
    element_system_faces: np.ndarray
    block_size: int

    @property
    def num_elements(self) -> int:
        return int(self.local_matrices.shape[0])

    @property
    def num_local_faces(self) -> int:
        return int(self.element_system_faces.shape[1])

    @property
    def local_size(self) -> int:
        return int(self.local_matrices.shape[1])

    @property
    def num_system_faces(self) -> int:
        valid = self.element_system_faces[self.element_system_faces >= 0]
        return int(np.max(valid) + 1) if valid.size else 0

    @property
    def num_dofs(self) -> int:
        return self.num_system_faces * self.block_size


@dataclass(frozen=True)
class FaceAdditiveSchwarzPreconditioner:
    """One-element overlapping additive-Schwarz preconditioner.

    Parameters
    ----------
    local_matrices
        Dense Schwarz matrices with shape ``(NE, Nlfe*b, Nlfe*b)``.
    inverse_matrices
        Their inverses with the same shape.
    element_system_faces
        Shape ``(NE, Nlfe)``.  Entry ``[e, lf]`` is the face-row index in the
        supplied system.  Eliminated boundary faces are represented by ``-1``.
    inverse_residuals
        Per-element infinity-norm residuals
        ``||P_e P_e^{-1} - I||_inf``.
    block_size
        Number of trace degrees of freedom on one face.
    """

    local_matrices: np.ndarray
    inverse_matrices: np.ndarray
    element_system_faces: np.ndarray
    inverse_residuals: np.ndarray
    block_size: int

    @property
    def num_elements(self) -> int:
        return int(self.inverse_matrices.shape[0])

    @property
    def num_local_faces(self) -> int:
        return int(self.element_system_faces.shape[1])

    @property
    def local_size(self) -> int:
        return self.num_local_faces * self.block_size

    @property
    def num_system_faces(self) -> int:
        valid = self.element_system_faces[self.element_system_faces >= 0]
        return int(np.max(valid) + 1) if valid.size else 0

    @property
    def num_dofs(self) -> int:
        return self.num_system_faces * self.block_size

    @property
    def maximum_inverse_residual(self) -> float:
        return float(np.max(self.inverse_residuals, initial=0.0))

    def restrict(self, vector: np.ndarray) -> np.ndarray:
        """Gather a global face vector into all element subdomains.

        The returned array has shape ``(NE, Nlfe, b)``.  Entries associated
        with eliminated boundary faces are zero.
        """

        array = np.asarray(vector, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("additive-Schwarz input contains non-finite values")

        if array.ndim == 1:
            expected = (self.num_dofs,)
            if array.shape != expected:
                raise ValueError(
                    f"flat vector must have shape {expected}; got {array.shape}"
                )
            face_vector = array.reshape(self.num_system_faces, self.block_size)
        else:
            expected = (self.num_system_faces, self.block_size)
            if array.shape != expected:
                raise ValueError(
                    f"face-major vector must have shape {expected}; got {array.shape}"
                )
            face_vector = array

        element_vector = np.zeros(
            (self.num_elements, self.num_local_faces, self.block_size),
            dtype=np.float64,
        )
        active = self.element_system_faces >= 0
        element_vector[active] = face_vector[self.element_system_faces[active]]
        return np.ascontiguousarray(element_vector)

    def prolong(self, element_vector: np.ndarray, *, flat: bool = False) -> np.ndarray:
        """Scatter-add element contributions back to global face rows."""

        element_vector = np.asarray(element_vector, dtype=np.float64)
        expected = (self.num_elements, self.num_local_faces, self.block_size)
        if element_vector.shape != expected:
            raise ValueError(
                f"element_vector must have shape {expected}; got {element_vector.shape}"
            )
        if not np.all(np.isfinite(element_vector)):
            raise ValueError("element_vector contains non-finite values")

        result = np.zeros(
            (self.num_system_faces, self.block_size),
            dtype=np.float64,
        )
        active = self.element_system_faces >= 0
        np.add.at(
            result,
            self.element_system_faces[active],
            element_vector[active],
        )
        result = np.ascontiguousarray(result)
        return result.reshape(-1) if flat else result

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply the ASM restriction--local solve--prolongation cycle.

        ``vector`` may be flat or face-major.  The output preserves its shape.
        Contributions on a face shared by two elements are added, as in the
        classical additive-Schwarz operator ``sum_e R_e^T P_e^{-1} R_e``.
        """

        array = np.asarray(vector, dtype=np.float64)
        flat_input = array.ndim == 1
        element_rhs = self.restrict(array)
        element_rhs_flat = element_rhs.reshape(self.num_elements, self.local_size)

        element_solution_flat = np.einsum(
            "eij,ej->ei",
            self.inverse_matrices,
            element_rhs_flat,
            optimize=True,
        )
        element_solution = element_solution_flat.reshape(
            self.num_elements,
            self.num_local_faces,
            self.block_size,
        )
        return self.prolong(element_solution, flat=flat_input)

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        """Alias for :meth:`apply`, suitable for ``restarted_gmres``."""

        return self.apply(vector)


def _validate_inputs(
    system: FaceDenseSystem,
    element_blocks: np.ndarray,
    loc2glob_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    if not isinstance(system, FaceDenseSystem):
        raise TypeError("system must be a FaceDenseSystem")

    element_blocks = np.asarray(element_blocks, dtype=np.float64)
    loc2glob_face = np.asarray(loc2glob_face, dtype=np.int64)
    if loc2glob_face.ndim != 2:
        raise ValueError("loc2glob_face must have shape (NE, Nlfe)")
    if loc2glob_face.size == 0:
        raise ValueError("loc2glob_face must be non-empty")
    if np.min(loc2glob_face) < 0:
        raise ValueError("loc2glob_face cannot contain negative face ids")

    num_elements, num_local_faces = loc2glob_face.shape
    block_size = system.block_size
    expected = (
        num_elements,
        num_local_faces,
        num_local_faces,
        block_size,
        block_size,
    )
    if element_blocks.shape != expected:
        raise ValueError(
            f"element_blocks must have shape {expected}; got {element_blocks.shape}"
        )
    if not np.all(np.isfinite(element_blocks)):
        raise ValueError("element_blocks contains non-finite values")

    num_global_faces = int(system.global_to_local.shape[0])
    if int(np.max(loc2glob_face)) >= num_global_faces:
        raise ValueError("loc2glob_face contains a face outside system.global_to_local")

    expected_diagonal_neighbors = np.arange(system.num_rows, dtype=np.int64)
    if not np.array_equal(system.neighbors[:, 0], expected_diagonal_neighbors):
        raise ValueError(
            "additive Schwarz requires slot zero to contain every diagonal face block"
        )

    return (
        np.ascontiguousarray(element_blocks),
        np.ascontiguousarray(loc2glob_face),
        num_elements,
        num_local_faces,
        block_size,
    )

def build_face_additive_schwarz_local_matrices(
    system: FaceDenseSystem,
    element_blocks: np.ndarray,
    loc2glob_face: np.ndarray,
) -> FaceAdditiveSchwarzLocalMatrices:
    r"""Construct the uniform one-element Schwarz matrices without inversion.

    The construction is identical to the one used by
    :func:`build_face_additive_schwarz_preconditioner`: active diagonal blocks
    are enriched with the assembled global diagonal, eliminated Dirichlet faces
    are decoupled with identity blocks, and penalty boundary rows reproduce the
    global row replacement.

    Keeping this operation independent of factorization is essential for GPU
    setup: the returned matrices can be transferred once and inverted or solved
    with batched accelerator routines.
    """

    (
        element_blocks,
        loc2glob_face,
        num_elements,
        num_local_faces,
        block_size,
    ) = _validate_inputs(system, element_blocks, loc2glob_face)

    element_system_faces = system.global_to_local[loc2glob_face]
    local_blocks = element_blocks.copy()

    for element in range(num_elements):
        for local_face in range(num_local_faces):
            system_face = int(element_system_faces[element, local_face])
            if system_face >= 0:
                local_blocks[element, local_face, local_face] = system.blocks[
                    system_face, 0
                ]

    num_global_faces = int(system.global_to_local.shape[0])
    incidence_count = np.bincount(
        loc2glob_face.reshape(-1),
        minlength=num_global_faces,
    )

    identity = np.eye(block_size, dtype=np.float64)
    for element in range(num_elements):
        for local_face in range(num_local_faces):
            global_face = int(loc2glob_face[element, local_face])
            system_face = int(element_system_faces[element, local_face])

            if system_face < 0:
                local_blocks[element, local_face, :, :, :] = 0.0
                local_blocks[element, :, local_face, :, :] = 0.0
                local_blocks[element, local_face, local_face] = identity
            elif system.mode == "penalty" and incidence_count[global_face] == 1:
                local_blocks[element, local_face, :, :, :] = 0.0
                local_blocks[element, local_face, local_face] = system.blocks[
                    system_face, 0
                ]

    local_size = num_local_faces * block_size
    local_matrices = (
        local_blocks.transpose(0, 1, 3, 2, 4)
        .reshape(num_elements, local_size, local_size)
        .copy()
    )

    return FaceAdditiveSchwarzLocalMatrices(
        local_matrices=np.ascontiguousarray(local_matrices),
        element_system_faces=np.ascontiguousarray(element_system_faces),
        block_size=block_size,
    )

def build_face_additive_schwarz_preconditioner(
    system: FaceDenseSystem,
    element_blocks: np.ndarray,
    loc2glob_face: np.ndarray,
    *,
    inverse_residual_tolerance: float | None = None,
) -> FaceAdditiveSchwarzPreconditioner:
    r"""Build the one-element overlapping ASM preconditioner.

    Matrix construction is delegated to
    :func:`build_face_additive_schwarz_local_matrices`; this routine then
    computes and validates the NumPy reference inverses.
    """

    if inverse_residual_tolerance is not None:
        if (
            inverse_residual_tolerance < 0.0
            or not np.isfinite(inverse_residual_tolerance)
        ):
            raise ValueError(
                "inverse_residual_tolerance must be finite and non-negative"
            )

    local = build_face_additive_schwarz_local_matrices(
        system,
        element_blocks,
        loc2glob_face,
    )
    local_matrices = local.local_matrices
    num_elements = local.num_elements
    local_size = local.local_size

    inverse_matrices = np.empty_like(local_matrices)
    for element in range(num_elements):
        try:
            inverse_matrices[element] = np.linalg.inv(local_matrices[element])
        except np.linalg.LinAlgError as exc:
            active_faces = local.element_system_faces[element].tolist()
            raise np.linalg.LinAlgError(
                "additive-Schwarz local matrix is singular: "
                f"element={element}, system_faces={active_faces}"
            ) from exc

    identity_local = np.eye(local_size, dtype=np.float64)
    products = np.matmul(local_matrices, inverse_matrices)
    inverse_residuals = np.max(
        np.sum(np.abs(products - identity_local[None, :, :]), axis=2),
        axis=1,
    )
    if not np.all(np.isfinite(inverse_residuals)):
        raise FloatingPointError(
            "additive-Schwarz inversion produced non-finite residuals"
        )
    if (
        inverse_residual_tolerance is not None
        and np.any(inverse_residuals > inverse_residual_tolerance)
    ):
        element = int(np.argmax(inverse_residuals))
        raise FloatingPointError(
            "additive-Schwarz inverse failed the requested residual tolerance: "
            f"element={element}, residual={inverse_residuals[element]:.3e}, "
            f"tolerance={inverse_residual_tolerance:.3e}"
        )

    return FaceAdditiveSchwarzPreconditioner(
        local_matrices=local_matrices,
        inverse_matrices=np.ascontiguousarray(inverse_matrices),
        element_system_faces=local.element_system_faces,
        inverse_residuals=np.ascontiguousarray(inverse_residuals),
        block_size=local.block_size,
    )

__all__ = [
    "FaceAdditiveSchwarzLocalMatrices",
    "FaceAdditiveSchwarzPreconditioner",
    "build_face_additive_schwarz_local_matrices",
    "build_face_additive_schwarz_preconditioner",
]