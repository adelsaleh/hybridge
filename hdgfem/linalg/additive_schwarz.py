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
from scipy.sparse import bsr_matrix, csr_matrix, isspmatrix_bsr

from hdgfem.assembly.face_dense import FaceDenseSystem


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
        """Return the number of elements."""
        return int(self.local_matrices.shape[0])

    @property
    def num_local_faces(self) -> int:
        """Return the number of local faces per element."""
        return int(self.element_system_faces.shape[1])

    @property
    def local_size(self) -> int:
        """Return the element-patch vector size."""
        return int(self.local_matrices.shape[1])

    @property
    def num_system_faces(self) -> int:
        """Return the number of faces in the reduced system."""
        valid = self.element_system_faces[self.element_system_faces >= 0]
        return int(np.max(valid) + 1) if valid.size else 0

    @property
    def num_dofs(self) -> int:
        """Return the number of scalar degrees of freedom."""
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
        """Return the number of elements."""
        return int(self.inverse_matrices.shape[0])

    @property
    def num_local_faces(self) -> int:
        """Return the number of local faces per element."""
        return int(self.element_system_faces.shape[1])

    @property
    def local_size(self) -> int:
        """Return the element-patch vector size."""
        return self.num_local_faces * self.block_size

    @property
    def num_system_faces(self) -> int:
        """Return the number of faces in the reduced system."""
        valid = self.element_system_faces[self.element_system_faces >= 0]
        return int(np.max(valid) + 1) if valid.size else 0

    @property
    def num_dofs(self) -> int:
        """Return the number of scalar degrees of freedom."""
        return self.num_system_faces * self.block_size

    @property
    def maximum_inverse_residual(self) -> float:
        """Return the largest local inverse residual."""
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
    """Validate shapes, dtypes, devices, and solver parameters."""
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



def _validate_bsr_patch_inputs(
    matrix: bsr_matrix,
    patches: np.ndarray,
    *,
    num_local_faces: int | None = None,
) -> np.ndarray:
    """Validate reduced face maps for either element or wider Schwarz patches."""
    if not isspmatrix_bsr(matrix):
        raise TypeError("matrix must be a SciPy BSR matrix")
    block_size, block_columns = matrix.blocksize
    if matrix.shape[0] != matrix.shape[1] or block_size != block_columns:
        raise ValueError("matrix and its BSR blocks must be square")
    if not matrix.has_canonical_format:
        raise ValueError("matrix must have sorted BSR indices without duplicates")
    if matrix.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("matrix must contain real float32 or float64 values")
    _require_finite_patch_values(matrix.data, "matrix contains non-finite values")
    faces = np.asarray(patches)
    if faces.ndim != 2 or not all(faces.shape):
        raise ValueError("patches must have nonempty shape (num_patches, width)")
    if num_local_faces is not None and faces.shape[1] != num_local_faces:
        raise ValueError(f"element_system_faces must have nonempty shape (NE, {num_local_faces})")
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("element_system_faces must contain integer face indices")
    num_faces = matrix.shape[0] // block_size
    if np.any(faces < -1) or np.any(faces >= num_faces):
        raise ValueError("element_system_faces contains an invalid system face")
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    for row in range(faces.shape[1]):
        for column in range(row):
            if np.any((faces[:, row] >= 0) & (faces[:, row] == faces[:, column])):
                raise ValueError("a patch cannot repeat an active system face")
    if np.any(np.bincount(faces[faces >= 0], minlength=num_faces) == 0):
        raise ValueError("patches must cover every matrix face row")
    return faces


def _require_finite_patch_values(values: np.ndarray, message: str) -> None:
    """Bound temporary finiteness masks even for nine-face inverse tensors."""
    values_per_item = int(np.prod(values.shape[1:], dtype=np.int64))
    batch_size = max(1, (1 << 22) // max(1, values_per_item))
    for start in range(0, values.shape[0], batch_size):
        if not np.all(np.isfinite(values[start : start + batch_size])):
            raise ValueError(message)


def _bsr_patch_positions(
    indptr: np.ndarray,
    indices: np.ndarray,
    faces: np.ndarray,
    *,
    allow_missing: bool = False,
    require_covered: bool = False,
) -> np.ndarray:
    """Look up patch face pairs using only a block-level sparse graph."""
    width = faces.shape[1]
    positions = np.full((faces.shape[0], width, width), -1, dtype=np.int64)
    covered = np.zeros(indices.size, dtype=bool) if require_covered else None
    for row in range(width):
        for column in range(width):
            active = (faces[:, row] >= 0) & (faces[:, column] >= 0)
            if not np.any(active):
                continue
            row_faces = faces[active, row]
            column_faces = faces[active, column]
            left = indptr[row_faces].astype(np.int64)
            end = indptr[row_faces + 1].astype(np.int64)
            right = end.copy()
            # Batched lower_bound within each short face row. Only O(Npatch)
            # integer workspace is needed, even for millions of BSR blocks.
            while np.any(left < right):
                searching = left < right
                middle = left[searching] + (right[searching] - left[searching]) // 2
                below = indices[middle] < column_faces[searching]
                left[searching] = np.where(below, middle + 1, left[searching])
                right[searching] = np.where(below, right[searching], middle)
            found = left < end
            found[found] &= indices[left[found]] == column_faces[found]
            if (not allow_missing or row == column) and not np.all(found):
                missing = int(np.flatnonzero(~found)[0])
                raise ValueError(
                    "matrix pattern is missing an active element face pair: "
                    f"row={row_faces[missing]}, column={column_faces[missing]}"
                )
            positions[active, row, column] = np.where(found, left, -1)
            if covered is not None:
                covered[left[found]] = True
    if covered is not None and not np.all(covered):
        raise ValueError("matrix pattern contains blocks outside the element face pairs")
    return positions


def _gather_bsr_patch_matrices(
    matrix: bsr_matrix, faces: np.ndarray, positions: np.ndarray
) -> FaceAdditiveSchwarzLocalMatrices:
    block_size = matrix.blocksize[0]
    width = faces.shape[1]
    local_size = width * block_size
    local_matrices = np.zeros((faces.shape[0], local_size, local_size), dtype=np.float64)
    identity = np.eye(block_size, dtype=np.float64)
    for start in range(0, faces.shape[0], 65536):
        stop = min(start + 65536, faces.shape[0])
        for row in range(width):
            row_slice = slice(row * block_size, (row + 1) * block_size)
            for column in range(width):
                column_slice = slice(column * block_size, (column + 1) * block_size)
                local_block = local_matrices[start:stop, row_slice, column_slice]
                slots = positions[start:stop, row, column]
                present = slots >= 0
                local_block[present] = matrix.data[slots[present]]
                if row == column:
                    local_block[faces[start:stop, row] < 0] = identity
    return FaceAdditiveSchwarzLocalMatrices(
        local_matrices=local_matrices,
        element_system_faces=faces,
        block_size=block_size,
    )


def _assemble_bsr_patch_correction(
    matrix: bsr_matrix,
    local: FaceAdditiveSchwarzLocalMatrices,
    inverse_matrices: np.ndarray,
    positions: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> bsr_matrix:
    block_size = matrix.blocksize[0]
    if local.block_size != block_size:
        raise ValueError("local block size does not match the matrix")
    faces = local.element_system_faces
    width = faces.shape[1]
    inverse_matrices = np.asarray(inverse_matrices)
    expected = (faces.shape[0], width * block_size, width * block_size)
    if inverse_matrices.shape != expected:
        raise ValueError(f"inverse_matrices must have shape {expected}")
    if inverse_matrices.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("inverse_matrices must contain real float32 or float64 values")
    _require_finite_patch_values(
        inverse_matrices, "inverse_matrices contains non-finite values"
    )
    values = np.zeros(
        (indices.size, block_size, block_size),
        dtype=np.result_type(matrix.dtype, inverse_matrices.dtype),
    )
    for start in range(0, faces.shape[0], 65536):
        stop = min(start + 65536, faces.shape[0])
        inverse_batch = inverse_matrices[start:stop]
        for row in range(width):
            row_slice = slice(row * block_size, (row + 1) * block_size)
            for column in range(width):
                column_slice = slice(column * block_size, (column + 1) * block_size)
                slots = positions[start:stop, row, column]
                active = slots >= 0
                np.add.at(
                    values,
                    slots[active],
                    inverse_batch[:, row_slice, column_slice][active],
                )
    return bsr_matrix(
        (values, indices.copy(), indptr.copy()), shape=matrix.shape
    )


def build_bsr_face_additive_schwarz_local_matrices(
    matrix: bsr_matrix,
    element_system_faces: np.ndarray,
) -> FaceAdditiveSchwarzLocalMatrices:
    """Gather exact three-face principal patches from an assembled BSR matrix.

    element_system_faces contains reduced face indices in the same order as
    matrix rows; use global_to_free[mesh.loc2glob_edge] after boundary
    elimination. Eliminated faces are -1 and receive decoupled identity blocks.
    All active blocks, including global diagonals, are copied once.

    The input must use sorted, duplicate-free BSR storage. Its block pattern
    must equal the union of active element face pairs, including structural
    zeros. The returned FP64 matrices can be inverted separately in batches.
    """
    faces = _validate_bsr_patch_inputs(
        matrix, element_system_faces, num_local_faces=3
    )
    positions = _bsr_patch_positions(
        matrix.indptr, matrix.indices, faces, require_covered=True
    )
    return _gather_bsr_patch_matrices(matrix, faces, positions)


def assemble_bsr_face_additive_schwarz_correction(
    matrix: bsr_matrix,
    local: FaceAdditiveSchwarzLocalMatrices,
    inverse_matrices: np.ndarray,
) -> bsr_matrix:
    """Assemble element ASM while preserving exactly the original BSR pattern.

    This applies the same unweighted restriction/prolongation as
    FaceAdditiveSchwarzPreconditioner.apply once during setup. The input block
    pattern must equal the union of three-face element cliques. Eliminated
    identity-padded faces never contribute. Numeric accumulation is bounded
    and uses np.add.at as in assemble_global_face_blocks.
    """
    if not isinstance(local, FaceAdditiveSchwarzLocalMatrices):
        raise TypeError("local must be FaceAdditiveSchwarzLocalMatrices")
    faces = _validate_bsr_patch_inputs(
        matrix, local.element_system_faces, num_local_faces=3
    )
    positions = _bsr_patch_positions(
        matrix.indptr, matrix.indices, faces, require_covered=True
    )
    return _assemble_bsr_patch_correction(
        matrix, local, inverse_matrices, positions, matrix.indices, matrix.indptr
    )


def build_bsr_additive_schwarz_local_matrices(
    matrix: bsr_matrix,
    patches: np.ndarray,
) -> FaceAdditiveSchwarzLocalMatrices:
    """Gather exact principal matrices for arbitrary fixed-width face patches.

    patches has shape (num_patches, width) with reduced matrix face indices.
    Active indices must be distinct within each patch; -1 denotes padding.
    Every matrix face must occur in at least one patch. The returned existing
    local-matrix container uses element_system_faces for this patch map and
    num_elements for the number of patches.

    Missing off-diagonal BSR entries are zero in the principal matrix. Missing
    active diagonals are rejected. Padded faces receive decoupled identities.
    For an SPD input, the active principal matrices are SPD; this routine does
    not perform factorization or an independent global SPD test.
    """
    faces = _validate_bsr_patch_inputs(matrix, patches)
    positions = _bsr_patch_positions(
        matrix.indptr, matrix.indices, faces, allow_missing=True
    )
    return _gather_bsr_patch_matrices(matrix, faces, positions)


def assemble_bsr_additive_schwarz_correction(
    matrix: bsr_matrix,
    local: FaceAdditiveSchwarzLocalMatrices,
    inverse_matrices: np.ndarray,
) -> bsr_matrix:
    """Assemble wider-patch ASM, including inverse fill, as a BSR correction.

    The result is sum_p R_p.T @ inverse_matrices[p] @ R_p with the same block
    size as matrix. Its block graph is the union of patch cliques, so fill
    absent from the original A is retained. The symbolic incidence product
    operates only on face indices, never on expanded scalar coefficients.
    Numeric values are accumulated in bounded batches with the same helper
    used for the strict three-face element correction.
    """
    if not isinstance(local, FaceAdditiveSchwarzLocalMatrices):
        raise TypeError("local must be FaceAdditiveSchwarzLocalMatrices")
    faces = _validate_bsr_patch_inputs(matrix, local.element_system_faces)
    num_faces = matrix.shape[0] // matrix.blocksize[0]
    active = faces >= 0
    patch_rows = np.broadcast_to(
        np.arange(faces.shape[0], dtype=np.int64)[:, None], faces.shape
    )
    incidence = csr_matrix(
        (np.ones(np.count_nonzero(active), dtype=bool),
         (patch_rows[active], faces[active])),
        shape=(faces.shape[0], num_faces),
    )
    graph = (incidence.T @ incidence).tocsr()
    graph.sort_indices()
    positions = _bsr_patch_positions(graph.indptr, graph.indices, faces)
    return _assemble_bsr_patch_correction(
        matrix, local, inverse_matrices, positions, graph.indices, graph.indptr
    )


__all__ = [
    "FaceAdditiveSchwarzLocalMatrices",
    "FaceAdditiveSchwarzPreconditioner",
    "build_face_additive_schwarz_local_matrices",
    "build_face_additive_schwarz_preconditioner",
    "build_bsr_face_additive_schwarz_local_matrices",
    "assemble_bsr_face_additive_schwarz_correction",
    "build_bsr_additive_schwarz_local_matrices",
    "assemble_bsr_additive_schwarz_correction",
]
