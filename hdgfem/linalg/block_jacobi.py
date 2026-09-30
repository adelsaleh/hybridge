"""CPU reference block-Jacobi preconditioner for face-dense HDG systems.

The condensed HDG matrix stores one small dense diagonal block per trace face
in slot zero.  The block-Jacobi preconditioner is therefore

    P^{-1} = diag(K_00^{-1}, K_11^{-1}, ..., K_FF^{-1}).

This module explicitly stores those inverse face blocks.  That is the same data
layout intended for the first GPU implementation, where setup will use batched
LU/inversion and application will use a batched dense matrix-vector product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hdgfem.assembly.face_dense import FaceDenseSystem


@dataclass(frozen=True)
class FaceBlockJacobiPreconditioner:
    """Inverse diagonal face blocks and their CPU application.

    Parameters
    ----------
    inverse_blocks
        Array of shape ``(Nface, b, b)``.  Entry ``inverse_blocks[f]`` is the
        inverse of the diagonal block associated with face row ``f``.
    inverse_residuals
        Infinity-norm residuals ``||K_ff K_ff^{-1} - I||_inf`` for every face.
        They are setup diagnostics only and are not used during application.
    """

    inverse_blocks: np.ndarray
    inverse_residuals: np.ndarray

    @property
    def num_faces(self) -> int:
        """Return the number of faces."""
        return int(self.inverse_blocks.shape[0])

    @property
    def block_size(self) -> int:
        """Return the dense face-block size."""
        return int(self.inverse_blocks.shape[1])

    @property
    def num_dofs(self) -> int:
        """Return the number of scalar degrees of freedom."""
        return self.num_faces * self.block_size

    @property
    def maximum_inverse_residual(self) -> float:
        """Return the largest local inverse residual."""
        return float(np.max(self.inverse_residuals, initial=0.0))

    def apply(self, vector: np.ndarray) -> np.ndarray:
        """Apply the inverse diagonal blocks, preserving the input shape.

        ``vector`` may be flat with shape ``(Nface*b,)`` or face-major with
        shape ``(Nface, b)``.  The output has the same shape.
        """

        array = np.asarray(vector, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("block-Jacobi input contains non-finite values")

        flat_input = array.ndim == 1
        if flat_input:
            expected = (self.num_dofs,)
            if array.shape != expected:
                raise ValueError(
                    f"flat vector must have shape {expected}; got {array.shape}"
                )
            face_vector = array.reshape(self.num_faces, self.block_size)
        else:
            expected = (self.num_faces, self.block_size)
            if array.shape != expected:
                raise ValueError(
                    f"face-major vector must have shape {expected}; got {array.shape}"
                )
            face_vector = array

        result = np.einsum(
            "fij,fj->fi",
            self.inverse_blocks,
            face_vector,
            optimize=True,
        )
        result = np.ascontiguousarray(result)
        return result.reshape(-1) if flat_input else result

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        """Alias for :meth:`apply`, suitable for ``restarted_gmres``."""

        return self.apply(vector)


def build_face_block_jacobi_preconditioner(
    system: FaceDenseSystem,
    *,
    inverse_residual_tolerance: float | None = None,
) -> FaceBlockJacobiPreconditioner:
    r"""Build ``P^{-1}`` from the diagonal face blocks of ``system``.

    Slot zero must be the diagonal block, so the topology invariant

    ``system.neighbors[f, 0] == f``

    is verified before inversion.  Each block is inverted independently with
    NumPy's batched dense inverse.  Singular blocks raise ``LinAlgError`` with
    the offending face reported.

    Parameters
    ----------
    system
        Penalty or Dirichlet-eliminated face-dense system.
    inverse_residual_tolerance
        Optional upper bound on
        ``||K_ff K_ff^{-1} - I||_inf``.  This is mainly useful in validation;
        by default, finite residuals are recorded but not thresholded.
    """

    if not isinstance(system, FaceDenseSystem):
        raise TypeError("system must be a FaceDenseSystem")
    if inverse_residual_tolerance is not None:
        if (
            inverse_residual_tolerance < 0.0
            or not np.isfinite(inverse_residual_tolerance)
        ):
            raise ValueError(
                "inverse_residual_tolerance must be finite and non-negative"
            )

    expected_diagonal_neighbors = np.arange(system.num_rows, dtype=np.int64)
    if not np.array_equal(
        system.neighbors[:, 0],
        expected_diagonal_neighbors,
    ):
        raise ValueError(
            "block-Jacobi requires slot zero to contain every diagonal face block"
        )

    diagonal_blocks = np.asarray(system.blocks[:, 0], dtype=np.float64)
    expected_shape = (
        system.num_rows,
        system.block_size,
        system.block_size,
    )
    if diagonal_blocks.shape != expected_shape:
        raise ValueError(
            f"diagonal blocks must have shape {expected_shape}; "
            f"got {diagonal_blocks.shape}"
        )
    if not np.all(np.isfinite(diagonal_blocks)):
        raise ValueError("diagonal face blocks contain non-finite values")

    # Invert one block at a time so a singular block can be identified clearly.
    inverse_blocks = np.empty_like(diagonal_blocks)
    for face in range(system.num_rows):
        try:
            inverse_blocks[face] = np.linalg.inv(diagonal_blocks[face])
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"diagonal face block {face} is singular"
            ) from exc

    identity = np.eye(system.block_size, dtype=np.float64)
    products = np.matmul(diagonal_blocks, inverse_blocks)
    inverse_residuals = np.max(
        np.sum(np.abs(products - identity[None, :, :]), axis=2),
        axis=1,
    )
    if not np.all(np.isfinite(inverse_residuals)):
        raise FloatingPointError("block-Jacobi inversion produced non-finite values")

    if (
        inverse_residual_tolerance is not None
        and np.any(inverse_residuals > inverse_residual_tolerance)
    ):
        face = int(np.argmax(inverse_residuals))
        raise FloatingPointError(
            "diagonal face-block inverse failed the requested residual tolerance: "
            f"face={face}, residual={inverse_residuals[face]:.3e}, "
            f"tolerance={inverse_residual_tolerance:.3e}"
        )

    return FaceBlockJacobiPreconditioner(
        inverse_blocks=np.ascontiguousarray(inverse_blocks),
        inverse_residuals=np.ascontiguousarray(inverse_residuals),
    )


__all__ = [
    "FaceBlockJacobiPreconditioner",
    "build_face_block_jacobi_preconditioner",
]
