"""Low-level cuBLAS batched LU inversion for small dense matrices.

The face block-Jacobi and one-element additive-Schwarz preconditioners contain
large batches of small square matrices.  This module exposes the explicit
``getrfBatched`` + ``getriBatched`` setup path used by cuBLAS, while keeping
CuPy arrays as the memory owner and stream/device integration layer.

The input matrices are C-contiguous.  cuBLAS interprets each raw matrix buffer
as column-major, hence it factorizes the transpose of the logical C-order
matrix.  ``getriBatched`` then writes the inverse transpose in column-major
storage, whose bytes are exactly the C-order representation of the desired
logical inverse.  The returned CuPy array therefore has the expected
``inverse[batch] == numpy.linalg.inv(input[batch])`` interpretation without an
extra transpose or copy.  A residual check guards this layout argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hybridge.runtime.optional import require_cupy_device


@dataclass(frozen=True)
class CuBLASBatchedInverseResult:
    """Result and diagnostics from batched cuBLAS inversion.

    Parameters
    ----------
    inverse_matrices
        C-contiguous CuPy array of shape ``(batch, n, n)``.
    inverse_residuals
        Device array containing
        ``||A_i A_i^{-1} - I||_inf`` for every batch matrix.
    factorization_info
        Host copy of the per-batch ``getrfBatched`` status array.
    inversion_info
        Host copy of the per-batch ``getriBatched`` status array.
    """

    inverse_matrices: Any
    inverse_residuals: Any
    factorization_info: np.ndarray
    inversion_info: np.ndarray

    @property
    def maximum_inverse_residual(self) -> float:
        """Return the largest local inverse residual."""
        cp = require_cupy_device()
        return float(cp.max(self.inverse_residuals).item())


def _validate_input_matrices(cp: Any, matrices: Any) -> tuple[int, int]:
    """Validate shapes, dtypes, devices, and solver parameters."""
    if not isinstance(matrices, cp.ndarray):
        raise TypeError("matrices must be a CuPy array")
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("matrices must have shape (batch, n, n)")
    if matrices.shape[0] <= 0 or matrices.shape[1] <= 0:
        raise ValueError("the matrix batch and matrix size must be positive")
    if matrices.dtype not in (cp.float32, cp.float64):
        raise TypeError("matrices must use float32 or float64")
    if not matrices.flags.c_contiguous:
        raise ValueError("matrices must be C-contiguous")
    if not bool(cp.all(cp.isfinite(matrices)).item()):
        raise FloatingPointError("matrices contain non-finite values")
    return int(matrices.shape[0]), int(matrices.shape[1])


def _device_pointer_array(cp: Any, matrices: Any) -> Any:
    """Return a device array of pointers to every matrix in a batch."""

    batch = int(matrices.shape[0])
    stride_bytes = int(matrices.strides[0])
    pointers = cp.arange(batch, dtype=cp.uintp)
    pointers *= cp.uintp(stride_bytes)
    pointers += cp.uintp(int(matrices.data.ptr))
    return cp.ascontiguousarray(pointers)


def _copy_and_check_info(
    cp: Any,
    info_device: Any,
    *,
    stage: str,
    label: str,
) -> np.ndarray:
    """Copy a cuBLAS info array to the host and report the first failure."""

    info = np.ascontiguousarray(cp.asnumpy(info_device), dtype=np.int32)
    nonzero = np.flatnonzero(info)
    if nonzero.size == 0:
        return info

    batch = int(nonzero[0])
    value = int(info[batch])
    if value < 0:
        raise RuntimeError(
            f"{label} {stage} reported an invalid argument for batch {batch}: "
            f"parameter {-value}"
        )
    raise np.linalg.LinAlgError(
        f"{label} {stage} failed for batch {batch}: "
        f"U({value},{value}) is zero"
    )


def invert_batched_cublas(
    matrices: Any,
    *,
    label: str = "batched matrices",
) -> CuBLASBatchedInverseResult:
    """Invert a batch with explicit cuBLAS ``getrf/getriBatched`` calls.

    The function uses the current CuPy stream and the cuBLAS handle owned by
    the current CuPy device.  It performs pivoted LU factorization, checks the
    per-batch factorization status, computes the inverse into a distinct output
    batch, checks the inversion status, and finally evaluates an infinity-norm
    inverse residual for every matrix.

    Notes
    -----
    This backend is NVIDIA CUDA specific.  CuPy's ROCm build does not expose
    cuBLAS and is rejected explicitly.
    """

    cp = require_cupy_device()
    if bool(getattr(cp.cuda.runtime, "is_hip", False)):
        raise RuntimeError("the explicit cuBLAS inverse backend requires CUDA")

    batch, size = _validate_input_matrices(cp, matrices)

    try:
        from cupy_backends.cuda.libs import cublas
    except Exception as error:  # pragma: no cover - installation dependent.
        raise RuntimeError("CuPy does not expose the low-level cuBLAS module") from error

    required = (
        "sgetrfBatched",
        "dgetrfBatched",
        "sgetriBatched",
        "dgetriBatched",
    )
    missing = [name for name in required if not hasattr(cublas, name)]
    if missing:
        raise RuntimeError(
            "the installed CuPy build lacks required cuBLAS wrappers: "
            + ", ".join(missing)
        )

    # getrfBatched is in-place; preserve the original matrices for the residual
    # check and for the preconditioner object's diagnostic/local-matrix storage.
    lu_matrices = matrices.copy(order="C")
    inverse_matrices = cp.empty_like(matrices, order="C")

    lu_pointers = _device_pointer_array(cp, lu_matrices)
    inverse_pointers = _device_pointer_array(cp, inverse_matrices)
    pivots = cp.empty((batch, size), dtype=cp.int32)
    factorization_info_device = cp.empty(batch, dtype=cp.int32)
    inversion_info_device = cp.empty(batch, dtype=cp.int32)

    handle = cp.cuda.device.get_cublas_handle()
    if matrices.dtype == cp.float32:
        getrf = cublas.sgetrfBatched
        getri = cublas.sgetriBatched
    else:
        getrf = cublas.dgetrfBatched
        getri = cublas.dgetriBatched

    getrf(
        handle,
        size,
        int(lu_pointers.data.ptr),
        size,
        int(pivots.data.ptr),
        int(factorization_info_device.data.ptr),
        batch,
    )
    factorization_info = _copy_and_check_info(
        cp,
        factorization_info_device,
        stage="getrfBatched",
        label=label,
    )

    getri(
        handle,
        size,
        int(lu_pointers.data.ptr),
        size,
        int(pivots.data.ptr),
        int(inverse_pointers.data.ptr),
        size,
        int(inversion_info_device.data.ptr),
        batch,
    )
    inversion_info = _copy_and_check_info(
        cp,
        inversion_info_device,
        stage="getriBatched",
        label=label,
    )

    if not bool(cp.all(cp.isfinite(inverse_matrices)).item()):
        raise FloatingPointError(f"{label} cuBLAS inversion produced non-finite values")

    identity = cp.eye(size, dtype=matrices.dtype)
    products = cp.matmul(matrices, inverse_matrices)
    inverse_residuals = cp.max(
        cp.sum(cp.abs(products - identity[None, :, :]), axis=2),
        axis=1,
    )
    inverse_residuals = cp.ascontiguousarray(inverse_residuals)
    if not bool(cp.all(cp.isfinite(inverse_residuals)).item()):
        raise FloatingPointError(f"{label} cuBLAS inverse residuals are non-finite")

    return CuBLASBatchedInverseResult(
        inverse_matrices=cp.ascontiguousarray(inverse_matrices),
        inverse_residuals=inverse_residuals,
        factorization_info=factorization_info,
        inversion_info=inversion_info,
    )


@dataclass
class BatchedLUWorkspace:
    """Reusable pivots, status flags, and pointer arrays for batched LU solves.

    The pointer arrays are tied to the buffers they were built for; they are
    rebuilt whenever the matrix or right-hand-side buffer, shape, or dtype
    changes. Shared by the cuBLAS and MAGMA batched solvers.
    """

    pivots: Any | None = None
    info: Any | None = None
    matrix_pointers: Any | None = None
    rhs_pointers: Any | None = None
    pivot_pointers: Any | None = None
    signature: tuple | None = None

    def ensure(self, cp: Any, matrices: Any, rhs: Any) -> None:
        """Allocate or refresh the arrays for ``matrices`` and ``rhs``."""
        signature = (
            int(matrices.data.ptr), matrices.shape, str(matrices.dtype),
            int(rhs.data.ptr), rhs.shape,
        )
        if signature == self.signature:
            return
        batch, size = int(matrices.shape[0]), int(matrices.shape[1])
        self.pivots = cp.empty((batch, size), dtype=cp.int32)
        self.info = cp.zeros(batch, dtype=cp.int32)
        self.matrix_pointers = _device_pointer_array(cp, matrices)
        self.rhs_pointers = _device_pointer_array(cp, rhs)
        self.pivot_pointers = _device_pointer_array(cp, self.pivots)
        self.signature = signature


def validate_batched_lu_operands(cp: Any, matrices: Any, rhs: Any) -> tuple[int, int, int]:
    """Check the column-major batched LU layout and return ``(batch, n, nrhs)``."""
    if not isinstance(matrices, cp.ndarray) or not isinstance(rhs, cp.ndarray):
        raise TypeError("matrices and rhs must be CuPy arrays")
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("matrices must have shape (batch, n, n)")
    if rhs.ndim != 3 or rhs.shape[0] != matrices.shape[0] or rhs.shape[2] != matrices.shape[1]:
        raise ValueError("rhs must have shape (batch, nrhs, n)")
    if matrices.dtype not in (cp.float32, cp.float64) or rhs.dtype != matrices.dtype:
        raise TypeError("matrices and rhs must share float32 or float64")
    if not (matrices.flags.c_contiguous and rhs.flags.c_contiguous):
        raise ValueError("matrices and rhs must be C-contiguous")
    return int(matrices.shape[0]), int(matrices.shape[1]), int(rhs.shape[1])


def lu_solve_batched_cublas(
    matrices: Any,
    rhs: Any,
    *,
    trans: bool = False,
    workspace: BatchedLUWorkspace | None = None,
    check_info: bool = True,
    label: str = "batched matrices",
) -> BatchedLUWorkspace:
    """Solve ``A_b X_b = B_b`` in place with cuBLAS ``getrf/getrsBatched``.

    Layout: ``matrices[b]`` holds ``A_b`` in column-major order (the bytes of
    its C-order transpose) and ``rhs[b]``, shaped ``(nrhs, n)``, holds ``B_b``
    column-major. With ``trans=True`` the stored matrix is used transposed,
    which solves with a C-order ``matrices[b]`` directly. The matrices are
    overwritten by their LU factors and ``rhs`` by the solutions, on the
    current CuPy stream.

    ``check_info=False`` skips the host copy of the per-batch status (and its
    stream synchronization), for timing; the statuses stay in
    ``workspace.info``.
    """
    cp = require_cupy_device()
    from cupy_backends.cuda.libs import cublas

    batch, size, nrhs = validate_batched_lu_operands(cp, matrices, rhs)
    workspace = BatchedLUWorkspace() if workspace is None else workspace
    workspace.ensure(cp, matrices, rhs)
    handle = cp.cuda.device.get_cublas_handle()
    single = matrices.dtype == cp.float32
    getrf = cublas.sgetrfBatched if single else cublas.dgetrfBatched
    getrs = cublas.sgetrsBatched if single else cublas.dgetrsBatched
    getrf(handle, size, int(workspace.matrix_pointers.data.ptr), size,
          int(workspace.pivots.data.ptr), int(workspace.info.data.ptr), batch)
    if check_info:
        _copy_and_check_info(cp, workspace.info, stage="getrfBatched", label=label)
    # getrsBatched reports only argument errors, through a host integer.
    status = np.zeros(1, dtype=np.int32)
    getrs(handle, cublas.CUBLAS_OP_T if trans else cublas.CUBLAS_OP_N,
          size, nrhs, int(workspace.matrix_pointers.data.ptr), size,
          int(workspace.pivots.data.ptr), int(workspace.rhs_pointers.data.ptr),
          size, status.ctypes.data, batch)
    if status[0]:
        raise RuntimeError(f"{label} getrsBatched rejected argument {-int(status[0])}")
    return workspace


__all__ = [
    "BatchedLUWorkspace",
    "CuBLASBatchedInverseResult",
    "invert_batched_cublas",
    "lu_solve_batched_cublas",
    "validate_batched_lu_operands",
]
