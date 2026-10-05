"""CuPy implementation of the HDG face-dense matrix--vector product.

This module implements only the first GPU operator layer.  It transfers an
already assembled :class:`~hybridge.linalg.face_dense.FaceDenseSystem` to a
CUDA device, gathers the neighbouring face vectors into a fixed-width extended
layout, and applies the face-row dense matrices in a batch.

No COO/CSR matrix is created on the device.  The two main device arrays are

``matrix_batches[f, i, s*b + j]``
    Dense row-face matrix with shape ``(NF, b, S*b)``.

``x_extended[f, s, j]``
    Gathered input vector with shape ``(NF, S, b)``.

The default ``"matmul"`` implementation evaluates the batch through
``cupy.matmul``, which dispatches the dense products to CuPy/cuBLAS.  A simple
``"raw"`` CUDA kernel is also provided as an independent correctness path.  It
is not intended to be selected on performance grounds before profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from hybridge.linalg.face_dense import FaceDenseSystem
from hybridge.runtime.optional import device_arrays_overlap, require_cupy_device


GPUFaceMatvecImplementation = Literal["matmul", "raw", "raw_fused"]


@dataclass(frozen=True)
class FaceDenseBatchLayout:
    """Host-side arrays prepared for the GPU face-dense operator.

    Attributes
    ----------
    matrix_batches
        C-contiguous array of shape ``(NF, b, S*b)``.  For each face this is
        the horizontal concatenation ``[K_f0 K_f1 ... K_f,S-1]``.
    neighbors
        C-contiguous ``int32`` array of shape ``(NF, S)``.  Unused slots remain
        ``-1`` and are gathered as zero vectors by the CUDA kernel.
    """

    matrix_batches: np.ndarray
    neighbors: np.ndarray

    @property
    def num_rows(self) -> int:
        """Return the number of face rows."""
        return int(self.matrix_batches.shape[0])

    @property
    def block_size(self) -> int:
        """Return the dense face-block size."""
        return int(self.matrix_batches.shape[1])

    @property
    def num_slots(self) -> int:
        """Return the number of stored neighbor slots per row."""
        return int(self.neighbors.shape[1])

    @property
    def num_dofs(self) -> int:
        """Return the number of scalar degrees of freedom."""
        return self.num_rows * self.block_size


def prepare_face_dense_batch_layout(
    blocks: np.ndarray,
    neighbors: np.ndarray,
) -> FaceDenseBatchLayout:
    """Validate and reshape CPU face blocks for batched GPU evaluation.

    The CPU matrix is stored as ``blocks[f, s, i, j]``.  A dense face row must
    instead be viewed as

    ``matrix_batches[f, i, s*b + j]``.

    This function performs that permutation once during setup; no matrix-layout
    transformation is needed inside a Krylov iteration.
    """

    blocks = np.asarray(blocks)
    neighbors = np.asarray(neighbors)

    if blocks.ndim != 4:
        raise ValueError("blocks must have shape (NF, S, b, b)")
    num_rows, num_slots, row_size, column_size = blocks.shape
    if row_size != column_size:
        raise ValueError("face blocks must be square")
    if neighbors.shape != (num_rows, num_slots):
        raise ValueError(
            "neighbors must have shape matching the first two block axes: "
            f"expected {(num_rows, num_slots)}, got {neighbors.shape}"
        )
    if num_rows == 0 or num_slots == 0 or row_size == 0:
        raise ValueError("face-dense arrays must be non-empty")
    if blocks.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("blocks must use float32 or float64")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("blocks contain non-finite values")

    if not np.issubdtype(neighbors.dtype, np.integer):
        raise TypeError("neighbors must use an integer dtype")
    if np.any(neighbors < -1):
        raise ValueError("neighbors may contain only valid row ids or -1")
    valid = neighbors >= 0
    if np.any(neighbors[valid] >= num_rows):
        raise ValueError("neighbors contains an index outside the face system")
    if num_rows > np.iinfo(np.int32).max:
        raise OverflowError("the CUDA gather kernel uses int32 face indices")

    # (NF, S, i, j) -> (NF, i, S, j) -> (NF, i, S*b)
    matrix_batches = (
        blocks.transpose(0, 2, 1, 3)
        .reshape(num_rows, row_size, num_slots * row_size)
        .copy(order="C")
    )

    return FaceDenseBatchLayout(
        matrix_batches=np.ascontiguousarray(matrix_batches),
        neighbors=np.ascontiguousarray(neighbors, dtype=np.int32),
    )


_GATHER_KERNEL_SOURCE = r"""
extern "C" __global__
void gather_face_neighbors_f32(
    const int num_rows,
    const int num_slots,
    const int block_size,
    const int* __restrict__ neighbors,
    const float* __restrict__ x,
    float* __restrict__ x_extended)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned long long total =
        (unsigned long long) num_rows * num_slots * block_size;
    if (index >= total) return;

    const int local_dof = (int) (index % block_size);
    const int slot = (int) ((index / block_size) % num_slots);
    const int row_face =
        (int) (index / ((unsigned long long) num_slots * block_size));
    const int column_face = neighbors[row_face * num_slots + slot];

    x_extended[index] =
        (column_face >= 0) ? x[column_face * block_size + local_dof] : 0.0f;
}

extern "C" __global__
void gather_face_neighbors_f64(
    const int num_rows,
    const int num_slots,
    const int block_size,
    const int* __restrict__ neighbors,
    const double* __restrict__ x,
    double* __restrict__ x_extended)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned long long total =
        (unsigned long long) num_rows * num_slots * block_size;
    if (index >= total) return;

    const int local_dof = (int) (index % block_size);
    const int slot = (int) ((index / block_size) % num_slots);
    const int row_face =
        (int) (index / ((unsigned long long) num_slots * block_size));
    const int column_face = neighbors[row_face * num_slots + slot];

    x_extended[index] =
        (column_face >= 0) ? x[column_face * block_size + local_dof] : 0.0;
}
"""


_RAW_MATVEC_KERNEL_SOURCE = r"""
extern "C" __global__
void face_dense_matvec_f32(
    const int num_rows,
    const int block_size,
    const int extended_size,
    const float* __restrict__ matrix_batches,
    const float* __restrict__ x_extended,
    float* __restrict__ y)
{
    const unsigned long long output_index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned long long total_outputs =
        (unsigned long long) num_rows * block_size;
    if (output_index >= total_outputs) return;

    const int row_face = (int) (output_index / block_size);
    const int row_dof = (int) (output_index % block_size);
    const unsigned long long matrix_offset =
        ((unsigned long long) row_face * block_size + row_dof) * extended_size;
    const unsigned long long vector_offset =
        (unsigned long long) row_face * extended_size;

    float value = 0.0f;
    for (int column = 0; column < extended_size; ++column) {
        value += matrix_batches[matrix_offset + column]
               * x_extended[vector_offset + column];
    }
    y[output_index] = value;
}

extern "C" __global__
void face_dense_matvec_f64(
    const int num_rows,
    const int block_size,
    const int extended_size,
    const double* __restrict__ matrix_batches,
    const double* __restrict__ x_extended,
    double* __restrict__ y)
{
    const unsigned long long output_index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned long long total_outputs =
        (unsigned long long) num_rows * block_size;
    if (output_index >= total_outputs) return;

    const int row_face = (int) (output_index / block_size);
    const int row_dof = (int) (output_index % block_size);
    const unsigned long long matrix_offset =
        ((unsigned long long) row_face * block_size + row_dof) * extended_size;
    const unsigned long long vector_offset =
        (unsigned long long) row_face * extended_size;

    double value = 0.0;
    for (int column = 0; column < extended_size; ++column) {
        value += matrix_batches[matrix_offset + column]
               * x_extended[vector_offset + column];
    }
    y[output_index] = value;
}
"""

_FUSED_RAW_MATVEC_KERNEL_SOURCE = r"""
extern "C" __global__
void face_dense_matvec_fused_f32(
    const int num_rows,
    const int num_slots,
    const int block_size,
    const float* __restrict__ matrix_batches,
    const int* __restrict__ neighbors,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    const unsigned long long output_index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned long long total_outputs =
        (unsigned long long) num_rows * block_size;
    if (output_index >= total_outputs) return;

    const int row_face = (int) (output_index / block_size);
    const int row_dof = (int) (output_index % block_size);
    const int extended_size = num_slots * block_size;
    const unsigned long long matrix_offset =
        ((unsigned long long) row_face * block_size + row_dof)
        * extended_size;

    float value = 0.0f;
    for (int slot = 0; slot < num_slots; ++slot) {
        const int column_face = neighbors[row_face * num_slots + slot];
        if (column_face < 0) continue;
        const unsigned long long x_offset =
            (unsigned long long) column_face * block_size;
        const unsigned long long block_offset =
            matrix_offset + (unsigned long long) slot * block_size;
        for (int column_dof = 0; column_dof < block_size; ++column_dof) {
            value += matrix_batches[block_offset + column_dof]
                   * x[x_offset + column_dof];
        }
    }
    y[output_index] = value;
}

extern "C" __global__
void face_dense_matvec_fused_f64(
    const int num_rows,
    const int num_slots,
    const int block_size,
    const double* __restrict__ matrix_batches,
    const int* __restrict__ neighbors,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    const unsigned long long output_index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned long long total_outputs =
        (unsigned long long) num_rows * block_size;
    if (output_index >= total_outputs) return;

    const int row_face = (int) (output_index / block_size);
    const int row_dof = (int) (output_index % block_size);
    const int extended_size = num_slots * block_size;
    const unsigned long long matrix_offset =
        ((unsigned long long) row_face * block_size + row_dof)
        * extended_size;

    double value = 0.0;
    for (int slot = 0; slot < num_slots; ++slot) {
        const int column_face = neighbors[row_face * num_slots + slot];
        if (column_face < 0) continue;
        const unsigned long long x_offset =
            (unsigned long long) column_face * block_size;
        const unsigned long long block_offset =
            matrix_offset + (unsigned long long) slot * block_size;
        for (int column_dof = 0; column_dof < block_size; ++column_dof) {
            value += matrix_batches[block_offset + column_dof]
                   * x[x_offset + column_dof];
        }
    }
    y[output_index] = value;
}
"""



class CuPyFaceDenseOperator:
    """GPU-resident face-dense HDG operator.

    Construct instances with :meth:`from_system`.  Inputs to :meth:`matvec`
    and :meth:`matvec_into` must already be CuPy arrays on the same device;
    implicit host/device transfers are intentionally forbidden inside the
    iterative operator.
    """

    def __init__(
        self,
        *,
        matrix_batches,
        neighbors,
        implementation: GPUFaceMatvecImplementation,
        device_id: int,
    ) -> None:
        """Initialize the instance and validate persistent storage."""
        cp = require_cupy_device()
        if implementation not in ("matmul", "raw", "raw_fused"):
            raise ValueError(
                "implementation must be 'matmul', 'raw', or 'raw_fused'"
            )
        if matrix_batches.ndim != 3:
            raise ValueError("matrix_batches must have shape (NF, b, S*b)")
        if neighbors.ndim != 2:
            raise ValueError("neighbors must have shape (NF, S)")

        num_rows, block_size, extended_size = matrix_batches.shape
        num_slots = int(neighbors.shape[1])
        if neighbors.shape[0] != num_rows:
            raise ValueError("matrix_batches and neighbors disagree on NF")
        if extended_size != num_slots * block_size:
            raise ValueError("matrix_batches has an incompatible extended axis")
        if matrix_batches.dtype not in (cp.float32, cp.float64):
            raise TypeError("matrix_batches must use float32 or float64")
        if neighbors.dtype != cp.int32:
            raise TypeError("neighbors must use int32 on the device")

        self._cp = cp
        self.matrix_batches = matrix_batches
        self.neighbors = neighbors
        self.implementation = implementation
        self.device_id = int(device_id)
        self.num_rows = int(num_rows)
        self.num_slots = num_slots
        self.block_size = int(block_size)
        self.extended_size = int(extended_size)
        self.num_dofs = self.num_rows * self.block_size
        self.dtype = matrix_batches.dtype

        # The fused raw path reads neighbouring face values directly and
        # therefore needs neither the gathered-vector buffer nor a temporary
        # batched-matmul output.  The two-stage paths keep their reusable
        # workspaces for validation and comparison.
        self._x_extended = None
        if implementation != "raw_fused":
            self._x_extended = cp.empty(
                (self.num_rows, self.num_slots, self.block_size),
                dtype=self.dtype,
            )
        self._matmul_output = None
        if implementation == "matmul":
            self._matmul_output = cp.empty(
                (self.num_rows, self.block_size, 1),
                dtype=self.dtype,
            )

        suffix = "f32" if self.dtype == cp.float32 else "f64"
        self._gather_kernel = cp.RawKernel(
            _GATHER_KERNEL_SOURCE,
            f"gather_face_neighbors_{suffix}",
        )
        self._raw_matvec_kernel = None
        self._fused_raw_matvec_kernel = None
        if implementation == "raw":
            self._raw_matvec_kernel = cp.RawKernel(
                _RAW_MATVEC_KERNEL_SOURCE,
                f"face_dense_matvec_{suffix}",
            )
        elif implementation == "raw_fused":
            self._fused_raw_matvec_kernel = cp.RawKernel(
                _FUSED_RAW_MATVEC_KERNEL_SOURCE,
                f"face_dense_matvec_fused_{suffix}",
            )

    @classmethod
    def from_device_blocks(
        cls,
        blocks,
        neighbors,
        *,
        implementation: GPUFaceMatvecImplementation = "raw_fused",
        device_id: int | None = None,
    ) -> "CuPyFaceDenseOperator":
        """Construct directly from GPU-resident ``(NF,S,b,b)`` blocks.

        This setup path is used by the GPU assembly pipeline.  The only
        numerical transformation is the one-time device-side permutation to
        ``(NF,b,S*b)``; global blocks are never copied back to the CPU.
        """

        cp = require_cupy_device()
        if not isinstance(blocks, cp.ndarray):
            raise TypeError("blocks must be a CuPy array")
        if blocks.ndim != 4:
            raise ValueError("blocks must have shape (NF, S, b, b)")
        num_rows, num_slots, row_size, column_size = blocks.shape
        if row_size != column_size or min(blocks.shape) <= 0:
            raise ValueError("face blocks must be non-empty and square")
        if blocks.dtype not in (cp.float32, cp.float64):
            raise TypeError("blocks must use float32 or float64")
        selected_device = int(blocks.device.id) if device_id is None else int(device_id)
        if int(blocks.device.id) != selected_device:
            raise ValueError("blocks are on a different CUDA device")
        if isinstance(neighbors, cp.ndarray):
            neighbors_device = neighbors
            if int(neighbors_device.device.id) != selected_device:
                raise ValueError("neighbors are on a different CUDA device")
            if neighbors_device.dtype != cp.int32:
                neighbors_device = neighbors_device.astype(cp.int32, copy=True)
            elif not neighbors_device.flags.c_contiguous:
                neighbors_device = cp.ascontiguousarray(neighbors_device)
        else:
            neighbors_host = np.ascontiguousarray(neighbors, dtype=np.int32)
            with cp.cuda.Device(selected_device):
                neighbors_device = cp.asarray(neighbors_host)
        if neighbors_device.shape != (num_rows, num_slots):
            raise ValueError("neighbors must have shape (NF, S)")

        with cp.cuda.Device(selected_device):
            matrix_batches = cp.ascontiguousarray(
                blocks.transpose(0, 2, 1, 3).reshape(
                    num_rows,
                    row_size,
                    num_slots * row_size,
                )
            )
            return cls(
                matrix_batches=matrix_batches,
                neighbors=neighbors_device,
                implementation=implementation,
                device_id=selected_device,
            )

    @classmethod
    def from_system(
        cls,
        system: FaceDenseSystem,
        *,
        implementation: GPUFaceMatvecImplementation = "matmul",
        dtype: np.dtype | type | None = None,
        device_id: int | None = None,
    ) -> "CuPyFaceDenseOperator":
        """Prepare, transfer, and allocate a GPU operator from a CPU system."""

        cp = require_cupy_device()
        if not isinstance(system, FaceDenseSystem):
            raise TypeError("system must be a FaceDenseSystem")

        requested_dtype = np.dtype(system.blocks.dtype if dtype is None else dtype)
        if requested_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("dtype must be float32 or float64")

        host_blocks = np.asarray(system.blocks, dtype=requested_dtype)
        layout = prepare_face_dense_batch_layout(host_blocks, system.neighbors)

        selected_device = (
            int(cp.cuda.Device().id) if device_id is None else int(device_id)
        )
        with cp.cuda.Device(selected_device):
            matrix_batches = cp.asarray(layout.matrix_batches)
            neighbors = cp.asarray(layout.neighbors)
            return cls(
                matrix_batches=matrix_batches,
                neighbors=neighbors,
                implementation=implementation,
                device_id=selected_device,
            )

    @property
    def x_extended(self):
        """Reusable gathered-vector buffer with shape ``(NF, S, b)``.

        The fused raw implementation deliberately does not allocate this
        buffer.  Accessing it in that mode is therefore an error rather than a
        hidden allocation.
        """

        if self._x_extended is None:
            raise RuntimeError("raw_fused does not use a gathered-vector buffer")
        return self._x_extended

    @property
    def uses_gather_buffer(self) -> bool:
        """Whether the selected implementation materializes neighbour values."""

        return self._x_extended is not None

    @property
    def workspace_bytes(self) -> int:
        """Bytes owned by operator-only temporary device workspaces."""

        total = 0
        if self._x_extended is not None:
            total += int(self._x_extended.nbytes)
        if self._matmul_output is not None:
            total += int(self._matmul_output.nbytes)
        return total

    def _validate_device_vector(self, vector, *, name: str):
        """Validate shapes, dtypes, devices, and solver parameters."""
        cp = self._cp
        if not isinstance(vector, cp.ndarray):
            raise TypeError(
                f"{name} must be a CuPy array; explicit host/device transfers "
                "must happen outside the Krylov operator"
            )
        if int(vector.device.id) != self.device_id:
            raise ValueError(
                f"{name} is on CUDA device {vector.device.id}, but the operator "
                f"is on device {self.device_id}"
            )
        if vector.dtype != self.dtype:
            raise TypeError(
                f"{name} must have dtype {self.dtype}; got {vector.dtype}"
            )
        if vector.size != self.num_dofs:
            raise ValueError(
                f"{name} must contain {self.num_dofs} values; got {vector.shape}"
            )
        if vector.ndim not in (1, 2):
            raise ValueError(f"{name} must be flat or face-major")
        if vector.ndim == 2 and vector.shape != (
            self.num_rows,
            self.block_size,
        ):
            raise ValueError(
                f"{name} face-major shape must be "
                f"({self.num_rows}, {self.block_size}); got {vector.shape}"
            )
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector.reshape(self.num_rows, self.block_size)

    def gather_neighbors_into(self, x, out) -> None:
        """Gather ``x[neighbors[f,s]]`` into a preallocated device array."""

        cp = self._cp
        x_faces = self._validate_device_vector(x, name="x")
        if not isinstance(out, cp.ndarray):
            raise TypeError("out must be a CuPy array")
        expected = (self.num_rows, self.num_slots, self.block_size)
        if out.shape != expected or out.dtype != self.dtype:
            raise ValueError(
                f"out must have shape {expected} and dtype {self.dtype}"
            )
        if not out.flags.c_contiguous:
            raise ValueError("out must be C-contiguous")

        threads = 256
        total = self.num_rows * self.num_slots * self.block_size
        blocks = (total + threads - 1) // threads
        self._gather_kernel(
            (blocks,),
            (threads,),
            (
                np.int32(self.num_rows),
                np.int32(self.num_slots),
                np.int32(self.block_size),
                self.neighbors,
                x_faces,
                out,
            ),
        )

    def gather_neighbors(self, x):
        """Return a newly allocated gathered vector (validation convenience)."""

        out = self._cp.empty(
            (self.num_rows, self.num_slots, self.block_size),
            dtype=self.dtype,
        )
        self.gather_neighbors_into(x, out)
        return out

    def matvec_into(self, x, out) -> None:
        """Compute ``out = K @ x`` without allocating per-iteration buffers."""

        cp = self._cp
        self._validate_device_vector(x, name="x")
        out_faces = self._validate_device_vector(out, name="out")
        if device_arrays_overlap(x, out):
            raise ValueError("x and out must not alias")

        threads = 256
        blocks = (self.num_dofs + threads - 1) // threads

        if self.implementation == "raw_fused":
            assert self._fused_raw_matvec_kernel is not None
            self._fused_raw_matvec_kernel(
                (blocks,),
                (threads,),
                (
                    np.int32(self.num_rows),
                    np.int32(self.num_slots),
                    np.int32(self.block_size),
                    self.matrix_batches,
                    self.neighbors,
                    x.reshape(-1),
                    out_faces,
                ),
            )
            return

        assert self._x_extended is not None
        self.gather_neighbors_into(x, self._x_extended)

        if self.implementation == "matmul":
            assert self._matmul_output is not None
            cp.matmul(
                self.matrix_batches,
                self._x_extended.reshape(
                    self.num_rows,
                    self.extended_size,
                    1,
                ),
                out=self._matmul_output,
            )
            out_faces[...] = self._matmul_output[:, :, 0]
            return

        assert self._raw_matvec_kernel is not None
        self._raw_matvec_kernel(
            (blocks,),
            (threads,),
            (
                np.int32(self.num_rows),
                np.int32(self.block_size),
                np.int32(self.extended_size),
                self.matrix_batches,
                self._x_extended,
                out_faces,
            ),
        )

    def matvec(self, x):
        """Allocate and return ``K @ x`` while preserving the input shape."""

        cp = self._cp
        self._validate_device_vector(x, name="x")
        out = cp.empty_like(x)
        self.matvec_into(x, out)
        return out

    def to_device(self, array, *, preserve_shape: bool = True):
        """Explicitly copy a compatible host vector to this operator's device."""

        cp = self._cp
        host = np.asarray(array, dtype=np.dtype(self.dtype.name))
        if host.size != self.num_dofs:
            raise ValueError(
                f"array must contain {self.num_dofs} values; got {host.shape}"
            )
        target_shape = host.shape if preserve_shape else (self.num_dofs,)
        with cp.cuda.Device(self.device_id):
            return cp.asarray(np.ascontiguousarray(host.reshape(target_shape)))

    def to_host(self, array) -> np.ndarray:
        """Explicitly copy an array from this operator's device to NumPy."""

        cp = self._cp
        if not isinstance(array, cp.ndarray):
            raise TypeError("array must be a CuPy array")
        if int(array.device.id) != self.device_id:
            raise ValueError(
                f"array is on CUDA device {array.device.id}, but the operator "
                f"is on device {self.device_id}"
            )
        return cp.asnumpy(array)

    def matvec_host(self, x: np.ndarray) -> np.ndarray:
        """Host-to-device validation helper; do not use inside GMRES."""

        x_device = self.to_device(x)
        return self.to_host(self.matvec(x_device))

    def synchronize(self) -> None:
        """Synchronize the operator's CUDA device (mainly for validation)."""

        with self._cp.cuda.Device(self.device_id):
            self._cp.cuda.get_current_stream().synchronize()


__all__ = [
    "CuPyFaceDenseOperator",
    "FaceDenseBatchLayout",
    "GPUFaceMatvecImplementation",
    "prepare_face_dense_batch_layout",
]