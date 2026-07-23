"""CUDA preconditioners for face-dense HDG systems.

The iterative application paths are device resident.  During this correctness
stage the small dense inverse blocks are assembled and inverted by the already
validated NumPy reference builders, then transferred once to the selected CUDA
device.  Replacing setup with batched cuBLAS/cuSOLVER factorization later does
not change the application interfaces implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..assembly.face_dense import FaceDenseSystem
from ..linalg.additive_schwarz import (
    build_face_additive_schwarz_preconditioner,
)
from ..linalg.block_jacobi import build_face_block_jacobi_preconditioner
from .cupy import require_cupy_device


@dataclass
class CuPyFaceBlockJacobiPreconditioner:
    """Device-resident inverse face blocks with batched application."""

    inverse_blocks: Any
    device_id: int

    def __post_init__(self) -> None:
        cp = require_cupy_device()
        if not isinstance(self.inverse_blocks, cp.ndarray):
            raise TypeError("inverse_blocks must be a CuPy array")
        if self.inverse_blocks.ndim != 3:
            raise ValueError("inverse_blocks must have shape (NF, b, b)")
        if self.inverse_blocks.shape[1] != self.inverse_blocks.shape[2]:
            raise ValueError("inverse face blocks must be square")
        if self.inverse_blocks.dtype not in (cp.float32, cp.float64):
            raise TypeError("inverse_blocks must use float32 or float64")
        if int(self.inverse_blocks.device.id) != int(self.device_id):
            raise ValueError("inverse_blocks is on the wrong CUDA device")
        if not self.inverse_blocks.flags.c_contiguous:
            raise ValueError("inverse_blocks must be C-contiguous")

        self.device_id = int(self.device_id)
        self._cp = cp
        self._matmul_output = cp.empty(
            (self.num_faces, self.block_size, 1),
            dtype=self.dtype,
        )

    @classmethod
    def from_system(
        cls,
        system: FaceDenseSystem,
        *,
        device_id: int | None = None,
        inverse_residual_tolerance: float | None = None,
    ) -> "CuPyFaceBlockJacobiPreconditioner":
        cp = require_cupy_device()
        if not isinstance(system, FaceDenseSystem):
            raise TypeError("system must be a FaceDenseSystem")
        selected_device = (
            int(cp.cuda.Device().id) if device_id is None else int(device_id)
        )
        cpu_preconditioner = build_face_block_jacobi_preconditioner(
            system,
            inverse_residual_tolerance=inverse_residual_tolerance,
        )
        with cp.cuda.Device(selected_device):
            inverse_blocks = cp.asarray(cpu_preconditioner.inverse_blocks)
        return cls(inverse_blocks=inverse_blocks, device_id=selected_device)

    @property
    def num_faces(self) -> int:
        return int(self.inverse_blocks.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.inverse_blocks.shape[1])

    @property
    def num_dofs(self) -> int:
        return self.num_faces * self.block_size

    @property
    def dtype(self) -> Any:
        return self.inverse_blocks.dtype

    def _validate_vector(self, vector: Any, *, name: str) -> Any:
        cp = self._cp
        if not isinstance(vector, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        if int(vector.device.id) != self.device_id:
            raise ValueError(f"{name} is on the wrong CUDA device")
        if vector.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}")
        if vector.size != self.num_dofs:
            raise ValueError(f"{name} must contain {self.num_dofs} values")
        if vector.ndim not in (1, 2):
            raise ValueError(f"{name} must be flat or face-major")
        if vector.ndim == 2 and vector.shape != (
            self.num_faces,
            self.block_size,
        ):
            raise ValueError(f"{name} has an incompatible face-major shape")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector.reshape(self.num_faces, self.block_size)

    def apply_into(self, x: Any, out: Any) -> None:
        """Compute ``out = P^{-1} x`` without iteration-time allocation."""

        cp = self._cp
        x_faces = self._validate_vector(x, name="x")
        out_faces = self._validate_vector(out, name="out")
        if cp.shares_memory(x, out):
            raise ValueError("x and out must not alias")
        cp.matmul(
            self.inverse_blocks,
            x_faces.reshape(self.num_faces, self.block_size, 1),
            out=self._matmul_output,
        )
        out_faces[...] = self._matmul_output[:, :, 0]

    def apply(self, x: Any) -> Any:
        out = self._cp.empty_like(x)
        self.apply_into(x, out)
        return out


@dataclass(frozen=True)
class AdditiveSchwarzBatchLayout:
    """Host-side arrays prepared for the CUDA one-element ASM operator.

    Parameters
    ----------
    inverse_matrices
        C-contiguous array with shape ``(NE, L, L)``, where
        ``L = Nlfe * b``.
    element_system_faces
        C-contiguous ``int32`` array with shape ``(NE, Nlfe)``.  A value of
        ``-1`` marks a directly eliminated boundary face.
    inverse_residuals
        Infinity-norm setup residuals from the CPU reference inversions.
    block_size
        Number of trace unknowns per face.
    num_system_faces
        Number of face rows in the supplied penalty or eliminated system.
    """

    inverse_matrices: np.ndarray
    element_system_faces: np.ndarray
    inverse_residuals: np.ndarray # Per-element errors of the CPU-computed local inverses, measured as ``||P_e @ inverse_matrices[e] - I||_inf``
    block_size: int
    num_system_faces: int

    @property
    def num_elements(self) -> int:
        return int(self.inverse_matrices.shape[0])

    @property
    def num_local_faces(self) -> int:
        return int(self.element_system_faces.shape[1])

    @property
    def local_size(self) -> int:
        return int(self.inverse_matrices.shape[1])

    @property
    def num_dofs(self) -> int:
        return self.num_system_faces * self.block_size

    @property
    def maximum_inverse_residual(self) -> float:
        return float(np.max(self.inverse_residuals, initial=0.0))


def prepare_face_additive_schwarz_batch_layout(
    system: FaceDenseSystem,
    element_blocks: np.ndarray,
    loc2glob_face: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
    inverse_residual_tolerance: float | None = None,
) -> AdditiveSchwarzBatchLayout:
    """Build and validate the host data transferred by the CUDA ASM backend.

    This function is deliberately CUDA-independent and is therefore exercised
    by the regular CPU test suite.  The mathematical construction is delegated
    to :func:`build_face_additive_schwarz_preconditioner`, ensuring that the
    CUDA backend uses exactly the same enriched local matrices and boundary
    treatment as the validated CPU implementation.
    """

    if not isinstance(system, FaceDenseSystem):
        raise TypeError("system must be a FaceDenseSystem")

    requested_dtype = np.dtype(system.blocks.dtype if dtype is None else dtype)
    if requested_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64")

    cpu_preconditioner = build_face_additive_schwarz_preconditioner(
        system,
        element_blocks,
        loc2glob_face,
        inverse_residual_tolerance=inverse_residual_tolerance,
    )
    inverse_matrices = np.ascontiguousarray(
        cpu_preconditioner.inverse_matrices,
        dtype=requested_dtype,
    )
    element_system_faces64 = np.asarray(
        cpu_preconditioner.element_system_faces,
        dtype=np.int64,
    )
    if element_system_faces64.size == 0:
        raise ValueError("element_system_faces must be non-empty")
    if np.any(element_system_faces64 < -1):
        raise ValueError("element_system_faces may contain only row ids or -1")
    active = element_system_faces64 >= 0
    if np.any(element_system_faces64[active] >= system.num_rows):
        raise ValueError("element_system_faces contains an out-of-range row id")
    if system.num_rows > np.iinfo(np.int32).max:
        raise OverflowError("the CUDA ASM kernels use int32 face indices")

    element_system_faces = np.ascontiguousarray(
        element_system_faces64,
        dtype=np.int32,
    )
    inverse_residuals = np.ascontiguousarray(
        cpu_preconditioner.inverse_residuals,
        dtype=np.float64,
    )

    return AdditiveSchwarzBatchLayout(
        inverse_matrices=inverse_matrices,
        element_system_faces=element_system_faces,
        inverse_residuals=inverse_residuals,
        block_size=int(cpu_preconditioner.block_size),
        num_system_faces=int(system.num_rows),
    )


_ASM_KERNEL_SOURCE = r"""
__device__ __forceinline__ float asm_atomic_add(float* address, float value)
{
    return atomicAdd(address, value);
}

__device__ __forceinline__ double asm_atomic_add(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    unsigned long long int* address_as_ull =
        (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(
                value + __longlong_as_double(assumed)
            )
        );
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

extern "C" __global__
void restrict_element_faces_f32(
    const unsigned long long total,
    const int num_local_faces,
    const int block_size,
    const int* __restrict__ element_system_faces,
    const float* __restrict__ global_vector,
    float* __restrict__ element_vector)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total) return;

    const int local_dof = (int) (index % block_size);
    const unsigned long long element_face = index / block_size;
    const int system_face = element_system_faces[element_face];
    element_vector[index] = (system_face >= 0)
        ? global_vector[(unsigned long long) system_face * block_size + local_dof]
        : 0.0f;
}

extern "C" __global__
void restrict_element_faces_f64(
    const unsigned long long total,
    const int num_local_faces,
    const int block_size,
    const int* __restrict__ element_system_faces,
    const double* __restrict__ global_vector,
    double* __restrict__ element_vector)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total) return;

    const int local_dof = (int) (index % block_size);
    const unsigned long long element_face = index / block_size;
    const int system_face = element_system_faces[element_face];
    element_vector[index] = (system_face >= 0)
        ? global_vector[(unsigned long long) system_face * block_size + local_dof]
        : 0.0;
}

extern "C" __global__
void prolong_element_faces_f32(
    const unsigned long long total,
    const int num_local_faces,
    const int block_size,
    const int* __restrict__ element_system_faces,
    const float* __restrict__ element_vector,
    float* __restrict__ global_vector)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total) return;

    const int local_dof = (int) (index % block_size);
    const unsigned long long element_face = index / block_size;
    const int system_face = element_system_faces[element_face];
    if (system_face >= 0) {
        asm_atomic_add(
            global_vector + (unsigned long long) system_face * block_size + local_dof,
            element_vector[index]
        );
    }
}

extern "C" __global__
void prolong_element_faces_f64(
    const unsigned long long total,
    const int num_local_faces,
    const int block_size,
    const int* __restrict__ element_system_faces,
    const double* __restrict__ element_vector,
    double* __restrict__ global_vector)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total) return;

    const int local_dof = (int) (index % block_size);
    const unsigned long long element_face = index / block_size;
    const int system_face = element_system_faces[element_face];
    if (system_face >= 0) {
        asm_atomic_add(
            global_vector + (unsigned long long) system_face * block_size + local_dof,
            element_vector[index]
        );
    }
}
"""


@dataclass
class CuPyFaceAdditiveSchwarzPreconditioner:
    r"""Device implementation of one-element overlapping additive Schwarz.

    Application is the fully device-resident sequence

    ``element_rhs = R @ x``
        CUDA gather kernel;
    ``element_solution = inverse_matrices @ element_rhs``
        batched dense matrix--vector products through ``cupy.matmul``;
    ``out = sum_e R_e.T @ element_solution_e``
        CUDA scatter-add kernel using atomics on shared faces.

    The CPU is involved only during setup, when the validated reference builder
    constructs and inverts the enriched local matrices.
    """

    inverse_matrices: Any
    element_system_faces: Any
    block_size: int
    num_system_faces: int
    device_id: int

    def __post_init__(self) -> None:
        cp = require_cupy_device()
        if not isinstance(self.inverse_matrices, cp.ndarray):
            raise TypeError("inverse_matrices must be a CuPy array")
        if self.inverse_matrices.ndim != 3:
            raise ValueError("inverse_matrices must have shape (NE, L, L)")
        if self.inverse_matrices.shape[1] != self.inverse_matrices.shape[2]:
            raise ValueError("ASM inverse matrices must be square")
        if self.inverse_matrices.dtype not in (cp.float32, cp.float64):
            raise TypeError("inverse_matrices must use float32 or float64")
        if not self.inverse_matrices.flags.c_contiguous:
            raise ValueError("inverse_matrices must be C-contiguous")

        if not isinstance(self.element_system_faces, cp.ndarray):
            raise TypeError("element_system_faces must be a CuPy array")
        if self.element_system_faces.ndim != 2:
            raise ValueError("element_system_faces must have shape (NE, Nlfe)")
        if self.element_system_faces.dtype != cp.int32:
            raise TypeError("element_system_faces must use int32")
        if not self.element_system_faces.flags.c_contiguous:
            raise ValueError("element_system_faces must be C-contiguous")
        if self.element_system_faces.shape[0] != self.inverse_matrices.shape[0]:
            raise ValueError("ASM matrix and connectivity batches disagree")

        self.device_id = int(self.device_id)
        if int(self.inverse_matrices.device.id) != self.device_id:
            raise ValueError("inverse_matrices is on the wrong CUDA device")
        if int(self.element_system_faces.device.id) != self.device_id:
            raise ValueError("element_system_faces is on the wrong CUDA device")

        self.block_size = int(self.block_size)
        self.num_system_faces = int(self.num_system_faces)
        if self.block_size <= 0 or self.num_system_faces <= 0:
            raise ValueError("block_size and num_system_faces must be positive")
        if self.local_size != self.num_local_faces * self.block_size:
            raise ValueError(
                "local matrix size must equal num_local_faces * block_size"
            )

        # Direct construction is supported, so validate device connectivity at
        # setup time.  These scalar reads synchronize once, never in GMRES.
        minimum_face = int(cp.min(self.element_system_faces).item())
        maximum_face = int(cp.max(self.element_system_faces).item())
        if minimum_face < -1:
            raise ValueError("element_system_faces may contain only row ids or -1")
        if maximum_face >= self.num_system_faces:
            raise ValueError("element_system_faces contains an out-of-range row id")

        self._cp = cp
        self._element_rhs = cp.empty(
            (self.num_elements, self.local_size),
            dtype=self.dtype,
        )
        self._matmul_output = cp.empty(
            (self.num_elements, self.local_size, 1),
            dtype=self.dtype,
        )

        suffix = "f32" if self.dtype == cp.float32 else "f64"
        self._restrict_kernel = cp.RawKernel(
            _ASM_KERNEL_SOURCE,
            f"restrict_element_faces_{suffix}",
        )
        self._prolong_kernel = cp.RawKernel(
            _ASM_KERNEL_SOURCE,
            f"prolong_element_faces_{suffix}",
        )
        self._threads_per_block = 256
        self._kernel_blocks = (
            self.num_local_dofs + self._threads_per_block - 1
        ) // self._threads_per_block

    @classmethod
    def from_system(
        cls,
        system: FaceDenseSystem,
        element_blocks: np.ndarray,
        loc2glob_face: np.ndarray,
        *,
        dtype: np.dtype | type | None = None,
        device_id: int | None = None,
        inverse_residual_tolerance: float | None = None,
    ) -> "CuPyFaceAdditiveSchwarzPreconditioner":
        """Build CPU local inverses, transfer them, and allocate GPU buffers."""

        cp = require_cupy_device()
        layout = prepare_face_additive_schwarz_batch_layout(
            system,
            element_blocks,
            loc2glob_face,
            dtype=dtype,
            inverse_residual_tolerance=inverse_residual_tolerance,
        )
        selected_device = (
            int(cp.cuda.Device().id) if device_id is None else int(device_id)
        )
        with cp.cuda.Device(selected_device):
            inverse_matrices = cp.asarray(layout.inverse_matrices)
            element_system_faces = cp.asarray(layout.element_system_faces)
            return cls(
                inverse_matrices=inverse_matrices,
                element_system_faces=element_system_faces,
                block_size=layout.block_size,
                num_system_faces=layout.num_system_faces,
                device_id=selected_device,
            )

    @property
    def num_elements(self) -> int:
        return int(self.inverse_matrices.shape[0])

    @property
    def num_local_faces(self) -> int:
        return int(self.element_system_faces.shape[1])

    @property
    def local_size(self) -> int:
        return int(self.inverse_matrices.shape[1])

    @property
    def num_local_dofs(self) -> int:
        return self.num_elements * self.local_size

    @property
    def num_dofs(self) -> int:
        return self.num_system_faces * self.block_size

    @property
    def dtype(self) -> Any:
        return self.inverse_matrices.dtype

    @property
    def restricted_buffer(self) -> Any:
        """Reusable element-restricted vector with shape ``(NE, Nlfe, b)``."""

        return self._element_rhs.reshape(
            self.num_elements,
            self.num_local_faces,
            self.block_size,
        )

    @property
    def local_solution_buffer(self) -> Any:
        """Reusable local-solve output with shape ``(NE, Nlfe, b)``."""

        return self._matmul_output[:, :, 0].reshape(
            self.num_elements,
            self.num_local_faces,
            self.block_size,
        )

    def _validate_global_vector(self, vector: Any, *, name: str) -> Any:
        cp = self._cp
        if not isinstance(vector, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        if int(vector.device.id) != self.device_id:
            raise ValueError(f"{name} is on the wrong CUDA device")
        if vector.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}")
        if vector.size != self.num_dofs:
            raise ValueError(f"{name} must contain {self.num_dofs} values")
        if vector.ndim not in (1, 2):
            raise ValueError(f"{name} must be flat or face-major")
        if vector.ndim == 2 and vector.shape != (
            self.num_system_faces,
            self.block_size,
        ):
            raise ValueError(f"{name} has an incompatible face-major shape")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector.reshape(-1)

    def _validate_element_vector(self, vector: Any, *, name: str) -> Any:
        cp = self._cp
        if not isinstance(vector, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        if int(vector.device.id) != self.device_id:
            raise ValueError(f"{name} is on the wrong CUDA device")
        if vector.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}")
        if vector.size != self.num_local_dofs:
            raise ValueError(
                f"{name} must contain {self.num_local_dofs} local values"
            )
        valid_shapes = {
            (self.num_elements, self.local_size),
            (self.num_elements, self.num_local_faces, self.block_size),
        }
        if vector.shape not in valid_shapes:
            raise ValueError(f"{name} has an incompatible element-major shape")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector.reshape(-1)

    def _launch_restrict(self, x_flat: Any, element_flat: Any) -> None:
        self._restrict_kernel(
            (self._kernel_blocks,),
            (self._threads_per_block,),
            (
                np.uint64(self.num_local_dofs),
                np.int32(self.num_local_faces),
                np.int32(self.block_size),
                self.element_system_faces,
                x_flat,
                element_flat,
            ),
        )

    def _launch_prolong(self, element_flat: Any, out_flat: Any) -> None:
        self._prolong_kernel(
            (self._kernel_blocks,),
            (self._threads_per_block,),
            (
                np.uint64(self.num_local_dofs),
                np.int32(self.num_local_faces),
                np.int32(self.block_size),
                self.element_system_faces,
                element_flat,
                out_flat,
            ),
        )

    def restrict_into(self, x: Any, out: Any) -> None:
        """Gather a global face vector into a preallocated element buffer."""

        cp = self._cp
        x_flat = self._validate_global_vector(x, name="x")
        out_flat = self._validate_element_vector(out, name="out")
        if cp.shares_memory(x, out):
            raise ValueError("x and out must not alias")
        self._launch_restrict(x_flat, out_flat)

    def restrict(self, x: Any) -> Any:
        """Return ``R_e x`` with shape ``(NE, Nlfe, b)``."""

        out = self._cp.empty(
            (self.num_elements, self.num_local_faces, self.block_size),
            dtype=self.dtype,
        )
        self.restrict_into(x, out)
        return out

    def prolong_into(self, element_vector: Any, out: Any) -> None:
        """Scatter-add element vectors into a preallocated global vector."""

        cp = self._cp
        element_flat = self._validate_element_vector(
            element_vector,
            name="element_vector",
        )
        out_flat = self._validate_global_vector(out, name="out")
        if cp.shares_memory(element_vector, out):
            raise ValueError("element_vector and out must not alias")
        out_flat.fill(0)
        self._launch_prolong(element_flat, out_flat)

    def prolong(self, element_vector: Any, *, flat: bool = False) -> Any:
        """Return ``sum_e R_e.T element_vector_e`` on the GPU."""

        shape = (self.num_dofs,) if flat else (
            self.num_system_faces,
            self.block_size,
        )
        out = self._cp.empty(shape, dtype=self.dtype)
        self.prolong_into(element_vector, out)
        return out

    def apply_into(self, x: Any, out: Any) -> None:
        """Apply ``sum_e R_e.T P_e^{-1} R_e`` without buffer allocation."""

        cp = self._cp
        x_flat = self._validate_global_vector(x, name="x")
        out_flat = self._validate_global_vector(out, name="out")
        if cp.shares_memory(x, out):
            raise ValueError("x and out must not alias")

        self._launch_restrict(x_flat, self._element_rhs.reshape(-1))
        cp.matmul(
            self.inverse_matrices,
            self._element_rhs.reshape(self.num_elements, self.local_size, 1),
            out=self._matmul_output,
        )
        out_flat.fill(0)
        self._launch_prolong(self._matmul_output.reshape(-1), out_flat)

    def apply(self, x: Any) -> Any:
        """Allocate an output vector and apply the CUDA ASM preconditioner."""

        out = self._cp.empty_like(x)
        self.apply_into(x, out)
        return out


__all__ = [
    "AdditiveSchwarzBatchLayout",
    "CuPyFaceAdditiveSchwarzPreconditioner",
    "CuPyFaceBlockJacobiPreconditioner",
    "prepare_face_additive_schwarz_batch_layout",
]
