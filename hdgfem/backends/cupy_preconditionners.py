"""CUDA preconditioners for face-dense HDG systems.

Three local-solver paths are available for validation and later profiling:
validated CPU inverses transferred to the device, public CuPy batched GPU
inversion, explicit cuBLAS ``getrfBatched``/``getriBatched`` inversion, and
direct batched GPU solves at every application.  The inverse paths are the
production candidates; the direct solve path intentionally exposes the cost of
repeated public-API factorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hdgfem.assembly.face_dense import FaceDenseSystem
from hdgfem.linalg.additive_schwarz import (
    build_face_additive_schwarz_local_matrices,
    build_face_additive_schwarz_preconditioner,
)
from hdgfem.linalg.block_jacobi import build_face_block_jacobi_preconditioner
from hdgfem.backends.cublas_batched import invert_batched_cublas
from hdgfem.runtime.optional import device_arrays_overlap, require_cupy_device
from hdgfem.backends.cupy import solve_batched_vectors


_BATCHED_DENSE_MV_KERNEL_SOURCE = r"""
extern "C" __global__
void batched_dense_mv_f32(
    const unsigned long long total_rows,
    const int matrix_size,
    const float* __restrict__ matrices,
    const float* __restrict__ vectors,
    float* __restrict__ output)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total_rows) return;

    const unsigned long long batch = index / matrix_size;
    const int row = (int) (index - batch * matrix_size);
    const unsigned long long matrix_offset =
        batch * (unsigned long long) matrix_size * matrix_size
        + (unsigned long long) row * matrix_size;
    const unsigned long long vector_offset = batch * matrix_size;

    float value = 0.0f;
    for (int column = 0; column < matrix_size; ++column) {
        value += matrices[matrix_offset + column]
            * vectors[vector_offset + column];
    }
    output[index] = value;
}

extern "C" __global__
void batched_dense_mv_f64(
    const unsigned long long total_rows,
    const int matrix_size,
    const double* __restrict__ matrices,
    const double* __restrict__ vectors,
    double* __restrict__ output)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total_rows) return;

    const unsigned long long batch = index / matrix_size;
    const int row = (int) (index - batch * matrix_size);
    const unsigned long long matrix_offset =
        batch * (unsigned long long) matrix_size * matrix_size
        + (unsigned long long) row * matrix_size;
    const unsigned long long vector_offset = batch * matrix_size;

    double value = 0.0;
    for (int column = 0; column < matrix_size; ++column) {
        value += matrices[matrix_offset + column]
            * vectors[vector_offset + column];
    }
    output[index] = value;
}
"""


def _build_raw_batched_mv_kernel(cp: Any, dtype: Any) -> Any:
    """Build the requested preconditioner or device helper."""
    suffix = "f32" if dtype == cp.float32 else "f64"
    return cp.RawKernel(
        _BATCHED_DENSE_MV_KERNEL_SOURCE,
        f"batched_dense_mv_{suffix}",
    )


def _launch_raw_batched_mv(
    kernel: Any,
    matrices: Any,
    vectors: Any,
    output: Any,
    *,
    matrix_size: int,
) -> None:
    """Launch the corresponding preallocated CUDA kernel."""
    total_rows = int(matrices.shape[0]) * int(matrix_size)
    threads = 256
    blocks = (total_rows + threads - 1) // threads
    kernel(
        (blocks,),
        (threads,),
        (
            np.uint64(total_rows),
            np.int32(matrix_size),
            matrices,
            vectors,
            output,
        ),
    )


def _gpu_batched_inverse(cp: Any, matrices: Any, *, label: str) -> tuple[Any, Any]:
    """Compute batched inverses and residuals on the active GPU.

    ``cupy.linalg.solve`` accepts ``(..., M, M)`` batches.  Solving against a
    batched identity exposes the same LU-based operation while keeping this
    implementation on CuPy's public API.
    """

    size = int(matrices.shape[1])
    identity = cp.broadcast_to(
        cp.eye(size, dtype=matrices.dtype),
        matrices.shape,
    ).copy()
    try:
        inverse = cp.linalg.solve(matrices, identity)
    except Exception as error:  # pragma: no cover - CUDA/cuSOLVER dependent.
        raise np.linalg.LinAlgError(f"{label} GPU inversion failed") from error
    inverse = cp.ascontiguousarray(inverse)
    if not bool(cp.all(cp.isfinite(inverse)).item()):
        raise FloatingPointError(f"{label} GPU inversion produced non-finite values")
    products = cp.matmul(matrices, inverse)
    residuals = cp.max(
        cp.sum(cp.abs(products - identity), axis=2),
        axis=1,
    )
    if not bool(cp.all(cp.isfinite(residuals)).item()):
        raise FloatingPointError(f"{label} GPU inverse residuals are non-finite")
    return inverse, residuals


def _check_gpu_inverse_tolerance(
    cp: Any,
    residuals: Any,
    tolerance: float | None,
    *,
    label: str,
) -> None:
    """Execute the ``_check_gpu_inverse_tolerance`` numerical helper."""
    if tolerance is None:
        return
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("inverse_residual_tolerance must be finite and non-negative")
    maximum = float(cp.max(residuals).item())
    if maximum > tolerance:
        index = int(cp.argmax(residuals).item())
        raise FloatingPointError(
            f"{label} GPU inverse failed the requested residual tolerance: "
            f"batch={index}, residual={maximum:.3e}, tolerance={tolerance:.3e}"
        )


@dataclass
class CuPyFaceBlockJacobiPreconditioner:
    """CUDA block-Jacobi with selectable local-solver setup.

    ``local_solver`` may be:

    ``"cpu_inverse"``
        Build the inverse blocks with the validated NumPy reference and
        transfer them once.
    ``"gpu_inverse"``
        Transfer diagonal blocks, compute their batched inverse on the GPU
        with :func:`cupy.linalg.solve`, then apply with batched matmul.
    ``"cublas_inverse"``
        Factor and invert the diagonal blocks with explicit low-level cuBLAS
        ``getrfBatched`` and ``getriBatched`` calls, retaining per-batch status
        arrays and applying the resulting inverse with batched matmul.
    ``"gpu_solve"``
        Keep the diagonal blocks and call the batched GPU dense solver during
        every application.  This is a correctness/profiling baseline; CuPy's
        public API refactorizes and allocates on each call.
    """

    inverse_blocks: Any | None
    device_id: int
    local_blocks: Any | None = None
    local_solver: str = "external_inverse"
    inverse_residuals: Any | None = None
    factorization_info: np.ndarray | None = None
    inversion_info: np.ndarray | None = None
    application: str = "matmul"

    def __post_init__(self) -> None:
        """Validate and normalize the initialized data."""
        cp = require_cupy_device()
        valid_modes = {
            "external_inverse",
            "cpu_inverse",
            "gpu_inverse",
            "cublas_inverse",
            "gpu_solve",
        }
        if self.local_solver not in valid_modes:
            raise ValueError(f"unsupported block-Jacobi local_solver: {self.local_solver}")
        use_inverse = self.local_solver != "gpu_solve"
        if self.application not in {"matmul", "raw"}:
            raise ValueError(f"unsupported block-Jacobi application: {self.application}")
        if not use_inverse and self.application != "matmul":
            raise ValueError("raw application requires precomputed inverse blocks")
        matrices = self.inverse_blocks if use_inverse else self.local_blocks
        if not isinstance(matrices, cp.ndarray):
            name = "inverse_blocks" if use_inverse else "local_blocks"
            raise TypeError(f"{name} must be a CuPy array")
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("face blocks must have shape (NF, b, b)")
        if matrices.dtype not in (cp.float32, cp.float64):
            raise TypeError("face blocks must use float32 or float64")
        if not matrices.flags.c_contiguous:
            raise ValueError("face blocks must be C-contiguous")

        self.device_id = int(self.device_id)
        if int(matrices.device.id) != self.device_id:
            raise ValueError("face blocks are on the wrong CUDA device")
        self._cp = cp
        self._batch_matrices = matrices
        self._output = cp.empty((self.num_faces, self.block_size), dtype=self.dtype)
        self._matmul_output = (
            cp.empty((self.num_faces, self.block_size, 1), dtype=self.dtype)
            if use_inverse and self.application == "matmul"
            else None
        )
        self._raw_apply_kernel = (
            _build_raw_batched_mv_kernel(cp, self.dtype)
            if use_inverse and self.application == "raw"
            else None
        )

    @classmethod
    def from_system(
        cls,
        system: FaceDenseSystem,
        *,
        device_id: int | None = None,
        dtype: np.dtype | type | None = None,
        local_solver: str = "cpu_inverse",
        application: str = "matmul",
        inverse_residual_tolerance: float | None = None,
    ) -> "CuPyFaceBlockJacobiPreconditioner":
        """Construct the object from the supplied operator or system."""
        cp = require_cupy_device()
        if not isinstance(system, FaceDenseSystem):
            raise TypeError("system must be a FaceDenseSystem")
        if local_solver not in {
            "cpu_inverse",
            "gpu_inverse",
            "cublas_inverse",
            "gpu_solve",
        }:
            raise ValueError(f"unsupported block-Jacobi local_solver: {local_solver}")
        requested_dtype = np.dtype(system.blocks.dtype if dtype is None else dtype)
        if requested_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("dtype must be float32 or float64")
        selected_device = (
            int(cp.cuda.Device().id) if device_id is None else int(device_id)
        )

        if local_solver == "cpu_inverse":
            cpu = build_face_block_jacobi_preconditioner(
                system,
                inverse_residual_tolerance=inverse_residual_tolerance,
            )
            with cp.cuda.Device(selected_device):
                inverse_blocks = cp.asarray(
                    cpu.inverse_blocks,
                    dtype=requested_dtype,
                    order="C",
                )
            return cls(
                inverse_blocks=inverse_blocks,
                device_id=selected_device,
                local_solver=local_solver,
                inverse_residuals=cp.asarray(cpu.inverse_residuals),
                application=application,
            )

        diagonal_host = np.ascontiguousarray(
            system.blocks[:, 0],
            dtype=requested_dtype,
        )
        with cp.cuda.Device(selected_device):
            local_blocks = cp.asarray(diagonal_host)
            if local_solver == "gpu_solve":
                if inverse_residual_tolerance is not None:
                    raise ValueError(
                        "inverse_residual_tolerance is not applicable to gpu_solve"
                    )
                return cls(
                    inverse_blocks=None,
                    local_blocks=local_blocks,
                    device_id=selected_device,
                    local_solver=local_solver,
                    application=application,
                )
            factorization_info = None
            inversion_info = None
            if local_solver == "cublas_inverse":
                cublas_result = invert_batched_cublas(
                    local_blocks,
                    label="block-Jacobi diagonal blocks",
                )
                inverse_blocks = cublas_result.inverse_matrices
                residuals = cublas_result.inverse_residuals
                factorization_info = cublas_result.factorization_info
                inversion_info = cublas_result.inversion_info
            else:
                inverse_blocks, residuals = _gpu_batched_inverse(
                    cp,
                    local_blocks,
                    label="block-Jacobi diagonal blocks",
                )
            _check_gpu_inverse_tolerance(
                cp,
                residuals,
                inverse_residual_tolerance,
                label="block-Jacobi",
            )
            return cls(
                inverse_blocks=inverse_blocks,
                local_blocks=local_blocks,
                device_id=selected_device,
                local_solver=local_solver,
                inverse_residuals=residuals,
                factorization_info=factorization_info,
                inversion_info=inversion_info,
                application=application,
            )

    @property
    def num_faces(self) -> int:
        """Return the number of faces."""
        return int(self._batch_matrices.shape[0])

    @property
    def block_size(self) -> int:
        """Return the dense face-block size."""
        return int(self._batch_matrices.shape[1])

    @property
    def num_dofs(self) -> int:
        """Return the number of scalar degrees of freedom."""
        return self.num_faces * self.block_size

    @property
    def dtype(self) -> Any:
        """Return the scalar dtype."""
        return self._batch_matrices.dtype

    @property
    def allocates_during_apply(self) -> bool:
        """Return whether an application allocates device storage."""
        return self.local_solver == "gpu_solve"

    @property
    def maximum_inverse_residual(self) -> float | None:
        """Return the largest local inverse residual."""
        if self.inverse_residuals is None:
            return None
        return float(self._cp.max(self.inverse_residuals).item())

    def _validate_vector(self, vector: Any, *, name: str) -> Any:
        """Validate shapes, dtypes, devices, and solver parameters."""
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
        if vector.ndim == 2 and vector.shape != (self.num_faces, self.block_size):
            raise ValueError(f"{name} has an incompatible face-major shape")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector.reshape(self.num_faces, self.block_size)

    def apply_into(self, x: Any, out: Any) -> None:
        """Apply the preconditioner using reusable storage."""
        cp = self._cp
        x_faces = self._validate_vector(x, name="x")
        out_faces = self._validate_vector(out, name="out")
        if device_arrays_overlap(x, out):
            raise ValueError("x and out must not alias")

        if self.local_solver == "gpu_solve":
            # Public CuPy batched solve currently has no ``out`` parameter and
            # does not expose reusable LU factors.  Keep this path as a
            # correctness/performance comparison, not the production default.
            solved = solve_batched_vectors(cp, self.local_blocks, x_faces)
            self._output[...] = solved
            out_faces[...] = self._output
            return

        if self.application == "raw":
            assert self._raw_apply_kernel is not None
            _launch_raw_batched_mv(
                self._raw_apply_kernel,
                self.inverse_blocks,
                x_faces,
                out_faces,
                matrix_size=self.block_size,
            )
            return

        cp.matmul(
            self.inverse_blocks,
            x_faces.reshape(self.num_faces, self.block_size, 1),
            out=self._matmul_output,
        )
        out_faces[...] = self._matmul_output[:, :, 0]

    def apply(self, x: Any) -> Any:
        """Apply the preconditioner using reusable storage."""
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
    inverse_residuals: np.ndarray
    block_size: int
    num_system_faces: int

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
        return int(self.inverse_matrices.shape[1])

    @property
    def num_dofs(self) -> int:
        """Return the number of scalar degrees of freedom."""
        return self.num_system_faces * self.block_size

    @property
    def maximum_inverse_residual(self) -> float:
        """Return the largest local inverse residual."""
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


@dataclass(frozen=True)
class AdditiveSchwarzMatrixLayout:
    """Host-side one-element ASM matrices before accelerator inversion."""

    local_matrices: np.ndarray
    element_system_faces: np.ndarray
    block_size: int
    num_system_faces: int

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
    def num_dofs(self) -> int:
        """Return the number of scalar degrees of freedom."""
        return self.num_system_faces * self.block_size


def prepare_face_additive_schwarz_matrix_layout(
    system: FaceDenseSystem,
    element_blocks: np.ndarray,
    loc2glob_face: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> AdditiveSchwarzMatrixLayout:
    """Prepare enriched local matrices without computing CPU inverses."""

    if not isinstance(system, FaceDenseSystem):
        raise TypeError("system must be a FaceDenseSystem")
    requested_dtype = np.dtype(system.blocks.dtype if dtype is None else dtype)
    if requested_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("dtype must be float32 or float64")

    local = build_face_additive_schwarz_local_matrices(
        system,
        element_blocks,
        loc2glob_face,
    )
    faces64 = np.asarray(local.element_system_faces, dtype=np.int64)
    if faces64.size == 0:
        raise ValueError("element_system_faces must be non-empty")
    if np.any(faces64 < -1):
        raise ValueError("element_system_faces may contain only row ids or -1")
    active = faces64 >= 0
    if np.any(faces64[active] >= system.num_rows):
        raise ValueError("element_system_faces contains an out-of-range row id")
    if system.num_rows > np.iinfo(np.int32).max:
        raise OverflowError("the CUDA ASM kernels use int32 face indices")

    return AdditiveSchwarzMatrixLayout(
        local_matrices=np.ascontiguousarray(
            local.local_matrices,
            dtype=requested_dtype,
        ),
        element_system_faces=np.ascontiguousarray(faces64, dtype=np.int32),
        block_size=int(local.block_size),
        num_system_faces=int(system.num_rows),
    )


def build_face_additive_schwarz_incidence_slots(
    element_system_faces: np.ndarray,
    num_system_faces: int,
) -> np.ndarray:
    """Build a race-free face-to-element incidence table for one-element ASM.

    The returned ``int32`` array has shape ``(num_system_faces, 2)``.  Each
    non-negative entry is the flattened element-local-face slot
    ``element * Nlfe + local_face`` contributing to that system face.  Boundary
    faces have one active slot and interior manifold faces have two.

    A face with more than two incident elements is rejected because the fused
    CUDA prolongation kernel intentionally targets conforming manifold meshes.
    """

    faces = np.asarray(element_system_faces, dtype=np.int64)
    if faces.ndim != 2 or faces.size == 0:
        raise ValueError("element_system_faces must have shape (NE, Nlfe)")
    num_system_faces = int(num_system_faces)
    if num_system_faces <= 0:
        raise ValueError("num_system_faces must be positive")
    if np.any(faces < -1):
        raise ValueError("element_system_faces may contain only row ids or -1")
    active = faces >= 0
    if np.any(faces[active] >= num_system_faces):
        raise ValueError("element_system_faces contains an out-of-range row id")
    if faces.shape[0] * faces.shape[1] > np.iinfo(np.int32).max:
        raise OverflowError("flattened element-face slots exceed int32 capacity")

    slots = np.full((num_system_faces, 2), -1, dtype=np.int32)
    counts = np.zeros(num_system_faces, dtype=np.int8)
    num_local_faces = int(faces.shape[1])
    for element in range(int(faces.shape[0])):
        for local_face in range(num_local_faces):
            system_face = int(faces[element, local_face])
            if system_face < 0:
                continue
            incidence = int(counts[system_face])
            if incidence >= 2:
                raise ValueError(
                    "fused ASM requires at most two incident elements per system face"
                )
            slots[system_face, incidence] = element * num_local_faces + local_face
            counts[system_face] = incidence + 1

    missing = np.flatnonzero(counts == 0)
    if missing.size:
        raise ValueError(
            "every system face must occur in element_system_faces; "
            f"first missing row is {int(missing[0])}"
        )
    return np.ascontiguousarray(slots)


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


_FUSED_ASM_KERNEL_SOURCE = r"""
extern "C" __global__
void fused_asm_local_f32(
    const unsigned long long total_rows,
    const int num_local_faces,
    const int block_size,
    const int local_size,
    const int* __restrict__ element_system_faces,
    const float* __restrict__ inverse_matrices,
    const float* __restrict__ global_vector,
    float* __restrict__ local_output)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total_rows) return;

    const unsigned long long element = index / local_size;
    const int row = (int) (index - element * local_size);
    const unsigned long long matrix_offset =
        element * (unsigned long long) local_size * local_size
        + (unsigned long long) row * local_size;
    const unsigned long long face_offset = element * num_local_faces;

    float value = 0.0f;
    for (int column = 0; column < local_size; ++column) {
        const int local_face = column / block_size;
        const int local_dof = column - local_face * block_size;
        const int system_face = element_system_faces[face_offset + local_face];
        const float input = (system_face >= 0)
            ? global_vector[(unsigned long long) system_face * block_size + local_dof]
            : 0.0f;
        value += inverse_matrices[matrix_offset + column] * input;
    }
    local_output[index] = value;
}

extern "C" __global__
void fused_asm_local_f64(
    const unsigned long long total_rows,
    const int num_local_faces,
    const int block_size,
    const int local_size,
    const int* __restrict__ element_system_faces,
    const double* __restrict__ inverse_matrices,
    const double* __restrict__ global_vector,
    double* __restrict__ local_output)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total_rows) return;

    const unsigned long long element = index / local_size;
    const int row = (int) (index - element * local_size);
    const unsigned long long matrix_offset =
        element * (unsigned long long) local_size * local_size
        + (unsigned long long) row * local_size;
    const unsigned long long face_offset = element * num_local_faces;

    double value = 0.0;
    for (int column = 0; column < local_size; ++column) {
        const int local_face = column / block_size;
        const int local_dof = column - local_face * block_size;
        const int system_face = element_system_faces[face_offset + local_face];
        const double input = (system_face >= 0)
            ? global_vector[(unsigned long long) system_face * block_size + local_dof]
            : 0.0;
        value += inverse_matrices[matrix_offset + column] * input;
    }
    local_output[index] = value;
}

extern "C" __global__
void race_free_asm_prolong_f32(
    const unsigned long long total,
    const int num_local_faces,
    const int block_size,
    const int local_size,
    const int* __restrict__ face_element_slots,
    const float* __restrict__ local_vector,
    float* __restrict__ global_vector)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total) return;

    const unsigned long long system_face = index / block_size;
    const int local_dof = (int) (index - system_face * block_size);
    float value = 0.0f;
    #pragma unroll
    for (int side = 0; side < 2; ++side) {
        const int element_face = face_element_slots[system_face * 2 + side];
        if (element_face >= 0) {
            const int element = element_face / num_local_faces;
            const int local_face = element_face - element * num_local_faces;
            const unsigned long long local_index =
                (unsigned long long) element * local_size
                + (unsigned long long) local_face * block_size + local_dof;
            value += local_vector[local_index];
        }
    }
    global_vector[index] = value;
}

extern "C" __global__
void race_free_asm_prolong_f64(
    const unsigned long long total,
    const int num_local_faces,
    const int block_size,
    const int local_size,
    const int* __restrict__ face_element_slots,
    const double* __restrict__ local_vector,
    double* __restrict__ global_vector)
{
    const unsigned long long index =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= total) return;

    const unsigned long long system_face = index / block_size;
    const int local_dof = (int) (index - system_face * block_size);
    double value = 0.0;
    #pragma unroll
    for (int side = 0; side < 2; ++side) {
        const int element_face = face_element_slots[system_face * 2 + side];
        if (element_face >= 0) {
            const int element = element_face / num_local_faces;
            const int local_face = element_face - element * num_local_faces;
            const unsigned long long local_index =
                (unsigned long long) element * local_size
                + (unsigned long long) local_face * block_size + local_dof;
            value += local_vector[local_index];
        }
    }
    global_vector[index] = value;
}
"""


@dataclass
class CuPyFaceAdditiveSchwarzPreconditioner:
    r"""Device one-element ASM with selectable local dense solver.

    ``cpu_inverse`` transfers validated NumPy inverses. ``gpu_inverse`` builds
    the inverse batch through CuPy's public solver. ``cublas_inverse`` calls
    explicit cuBLAS ``getrfBatched``/``getriBatched`` and retains the per-batch
    status arrays. Both inverse modes use allocation-free batched matmul during
    GMRES. ``gpu_solve`` calls the public batched
    :func:`cupy.linalg.solve` at every application; it is retained as an
    independent correctness and profiling baseline, not the production default.
    """

    inverse_matrices: Any | None
    element_system_faces: Any
    block_size: int
    num_system_faces: int
    device_id: int
    local_matrices: Any | None = None
    local_solver: str = "external_inverse"
    inverse_residuals: Any | None = None
    factorization_info: np.ndarray | None = None
    inversion_info: np.ndarray | None = None
    application: str = "matmul"
    face_element_slots: Any | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the initialized data."""
        cp = require_cupy_device()
        valid_modes = {
            "external_inverse",
            "cpu_inverse",
            "gpu_inverse",
            "cublas_inverse",
            "gpu_solve",
        }
        if self.local_solver not in valid_modes:
            raise ValueError(f"unsupported ASM local_solver: {self.local_solver}")
        use_inverse = self.local_solver != "gpu_solve"
        if self.application not in {"matmul", "raw", "fused"}:
            raise ValueError(f"unsupported ASM application: {self.application}")
        if not use_inverse and self.application != "matmul":
            raise ValueError(
                "raw and fused applications require precomputed inverse matrices"
            )
        matrices = self.inverse_matrices if use_inverse else self.local_matrices
        matrix_name = "inverse_matrices" if use_inverse else "local_matrices"
        if not isinstance(matrices, cp.ndarray):
            raise TypeError(f"{matrix_name} must be a CuPy array")
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError("ASM matrices must have shape (NE, L, L)")
        if matrices.dtype not in (cp.float32, cp.float64):
            raise TypeError("ASM matrices must use float32 or float64")
        if not matrices.flags.c_contiguous:
            raise ValueError("ASM matrices must be C-contiguous")

        if not isinstance(self.element_system_faces, cp.ndarray):
            raise TypeError("element_system_faces must be a CuPy array")
        if self.element_system_faces.ndim != 2:
            raise ValueError("element_system_faces must have shape (NE, Nlfe)")
        if self.element_system_faces.dtype != cp.int32:
            raise TypeError("element_system_faces must use int32")
        if not self.element_system_faces.flags.c_contiguous:
            raise ValueError("element_system_faces must be C-contiguous")
        if self.element_system_faces.shape[0] != matrices.shape[0]:
            raise ValueError("ASM matrix and connectivity batches disagree")

        self.device_id = int(self.device_id)
        if int(matrices.device.id) != self.device_id:
            raise ValueError("ASM matrices are on the wrong CUDA device")
        if int(self.element_system_faces.device.id) != self.device_id:
            raise ValueError("element_system_faces is on the wrong CUDA device")
        if self.local_matrices is not None and int(self.local_matrices.device.id) != self.device_id:
            raise ValueError("local_matrices is on the wrong CUDA device")
        if self.inverse_matrices is not None and int(self.inverse_matrices.device.id) != self.device_id:
            raise ValueError("inverse_matrices is on the wrong CUDA device")

        self.block_size = int(self.block_size)
        self.num_system_faces = int(self.num_system_faces)
        if self.block_size <= 0 or self.num_system_faces <= 0:
            raise ValueError("block_size and num_system_faces must be positive")

        self._cp = cp
        self._batch_matrices = matrices
        if self.local_size != self.num_local_faces * self.block_size:
            raise ValueError("local matrix size must equal num_local_faces * block_size")
        minimum_face = int(cp.min(self.element_system_faces).item())
        maximum_face = int(cp.max(self.element_system_faces).item())
        if minimum_face < -1:
            raise ValueError("element_system_faces may contain only row ids or -1")
        if maximum_face >= self.num_system_faces:
            raise ValueError("element_system_faces contains an out-of-range row id")

        if self.application == "fused":
            if self.face_element_slots is None:
                host_slots = build_face_additive_schwarz_incidence_slots(
                    cp.asnumpy(self.element_system_faces),
                    self.num_system_faces,
                )
                self.face_element_slots = cp.asarray(host_slots)
            if not isinstance(self.face_element_slots, cp.ndarray):
                raise TypeError("face_element_slots must be a CuPy array")
            if self.face_element_slots.shape != (self.num_system_faces, 2):
                raise ValueError(
                    "face_element_slots must have shape (num_system_faces, 2)"
                )
            if self.face_element_slots.dtype != cp.int32:
                raise TypeError("face_element_slots must use int32")
            if not self.face_element_slots.flags.c_contiguous:
                raise ValueError("face_element_slots must be C-contiguous")
            if int(self.face_element_slots.device.id) != self.device_id:
                raise ValueError("face_element_slots is on the wrong CUDA device")

        self._element_rhs = (
            None
            if self.application == "fused"
            else cp.empty((self.num_elements, self.local_size), dtype=self.dtype)
        )
        self._local_output = cp.empty((self.num_elements, self.local_size), dtype=self.dtype)

        suffix = "f32" if self.dtype == cp.float32 else "f64"
        self._restrict_kernel = cp.RawKernel(
            _ASM_KERNEL_SOURCE,
            f"restrict_element_faces_{suffix}",
        )
        self._prolong_kernel = cp.RawKernel(
            _ASM_KERNEL_SOURCE,
            f"prolong_element_faces_{suffix}",
        )
        self._raw_apply_kernel = (
            _build_raw_batched_mv_kernel(cp, self.dtype)
            if use_inverse and self.application == "raw"
            else None
        )
        self._fused_local_kernel = (
            cp.RawKernel(_FUSED_ASM_KERNEL_SOURCE, f"fused_asm_local_{suffix}")
            if use_inverse and self.application == "fused"
            else None
        )
        self._race_free_prolong_kernel = (
            cp.RawKernel(
                _FUSED_ASM_KERNEL_SOURCE,
                f"race_free_asm_prolong_{suffix}",
            )
            if self.application == "fused"
            else None
        )
        self._threads_per_block = 256
        self._kernel_blocks = (
            self.num_local_dofs + self._threads_per_block - 1
        ) // self._threads_per_block
        self._global_kernel_blocks = (
            self.num_dofs + self._threads_per_block - 1
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
        local_solver: str = "cpu_inverse",
        application: str = "matmul",
        inverse_residual_tolerance: float | None = None,
    ) -> "CuPyFaceAdditiveSchwarzPreconditioner":
        """Construct the object from the supplied operator or system."""
        cp = require_cupy_device()
        if local_solver not in {
            "cpu_inverse",
            "gpu_inverse",
            "cublas_inverse",
            "gpu_solve",
        }:
            raise ValueError(f"unsupported ASM local_solver: {local_solver}")
        selected_device = (
            int(cp.cuda.Device().id) if device_id is None else int(device_id)
        )

        if local_solver == "cpu_inverse":
            layout = prepare_face_additive_schwarz_batch_layout(
                system,
                element_blocks,
                loc2glob_face,
                dtype=dtype,
                inverse_residual_tolerance=inverse_residual_tolerance,
            )
            face_slots = (
                build_face_additive_schwarz_incidence_slots(
                    layout.element_system_faces,
                    layout.num_system_faces,
                )
                if application == "fused"
                else None
            )
            with cp.cuda.Device(selected_device):
                return cls(
                    inverse_matrices=cp.asarray(layout.inverse_matrices),
                    element_system_faces=cp.asarray(layout.element_system_faces),
                    block_size=layout.block_size,
                    num_system_faces=layout.num_system_faces,
                    device_id=selected_device,
                    local_solver=local_solver,
                    inverse_residuals=cp.asarray(layout.inverse_residuals),
                    application=application,
                    face_element_slots=(
                        None if face_slots is None else cp.asarray(face_slots)
                    ),
                )

        layout = prepare_face_additive_schwarz_matrix_layout(
            system,
            element_blocks,
            loc2glob_face,
            dtype=dtype,
        )
        face_slots = (
            build_face_additive_schwarz_incidence_slots(
                layout.element_system_faces,
                layout.num_system_faces,
            )
            if application == "fused"
            else None
        )
        with cp.cuda.Device(selected_device):
            local_matrices = cp.asarray(layout.local_matrices)
            faces = cp.asarray(layout.element_system_faces)
            device_face_slots = None if face_slots is None else cp.asarray(face_slots)
            if local_solver == "gpu_solve":
                if inverse_residual_tolerance is not None:
                    raise ValueError(
                        "inverse_residual_tolerance is not applicable to gpu_solve"
                    )
                return cls(
                    inverse_matrices=None,
                    local_matrices=local_matrices,
                    element_system_faces=faces,
                    block_size=layout.block_size,
                    num_system_faces=layout.num_system_faces,
                    device_id=selected_device,
                    local_solver=local_solver,
                    application=application,
                    face_element_slots=device_face_slots,
                )

            factorization_info = None
            inversion_info = None
            if local_solver == "cublas_inverse":
                cublas_result = invert_batched_cublas(
                    local_matrices,
                    label="additive-Schwarz local matrices",
                )
                inverse_matrices = cublas_result.inverse_matrices
                residuals = cublas_result.inverse_residuals
                factorization_info = cublas_result.factorization_info
                inversion_info = cublas_result.inversion_info
            else:
                inverse_matrices, residuals = _gpu_batched_inverse(
                    cp,
                    local_matrices,
                    label="additive-Schwarz local matrices",
                )
            _check_gpu_inverse_tolerance(
                cp,
                residuals,
                inverse_residual_tolerance,
                label="additive-Schwarz",
            )
            return cls(
                inverse_matrices=inverse_matrices,
                local_matrices=local_matrices,
                element_system_faces=faces,
                block_size=layout.block_size,
                num_system_faces=layout.num_system_faces,
                device_id=selected_device,
                local_solver=local_solver,
                inverse_residuals=residuals,
                factorization_info=factorization_info,
                inversion_info=inversion_info,
                application=application,
                face_element_slots=device_face_slots,
            )

    @property
    def num_elements(self) -> int:
        """Return the number of elements."""
        return int(self._batch_matrices.shape[0])

    @property
    def num_local_faces(self) -> int:
        """Return the number of local faces per element."""
        return int(self.element_system_faces.shape[1])

    @property
    def local_size(self) -> int:
        """Return the element-patch vector size."""
        return int(self._batch_matrices.shape[1])

    @property
    def num_local_dofs(self) -> int:
        """Execute the ``num_local_dofs`` numerical helper."""
        return self.num_elements * self.local_size

    @property
    def num_dofs(self) -> int:
        """Return the number of scalar degrees of freedom."""
        return self.num_system_faces * self.block_size

    @property
    def dtype(self) -> Any:
        """Return the scalar dtype."""
        return self._batch_matrices.dtype

    @property
    def allocates_during_apply(self) -> bool:
        """Return whether an application allocates device storage."""
        return self.local_solver == "gpu_solve"

    @property
    def restricted_buffer(self) -> Any | None:
        """Restrict a global face vector to element-patch storage."""
        if self._element_rhs is None:
            return None
        return self._element_rhs.reshape(
            self.num_elements,
            self.num_local_faces,
            self.block_size,
        )

    @property
    def local_solution_buffer(self) -> Any:
        """Return the persistent local-solution buffer."""
        return self._local_output.reshape(
            self.num_elements,
            self.num_local_faces,
            self.block_size,
        )

    @property
    def restricted_workspace_bytes(self) -> int:
        """Return restricted-vector workspace storage in bytes."""
        return 0 if self._element_rhs is None else int(self._element_rhs.nbytes)

    @property
    def local_workspace_bytes(self) -> int:
        """Return local-vector workspace storage in bytes."""
        return int(self._local_output.nbytes)

    @property
    def workspace_bytes(self) -> int:
        """Return total persistent workspace storage in bytes."""
        return self.restricted_workspace_bytes + self.local_workspace_bytes

    @property
    def uses_race_free_prolongation(self) -> bool:
        """Return whether prolongation avoids atomic updates."""
        return self.application == "fused"

    @property
    def maximum_inverse_residual(self) -> float | None:
        """Return the largest local inverse residual."""
        if self.inverse_residuals is None:
            return None
        return float(self._cp.max(self.inverse_residuals).item())

    def _validate_global_vector(self, vector: Any, *, name: str) -> Any:
        """Validate shapes, dtypes, devices, and solver parameters."""
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
        """Validate shapes, dtypes, devices, and solver parameters."""
        cp = self._cp
        if not isinstance(vector, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        if int(vector.device.id) != self.device_id:
            raise ValueError(f"{name} is on the wrong CUDA device")
        if vector.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}")
        if vector.size != self.num_local_dofs:
            raise ValueError(f"{name} must contain {self.num_local_dofs} local values")
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
        """Launch the corresponding preallocated CUDA kernel."""
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
        """Launch the corresponding preallocated CUDA kernel."""
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

    def _launch_fused_local(self, x_flat: Any, local_flat: Any) -> None:
        """Launch the corresponding preallocated CUDA kernel."""
        if self._fused_local_kernel is None:
            raise RuntimeError("fused ASM local kernel is unavailable")
        self._fused_local_kernel(
            (self._kernel_blocks,),
            (self._threads_per_block,),
            (
                np.uint64(self.num_local_dofs),
                np.int32(self.num_local_faces),
                np.int32(self.block_size),
                np.int32(self.local_size),
                self.element_system_faces,
                self.inverse_matrices,
                x_flat,
                local_flat,
            ),
        )

    def _launch_race_free_prolong(self, element_flat: Any, out_flat: Any) -> None:
        """Launch the corresponding preallocated CUDA kernel."""
        if self._race_free_prolong_kernel is None or self.face_element_slots is None:
            raise RuntimeError("race-free ASM prolongation kernel is unavailable")
        self._race_free_prolong_kernel(
            (self._global_kernel_blocks,),
            (self._threads_per_block,),
            (
                np.uint64(self.num_dofs),
                np.int32(self.num_local_faces),
                np.int32(self.block_size),
                np.int32(self.local_size),
                self.face_element_slots,
                element_flat,
                out_flat,
            ),
        )

    def fused_local_into(self, x: Any, out: Any) -> None:
        """Execute the ``fused_local_into`` numerical helper."""
        if self.application != "fused":
            raise RuntimeError("fused_local_into requires application='fused'")
        x_flat = self._validate_global_vector(x, name="x")
        out_flat = self._validate_element_vector(out, name="out")
        if device_arrays_overlap(x, out):
            raise ValueError("x and out must not alias")
        self._launch_fused_local(x_flat, out_flat)

    def restrict_into(self, x: Any, out: Any) -> None:
        """Restrict a global face vector to element-patch storage."""
        cp = self._cp
        x_flat = self._validate_global_vector(x, name="x")
        out_flat = self._validate_element_vector(out, name="out")
        if device_arrays_overlap(x, out):
            raise ValueError("x and out must not alias")
        self._launch_restrict(x_flat, out_flat)

    def restrict(self, x: Any) -> Any:
        """Restrict a global face vector to element-patch storage."""
        out = self._cp.empty(
            (self.num_elements, self.num_local_faces, self.block_size),
            dtype=self.dtype,
        )
        self.restrict_into(x, out)
        return out

    def prolong_into(self, element_vector: Any, out: Any) -> None:
        """Prolong element-patch values to the global face vector."""
        element_flat = self._validate_element_vector(element_vector, name="element_vector")
        out_flat = self._validate_global_vector(out, name="out")
        if device_arrays_overlap(element_vector, out):
            raise ValueError("element_vector and out must not alias")
        if self.application == "fused":
            self._launch_race_free_prolong(element_flat, out_flat)
        else:
            out_flat.fill(0)
            self._launch_prolong(element_flat, out_flat)

    def prolong(self, element_vector: Any, *, flat: bool = False) -> Any:
        """Prolong element-patch values to the global face vector."""
        shape = (self.num_dofs,) if flat else (
            self.num_system_faces,
            self.block_size,
        )
        out = self._cp.empty(shape, dtype=self.dtype)
        self.prolong_into(element_vector, out)
        return out

    def apply_into(self, x: Any, out: Any) -> None:
        """Apply the preconditioner using reusable storage."""
        cp = self._cp
        x_flat = self._validate_global_vector(x, name="x")
        out_flat = self._validate_global_vector(out, name="out")
        if device_arrays_overlap(x, out):
            raise ValueError("x and out must not alias")

        if self.application == "fused":
            self._launch_fused_local(x_flat, self._local_output.reshape(-1))
            self._launch_race_free_prolong(
                self._local_output.reshape(-1),
                out_flat,
            )
            return

        assert self._element_rhs is not None
        self._launch_restrict(x_flat, self._element_rhs.reshape(-1))
        if self.local_solver == "gpu_solve":
            solved = solve_batched_vectors(
                cp,
                self.local_matrices,
                self._element_rhs,
            )
            self._local_output[...] = solved
        elif self.application == "raw":
            assert self._raw_apply_kernel is not None
            _launch_raw_batched_mv(
                self._raw_apply_kernel,
                self.inverse_matrices,
                self._element_rhs,
                self._local_output,
                matrix_size=self.local_size,
            )
        else:
            cp.matmul(
                self.inverse_matrices,
                self._element_rhs.reshape(self.num_elements, self.local_size, 1),
                out=self._local_output.reshape(self.num_elements, self.local_size, 1),
            )
        out_flat.fill(0)
        self._launch_prolong(self._local_output.reshape(-1), out_flat)

    def apply(self, x: Any) -> Any:
        """Apply the preconditioner using reusable storage."""
        out = self._cp.empty_like(x)
        self.apply_into(x, out)
        return out


__all__ = [
    "AdditiveSchwarzBatchLayout",
    "AdditiveSchwarzMatrixLayout",
    "CuPyFaceAdditiveSchwarzPreconditioner",
    "CuPyFaceBlockJacobiPreconditioner",
    "build_face_additive_schwarz_incidence_slots",
    "prepare_face_additive_schwarz_batch_layout",
    "prepare_face_additive_schwarz_matrix_layout",
]