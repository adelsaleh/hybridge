"""CUDA preconditioners for face-dense HDG systems.

Three local-solver paths are available for validation and later profiling:
validated CPU inverses transferred to the device, batched GPU inversion during
setup followed by dense matvec application, and direct batched GPU solves at
every application.  The inverse paths are the production candidates; the direct
solve path intentionally exposes the cost of repeated public-API factorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..assembly.face_dense import FaceDenseSystem
from ..linalg.additive_schwarz import (
    build_face_additive_schwarz_local_matrices,
    build_face_additive_schwarz_preconditioner,
)
from ..linalg.block_jacobi import build_face_block_jacobi_preconditioner
from .cupy import require_cupy_device

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

    def __post_init__(self) -> None:
        cp = require_cupy_device()
        valid_modes = {"external_inverse", "cpu_inverse", "gpu_inverse", "gpu_solve"}
        if self.local_solver not in valid_modes:
            raise ValueError(f"unsupported block-Jacobi local_solver: {self.local_solver}")
        use_inverse = self.local_solver != "gpu_solve"
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
            if use_inverse
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
        inverse_residual_tolerance: float | None = None,
    ) -> "CuPyFaceBlockJacobiPreconditioner":
        cp = require_cupy_device()
        if not isinstance(system, FaceDenseSystem):
            raise TypeError("system must be a FaceDenseSystem")
        if local_solver not in {"cpu_inverse", "gpu_inverse", "gpu_solve"}:
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
                )
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
            )

    @property
    def num_faces(self) -> int:
        return int(self._batch_matrices.shape[0])

    @property
    def block_size(self) -> int:
        return int(self._batch_matrices.shape[1])

    @property
    def num_dofs(self) -> int:
        return self.num_faces * self.block_size

    @property
    def dtype(self) -> Any:
        return self._batch_matrices.dtype

    @property
    def allocates_during_apply(self) -> bool:
        return self.local_solver == "gpu_solve"

    @property
    def maximum_inverse_residual(self) -> float | None:
        if self.inverse_residuals is None:
            return None
        return float(self._cp.max(self.inverse_residuals).item())

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
        if vector.ndim == 2 and vector.shape != (self.num_faces, self.block_size):
            raise ValueError(f"{name} has an incompatible face-major shape")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector.reshape(self.num_faces, self.block_size)

    def apply_into(self, x: Any, out: Any) -> None:
        cp = self._cp
        x_faces = self._validate_vector(x, name="x")
        out_faces = self._validate_vector(out, name="out")
        if cp.shares_memory(x, out):
            raise ValueError("x and out must not alias")

        if self.local_solver == "gpu_solve":
            # Public CuPy batched solve currently has no ``out`` parameter and
            # does not expose reusable LU factors.  Keep this path as a
            # correctness/performance comparison, not the production default.
            solved = cp.linalg.solve(self.local_blocks, x_faces)
            self._output[...] = solved
            out_faces[...] = self._output
            return

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
    inverse_residuals: np.ndarray
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

@dataclass(frozen=True)
class AdditiveSchwarzMatrixLayout:
    """Host-side one-element ASM matrices before accelerator inversion."""

    local_matrices: np.ndarray
    element_system_faces: np.ndarray
    block_size: int
    num_system_faces: int

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
    def num_dofs(self) -> int:
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

@dataclass
class CuPyFaceAdditiveSchwarzPreconditioner:
    r"""Device one-element ASM with selectable local dense solver.

    ``element_rhs = R @ x``
        CUDA gather kernel;
    ``element_solution = inverse_matrices @ element_rhs``
        batched dense matrix--vector products through ``cupy.matmul``;
    ``out = sum_e R_e.T @ element_solution_e``
        CUDA scatter-add kernel using atomics on shared faces.

    ``cpu_inverse`` transfers validated NumPy inverses. ``gpu_inverse`` builds
    the inverse batch on the GPU during setup and then uses allocation-free
    batched matmul during GMRES. ``gpu_solve`` calls the public batched
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

    def __post_init__(self) -> None:
        cp = require_cupy_device()
        valid_modes = {"external_inverse", "cpu_inverse", "gpu_inverse", "gpu_solve"}
        if self.local_solver not in valid_modes:
            raise ValueError(f"unsupported ASM local_solver: {self.local_solver}")
        use_inverse = self.local_solver != "gpu_solve"
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

        self._element_rhs = cp.empty((self.num_elements, self.local_size), dtype=self.dtype)
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
        local_solver: str = "cpu_inverse",
        inverse_residual_tolerance: float | None = None,
    ) -> "CuPyFaceAdditiveSchwarzPreconditioner":
        cp = require_cupy_device()
        if local_solver not in {"cpu_inverse", "gpu_inverse", "gpu_solve"}:
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
            with cp.cuda.Device(selected_device):
                return cls(
                    inverse_matrices=cp.asarray(layout.inverse_matrices),
                    element_system_faces=cp.asarray(layout.element_system_faces),
                    block_size=layout.block_size,
                    num_system_faces=layout.num_system_faces,
                    device_id=selected_device,
                    local_solver=local_solver,
                    inverse_residuals=cp.asarray(layout.inverse_residuals),
                )

        layout = prepare_face_additive_schwarz_matrix_layout(
            system,
            element_blocks,
            loc2glob_face,
            dtype=dtype,
        )
        with cp.cuda.Device(selected_device):
            local_matrices = cp.asarray(layout.local_matrices)
            faces = cp.asarray(layout.element_system_faces)
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
                )

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
            )

    @property
    def num_elements(self) -> int:
        return int(self._batch_matrices.shape[0])

    @property
    def num_local_faces(self) -> int:
        return int(self.element_system_faces.shape[1])

    @property
    def local_size(self) -> int:
        return int(self._batch_matrices.shape[1])

    @property
    def num_local_dofs(self) -> int:
        return self.num_elements * self.local_size

    @property
    def num_dofs(self) -> int:
        return self.num_system_faces * self.block_size

    @property
    def dtype(self) -> Any:
        return self._batch_matrices.dtype

    @property
    def allocates_during_apply(self) -> bool:
        return self.local_solver == "gpu_solve"

    @property
    def restricted_buffer(self) -> Any:
        return self._element_rhs.reshape(
            self.num_elements,
            self.num_local_faces,
            self.block_size,
        )

    @property
    def local_solution_buffer(self) -> Any:
        return self._local_output.reshape(
            self.num_elements,
            self.num_local_faces,
            self.block_size,
        )

    @property
    def maximum_inverse_residual(self) -> float | None:
        if self.inverse_residuals is None:
            return None
        return float(self._cp.max(self.inverse_residuals).item())

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
        cp = self._cp
        x_flat = self._validate_global_vector(x, name="x")
        out_flat = self._validate_element_vector(out, name="out")
        if cp.shares_memory(x, out):
            raise ValueError("x and out must not alias")
        self._launch_restrict(x_flat, out_flat)

    def restrict(self, x: Any) -> Any:
        out = self._cp.empty(
            (self.num_elements, self.num_local_faces, self.block_size),
            dtype=self.dtype,
        )
        self.restrict_into(x, out)
        return out

    def prolong_into(self, element_vector: Any, out: Any) -> None:
        cp = self._cp
        element_flat = self._validate_element_vector(element_vector, name="element_vector")
        out_flat = self._validate_global_vector(out, name="out")
        if cp.shares_memory(element_vector, out):
            raise ValueError("element_vector and out must not alias")
        out_flat.fill(0)
        self._launch_prolong(element_flat, out_flat)

    def prolong(self, element_vector: Any, *, flat: bool = False) -> Any:
        shape = (self.num_dofs,) if flat else (
            self.num_system_faces,
            self.block_size,
        )
        out = self._cp.empty(shape, dtype=self.dtype)
        self.prolong_into(element_vector, out)
        return out

    def apply_into(self, x: Any, out: Any) -> None:
        cp = self._cp
        x_flat = self._validate_global_vector(x, name="x")
        out_flat = self._validate_global_vector(out, name="out")
        if cp.shares_memory(x, out):
            raise ValueError("x and out must not alias")

        self._launch_restrict(x_flat, self._element_rhs.reshape(-1))
        if self.local_solver == "gpu_solve":
            solved = cp.linalg.solve(self.local_matrices, self._element_rhs)
            self._local_output[...] = solved
        else:
            cp.matmul(
                self.inverse_matrices,
                self._element_rhs.reshape(self.num_elements, self.local_size, 1),
                out=self._local_output.reshape(self.num_elements, self.local_size, 1),
            )
        out_flat.fill(0)
        self._launch_prolong(self._local_output.reshape(-1), out_flat)

    def apply(self, x: Any) -> Any:
        out = self._cp.empty_like(x)
        self.apply_into(x, out)
        return out


__all__ = [
    "AdditiveSchwarzBatchLayout",
    "AdditiveSchwarzMatrixLayout",
    "CuPyFaceAdditiveSchwarzPreconditioner",
    "CuPyFaceBlockJacobiPreconditioner",
    "prepare_face_additive_schwarz_batch_layout",
    "prepare_face_additive_schwarz_matrix_layout",
]
