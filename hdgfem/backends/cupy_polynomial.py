"""GPU harmonic-Ritz polynomial preconditioner.

The spectral setup runs one matrix-free Arnoldi cycle on the GPU and transfers
only the small Hessenberg data to the CPU.  Harmonic Ritz extraction and
conjugate-preserving Leja ordering are CPU operations.  Application then stays
entirely on one CUDA device and reuses the validated face-dense matvec plus an
optional block-Jacobi or additive-Schwarz action.

For a base preconditioner ``M^{-1}``, the implemented action is

    p(M^{-1} A) M^{-1} y.

Two fused CUDA update kernels evaluate real roots and complex-conjugate pairs
without complex device vectors or per-application allocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..linalg.polynomial import (
    RootBlock,
    conjugate_root_blocks,
    harmonic_ritz_values,
    leja_order_conjugate_preserving,
)
from .cupy import device_arrays_overlap, require_cupy_device
from .cupy_gmres import (
    CuPyOrthogonalization,
    CuPyVectorBLAS,
    DeviceMatvecOperator,
    DevicePreconditioner,
)


_SETUP_ORTHOGONALIZATIONS = {"mgs", "mgs2", "cgs", "cgs2"}


@dataclass(frozen=True)
class CuPyPolynomialSetup:
    """Small host-side output of the GPU Arnoldi spectral setup."""

    hessenberg: np.ndarray
    harmonic_ritz_values: np.ndarray
    ordered_roots: np.ndarray
    requested_degree: int
    effective_degree: int
    seed: int
    breakdown: bool
    orthogonalization: CuPyOrthogonalization
    frobenius_orthogonality_defect: float
    maximum_offdiagonal: float
    maximum_diagonal_error: float
    matvec_count: int
    base_preconditioner_count: int

    @property
    def degree(self) -> int:
        return int(self.ordered_roots.size)


_POLYNOMIAL_UPDATE_KERNEL_SOURCE = r"""
extern "C" __global__
void polynomial_real_update_f32(
    const unsigned long long total,
    const float inverse_theta,
    const float* __restrict__ bq,
    float* __restrict__ q,
    float* __restrict__ result)
{
    const unsigned long long i =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    const float qi = q[i];
    result[i] += inverse_theta * qi;
    q[i] = qi - inverse_theta * bq[i];
}

extern "C" __global__
void polynomial_real_update_f64(
    const unsigned long long total,
    const double inverse_theta,
    const double* __restrict__ bq,
    double* __restrict__ q,
    double* __restrict__ result)
{
    const unsigned long long i =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    const double qi = q[i];
    result[i] += inverse_theta * qi;
    q[i] = qi - inverse_theta * bq[i];
}

extern "C" __global__
void polynomial_pair_update_f32(
    const unsigned long long total,
    const float twice_a_over_denominator,
    const float inverse_denominator,
    const float* __restrict__ bq,
    const float* __restrict__ b2q,
    float* __restrict__ q,
    float* __restrict__ result)
{
    const unsigned long long i =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    const float qi = q[i];
    const float bqi = bq[i];
    result[i] += twice_a_over_denominator * qi
               - inverse_denominator * bqi;
    q[i] = qi - twice_a_over_denominator * bqi
              + inverse_denominator * b2q[i];
}

extern "C" __global__
void polynomial_pair_update_f64(
    const unsigned long long total,
    const double twice_a_over_denominator,
    const double inverse_denominator,
    const double* __restrict__ bq,
    const double* __restrict__ b2q,
    double* __restrict__ q,
    double* __restrict__ result)
{
    const unsigned long long i =
        (unsigned long long) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    const double qi = q[i];
    const double bqi = bq[i];
    result[i] += twice_a_over_denominator * qi
               - inverse_denominator * bqi;
    q[i] = qi - twice_a_over_denominator * bqi
              + inverse_denominator * b2q[i];
}
"""


def _orthogonality_metrics(gram: np.ndarray) -> tuple[float, float, float]:
    gram = np.asarray(gram, dtype=np.float64)
    dimension = gram.shape[0]
    defect = gram - np.eye(dimension, dtype=np.float64)
    frobenius = float(np.linalg.norm(defect, ord="fro"))
    diagonal = float(np.max(np.abs(np.diag(defect)), initial=0.0))
    offdiagonal = defect.copy()
    if dimension:
        offdiagonal[np.diag_indices(dimension)] = 0.0
    maximum_offdiagonal = float(np.max(np.abs(offdiagonal), initial=0.0))
    return frobenius, maximum_offdiagonal, diagonal


def _validate_protocol_compatibility(
    operator: DeviceMatvecOperator,
    base_preconditioner: DevicePreconditioner | None,
) -> tuple[Any, int, int]:
    cp = require_cupy_device()
    if not isinstance(operator, DeviceMatvecOperator):
        raise TypeError("operator must implement the device matvec protocol")
    num_dofs = int(operator.num_dofs)
    device_id = int(operator.device_id)
    dtype = cp.dtype(operator.dtype)
    if num_dofs <= 0:
        raise ValueError("operator.num_dofs must be positive")
    if dtype not in (cp.float32, cp.float64):
        raise TypeError("polynomial preconditioning supports float32 or float64")
    if base_preconditioner is not None:
        if not isinstance(base_preconditioner, DevicePreconditioner):
            raise TypeError("base_preconditioner must implement apply_into or be None")
        if int(base_preconditioner.num_dofs) != num_dofs:
            raise ValueError("base preconditioner size does not match the operator")
        if int(base_preconditioner.device_id) != device_id:
            raise ValueError("base preconditioner and operator use different devices")
        if cp.dtype(base_preconditioner.dtype) != dtype:
            raise TypeError("base preconditioner dtype does not match the operator")
    return dtype, device_id, num_dofs


def setup_polynomial_preconditioner_cupy(
    operator: DeviceMatvecOperator,
    *,
    degree: int,
    base_preconditioner: DevicePreconditioner | None = None,
    seed: int = 1729,
    initial_vector: np.ndarray | None = None,
    orthogonalization: CuPyOrthogonalization = "cgs2",
    breakdown_tolerance: float | None = None,
    pair_tolerance: float = 1.0e-10,
    profiler: Any | None = None,
) -> CuPyPolynomialSetup:
    """Run one GPU Arnoldi cycle and construct harmonic-Ritz/Leja roots.

    ``cgs2`` is the default for setup even when the outer GMRES uses faster
    one-pass CGS: harmonic Ritz values are more sensitive to basis defects than
    the final restarted solve observed in the current Poisson benchmarks.
    """

    cp = require_cupy_device()
    dtype, device_id, num_dofs = _validate_protocol_compatibility(
        operator, base_preconditioner
    )
    if isinstance(degree, bool) or int(degree) != degree or degree <= 0:
        raise ValueError("degree must be a positive integer")
    degree = int(degree)
    if degree > num_dofs:
        raise ValueError("degree cannot exceed operator.num_dofs")
    orthogonalization = str(orthogonalization).lower()  # type: ignore[assignment]
    if orthogonalization not in _SETUP_ORTHOGONALIZATIONS:
        raise ValueError("unsupported setup orthogonalization")
    if pair_tolerance <= 0.0 or not np.isfinite(pair_tolerance):
        raise ValueError("pair_tolerance must be positive and finite")
    if breakdown_tolerance is None:
        breakdown_tolerance = 100.0 * np.finfo(np.dtype(dtype.name)).eps
    if breakdown_tolerance < 0.0 or not np.isfinite(breakdown_tolerance):
        raise ValueError("breakdown_tolerance must be finite and non-negative")

    host_dtype = np.dtype(dtype.name)
    if initial_vector is None:
        initial_host = np.random.default_rng(seed).standard_normal(num_dofs)
    else:
        initial_host = np.asarray(initial_vector)
    if initial_host.size != num_dofs or not np.all(np.isfinite(initial_host)):
        raise ValueError("initial_vector has an invalid size or non-finite values")
    initial_host = np.ascontiguousarray(initial_host.reshape(-1), dtype=host_dtype)

    with cp.cuda.Device(device_id):
        basis = cp.empty((degree + 1, num_dofs), dtype=dtype)
        operator_output = cp.empty(num_dofs, dtype=dtype)
        work = cp.empty(num_dofs, dtype=dtype)
        coefficients_device = cp.empty(degree, dtype=dtype)
        coefficients_host = np.empty(degree, dtype=host_dtype)
        blas = CuPyVectorBLAS(dtype=dtype, device_id=device_id, profiler=profiler)

        cp.copyto(basis[0], cp.asarray(initial_host))
        initial_norm = blas.norm(basis[0])
        if initial_norm == 0.0:
            raise ValueError("initial_vector must be nonzero")
        blas.scal(1.0 / initial_norm, basis[0])

        hessenberg = np.zeros((degree + 1, degree), dtype=np.float64)
        effective_degree = degree
        breakdown = False
        matvec_count = 0
        base_count = 0

        def effective_apply(source: Any, destination: Any) -> None:
            nonlocal matvec_count, base_count
            operator.matvec_into(source, operator_output)
            matvec_count += 1
            if base_preconditioner is None:
                cp.copyto(destination, operator_output)
            else:
                base_preconditioner.apply_into(operator_output, destination)
                base_count += 1

        for column in range(degree):
            effective_apply(basis[column], work)
            work_before = blas.norm(work)
            active_dimension = column + 1
            active_basis = basis[:active_dimension]

            if orthogonalization in ("mgs", "mgs2"):
                passes = 2 if orthogonalization == "mgs2" else 1
                for _ in range(passes):
                    for row in range(active_dimension):
                        coefficient = blas.dot(basis[row], work)
                        hessenberg[row, column] += coefficient
                        blas.axpy(-coefficient, basis[row], work)
            else:
                passes = 2 if orthogonalization == "cgs2" else 1
                device_slice = coefficients_device[:active_dimension]
                host_slice = coefficients_host[:active_dimension]
                for _ in range(passes):
                    blas.basis_projection(active_basis, work, device_slice)
                    blas.copy_device_vector_to_host(device_slice, host_slice)
                    hessenberg[:active_dimension, column] += host_slice
                    blas.basis_correction(active_basis, device_slice, work)

            next_norm = blas.norm(work)
            hessenberg[column + 1, column] = next_norm
            threshold = breakdown_tolerance * max(work_before, 1.0)
            if next_norm <= threshold:
                effective_degree = column + 1
                breakdown = True
                break
            cp.copyto(basis[column + 1], work)
            blas.scal(1.0 / next_norm, basis[column + 1])

        active_basis = basis[:effective_degree]
        gram_host = cp.asnumpy(active_basis @ active_basis.T)
        frobenius, maximum_offdiagonal, maximum_diagonal_error = (
            _orthogonality_metrics(gram_host)
        )
        used_hessenberg = np.ascontiguousarray(
            hessenberg[: effective_degree + 1, :effective_degree]
        )

    values = harmonic_ritz_values(used_hessenberg)
    roots = leja_order_conjugate_preserving(values, tolerance=pair_tolerance)
    root_scale = max(float(np.max(np.abs(roots))), 1.0)
    zero_tolerance = 100.0 * np.finfo(np.float64).eps * root_scale
    if np.any(np.abs(roots) <= zero_tolerance):
        raise np.linalg.LinAlgError("harmonic Ritz setup produced a zero root")

    return CuPyPolynomialSetup(
        hessenberg=used_hessenberg,
        harmonic_ritz_values=values,
        ordered_roots=roots,
        requested_degree=degree,
        effective_degree=effective_degree,
        seed=int(seed),
        breakdown=breakdown,
        orthogonalization=orthogonalization,  # type: ignore[arg-type]
        frobenius_orthogonality_defect=frobenius,
        maximum_offdiagonal=maximum_offdiagonal,
        maximum_diagonal_error=maximum_diagonal_error,
        matvec_count=matvec_count,
        base_preconditioner_count=base_count,
    )


class CuPyPolynomialPreconditioner:
    """Allocation-free device action ``p(M^{-1}A)M^{-1}``."""

    def __init__(
        self,
        operator: DeviceMatvecOperator,
        *,
        roots: np.ndarray,
        base_preconditioner: DevicePreconditioner | None = None,
        setup: CuPyPolynomialSetup | None = None,
        pair_tolerance: float = 1.0e-10,
    ) -> None:
        cp = require_cupy_device()
        dtype, device_id, num_dofs = _validate_protocol_compatibility(
            operator, base_preconditioner
        )
        roots_array = np.asarray(roots, dtype=np.complex128)
        if roots_array.ndim != 1 or roots_array.size == 0:
            raise ValueError("roots must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(roots_array)):
            raise ValueError("roots contain non-finite values")
        scale = max(float(np.max(np.abs(roots_array))), 1.0)
        if np.any(np.abs(roots_array) <= 100.0 * np.finfo(np.float64).eps * scale):
            raise np.linalg.LinAlgError("polynomial roots cannot be zero")

        self.operator = operator
        self.base_preconditioner = base_preconditioner
        self.roots = np.ascontiguousarray(roots_array)
        self.root_blocks: tuple[RootBlock, ...] = tuple(
            conjugate_root_blocks(self.roots, tolerance=pair_tolerance)
        )
        self.setup = setup
        self.num_dofs = num_dofs
        self.device_id = device_id
        self.dtype = dtype
        self.application_count = 0
        self.matvec_count = 0
        self.base_preconditioner_count = 0

        with cp.cuda.Device(device_id):
            self._q = cp.empty(num_dofs, dtype=dtype)
            self._bq = cp.empty(num_dofs, dtype=dtype)
            self._b2q = cp.empty(num_dofs, dtype=dtype)
            self._operator_output = cp.empty(num_dofs, dtype=dtype)
            module = cp.RawModule(
                code=_POLYNOMIAL_UPDATE_KERNEL_SOURCE,
                options=("--std=c++11",),
            )
            suffix = "f32" if dtype == cp.float32 else "f64"
            self._real_update = module.get_function(
                f"polynomial_real_update_{suffix}"
            )
            self._pair_update = module.get_function(
                f"polynomial_pair_update_{suffix}"
            )
        self._cp = cp

    @classmethod
    def from_operator(
        cls,
        operator: DeviceMatvecOperator,
        *,
        degree: int,
        base_preconditioner: DevicePreconditioner | None = None,
        seed: int = 1729,
        initial_vector: np.ndarray | None = None,
        setup_orthogonalization: CuPyOrthogonalization = "cgs2",
        breakdown_tolerance: float | None = None,
        pair_tolerance: float = 1.0e-10,
        profiler: Any | None = None,
    ) -> "CuPyPolynomialPreconditioner":
        setup = setup_polynomial_preconditioner_cupy(
            operator,
            degree=degree,
            base_preconditioner=base_preconditioner,
            seed=seed,
            initial_vector=initial_vector,
            orthogonalization=setup_orthogonalization,
            breakdown_tolerance=breakdown_tolerance,
            pair_tolerance=pair_tolerance,
            profiler=profiler,
        )
        return cls(
            operator,
            roots=setup.ordered_roots,
            base_preconditioner=base_preconditioner,
            setup=setup,
            pair_tolerance=pair_tolerance,
        )

    @property
    def degree(self) -> int:
        return int(self.roots.size)

    @property
    def matvecs_per_application(self) -> int:
        return self.degree

    @property
    def base_preconditioner_calls_per_application(self) -> int:
        return 0 if self.base_preconditioner is None else self.degree + 1

    @property
    def allocates_during_apply(self) -> bool:
        return False

    def _validate_vector(self, vector: Any, *, name: str) -> Any:
        cp = self._cp
        if not isinstance(vector, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        if int(vector.device.id) != self.device_id:
            raise ValueError(f"{name} is on the wrong CUDA device")
        if vector.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}")
        if vector.size != self.num_dofs or vector.ndim not in (1, 2):
            raise ValueError(f"{name} must contain {self.num_dofs} values")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector.reshape(-1)

    def _effective_apply(self, source: Any, destination: Any) -> None:
        if self.base_preconditioner is None:
            self.operator.matvec_into(source, destination)
        else:
            self.operator.matvec_into(source, self._operator_output)
            self.base_preconditioner.apply_into(self._operator_output, destination)
            self.base_preconditioner_count += 1
        self.matvec_count += 1

    def _launch_real_update(self, theta: float, result: Any) -> None:
        total = self.num_dofs
        threads = 256
        blocks = (total + threads - 1) // threads
        scalar = np.float32 if self.dtype == self._cp.float32 else np.float64
        self._real_update(
            (blocks,),
            (threads,),
            (
                np.uint64(total),
                scalar(1.0 / theta),
                self._bq,
                self._q,
                result,
            ),
        )

    def _launch_pair_update(self, a: float, b: float, result: Any) -> None:
        total = self.num_dofs
        threads = 256
        blocks = (total + threads - 1) // threads
        denominator = a * a + b * b
        scalar = np.float32 if self.dtype == self._cp.float32 else np.float64
        self._pair_update(
            (blocks,),
            (threads,),
            (
                np.uint64(total),
                scalar(2.0 * a / denominator),
                scalar(1.0 / denominator),
                self._bq,
                self._b2q,
                self._q,
                result,
            ),
        )

    def apply_into(self, x: Any, out: Any) -> None:
        """Apply the polynomial with fixed, reusable device workspaces."""

        x_flat = self._validate_vector(x, name="x")
        out_flat = self._validate_vector(out, name="out")
        if device_arrays_overlap(x_flat, out_flat):
            raise ValueError("x and out must not overlap")

        if self.base_preconditioner is None:
            self._cp.copyto(self._q, x_flat)
        else:
            self.base_preconditioner.apply_into(x_flat, self._q)
            self.base_preconditioner_count += 1
        out_flat.fill(0)

        for block in self.root_blocks:
            self._effective_apply(self._q, self._bq)
            if block.is_real:
                self._launch_real_update(float(block.first.real), out_flat)
                continue
            a = float(block.first.real)
            b = float(abs(block.first.imag))
            self._effective_apply(self._bq, self._b2q)
            self._launch_pair_update(a, b, out_flat)

        self.application_count += 1

    def apply(self, x: Any) -> Any:
        x_flat = self._validate_vector(x, name="x")
        out = self._cp.empty_like(x_flat)
        self.apply_into(x_flat, out)
        return out.reshape(x.shape)


__all__ = [
    "CuPyPolynomialPreconditioner",
    "CuPyPolynomialSetup",
    "setup_polynomial_preconditioner_cupy",
]
