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

from hybridge.linalg.polynomial import (
    RootBlock,
    conjugate_root_blocks,
    harmonic_ritz_values,
    leja_order_conjugate_preserving,
)
from hybridge.runtime.optional import device_arrays_overlap, require_cupy_device
from hybridge.linalg.gpu.gmres import (
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
        """Return the polynomial degree."""
        return int(self.ordered_roots.size)


@dataclass(frozen=True)
class CuPyPolynomialArnoldiProbe:
    """Reusable Arnoldi data for a family of polynomial degrees.

    A single probe of dimension ``maximum_degree`` contains every leading
    Arnoldi relation needed to construct degrees ``1, ..., maximum_degree``.
    Only the small Hessenberg and Gram matrices are retained on the host; the
    large Krylov basis is released after setup.
    """

    hessenberg: np.ndarray
    gram_matrix: np.ndarray
    requested_degree: int
    effective_degree: int
    seed: int
    breakdown: bool
    orthogonalization: CuPyOrthogonalization
    matvec_count: int
    base_preconditioner_count: int
    num_dofs: int
    device_id: int
    dtype_name: str
    uses_base_preconditioner: bool
    operator_identity: int
    base_preconditioner_identity: int | None

    @property
    def maximum_degree(self) -> int:
        """Return the largest supported polynomial degree."""
        return int(self.effective_degree)


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


_POLYNOMIAL_KERNEL_CACHE: dict[tuple[int, str], tuple[Any, Any, Any]] = {}


def _get_polynomial_update_kernels(cp: Any, *, dtype: Any, device_id: int):
    """Return cached RawModule kernels for one device and floating dtype."""

    key = (int(device_id), np.dtype(dtype.name).str)
    cached = _POLYNOMIAL_KERNEL_CACHE.get(key)
    if cached is not None:
        _, real_update, pair_update = cached
        return real_update, pair_update
    with cp.cuda.Device(device_id):
        module = cp.RawModule(
            code=_POLYNOMIAL_UPDATE_KERNEL_SOURCE,
            options=("--std=c++11",),
        )
        suffix = "f32" if dtype == cp.float32 else "f64"
        real_update = module.get_function(f"polynomial_real_update_{suffix}")
        pair_update = module.get_function(f"polynomial_pair_update_{suffix}")
    _POLYNOMIAL_KERNEL_CACHE[key] = (module, real_update, pair_update)
    return real_update, pair_update


def initialize_polynomial_kernels_cupy(
    *, dtype: Any = np.float64, device_id: int | None = None
) -> None:
    """Compile and launch the polynomial update kernels once.

    Benchmark scripts use this helper to separate one-time CUDA JIT cost from
    numerical Arnoldi setup.  Normal users do not need to call it because the
    preconditioner constructor initializes the same cache lazily.
    """

    cp = require_cupy_device()
    resolved_device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
    resolved_dtype = cp.dtype(dtype)
    if resolved_dtype not in (cp.float32, cp.float64):
        raise TypeError("polynomial kernels support float32 or float64")
    real_update, pair_update = _get_polynomial_update_kernels(
        cp, dtype=resolved_dtype, device_id=resolved_device
    )
    scalar = np.float32 if resolved_dtype == cp.float32 else np.float64
    with cp.cuda.Device(resolved_device):
        q = cp.ones(1, dtype=resolved_dtype)
        bq = cp.zeros(1, dtype=resolved_dtype)
        b2q = cp.zeros(1, dtype=resolved_dtype)
        result = cp.zeros(1, dtype=resolved_dtype)
        real_update(
            (1,),
            (1,),
            (np.uint64(1), scalar(1.0), bq, q, result),
        )
        pair_update(
            (1,),
            (1,),
            (np.uint64(1), scalar(1.0), scalar(1.0), bq, b2q, q, result),
        )
        cp.cuda.get_current_stream().synchronize()


def _orthogonality_metrics(gram: np.ndarray) -> tuple[float, float, float]:
    """Compute Arnoldi-basis orthogonality diagnostics."""
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
    """Validate shapes, dtypes, devices, and solver parameters."""
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


def setup_polynomial_arnoldi_probe_cupy(
    operator: DeviceMatvecOperator,
    *,
    maximum_degree: int,
    base_preconditioner: DevicePreconditioner | None = None,
    seed: int = 1729,
    initial_vector: np.ndarray | None = None,
    orthogonalization: CuPyOrthogonalization = "cgs2",
    breakdown_tolerance: float | None = None,
    profiler: Any | None = None,
) -> CuPyPolynomialArnoldiProbe:
    """Run one reusable GPU Arnoldi probe up to ``maximum_degree``.

    For a fixed initial vector, the first ``d`` columns are identical to an
    independent degree-``d`` Arnoldi setup.  A degree sweep can therefore run
    the expensive matrix-free probe once and derive every candidate from a
    leading Hessenberg submatrix on the CPU.
    """

    cp = require_cupy_device()
    dtype, device_id, num_dofs = _validate_protocol_compatibility(
        operator, base_preconditioner
    )
    if (
        isinstance(maximum_degree, bool)
        or int(maximum_degree) != maximum_degree
        or maximum_degree <= 0
    ):
        raise ValueError("maximum_degree must be a positive integer")
    maximum_degree = int(maximum_degree)
    if maximum_degree > num_dofs:
        raise ValueError("maximum_degree cannot exceed operator.num_dofs")
    orthogonalization = str(orthogonalization).lower()  # type: ignore[assignment]
    if orthogonalization not in _SETUP_ORTHOGONALIZATIONS:
        raise ValueError("unsupported setup orthogonalization")
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
        basis = cp.empty((maximum_degree + 1, num_dofs), dtype=dtype)
        operator_output = cp.empty(num_dofs, dtype=dtype)
        work = cp.empty(num_dofs, dtype=dtype)
        coefficients_device = cp.empty(maximum_degree, dtype=dtype)
        coefficients_host = np.empty(maximum_degree, dtype=host_dtype)
        blas = CuPyVectorBLAS(dtype=dtype, device_id=device_id, profiler=profiler)

        cp.copyto(basis[0], cp.asarray(initial_host))
        initial_norm = blas.norm(basis[0])
        if initial_norm == 0.0:
            raise ValueError("initial_vector must be nonzero")
        blas.scal(1.0 / initial_norm, basis[0])

        hessenberg = np.zeros(
            (maximum_degree + 1, maximum_degree), dtype=np.float64
        )
        effective_degree = maximum_degree
        breakdown = False
        matvec_count = 0
        base_count = 0

        def effective_apply(source: Any, destination: Any) -> None:
            """Apply the preconditioner using reusable storage."""
            nonlocal matvec_count, base_count
            operator.matvec_into(source, operator_output)
            matvec_count += 1
            if base_preconditioner is None:
                cp.copyto(destination, operator_output)
            else:
                base_preconditioner.apply_into(operator_output, destination)
                base_count += 1

        for column in range(maximum_degree):
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
        gram_host = np.ascontiguousarray(cp.asnumpy(active_basis @ active_basis.T))
        used_hessenberg = np.ascontiguousarray(
            hessenberg[: effective_degree + 1, :effective_degree]
        )

    return CuPyPolynomialArnoldiProbe(
        hessenberg=used_hessenberg,
        gram_matrix=gram_host,
        requested_degree=maximum_degree,
        effective_degree=effective_degree,
        seed=int(seed),
        breakdown=breakdown,
        orthogonalization=orthogonalization,  # type: ignore[arg-type]
        matvec_count=matvec_count,
        base_preconditioner_count=base_count,
        num_dofs=num_dofs,
        device_id=device_id,
        dtype_name=np.dtype(dtype.name).name,
        uses_base_preconditioner=base_preconditioner is not None,
        operator_identity=id(operator),
        base_preconditioner_identity=(
            None if base_preconditioner is None else id(base_preconditioner)
        ),
    )


def polynomial_setup_from_probe(
    probe: CuPyPolynomialArnoldiProbe,
    *,
    degree: int,
    pair_tolerance: float = 1.0e-10,
) -> CuPyPolynomialSetup:
    """Extract one harmonic-Ritz/Leja polynomial from a reusable probe."""

    if not isinstance(probe, CuPyPolynomialArnoldiProbe):
        raise TypeError("probe must be a CuPyPolynomialArnoldiProbe")
    if isinstance(degree, bool) or int(degree) != degree or degree <= 0:
        raise ValueError("degree must be a positive integer")
    degree = int(degree)
    if degree > probe.requested_degree:
        raise ValueError("degree exceeds the probe requested degree")
    if pair_tolerance <= 0.0 or not np.isfinite(pair_tolerance):
        raise ValueError("pair_tolerance must be positive and finite")

    effective_degree = min(degree, probe.effective_degree)
    if effective_degree <= 0:
        raise np.linalg.LinAlgError("Arnoldi probe has no usable dimension")
    used_hessenberg = np.ascontiguousarray(
        probe.hessenberg[: effective_degree + 1, :effective_degree]
    )
    values = harmonic_ritz_values(used_hessenberg)
    roots = leja_order_conjugate_preserving(values, tolerance=pair_tolerance)
    root_scale = max(float(np.max(np.abs(roots))), 1.0)
    zero_tolerance = 100.0 * np.finfo(np.float64).eps * root_scale
    if np.any(np.abs(roots) <= zero_tolerance):
        raise np.linalg.LinAlgError("harmonic Ritz setup produced a zero root")

    gram = probe.gram_matrix[:effective_degree, :effective_degree]
    frobenius, maximum_offdiagonal, maximum_diagonal_error = (
        _orthogonality_metrics(gram)
    )
    uses_base = probe.uses_base_preconditioner
    return CuPyPolynomialSetup(
        hessenberg=used_hessenberg,
        harmonic_ritz_values=values,
        ordered_roots=roots,
        requested_degree=degree,
        effective_degree=effective_degree,
        seed=probe.seed,
        breakdown=effective_degree < degree,
        orthogonalization=probe.orthogonalization,
        frobenius_orthogonality_defect=frobenius,
        maximum_offdiagonal=maximum_offdiagonal,
        maximum_diagonal_error=maximum_diagonal_error,
        matvec_count=effective_degree,
        base_preconditioner_count=effective_degree if uses_base else 0,
    )


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
    """Run one GPU Arnoldi cycle and construct harmonic-Ritz/Leja roots."""

    probe = setup_polynomial_arnoldi_probe_cupy(
        operator,
        maximum_degree=degree,
        base_preconditioner=base_preconditioner,
        seed=seed,
        initial_vector=initial_vector,
        orthogonalization=orthogonalization,
        breakdown_tolerance=breakdown_tolerance,
        profiler=profiler,
    )
    return polynomial_setup_from_probe(
        probe,
        degree=degree,
        pair_tolerance=pair_tolerance,
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
        """Initialize the instance and validate persistent storage."""
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
            self._real_update, self._pair_update = (
                _get_polynomial_update_kernels(
                    cp, dtype=dtype, device_id=device_id
                )
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
        """Construct the object from the supplied operator or system."""
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

    @classmethod
    def from_probe(
        cls,
        operator: DeviceMatvecOperator,
        *,
        probe: CuPyPolynomialArnoldiProbe,
        degree: int,
        base_preconditioner: DevicePreconditioner | None = None,
        pair_tolerance: float = 1.0e-10,
    ) -> "CuPyPolynomialPreconditioner":
        """Construct a degree candidate from a shared Arnoldi probe."""

        cp = require_cupy_device()
        dtype, device_id, num_dofs = _validate_protocol_compatibility(
            operator, base_preconditioner
        )
        if probe.num_dofs != num_dofs:
            raise ValueError("probe size does not match the operator")
        if probe.device_id != device_id:
            raise ValueError("probe and operator use different CUDA devices")
        if np.dtype(probe.dtype_name) != np.dtype(dtype.name):
            raise TypeError("probe dtype does not match the operator")
        if probe.operator_identity != id(operator):
            raise ValueError("probe was built for a different operator object")
        expected_base_identity = (
            None if base_preconditioner is None else id(base_preconditioner)
        )
        if probe.base_preconditioner_identity != expected_base_identity:
            raise ValueError("probe was built for a different base preconditioner")
        setup = polynomial_setup_from_probe(
            probe, degree=degree, pair_tolerance=pair_tolerance
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
        """Return the polynomial degree."""
        return int(self.roots.size)

    @property
    def matvecs_per_application(self) -> int:
        """Return the number of operator applications per preconditioner call."""
        return self.degree

    @property
    def base_preconditioner_calls_per_application(self) -> int:
        """Return the number of base-preconditioner calls per application."""
        return 0 if self.base_preconditioner is None else self.degree + 1

    @property
    def allocates_during_apply(self) -> bool:
        """Return whether an application allocates device storage."""
        return False

    @property
    def workspace_bytes(self) -> int:
        """Device workspace retained by the composite preconditioner."""

        own = sum(
            int(array.nbytes)
            for array in (
                self._q,
                self._bq,
                self._b2q,
                self._operator_output,
            )
        )
        base = int(getattr(self.base_preconditioner, "workspace_bytes", 0))
        return own + base

    def _validate_vector(self, vector: Any, *, name: str) -> Any:
        """Validate shapes, dtypes, devices, and solver parameters."""
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
        """Apply the preconditioner using reusable storage."""
        if self.base_preconditioner is None:
            self.operator.matvec_into(source, destination)
        else:
            self.operator.matvec_into(source, self._operator_output)
            self.base_preconditioner.apply_into(self._operator_output, destination)
            self.base_preconditioner_count += 1
        self.matvec_count += 1

    def _launch_real_update(self, theta: float, result: Any) -> None:
        """Launch the corresponding preallocated CUDA kernel."""
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
        """Launch the corresponding preallocated CUDA kernel."""
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
        """Apply the preconditioner using reusable storage."""
        x_flat = self._validate_vector(x, name="x")
        out = self._cp.empty_like(x_flat)
        self.apply_into(x_flat, out)
        return out.reshape(x.shape)


__all__ = [
    "CuPyPolynomialArnoldiProbe",
    "CuPyPolynomialPreconditioner",
    "CuPyPolynomialSetup",
    "initialize_polynomial_kernels_cupy",
    "polynomial_setup_from_probe",
    "setup_polynomial_arnoldi_probe_cupy",
    "setup_polynomial_preconditioner_cupy",
]
