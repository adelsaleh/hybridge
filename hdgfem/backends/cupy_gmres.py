"""Restarted GMRES with GPU vectors and CPU Hessenberg updates.

The large Krylov vectors remain on one CUDA device.  BLAS-1/2 operations are
performed through CuPy's cuBLAS wrappers, while the small Hessenberg matrix,
Givens coefficients, and triangular solve remain as NumPy arrays on the CPU.
This mirrors the hybrid GPU/CPU decomposition used in the HDG preconditioning
paper and keeps all transfers restricted to scalar Arnoldi coefficients and the
small restart coefficient vector.

This module is an optional CUDA backend: importing it does not require CuPy,
but constructing the BLAS adapter or calling the solver does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .cupy import require_cupy_device

CuPyGMRESStatus = Literal["converged", "max_iterations", "breakdown"]


@runtime_checkable
class DeviceMatvecOperator(Protocol):
    """Minimal device operator required by :func:`restarted_gmres_cupy`."""

    num_dofs: int
    dtype: Any
    device_id: int

    def matvec_into(self, x: Any, out: Any) -> None:
        """Compute ``out = A @ x`` on the device."""


@runtime_checkable
class DevicePreconditioner(Protocol):
    """Minimal left-preconditioner interface used by GPU GMRES."""

    num_dofs: int
    dtype: Any
    device_id: int

    def apply_into(self, x: Any, out: Any) -> None:
        """Compute ``out = M^{-1} @ x`` on the device."""


@dataclass(frozen=True)
class CuPyGMRESResult:
    """GPU GMRES result.

    ``solution`` is intentionally left on the GPU.  Use the operator's
    ``to_host`` method, or ``cupy.asnumpy``, only when an explicit transfer is
    desired.
    """

    solution: Any
    converged: bool
    status: CuPyGMRESStatus
    iterations: int
    restart_cycles: int
    residual_norm: float
    relative_residual: float
    true_residual_history: np.ndarray
    estimated_preconditioned_residual_history: np.ndarray
    matvec_count: int
    preconditioner_count: int
    dot_count: int
    axpy_count: int
    norm_count: int
    basis_update_count: int


class CuPyVectorBLAS:
    """Thin, explicit cuBLAS adapter for real CuPy vectors.

    CuPy's public ``cupy.cublas`` helper routes these operations to cuBLAS.
    Dot products and norms write directly into a reusable NumPy scalar because
    GMRES immediately needs those values on the CPU for the Hessenberg update.
    """

    def __init__(self, *, dtype: Any, device_id: int) -> None:
        cp = require_cupy_device()
        if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("GPU GMRES supports float32 and float64 only")
        self.cp = cp
        self.dtype = cp.dtype(dtype)
        self.device_id = int(device_id)
        self._host_scalar = np.empty((), dtype=np.dtype(self.dtype.name))
        # cupy.cublas is a high-level wrapper around cuBLAS BLAS-1/2 routines.
        from cupy import cublas as cupy_cublas

        self._cublas = cupy_cublas

    def _validate_vector(self, vector: Any, *, name: str) -> Any:
        cp = self.cp
        if not isinstance(vector, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        if int(vector.device.id) != self.device_id:
            raise ValueError(
                f"{name} is on CUDA device {vector.device.id}, expected "
                f"device {self.device_id}"
            )
        if vector.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}")
        if vector.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector

    def copy(self, source: Any, destination: Any) -> None:
        source = self._validate_vector(source, name="source")
        destination = self._validate_vector(destination, name="destination")
        if source.shape != destination.shape:
            raise ValueError("source and destination must have equal shape")
        self.cp.copyto(destination, source)

    def dot(self, x: Any, y: Any) -> float:
        x = self._validate_vector(x, name="x")
        y = self._validate_vector(y, name="y")
        if x.shape != y.shape:
            raise ValueError("dot vectors must have equal shape")
        self._cublas.dot(x, y, out=self._host_scalar)
        return float(self._host_scalar)

    def norm(self, x: Any) -> float:
        x = self._validate_vector(x, name="x")
        self._cublas.nrm2(x, out=self._host_scalar)
        return float(self._host_scalar)

    def axpy(self, alpha: float, x: Any, y: Any) -> None:
        x = self._validate_vector(x, name="x")
        y = self._validate_vector(y, name="y")
        if x.shape != y.shape:
            raise ValueError("AXPY vectors must have equal shape")
        self._cublas.axpy(alpha, x, y)

    def scal(self, alpha: float, x: Any) -> None:
        x = self._validate_vector(x, name="x")
        self._cublas.scal(alpha, x)

    def basis_update(
        self,
        basis_rows: Any,
        coefficients: Any,
        solution: Any,
    ) -> None:
        """Perform ``solution += basis_rows.T @ coefficients`` with GEMV."""

        cp = self.cp
        if not isinstance(basis_rows, cp.ndarray) or basis_rows.ndim != 2:
            raise TypeError("basis_rows must be a two-dimensional CuPy array")
        if int(basis_rows.device.id) != self.device_id:
            raise ValueError("basis_rows is on the wrong CUDA device")
        if basis_rows.dtype != self.dtype or not basis_rows.flags.c_contiguous:
            raise ValueError("basis_rows must be C-contiguous with matching dtype")
        coefficients = self._validate_vector(coefficients, name="coefficients")
        solution = self._validate_vector(solution, name="solution")
        if basis_rows.shape != (coefficients.size, solution.size):
            raise ValueError(
                "basis update shapes are incompatible: "
                f"basis={basis_rows.shape}, coefficients={coefficients.shape}, "
                f"solution={solution.shape}"
            )
        self._cublas.gemv(
            "T",
            1.0,
            basis_rows,
            coefficients,
            1.0,
            solution,
        )


def _compute_givens(a: float, b: float) -> tuple[float, float, float]:
    """Return ``c, s, r`` such that ``[[c,s],[-s,c]] [a,b]^T=[r,0]^T``."""

    r = float(np.hypot(a, b))
    if r == 0.0:
        return 1.0, 0.0, 0.0
    return a / r, b / r, r


def _apply_previous_givens(
    hessenberg: np.ndarray,
    cosines: np.ndarray,
    sines: np.ndarray,
    column: int,
) -> None:
    """Apply rotations ``0..column-1`` to one Hessenberg column in place."""

    for row in range(column):
        upper = hessenberg[row, column]
        lower = hessenberg[row + 1, column]
        hessenberg[row, column] = cosines[row] * upper + sines[row] * lower
        hessenberg[row + 1, column] = -sines[row] * upper + cosines[row] * lower


def _back_substitute_upper(
    upper: np.ndarray,
    rhs: np.ndarray,
    *,
    singular_tolerance: float,
) -> np.ndarray:
    """Solve a small upper-triangular system without invoking GPU code."""

    upper = np.asarray(upper, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if upper.ndim != 2 or upper.shape[0] != upper.shape[1]:
        raise ValueError("upper must be square")
    if rhs.shape != (upper.shape[0],):
        raise ValueError("rhs has an incompatible shape")

    result = rhs.copy()
    for row in range(upper.shape[0] - 1, -1, -1):
        diagonal = float(upper[row, row])
        scale = max(1.0, float(np.max(np.abs(upper[row, row:]), initial=0.0)))
        if abs(diagonal) <= singular_tolerance * scale:
            raise np.linalg.LinAlgError(
                f"GMRES triangular factor is singular at row {row}"
            )
        if row + 1 < upper.shape[0]:
            result[row] -= upper[row, row + 1 :] @ result[row + 1 :]
        result[row] /= diagonal
    return result


def _validate_restart_parameters(
    *,
    restart: int,
    max_iterations: int | None,
    num_dofs: int,
    rtol: float,
    atol: float,
    breakdown_tolerance: float | None,
) -> tuple[int, int, float]:
    if isinstance(restart, bool) or int(restart) != restart or restart <= 0:
        raise ValueError("restart must be a positive integer")
    restart = int(restart)
    if max_iterations is None:
        max_iterations = 10 * num_dofs
    if (
        isinstance(max_iterations, bool)
        or int(max_iterations) != max_iterations
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    if rtol < 0.0 or not np.isfinite(rtol):
        raise ValueError("rtol must be finite and non-negative")
    if atol < 0.0 or not np.isfinite(atol):
        raise ValueError("atol must be finite and non-negative")
    if breakdown_tolerance is None:
        breakdown_tolerance = 100.0 * np.finfo(np.float64).eps
    if breakdown_tolerance < 0.0 or not np.isfinite(breakdown_tolerance):
        raise ValueError("breakdown_tolerance must be finite and non-negative")
    return restart, int(max_iterations), float(breakdown_tolerance)


def restarted_gmres_cupy(
    operator: DeviceMatvecOperator,
    rhs: Any,
    *,
    x0: Any | None = None,
    restart: int = 30,
    max_iterations: int | None = None,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    preconditioner: DevicePreconditioner | None = None,
    reorthogonalize: bool = False,
    breakdown_tolerance: float | None = None,
) -> CuPyGMRESResult:
    r"""Solve ``A x = rhs`` with restarted left-preconditioned GPU GMRES.

    Large vectors and the Arnoldi basis stay on the GPU.  Modified
    Gram--Schmidt uses cuBLAS DOT and AXPY calls; norms use cuBLAS NRM2; the
    restart update uses one cuBLAS GEMV.  Only scalar Hessenberg coefficients
    and the final restart coefficient vector cross the PCIe boundary.

    The Givens residual is used as an inexpensive convergence trigger.  A true
    residual is always recomputed after each completed or early-terminated
    restart cycle before convergence is accepted.
    """

    cp = require_cupy_device()
    if not isinstance(operator, DeviceMatvecOperator):
        raise TypeError("operator must implement the device matvec protocol")
    if preconditioner is not None and not isinstance(
        preconditioner, DevicePreconditioner
    ):
        raise TypeError("preconditioner must implement apply_into or be None")

    num_dofs = int(operator.num_dofs)
    if num_dofs <= 0:
        raise ValueError("operator.num_dofs must be positive")
    dtype = cp.dtype(operator.dtype)
    device_id = int(operator.device_id)
    restart, max_iterations, breakdown_tolerance = _validate_restart_parameters(
        restart=restart,
        max_iterations=max_iterations,
        num_dofs=num_dofs,
        rtol=rtol,
        atol=atol,
        breakdown_tolerance=breakdown_tolerance,
    )

    if preconditioner is not None:
        if int(preconditioner.num_dofs) != num_dofs:
            raise ValueError("preconditioner size does not match the operator")
        if cp.dtype(preconditioner.dtype) != dtype:
            raise TypeError("preconditioner dtype does not match the operator")
        if int(preconditioner.device_id) != device_id:
            raise ValueError("preconditioner and operator use different devices")

    def validate_vector(vector: Any, *, name: str) -> Any:
        if not isinstance(vector, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        if int(vector.device.id) != device_id:
            raise ValueError(f"{name} is on the wrong CUDA device")
        if vector.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if vector.size != num_dofs:
            raise ValueError(f"{name} must contain {num_dofs} values")
        if vector.ndim not in (1, 2):
            raise ValueError(f"{name} must be flat or face-major")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector

    rhs = validate_vector(rhs, name="rhs")
    original_shape = rhs.shape

    with cp.cuda.Device(device_id):
        b = rhs.reshape(-1)
        if x0 is None:
            x = cp.zeros(num_dofs, dtype=dtype)
        else:
            x0 = validate_vector(x0, name="x0")
            x = x0.reshape(-1).copy()

        blas = CuPyVectorBLAS(dtype=dtype, device_id=device_id)
        basis = cp.empty((restart + 1, num_dofs), dtype=dtype)
        matvec_buffer = cp.empty(num_dofs, dtype=dtype)
        residual = cp.empty(num_dofs, dtype=dtype)
        work = cp.empty(num_dofs, dtype=dtype)
        preconditioned = cp.empty(num_dofs, dtype=dtype)
        coefficient_device = cp.empty(restart, dtype=dtype)

        matvec_count = 0
        preconditioner_count = 0
        dot_count = 0
        axpy_count = 0
        norm_count = 0
        basis_update_count = 0

        def apply_matvec(source: Any, destination: Any) -> None:
            nonlocal matvec_count
            operator.matvec_into(source, destination)
            matvec_count += 1

        def apply_preconditioner(source: Any, destination: Any) -> None:
            nonlocal preconditioner_count
            if preconditioner is None:
                blas.copy(source, destination)
            else:
                preconditioner.apply_into(source, destination)
                preconditioner_count += 1

        def compute_true_residual() -> float:
            nonlocal axpy_count, norm_count
            apply_matvec(x, matvec_buffer)
            blas.copy(b, residual)
            blas.axpy(-1.0, matvec_buffer, residual)
            axpy_count += 1
            value = blas.norm(residual)
            norm_count += 1
            return value

        b_norm = blas.norm(b)
        norm_count += 1
        denominator = max(b_norm, np.finfo(np.dtype(dtype.name)).eps)
        target = max(float(atol), float(rtol) * b_norm)

        true_norm = compute_true_residual()
        true_history: list[float] = [true_norm]
        estimated_history: list[float] = []

        def make_result(
            status: CuPyGMRESStatus,
            iterations: int,
            cycles: int,
            final_norm: float,
        ) -> CuPyGMRESResult:
            return CuPyGMRESResult(
                solution=x.reshape(original_shape),
                converged=status == "converged",
                status=status,
                iterations=iterations,
                restart_cycles=cycles,
                residual_norm=float(final_norm),
                relative_residual=float(final_norm) / denominator,
                true_residual_history=np.asarray(true_history, dtype=np.float64),
                estimated_preconditioned_residual_history=np.asarray(
                    estimated_history,
                    dtype=np.float64,
                ),
                matvec_count=matvec_count,
                preconditioner_count=preconditioner_count,
                dot_count=dot_count,
                axpy_count=axpy_count,
                norm_count=norm_count,
                basis_update_count=basis_update_count,
            )

        if true_norm <= target:
            return make_result("converged", 0, 0, true_norm)

        iterations = 0
        cycles = 0
        while iterations < max_iterations:
            cycles += 1

            # The true residual from the previous cycle is already in
            # ``residual``. Apply the left preconditioner and normalize it.
            apply_preconditioner(residual, preconditioned)
            beta = blas.norm(preconditioned)
            norm_count += 1
            estimated_history.append(beta)
            if beta <= breakdown_tolerance * max(1.0, true_norm):
                return make_result("breakdown", iterations, cycles, true_norm)

            blas.copy(preconditioned, basis[0])
            blas.scal(1.0 / beta, basis[0])

            cycle_dimension = min(restart, max_iterations - iterations)
            hessenberg = np.zeros(
                (cycle_dimension + 1, cycle_dimension),
                dtype=np.float64,
            )
            cosines = np.zeros(cycle_dimension, dtype=np.float64)
            sines = np.zeros(cycle_dimension, dtype=np.float64)
            least_squares_rhs = np.zeros(cycle_dimension + 1, dtype=np.float64)
            least_squares_rhs[0] = beta

            used_dimension = 0
            happy_breakdown = False
            for column in range(cycle_dimension):
                apply_matvec(basis[column], matvec_buffer)
                apply_preconditioner(matvec_buffer, work)

                passes = 2 if reorthogonalize else 1
                for _ in range(passes):
                    for row in range(column + 1):
                        coefficient = blas.dot(basis[row], work)
                        dot_count += 1
                        hessenberg[row, column] += coefficient
                        blas.axpy(-coefficient, basis[row], work)
                        axpy_count += 1

                next_norm = blas.norm(work)
                norm_count += 1
                hessenberg[column + 1, column] = next_norm
                arnoldi_scale = max(
                    1.0,
                    float(np.linalg.norm(hessenberg[: column + 1, column])),
                )
                happy_breakdown = (
                    next_norm <= breakdown_tolerance * arnoldi_scale
                )
                if not happy_breakdown:
                    blas.copy(work, basis[column + 1])
                    blas.scal(1.0 / next_norm, basis[column + 1])

                _apply_previous_givens(
                    hessenberg,
                    cosines,
                    sines,
                    column,
                )
                cosine, sine, diagonal = _compute_givens(
                    hessenberg[column, column],
                    hessenberg[column + 1, column],
                )
                cosines[column] = cosine
                sines[column] = sine
                hessenberg[column, column] = diagonal
                hessenberg[column + 1, column] = 0.0

                old_rhs = least_squares_rhs[column]
                least_squares_rhs[column] = cosine * old_rhs
                least_squares_rhs[column + 1] = -sine * old_rhs
                estimate = abs(least_squares_rhs[column + 1])
                estimated_history.append(estimate)

                iterations += 1
                used_dimension = column + 1

                # The estimate belongs to the preconditioned residual.  It is
                # used only as a trigger; the true residual is checked below.
                estimate_target = max(float(atol), float(rtol) * beta)
                if (
                    estimate <= estimate_target
                    or happy_breakdown
                    or used_dimension == cycle_dimension
                    or iterations == max_iterations
                ):
                    break

            if used_dimension == 0:
                return make_result("breakdown", iterations, cycles, true_norm)

            try:
                coefficients = _back_substitute_upper(
                    hessenberg[:used_dimension, :used_dimension],
                    least_squares_rhs[:used_dimension],
                    singular_tolerance=breakdown_tolerance,
                )
            except np.linalg.LinAlgError:
                return make_result("breakdown", iterations, cycles, true_norm)

            coefficient_device[:used_dimension].set(
                coefficients.astype(
                    np.dtype(dtype.name),
                    copy=False,
                )
            )
            blas.basis_update(
                basis[:used_dimension],
                coefficient_device[:used_dimension],
                x,
            )
            basis_update_count += 1

            true_norm = compute_true_residual()
            true_history.append(true_norm)
            if true_norm <= target:
                return make_result("converged", iterations, cycles, true_norm)
            if happy_breakdown:
                return make_result("breakdown", iterations, cycles, true_norm)

        return make_result("max_iterations", iterations, cycles, true_norm)


__all__ = [
    "CuPyGMRESResult",
    "CuPyVectorBLAS",
    "DeviceMatvecOperator",
    "DevicePreconditioner",
    "restarted_gmres_cupy",
]
