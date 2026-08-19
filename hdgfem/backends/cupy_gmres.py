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

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .cupy import device_arrays_overlap, require_cupy_device

CuPyGMRESStatus = Literal[
    "converged",
    "max_iterations",
    "breakdown",
    "stagnated",
    "diverged",
    "non_finite",
]
CuPyOrthogonalization = Literal["mgs", "mgs2", "cgs", "cgs2"]


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

    @property
    def num_dofs(self) -> int:
        """Return the number of device degrees of freedom."""
        ...

    @property
    def dtype(self) -> Any:
        """Return the device scalar dtype."""
        ...

    @property
    def device_id(self) -> int:
        """Return the CUDA device identifier."""
        ...

    def apply_into(self, x: Any, out: Any) -> None:
        """Compute ``out = M^{-1} @ x`` on the device."""




@dataclass
class CuPyGMRESWorkspace:
    """Reusable GPU and CPU storage for restarted GMRES.

    The workspace owns the Arnoldi basis, vector work buffers, short device
    coefficient arrays, and the small CPU Hessenberg/Givens arrays.  Reusing
    one instance across sequential solves avoids repeated large GPU
    allocations and makes multi-right-hand-side timings representative of the
    actual numerical work.

    A workspace is not thread-safe and must not be used by concurrent solves.
    ``restart_capacity`` may be larger than the restart selected by a solve.
    """

    num_dofs: int
    restart_capacity: int
    dtype: Any
    device_id: int
    basis: Any = field(repr=False)
    matvec_buffer: Any = field(repr=False)
    residual: Any = field(repr=False)
    work: Any = field(repr=False)
    preconditioned: Any = field(repr=False)
    coefficient_device: Any = field(repr=False)
    orthogonalization_coefficients_device: Any = field(repr=False)
    orthogonalization_coefficients_host: np.ndarray = field(repr=False)
    update_coefficients_host: np.ndarray = field(repr=False)
    hessenberg: np.ndarray = field(repr=False)
    cosines: np.ndarray = field(repr=False)
    sines: np.ndarray = field(repr=False)
    least_squares_rhs: np.ndarray = field(repr=False)

    @classmethod
    def allocate(
        cls,
        *,
        num_dofs: int,
        restart_capacity: int,
        dtype: Any,
        device_id: int = 0,
    ) -> "CuPyGMRESWorkspace":
        """Allocate reusable solver workspace."""
        cp = require_cupy_device()
        if isinstance(num_dofs, bool) or int(num_dofs) != num_dofs or num_dofs <= 0:
            raise ValueError("num_dofs must be a positive integer")
        if (
            isinstance(restart_capacity, bool)
            or int(restart_capacity) != restart_capacity
            or restart_capacity <= 0
        ):
            raise ValueError("restart_capacity must be a positive integer")
        dtype = cp.dtype(dtype)
        if np.dtype(dtype.name) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("GPU GMRES supports float32 and float64 only")
        num_dofs = int(num_dofs)
        restart_capacity = int(restart_capacity)
        device_id = int(device_id)
        host_dtype = np.dtype(dtype.name)
        with cp.cuda.Device(device_id):
            return cls(
                num_dofs=num_dofs,
                restart_capacity=restart_capacity,
                dtype=dtype,
                device_id=device_id,
                basis=cp.empty((restart_capacity + 1, num_dofs), dtype=dtype),
                matvec_buffer=cp.empty(num_dofs, dtype=dtype),
                residual=cp.empty(num_dofs, dtype=dtype),
                work=cp.empty(num_dofs, dtype=dtype),
                preconditioned=cp.empty(num_dofs, dtype=dtype),
                coefficient_device=cp.empty(restart_capacity, dtype=dtype),
                orthogonalization_coefficients_device=cp.empty(
                    restart_capacity, dtype=dtype
                ),
                orthogonalization_coefficients_host=np.empty(
                    restart_capacity, dtype=host_dtype
                ),
                update_coefficients_host=np.empty(
                    restart_capacity, dtype=host_dtype
                ),
                hessenberg=np.empty(
                    (restart_capacity + 1, restart_capacity), dtype=np.float64
                ),
                cosines=np.empty(restart_capacity, dtype=np.float64),
                sines=np.empty(restart_capacity, dtype=np.float64),
                least_squares_rhs=np.empty(
                    restart_capacity + 1, dtype=np.float64
                ),
            )

    def validate_for(
        self,
        *,
        num_dofs: int,
        restart: int,
        dtype: Any,
        device_id: int,
    ) -> None:
        """Validate shapes, dtypes, devices, and solver parameters."""
        cp = require_cupy_device()
        if self.num_dofs != int(num_dofs):
            raise ValueError("workspace size does not match the operator")
        if self.restart_capacity < int(restart):
            raise ValueError(
                "workspace restart_capacity is smaller than the requested restart"
            )
        if cp.dtype(self.dtype) != cp.dtype(dtype):
            raise TypeError("workspace dtype does not match the operator")
        if self.device_id != int(device_id):
            raise ValueError("workspace and operator use different CUDA devices")

    @property
    def device_arrays(self) -> tuple[Any, ...]:
        """Device arrays owned by the workspace."""

        return (
            self.basis,
            self.matvec_buffer,
            self.residual,
            self.work,
            self.preconditioned,
            self.coefficient_device,
            self.orthogonalization_coefficients_device,
        )

    @property
    def device_bytes(self) -> int:
        """Total bytes owned by the reusable device arrays."""

        return sum(int(array.nbytes) for array in self.device_arrays)


@dataclass(frozen=True)
class CuPyOrthogonalityRecord:
    """Orthogonality diagnostics for one completed restart cycle."""

    restart_cycle: int
    total_iterations: int
    basis_dimension: int
    frobenius_defect: float
    maximum_offdiagonal: float
    maximum_diagonal_error: float




@dataclass(frozen=True)
class CuPyGMRESCycleRecord:
    """Diagnostics for one completed restart cycle.

    ``true_residual_start`` and ``true_residual_end`` are unpreconditioned
    residual norms.  The solver recomputes the true residual at every restart
    boundary and uses it to seed the next cycle, so each completed cycle also
    acts as a residual-replacement step.
    """

    restart_cycle: int
    iteration_start: int
    iteration_end: int
    basis_dimension: int
    orthogonalization: CuPyOrthogonalization
    true_residual_start: float
    true_residual_end: float
    residual_reduction: float
    estimated_residual_end: float
    happy_breakdown: bool
    stagnation_count: int
    switched_to_cgs2: bool

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
    orthogonalization: CuPyOrthogonalization
    basis_projection_count: int
    basis_correction_count: int
    coefficient_d2h_count: int
    orthogonality_records: tuple[CuPyOrthogonalityRecord, ...]
    cycle_records: tuple[CuPyGMRESCycleRecord, ...]
    orthogonalization_history: tuple[CuPyOrthogonalization, ...]
    fallback_count: int
    true_residual_recomputations: int
    termination_reason: str


class CuPyRestartedGMRESSolver:
    """Reusable restarted-GMRES solver for sequential right-hand sides.

    The numerical configuration and :class:`CuPyGMRESWorkspace` are created
    once.  Call :meth:`solve` repeatedly with different right-hand sides and
    preallocated ``solution_out`` vectors to avoid large per-solve device
    allocations.
    """

    def __init__(
        self,
        operator: DeviceMatvecOperator,
        *,
        restart: int = 30,
        max_iterations: int | None = None,
        rtol: float = 1.0e-8,
        atol: float = 0.0,
        preconditioner: DevicePreconditioner | None = None,
        orthogonalization: CuPyOrthogonalization = "cgs",
        breakdown_tolerance: float | None = None,
        check_finite: bool = True,
        stagnation_cycles: int | None = None,
        stagnation_tolerance: float = 1.0e-3,
        divergence_factor: float = 1.0e6,
        cgs2_fallback_threshold: float | None = None,
        workspace: CuPyGMRESWorkspace | None = None,
    ) -> None:
        """Initialize the instance and validate persistent storage."""
        cp = require_cupy_device()
        self.operator = operator
        self.restart = int(restart)
        self.max_iterations = max_iterations
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.preconditioner = preconditioner
        self.orthogonalization = orthogonalization
        self.breakdown_tolerance = breakdown_tolerance
        self.check_finite = bool(check_finite)
        self.stagnation_cycles = stagnation_cycles
        self.stagnation_tolerance = float(stagnation_tolerance)
        self.divergence_factor = float(divergence_factor)
        self.cgs2_fallback_threshold = cgs2_fallback_threshold
        if workspace is None:
            workspace = CuPyGMRESWorkspace.allocate(
                num_dofs=int(operator.num_dofs),
                restart_capacity=self.restart,
                dtype=cp.dtype(operator.dtype),
                device_id=int(operator.device_id),
            )
        else:
            workspace.validate_for(
                num_dofs=int(operator.num_dofs),
                restart=self.restart,
                dtype=cp.dtype(operator.dtype),
                device_id=int(operator.device_id),
            )
        self.workspace = workspace

    @property
    def workspace_device_bytes(self) -> int:
        """Return persistent device-workspace storage in bytes."""
        return self.workspace.device_bytes

    def solve(
        self,
        rhs: Any,
        *,
        x0: Any | None = None,
        solution_out: Any | None = None,
        profiler: Any | None = None,
        monitor_orthogonality: bool = False,
    ) -> CuPyGMRESResult:
        """Solve the configured linear system."""
        return restarted_gmres_cupy(
            self.operator,
            rhs,
            x0=x0,
            restart=self.restart,
            max_iterations=self.max_iterations,
            rtol=self.rtol,
            atol=self.atol,
            preconditioner=self.preconditioner,
            orthogonalization=self.orthogonalization,
            breakdown_tolerance=self.breakdown_tolerance,
            check_finite=self.check_finite,
            stagnation_cycles=self.stagnation_cycles,
            stagnation_tolerance=self.stagnation_tolerance,
            divergence_factor=self.divergence_factor,
            cgs2_fallback_threshold=self.cgs2_fallback_threshold,
            profiler=profiler,
            monitor_orthogonality=monitor_orthogonality,
            workspace=self.workspace,
            solution_out=solution_out,
        )


class CuPyVectorBLAS:
    """Thin, explicit cuBLAS adapter for real CuPy vectors.

    CuPy's public ``cupy.cublas`` helper routes these operations to cuBLAS.
    Dot products and norms write directly into a reusable NumPy scalar because
    GMRES immediately needs those values on the CPU for the Hessenberg update.
    """

    def __init__(
        self,
        *,
        dtype: Any,
        device_id: int,
        profiler: Any | None = None,
    ) -> None:
        """Initialize the instance and validate persistent storage."""
        cp = require_cupy_device()
        if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("GPU GMRES supports float32 and float64 only")
        self.cp = cp
        self.dtype = cp.dtype(dtype)
        self.device_id = int(device_id)
        self._host_scalar = np.empty((), dtype=np.dtype(self.dtype.name))
        self.profiler = profiler
        # cupy.cublas is a high-level wrapper around cuBLAS BLAS-1/2 routines.
        from cupy import cublas as cupy_cublas

        self._cublas = cupy_cublas

    def _validate_vector(self, vector: Any, *, name: str) -> Any:
        """Validate shapes, dtypes, devices, and solver parameters."""
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
        """Copy a device vector into caller-provided storage."""
        source = self._validate_vector(source, name="source")
        destination = self._validate_vector(destination, name="destination")
        if source.shape != destination.shape:
            raise ValueError("source and destination must have equal shape")
        def operation() -> None:
            """Execute the captured vector operation."""
            self.cp.copyto(destination, source)

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call("copy", operation)

    def dot(self, x: Any, y: Any) -> float:
        """Return the device dot product."""
        x = self._validate_vector(x, name="x")
        y = self._validate_vector(y, name="y")
        if x.shape != y.shape:
            raise ValueError("dot vectors must have equal shape")
        def operation() -> None:
            """Execute the captured vector operation."""
            self._cublas.dot(x, y, out=self._host_scalar)

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call(
                "dot",
                operation,
                host_synchronizing=True,
            )
        return float(self._host_scalar)

    def norm(self, x: Any) -> float:
        """Return the device Euclidean norm."""
        x = self._validate_vector(x, name="x")
        def operation() -> None:
            """Execute the captured vector operation."""
            self._cublas.nrm2(x, out=self._host_scalar)

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call(
                "norm",
                operation,
                host_synchronizing=True,
            )
        return float(self._host_scalar)

    def axpy(self, alpha: float, x: Any, y: Any) -> None:
        """Apply the device AXPY update."""
        x = self._validate_vector(x, name="x")
        y = self._validate_vector(y, name="y")
        if x.shape != y.shape:
            raise ValueError("AXPY vectors must have equal shape")
        def operation() -> None:
            """Execute the captured vector operation."""
            self._cublas.axpy(alpha, x, y)

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call("axpy", operation)

    def scal(self, alpha: float, x: Any) -> None:
        """Scale a device vector in place."""
        x = self._validate_vector(x, name="x")
        def operation() -> None:
            """Execute the captured vector operation."""
            self._cublas.scal(alpha, x)

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call("scal", operation)

    def _validate_basis_matrix(self, matrix: Any, *, name: str) -> Any:
        """Validate shapes, dtypes, devices, and solver parameters."""
        cp = self.cp
        if not isinstance(matrix, cp.ndarray):
            raise TypeError(f"{name} must be a CuPy array")
        if int(matrix.device.id) != self.device_id:
            raise ValueError(f"{name} is on the wrong CUDA device")
        if matrix.dtype != self.dtype:
            raise TypeError(f"{name} must have dtype {self.dtype}")
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if not matrix.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return matrix

    def basis_projection(
        self,
        basis_rows: Any,
        vector: Any,
        coefficients: Any,
    ) -> None:
        """Compute ``coefficients = basis_rows @ vector`` with one GEMV."""

        basis_rows = self._validate_basis_matrix(
            basis_rows,
            name="basis_rows",
        )
        vector = self._validate_vector(vector, name="vector")
        coefficients = self._validate_vector(
            coefficients,
            name="coefficients",
        )
        if basis_rows.shape != (coefficients.size, vector.size):
            raise ValueError(
                "basis projection shapes are incompatible: "
                f"basis={basis_rows.shape}, vector={vector.shape}, "
                f"coefficients={coefficients.shape}"
            )

        def operation() -> None:
            """Execute the captured vector operation."""
            self._cublas.gemv(
                "N",
                1.0,
                basis_rows,
                vector,
                0.0,
                coefficients,
            )

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call("basis_projection", operation)

    def basis_correction(
        self,
        basis_rows: Any,
        coefficients: Any,
        vector: Any,
    ) -> None:
        """Compute ``vector -= basis_rows.T @ coefficients`` with one GEMV."""

        basis_rows = self._validate_basis_matrix(
            basis_rows,
            name="basis_rows",
        )
        coefficients = self._validate_vector(
            coefficients,
            name="coefficients",
        )
        vector = self._validate_vector(vector, name="vector")
        if basis_rows.shape != (coefficients.size, vector.size):
            raise ValueError(
                "basis correction shapes are incompatible: "
                f"basis={basis_rows.shape}, coefficients={coefficients.shape}, "
                f"vector={vector.shape}"
            )

        def operation() -> None:
            """Execute the captured vector operation."""
            self._cublas.gemv(
                "T",
                -1.0,
                basis_rows,
                coefficients,
                1.0,
                vector,
            )

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call("basis_correction", operation)

    def copy_device_vector_to_host(
        self,
        source: Any,
        destination: np.ndarray,
    ) -> None:
        """Copy one short device vector into a reusable NumPy buffer.

        The transfer is intentionally blocking because the CPU immediately
        accumulates the coefficients into the Hessenberg column.  CGS/CGS2
        performs one such transfer per orthogonalization pass rather than one
        scalar transfer per basis vector.
        """

        source = self._validate_vector(source, name="source")
        destination = np.asarray(destination)
        expected_dtype = np.dtype(self.dtype.name)
        if destination.dtype != expected_dtype:
            raise TypeError(
                f"destination must have dtype {expected_dtype}, got "
                f"{destination.dtype}"
            )
        if destination.shape != source.shape:
            raise ValueError("source and destination must have equal shape")
        if not destination.flags.c_contiguous:
            raise ValueError("destination must be C-contiguous")

        def operation() -> None:
            """Execute the captured vector operation."""
            source.get(out=destination, blocking=True)

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call(
                "orthogonalization_d2h",
                operation,
                host_synchronizing=True,
            )

    def basis_update(
        self,
        basis_rows: Any,
        coefficients: Any,
        solution: Any,
    ) -> None:
        """Perform ``solution += basis_rows.T @ coefficients`` with GEMV."""

        basis_rows = self._validate_basis_matrix(
            basis_rows,
            name="basis_rows",
        )
        coefficients = self._validate_vector(coefficients, name="coefficients")
        solution = self._validate_vector(solution, name="solution")
        if basis_rows.shape != (coefficients.size, solution.size):
            raise ValueError(
                "basis update shapes are incompatible: "
                f"basis={basis_rows.shape}, coefficients={coefficients.shape}, "
                f"solution={solution.shape}"
            )
        def operation() -> None:
            """Execute the captured vector operation."""
            self._cublas.gemv(
                "T",
                1.0,
                basis_rows,
                coefficients,
                1.0,
                solution,
            )

        if self.profiler is None:
            operation()
        else:
            self.profiler.record_gpu_call("basis_update", operation)



def _orthogonality_metrics_from_gram(
    gram: np.ndarray,
) -> tuple[float, float, float]:
    """Return ``||G-I||_F``, max off-diagonal, and max diagonal error.

    This helper is intentionally NumPy-only so that the metric definition can
    be unit-tested without a CUDA device.  ``gram`` is expected to be the small
    host copy of ``V @ V.T`` for one Arnoldi basis.
    """

    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square matrix")
    if not np.all(np.isfinite(gram)):
        raise ValueError("gram must contain only finite values")
    dimension = gram.shape[0]
    defect = gram - np.eye(dimension, dtype=np.float64)
    frobenius = float(np.linalg.norm(defect, ord="fro"))
    diagonal_error = float(
        np.max(np.abs(np.diag(defect)), initial=0.0)
    )
    offdiagonal = defect.copy()
    if dimension:
        offdiagonal[np.diag_indices(dimension)] = 0.0
    maximum_offdiagonal = float(
        np.max(np.abs(offdiagonal), initial=0.0)
    )
    return frobenius, maximum_offdiagonal, diagonal_error

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


def _resolve_orthogonalization(
    *,
    orthogonalization: CuPyOrthogonalization | None,
    reorthogonalize: bool,
) -> CuPyOrthogonalization:
    """Resolve the new orthogonalization selector and legacy boolean flag."""

    valid = {"mgs", "mgs2", "cgs", "cgs2"}
    if orthogonalization is None:
        return "mgs2" if reorthogonalize else "mgs"
    value = str(orthogonalization).lower()
    if value not in valid:
        raise ValueError(
            "orthogonalization must be one of mgs, mgs2, cgs, or cgs2"
        )
    if reorthogonalize:
        raise ValueError(
            "reorthogonalize cannot be combined with an explicit "
            "orthogonalization mode"
        )
    return value  # type: ignore[return-value]


def _validate_restart_parameters(
    *,
    restart: int,
    max_iterations: int | None,
    num_dofs: int,
    rtol: float,
    atol: float,
    breakdown_tolerance: float | None,
) -> tuple[int, int, float]:
    """Validate shapes, dtypes, devices, and solver parameters."""
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


def _validate_robustness_parameters(
    *,
    check_finite: bool,
    stagnation_cycles: int | None,
    stagnation_tolerance: float,
    divergence_factor: float,
    cgs2_fallback_threshold: float | None,
) -> tuple[bool, int | None, float, float, float | None]:
    """Validate production convergence safeguards without requiring CUDA."""

    check_finite = bool(check_finite)
    if stagnation_cycles is not None:
        if (
            isinstance(stagnation_cycles, bool)
            or int(stagnation_cycles) != stagnation_cycles
            or stagnation_cycles <= 0
        ):
            raise ValueError("stagnation_cycles must be a positive integer or None")
        stagnation_cycles = int(stagnation_cycles)
    stagnation_tolerance = float(stagnation_tolerance)
    if (
        not np.isfinite(stagnation_tolerance)
        or stagnation_tolerance < 0.0
        or stagnation_tolerance >= 1.0
    ):
        raise ValueError("stagnation_tolerance must be finite and in [0, 1)")
    divergence_factor = float(divergence_factor)
    if not np.isfinite(divergence_factor) or divergence_factor <= 1.0:
        raise ValueError("divergence_factor must be finite and greater than one")
    if cgs2_fallback_threshold is not None:
        cgs2_fallback_threshold = float(cgs2_fallback_threshold)
        if (
            not np.isfinite(cgs2_fallback_threshold)
            or cgs2_fallback_threshold < 0.0
        ):
            raise ValueError(
                "cgs2_fallback_threshold must be finite and non-negative or None"
            )
    return (
        check_finite,
        stagnation_cycles,
        stagnation_tolerance,
        divergence_factor,
        cgs2_fallback_threshold,
    )


def _updated_stagnation_count(
    previous_residual: float,
    current_residual: float,
    *,
    tolerance: float,
    previous_count: int,
) -> int:
    """Update the consecutive restart-cycle stagnation counter."""

    previous_residual = float(previous_residual)
    current_residual = float(current_residual)
    if not np.isfinite(previous_residual) or not np.isfinite(current_residual):
        return previous_count + 1
    required = previous_residual * (1.0 - float(tolerance))
    return previous_count + 1 if current_residual >= required else 0


def _termination_reason(status: CuPyGMRESStatus) -> str:
    """Execute the ``_termination_reason`` numerical helper."""
    return {
        "converged": "true residual satisfied the requested tolerance",
        "max_iterations": "maximum iteration count reached",
        "breakdown": "Arnoldi or triangular-solve breakdown",
        "stagnated": "stagnation: true residual failed to improve across restart cycles",
        "diverged": "true residual exceeded the configured divergence limit",
        "non_finite": "a residual, norm, or Hessenberg quantity became non-finite",
    }[status]


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
    orthogonalization: CuPyOrthogonalization | None = None,
    reorthogonalize: bool = False,
    breakdown_tolerance: float | None = None,
    check_finite: bool = True,
    stagnation_cycles: int | None = None,
    stagnation_tolerance: float = 1.0e-3,
    divergence_factor: float = 1.0e6,
    cgs2_fallback_threshold: float | None = None,
    profiler: Any | None = None,
    monitor_orthogonality: bool = False,
    workspace: CuPyGMRESWorkspace | None = None,
    solution_out: Any | None = None,
) -> CuPyGMRESResult:
    r"""Solve ``A x = rhs`` with restarted left-preconditioned GPU GMRES.

    Large vectors and the Arnoldi basis stay on the GPU.  ``mgs`` and
    ``mgs2`` use cuBLAS DOT/AXPY operations.  ``cgs`` and ``cgs2`` form all
    projection coefficients with one cuBLAS GEMV and apply the correction with
    a second GEMV per pass.  CGS modes transfer one short coefficient vector to
    the CPU per pass, rather than one scalar per basis vector.  Norms use
    cuBLAS NRM2 and the restart update uses one cuBLAS GEMV.

    When ``monitor_orthogonality`` is enabled, one small Gram matrix is
    computed and copied to the CPU after every restart cycle.  This diagnostic
    path is intentionally disabled by default because it allocates and
    synchronizes.

    The Givens residual is used as an inexpensive convergence trigger.  A true
    residual is always recomputed after each completed or early-terminated
    restart cycle before convergence is accepted.

    When ``profiler`` is supplied, CUDA events are inserted around each large
    operation and CPU timers are used for the Hessenberg/Givens work.  This is
    diagnostic instrumentation; benchmark an uninstrumented solve separately
    for the primary time-to-solution measurement.

    Pass a :class:`CuPyGMRESWorkspace` to reuse the Arnoldi basis and all work
    buffers across sequential solves.  Supplying ``solution_out`` additionally
    avoids allocating the solution vector; it must not overlap ``rhs``.
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
    if profiler is not None and int(profiler.device_id) != device_id:
        raise ValueError("profiler and operator use different CUDA devices")
    restart, max_iterations, breakdown_tolerance = _validate_restart_parameters(
        restart=restart,
        max_iterations=max_iterations,
        num_dofs=num_dofs,
        rtol=rtol,
        atol=atol,
        breakdown_tolerance=breakdown_tolerance,
    )
    (
        check_finite,
        stagnation_cycles,
        stagnation_tolerance,
        divergence_factor,
        cgs2_fallback_threshold,
    ) = _validate_robustness_parameters(
        check_finite=check_finite,
        stagnation_cycles=stagnation_cycles,
        stagnation_tolerance=stagnation_tolerance,
        divergence_factor=divergence_factor,
        cgs2_fallback_threshold=cgs2_fallback_threshold,
    )

    orthogonalization = _resolve_orthogonalization(
        orthogonalization=orthogonalization,
        reorthogonalize=reorthogonalize,
    )

    if preconditioner is not None:
        if int(preconditioner.num_dofs) != num_dofs:
            raise ValueError("preconditioner size does not match the operator")
        if cp.dtype(preconditioner.dtype) != dtype:
            raise TypeError("preconditioner dtype does not match the operator")
        if int(preconditioner.device_id) != device_id:
            raise ValueError("preconditioner and operator use different devices")

    def validate_vector(vector: Any, *, name: str) -> Any:
        """Validate shapes, dtypes, devices, and solver parameters."""
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
        if solution_out is None:
            if x0 is None:
                x = cp.zeros(num_dofs, dtype=dtype)
            else:
                x0 = validate_vector(x0, name="x0")
                x = x0.reshape(-1).copy()
        else:
            solution_out = validate_vector(solution_out, name="solution_out")
            if solution_out.shape != original_shape:
                raise ValueError("solution_out must have the same shape as rhs")
            if device_arrays_overlap(rhs, solution_out):
                raise ValueError("solution_out must not overlap rhs")
            x = solution_out.reshape(-1)
            if x0 is None:
                x.fill(0)
            else:
                x0 = validate_vector(x0, name="x0")
                x0_flat = x0.reshape(-1)
                same_storage = (
                    int(x0_flat.data.ptr) == int(x.data.ptr)
                    and int(x0_flat.nbytes) == int(x.nbytes)
                )
                if not same_storage and device_arrays_overlap(x0_flat, x):
                    raise ValueError(
                        "x0 and solution_out may be identical but must not "
                        "partially overlap"
                    )
                if not same_storage:
                    cp.copyto(x, x0_flat)

        if workspace is None:
            workspace = CuPyGMRESWorkspace.allocate(
                num_dofs=num_dofs,
                restart_capacity=restart,
                dtype=dtype,
                device_id=device_id,
            )
        else:
            if not isinstance(workspace, CuPyGMRESWorkspace):
                raise TypeError("workspace must be a CuPyGMRESWorkspace")
            workspace.validate_for(
                num_dofs=num_dofs,
                restart=restart,
                dtype=dtype,
                device_id=device_id,
            )

        for workspace_array in workspace.device_arrays:
            if device_arrays_overlap(rhs, workspace_array):
                raise ValueError("rhs must not overlap the GMRES workspace")
            if solution_out is not None and device_arrays_overlap(
                solution_out, workspace_array
            ):
                raise ValueError(
                    "solution_out must not overlap the GMRES workspace"
                )

        blas = CuPyVectorBLAS(
            dtype=dtype,
            device_id=device_id,
            profiler=profiler,
        )
        basis = workspace.basis[: restart + 1]
        matvec_buffer = workspace.matvec_buffer
        residual = workspace.residual
        work = workspace.work
        preconditioned = workspace.preconditioned
        coefficient_device = workspace.coefficient_device[:restart]
        orthogonalization_coefficients_device = (
            workspace.orthogonalization_coefficients_device[:restart]
        )
        orthogonalization_coefficients_host = (
            workspace.orthogonalization_coefficients_host[:restart]
        )

        matvec_count = 0
        preconditioner_count = 0
        dot_count = 0
        axpy_count = 0
        norm_count = 0
        basis_update_count = 0
        basis_projection_count = 0
        basis_correction_count = 0
        coefficient_d2h_count = 0
        true_residual_recomputations = 0
        fallback_count = 0
        active_orthogonalization = orthogonalization
        orthogonality_records: list[CuPyOrthogonalityRecord] = []
        cycle_records: list[CuPyGMRESCycleRecord] = []
        orthogonalization_history: list[CuPyOrthogonalization] = []
        consecutive_stagnation = 0

        def apply_matvec(source: Any, destination: Any) -> None:
            """Apply the face-block matrix-vector operator."""
            nonlocal matvec_count

            def operation() -> None:
                """Execute the captured vector operation."""
                operator.matvec_into(source, destination)

            if profiler is None:
                operation()
            else:
                profiler.record_gpu_call("matvec", operation)
            matvec_count += 1

        def apply_preconditioner(source: Any, destination: Any) -> None:
            """Apply the configured left preconditioner."""
            nonlocal preconditioner_count
            if preconditioner is None:
                blas.copy(source, destination)
            else:
                def operation() -> None:
                    """Execute the captured vector operation."""
                    preconditioner.apply_into(source, destination)

                if profiler is None:
                    operation()
                else:
                    profiler.record_gpu_call("preconditioner", operation)
                preconditioner_count += 1

        def compute_true_residual() -> float:
            """Compute the unpreconditioned true residual."""
            nonlocal axpy_count, norm_count, true_residual_recomputations
            apply_matvec(x, matvec_buffer)
            blas.copy(b, residual)
            blas.axpy(-1.0, matvec_buffer, residual)
            axpy_count += 1
            value = blas.norm(residual)
            norm_count += 1
            true_residual_recomputations += 1
            return value

        b_norm = blas.norm(b)
        norm_count += 1
        machine_epsilon = np.finfo(np.dtype(dtype.name)).eps
        denominator = (
            max(b_norm, machine_epsilon) if np.isfinite(b_norm) else 1.0
        )
        target = (
            max(float(atol), float(rtol) * b_norm)
            if np.isfinite(b_norm)
            else float(atol)
        )

        true_norm = compute_true_residual()
        initial_true_norm = true_norm
        true_history: list[float] = [true_norm]
        estimated_history: list[float] = []

        def make_result(
            status: CuPyGMRESStatus,
            iterations: int,
            cycles: int,
            final_norm: float,
        ) -> CuPyGMRESResult:
            """Build the immutable solver result record."""
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
                orthogonalization=orthogonalization,
                basis_projection_count=basis_projection_count,
                basis_correction_count=basis_correction_count,
                coefficient_d2h_count=coefficient_d2h_count,
                orthogonality_records=tuple(orthogonality_records),
                cycle_records=tuple(cycle_records),
                orthogonalization_history=tuple(orthogonalization_history),
                fallback_count=fallback_count,
                true_residual_recomputations=true_residual_recomputations,
                termination_reason=_termination_reason(status),
            )

        if not np.isfinite(b_norm) or not np.isfinite(true_norm):
            return make_result("non_finite", 0, 0, true_norm)
        if true_norm <= target:
            return make_result("converged", 0, 0, true_norm)

        iterations = 0
        cycles = 0
        while iterations < max_iterations:
            cycles += 1
            cycle_iteration_start = iterations
            cycle_true_start = true_norm
            cycle_orthogonalization = active_orthogonalization
            orthogonalization_history.append(cycle_orthogonalization)
            switched_to_cgs2 = False
            fallback_requested = False
            cycle_orthogonality_record: CuPyOrthogonalityRecord | None = None

            # The true residual from the previous cycle is already in
            # ``residual``. Apply the left preconditioner and normalize it.
            apply_preconditioner(residual, preconditioned)
            beta = blas.norm(preconditioned)
            norm_count += 1
            estimated_history.append(beta)
            if not np.isfinite(beta):
                return make_result("non_finite", iterations, cycles, true_norm)
            if beta <= breakdown_tolerance * max(1.0, true_norm):
                return make_result("breakdown", iterations, cycles, true_norm)

            blas.copy(preconditioned, basis[0])
            blas.scal(1.0 / beta, basis[0])

            cycle_dimension = min(restart, max_iterations - iterations)
            hessenberg = workspace.hessenberg[
                : cycle_dimension + 1, :cycle_dimension
            ]
            cosines = workspace.cosines[:cycle_dimension]
            sines = workspace.sines[:cycle_dimension]
            least_squares_rhs = workspace.least_squares_rhs[
                : cycle_dimension + 1
            ]
            hessenberg.fill(0.0)
            cosines.fill(0.0)
            sines.fill(0.0)
            least_squares_rhs.fill(0.0)
            least_squares_rhs[0] = beta

            used_dimension = 0
            happy_breakdown = False
            for column in range(cycle_dimension):
                apply_matvec(basis[column], matvec_buffer)
                apply_preconditioner(matvec_buffer, work)

                if cycle_orthogonalization in ("mgs", "mgs2"):
                    passes = 2 if cycle_orthogonalization == "mgs2" else 1
                    for _ in range(passes):
                        for row in range(column + 1):
                            coefficient = blas.dot(basis[row], work)
                            dot_count += 1
                            hessenberg[row, column] += coefficient
                            blas.axpy(-coefficient, basis[row], work)
                            axpy_count += 1
                else:
                    passes = 2 if cycle_orthogonalization == "cgs2" else 1
                    active_dimension = column + 1
                    active_basis = basis[:active_dimension]
                    device_coefficients = (
                        orthogonalization_coefficients_device[:active_dimension]
                    )
                    host_coefficients = (
                        orthogonalization_coefficients_host[:active_dimension]
                    )
                    for _ in range(passes):
                        blas.basis_projection(
                            active_basis,
                            work,
                            device_coefficients,
                        )
                        basis_projection_count += 1
                        blas.basis_correction(
                            active_basis,
                            device_coefficients,
                            work,
                        )
                        basis_correction_count += 1
                        blas.copy_device_vector_to_host(
                            device_coefficients,
                            host_coefficients,
                        )
                        coefficient_d2h_count += 1
                        hessenberg[:active_dimension, column] += (
                            host_coefficients
                        )

                next_norm = blas.norm(work)
                norm_count += 1
                if not np.isfinite(next_norm):
                    return make_result("non_finite", iterations, cycles, true_norm)
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

                def update_hessenberg_column() -> float:
                    """Update the projected Krylov data."""
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
                    return abs(least_squares_rhs[column + 1])

                if profiler is None:
                    estimate = update_hessenberg_column()
                else:
                    estimate = profiler.record_cpu_call(
                        "hessenberg_givens",
                        update_hessenberg_column,
                    )
                estimated_history.append(estimate)
                if not np.isfinite(estimate):
                    return make_result("non_finite", iterations, cycles, true_norm)
                if check_finite and not np.all(
                    np.isfinite(hessenberg[: column + 2, column])
                ):
                    return make_result("non_finite", iterations, cycles, true_norm)

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

            needs_orthogonality = monitor_orthogonality or (
                cgs2_fallback_threshold is not None
                and cycle_orthogonalization == "cgs"
            )
            if needs_orthogonality:
                # Include the newly generated Arnoldi vector unless a happy
                # breakdown prevented its normalization.
                basis_dimension = used_dimension + (0 if happy_breakdown else 1)
                active_basis = basis[:basis_dimension]

                def compute_orthogonality_record() -> CuPyOrthogonalityRecord:
                    """Compute the requested numerical diagnostic."""
                    gram_host = cp.asnumpy(active_basis @ active_basis.T)
                    frobenius, offdiagonal, diagonal = (
                        _orthogonality_metrics_from_gram(gram_host)
                    )
                    return CuPyOrthogonalityRecord(
                        restart_cycle=cycles,
                        total_iterations=iterations,
                        basis_dimension=basis_dimension,
                        frobenius_defect=frobenius,
                        maximum_offdiagonal=offdiagonal,
                        maximum_diagonal_error=diagonal,
                    )

                if profiler is None:
                    cycle_orthogonality_record = compute_orthogonality_record()
                else:
                    cycle_orthogonality_record = profiler.record_gpu_call(
                        "orthogonality_gram",
                        compute_orthogonality_record,
                        host_synchronizing=True,
                    )
                orthogonality_records.append(cycle_orthogonality_record)
                fallback_requested = bool(
                    cgs2_fallback_threshold is not None
                    and cycle_orthogonalization == "cgs"
                    and cycle_orthogonality_record.maximum_offdiagonal
                    > cgs2_fallback_threshold
                )

            def solve_small_system() -> np.ndarray:
                """Solve the configured linear system."""
                return _back_substitute_upper(
                    hessenberg[:used_dimension, :used_dimension],
                    least_squares_rhs[:used_dimension],
                    singular_tolerance=breakdown_tolerance,
                )

            try:
                if profiler is None:
                    coefficients = solve_small_system()
                else:
                    coefficients = profiler.record_cpu_call(
                        "back_substitution",
                        solve_small_system,
                    )
            except np.linalg.LinAlgError:
                return make_result("breakdown", iterations, cycles, true_norm)
            if check_finite and not np.all(np.isfinite(coefficients)):
                return make_result("non_finite", iterations, cycles, true_norm)

            coefficient_host = workspace.update_coefficients_host[
                :used_dimension
            ]
            coefficient_host[:] = coefficients

            def transfer_coefficients() -> None:
                """Transfer the small projected coefficients to host storage."""
                coefficient_device[:used_dimension].set(coefficient_host)

            if profiler is None:
                transfer_coefficients()
            else:
                profiler.record_gpu_call(
                    "coefficient_h2d",
                    transfer_coefficients,
                )
            blas.basis_update(
                basis[:used_dimension],
                coefficient_device[:used_dimension],
                x,
            )
            basis_update_count += 1

            true_norm = compute_true_residual()
            true_history.append(true_norm)
            if not np.isfinite(true_norm):
                cycle_records.append(
                    CuPyGMRESCycleRecord(
                        restart_cycle=cycles,
                        iteration_start=cycle_iteration_start,
                        iteration_end=iterations,
                        basis_dimension=used_dimension,
                        orthogonalization=cycle_orthogonalization,
                        true_residual_start=float(cycle_true_start),
                        true_residual_end=float(true_norm),
                        residual_reduction=float("nan"),
                        estimated_residual_end=float(estimated_history[-1]),
                        happy_breakdown=happy_breakdown,
                        stagnation_count=consecutive_stagnation,
                        switched_to_cgs2=switched_to_cgs2,
                    )
                )
                return make_result("non_finite", iterations, cycles, true_norm)

            converged_this_cycle = true_norm <= target
            if fallback_requested and not converged_this_cycle:
                active_orthogonalization = "cgs2"
                fallback_count += 1
                switched_to_cgs2 = True

            residual_reduction = (
                true_norm / cycle_true_start
                if cycle_true_start > 0.0
                else (0.0 if true_norm == 0.0 else float("inf"))
            )
            consecutive_stagnation = _updated_stagnation_count(
                cycle_true_start,
                true_norm,
                tolerance=stagnation_tolerance,
                previous_count=consecutive_stagnation,
            )
            cycle_records.append(
                CuPyGMRESCycleRecord(
                    restart_cycle=cycles,
                    iteration_start=cycle_iteration_start,
                    iteration_end=iterations,
                    basis_dimension=used_dimension,
                    orthogonalization=cycle_orthogonalization,
                    true_residual_start=float(cycle_true_start),
                    true_residual_end=float(true_norm),
                    residual_reduction=float(residual_reduction),
                    estimated_residual_end=float(estimated_history[-1]),
                    happy_breakdown=happy_breakdown,
                    stagnation_count=consecutive_stagnation,
                    switched_to_cgs2=switched_to_cgs2,
                )
            )

            if converged_this_cycle:
                return make_result("converged", iterations, cycles, true_norm)
            if true_norm > divergence_factor * max(initial_true_norm, target):
                return make_result("diverged", iterations, cycles, true_norm)
            if (
                stagnation_cycles is not None
                and consecutive_stagnation >= stagnation_cycles
            ):
                return make_result("stagnated", iterations, cycles, true_norm)
            if happy_breakdown:
                return make_result("breakdown", iterations, cycles, true_norm)

        return make_result("max_iterations", iterations, cycles, true_norm)


__all__ = [
    "CuPyGMRESCycleRecord",
    "CuPyGMRESResult",
    "CuPyGMRESWorkspace",
    "CuPyRestartedGMRESSolver",
    "CuPyOrthogonalityRecord",
    "CuPyOrthogonalization",
    "CuPyVectorBLAS",
    "DeviceMatvecOperator",
    "DevicePreconditioner",
    "_orthogonality_metrics_from_gram",
    "_resolve_orthogonalization",
    "_termination_reason",
    "_updated_stagnation_count",
    "_validate_robustness_parameters",
    "restarted_gmres_cupy",
]