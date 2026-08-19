"""CUDA-event profiling helpers for the face-dense HDG solver.

The routines in this module are deliberately separate from the validated
numerical kernels.  They benchmark preallocated calls with CUDA events, expose
stage-level timings for the face-dense operator and preconditioners, and can
optionally instrument GPU GMRES.

Two timing modes are important:

``benchmark_cuda_call``
    Low-perturbation timing for a repeated preallocated device operation.  All
    event pairs are recorded first and synchronized once at the end.

``CuPyGMRESProfiler``
    Fine-grained diagnostic instrumentation.  It records an event pair around
    every BLAS/operator call.  This is useful for attribution, but the event
    insertion overhead means its total solve time must not be used as the main
    time-to-solution number.  Measure an uninstrumented solve separately.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Iterator

import numpy as np

from .cupy import require_cupy_device, solve_batched_vectors


@dataclass(frozen=True)
class CudaTimingStats:
    """Summary of repeated CUDA-event samples, in milliseconds."""

    samples_ms: np.ndarray
    warmup: int
    repeats: int

    def __post_init__(self) -> None:
        """Validate and normalize the initialized data."""
        samples = np.asarray(self.samples_ms, dtype=np.float64)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("samples_ms must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(samples)) or np.any(samples < 0.0):
            raise ValueError("samples_ms must contain finite non-negative values")
        if int(self.warmup) < 0 or int(self.repeats) <= 0:
            raise ValueError("warmup must be non-negative and repeats positive")
        if samples.size != int(self.repeats):
            raise ValueError("samples_ms size must equal repeats")
        object.__setattr__(self, "samples_ms", np.ascontiguousarray(samples))
        object.__setattr__(self, "warmup", int(self.warmup))
        object.__setattr__(self, "repeats", int(self.repeats))

    @property
    def minimum_ms(self) -> float:
        """Return the minimum recorded time in milliseconds."""
        return float(np.min(self.samples_ms))

    @property
    def median_ms(self) -> float:
        """Return the median recorded time in milliseconds."""
        return float(np.median(self.samples_ms))

    @property
    def mean_ms(self) -> float:
        """Return the mean recorded time in milliseconds."""
        return float(np.mean(self.samples_ms))

    @property
    def standard_deviation_ms(self) -> float:
        """Return the timing standard deviation in milliseconds."""
        return float(np.std(self.samples_ms))

    @property
    def p90_ms(self) -> float:
        """Return the ninetieth-percentile time in milliseconds."""
        return float(np.percentile(self.samples_ms, 90.0))

    def to_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {
            "warmup": self.warmup,
            "repeats": self.repeats,
            "minimum_ms": self.minimum_ms,
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "standard_deviation_ms": self.standard_deviation_ms,
            "p90_ms": self.p90_ms,
        }
        if include_samples:
            result["samples_ms"] = self.samples_ms.tolist()
        return result


@dataclass(frozen=True)
class SetupTiming:
    """Synchronized wall-clock setup timing."""

    elapsed_ms: float

    def __post_init__(self) -> None:
        """Validate and normalize the initialized data."""
        if not np.isfinite(self.elapsed_ms) or self.elapsed_ms < 0.0:
            raise ValueError("elapsed_ms must be finite and non-negative")


@dataclass(frozen=True)
class FaceDenseOperatorProfile:
    """Timing breakdown for one GPU face-dense operator."""

    total: CudaTimingStats
    gather: CudaTimingStats
    dense_product: CudaTimingStats
    estimated_flops: int
    matrix_bytes: int
    vector_bytes: int

    @property
    def median_gflops(self) -> float:
        """Return the median measured throughput in GFLOP/s."""
        seconds = self.total.median_ms * 1.0e-3
        return 0.0 if seconds == 0.0 else self.estimated_flops / seconds / 1.0e9

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "total": self.total.to_dict(),
            "gather": self.gather.to_dict(),
            "dense_product": self.dense_product.to_dict(),
            "estimated_flops": self.estimated_flops,
            "matrix_bytes": self.matrix_bytes,
            "vector_bytes": self.vector_bytes,
            "median_gflops": self.median_gflops,
        }


@dataclass(frozen=True)
class AdditiveSchwarzProfile:
    """Timing breakdown for one GPU ASM application."""

    total: CudaTimingStats
    restriction: CudaTimingStats
    local_solve: CudaTimingStats
    prolongation: CudaTimingStats

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "total": self.total.to_dict(),
            "restriction": self.restriction.to_dict(),
            "local_solve": self.local_solve.to_dict(),
            "prolongation": self.prolongation.to_dict(),
        }


@dataclass(frozen=True)
class GMRESOperationTiming:
    """Aggregated timing for one operation category inside GPU GMRES."""

    category: str
    count: int
    gpu_time_ms: float
    host_wall_time_ms: float
    synchronizing_host_time_ms: float

    @property
    def mean_gpu_time_ms(self) -> float:
        """Return the mean recorded GPU time in milliseconds."""
        return 0.0 if self.count == 0 else self.gpu_time_ms / self.count

    @property
    def estimated_host_sync_overhead_ms(self) -> float:
        """Return the estimated host synchronization overhead in milliseconds."""
        return max(0.0, self.synchronizing_host_time_ms - self.gpu_time_ms)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            **asdict(self),
            "mean_gpu_time_ms": self.mean_gpu_time_ms,
            "estimated_host_sync_overhead_ms": self.estimated_host_sync_overhead_ms,
        }


@dataclass(frozen=True)
class GMRESProfilingSummary:
    """Fine-grained GMRES operation attribution."""

    operations: tuple[GMRESOperationTiming, ...]
    cpu_times_ms: dict[str, float]

    @property
    def total_gpu_operation_ms(self) -> float:
        """Return total attributed GPU-operation time in milliseconds."""
        return float(sum(item.gpu_time_ms for item in self.operations))

    @property
    def total_cpu_small_system_ms(self) -> float:
        """Return total CPU small-system time in milliseconds."""
        return float(sum(self.cpu_times_ms.values()))

    def operation(self, category: str) -> GMRESOperationTiming | None:
        """Execute the captured vector operation."""
        for item in self.operations:
            if item.category == category:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "operations": [item.to_dict() for item in self.operations],
            "cpu_times_ms": dict(self.cpu_times_ms),
            "total_gpu_operation_ms": self.total_gpu_operation_ms,
            "total_cpu_small_system_ms": self.total_cpu_small_system_ms,
        }


@dataclass
class _GPURecord:
    category: str
    start: Any
    stop: Any
    host_wall_ms: float
    host_synchronizing: bool


class CuPyGMRESProfiler:
    """Record CUDA events and CPU timings during one GPU GMRES solve.

    The profiler is intentionally opt-in.  Event insertion around every BLAS
    operation perturbs the solve, so use :func:`benchmark_cuda_call` around an
    *uninstrumented* GMRES call for the primary time-to-solution number.
    """

    def __init__(self, *, device_id: int | None = None) -> None:
        """Initialize the instance and validate persistent storage."""
        cp = require_cupy_device()
        self._cp = cp
        self.device_id = (
            int(cp.cuda.Device().id) if device_id is None else int(device_id)
        )
        self._gpu_records: list[_GPURecord] = []
        self._cpu_times_ms: defaultdict[str, float] = defaultdict(float)
        self._finalized = False

    def record_gpu_call(
        self,
        category: str,
        function: Callable[[], Any],
        *,
        host_synchronizing: bool = False,
    ) -> Any:
        """Record one attributed operation sample."""
        if self._finalized:
            raise RuntimeError("cannot record after profiler finalization")
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            start = cp.cuda.Event()
            stop = cp.cuda.Event()
            start.record()
            wall_start = perf_counter()
            result = function()
            host_wall_ms = 1.0e3 * (perf_counter() - wall_start)
            stop.record()
        self._gpu_records.append(
            _GPURecord(
                category=str(category),
                start=start,
                stop=stop,
                host_wall_ms=float(host_wall_ms),
                host_synchronizing=bool(host_synchronizing),
            )
        )
        return result

    @contextmanager
    def cpu_timer(self, category: str) -> Iterator[None]:
        """Measure an attributed CPU operation with a context manager."""
        if self._finalized:
            raise RuntimeError("cannot record after profiler finalization")
        start = perf_counter()
        try:
            yield
        finally:
            self._cpu_times_ms[str(category)] += 1.0e3 * (perf_counter() - start)

    def record_cpu_call(self, category: str, function: Callable[[], Any]) -> Any:
        """Record one attributed operation sample."""
        with self.cpu_timer(category):
            return function()

    def finalize(self) -> GMRESProfilingSummary:
        """Finalize and return the collected profiling summary."""
        if self._finalized:
            raise RuntimeError("profiler has already been finalized")
        self._finalized = True
        cp = self._cp
        with cp.cuda.Device(self.device_id):
            cp.cuda.get_current_stream().synchronize()

        grouped: dict[str, dict[str, float | int]] = {}
        for record in self._gpu_records:
            values = grouped.setdefault(
                record.category,
                {
                    "count": 0,
                    "gpu_time_ms": 0.0,
                    "host_wall_time_ms": 0.0,
                    "synchronizing_host_time_ms": 0.0,
                },
            )
            values["count"] = int(values["count"]) + 1
            values["gpu_time_ms"] = float(values["gpu_time_ms"]) + float(
                cp.cuda.get_elapsed_time(record.start, record.stop)
            )
            values["host_wall_time_ms"] = float(values["host_wall_time_ms"]) + (
                record.host_wall_ms
            )
            if record.host_synchronizing:
                values["synchronizing_host_time_ms"] = float(
                    values["synchronizing_host_time_ms"]
                ) + record.host_wall_ms

        operations = tuple(
            GMRESOperationTiming(
                category=category,
                count=int(values["count"]),
                gpu_time_ms=float(values["gpu_time_ms"]),
                host_wall_time_ms=float(values["host_wall_time_ms"]),
                synchronizing_host_time_ms=float(
                    values["synchronizing_host_time_ms"]
                ),
            )
            for category, values in sorted(grouped.items())
        )
        return GMRESProfilingSummary(
            operations=operations,
            cpu_times_ms=dict(sorted(self._cpu_times_ms.items())),
        )


def benchmark_cuda_call(
    function: Callable[[], Any],
    *,
    warmup: int = 10,
    repeats: int = 100,
    device_id: int | None = None,
) -> CudaTimingStats:
    """Benchmark a preallocated CUDA call using one synchronization boundary.

    The callable must enqueue its work on the current stream.  Event objects are
    allocated before timing, all repetitions are then recorded back-to-back,
    and the stream is synchronized once.  This avoids a host synchronization
    between individual samples.
    """

    if isinstance(warmup, bool) or int(warmup) != warmup or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    warmup = int(warmup)
    repeats = int(repeats)

    cp = require_cupy_device()
    selected_device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
    with cp.cuda.Device(selected_device):
        stream = cp.cuda.get_current_stream()
        for _ in range(warmup):
            function()
        stream.synchronize()

        starts = [cp.cuda.Event() for _ in range(repeats)]
        stops = [cp.cuda.Event() for _ in range(repeats)]
        for index in range(repeats):
            starts[index].record()
            function()
            stops[index].record()
        stream.synchronize()
        samples = np.asarray(
            [cp.cuda.get_elapsed_time(start, stop) for start, stop in zip(starts, stops)],
            dtype=np.float64,
        )
    return CudaTimingStats(samples_ms=samples, warmup=warmup, repeats=repeats)


def time_synchronized_setup(
    builder: Callable[[], Any],
    *,
    device_id: int | None = None,
) -> tuple[Any, SetupTiming]:
    """Run one setup callable with synchronization before and after it."""

    cp = require_cupy_device()
    selected_device = int(cp.cuda.Device().id) if device_id is None else int(device_id)
    with cp.cuda.Device(selected_device):
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        start = perf_counter()
        result = builder()
        stream.synchronize()
        elapsed_ms = 1.0e3 * (perf_counter() - start)
    return result, SetupTiming(elapsed_ms=float(elapsed_ms))


def _operator_dense_product_into(operator: Any, out: Any) -> None:
    """Apply only the dense-product stage to an already gathered vector."""

    cp = operator._cp
    out_faces = operator._validate_device_vector(out, name="out")
    if operator.implementation == "matmul":
        cp.matmul(
            operator.matrix_batches,
            operator.x_extended.reshape(
                operator.num_rows,
                operator.extended_size,
                1,
            ),
            out=operator._matmul_output,
        )
        out_faces[...] = operator._matmul_output[:, :, 0]
        return

    threads = 256
    blocks = (operator.num_dofs + threads - 1) // threads
    if operator._raw_matvec_kernel is None:
        raise RuntimeError("raw matvec kernel has not been initialized")
    operator._raw_matvec_kernel(
        (blocks,),
        (threads,),
        (
            np.int32(operator.num_rows),
            np.int32(operator.block_size),
            np.int32(operator.extended_size),
            operator.matrix_batches,
            operator.x_extended,
            out_faces,
        ),
    )


def _zero_cuda_timing(*, warmup: int, repeats: int) -> CudaTimingStats:
    """Return an explicit zero-cost stage for a fused implementation."""

    return CudaTimingStats(
        samples_ms=np.zeros(int(repeats), dtype=np.float64),
        warmup=int(warmup),
        repeats=int(repeats),
    )


def profile_face_dense_operator(
    operator: Any,
    x: Any,
    out: Any,
    *,
    warmup: int = 10,
    repeats: int = 100,
) -> FaceDenseOperatorProfile:
    """Profile gather, dense product, and complete face-dense matvec."""

    operator._validate_device_vector(x, name="x")
    operator._validate_device_vector(out, name="out")

    if operator.implementation == "raw_fused":
        gather = _zero_cuda_timing(warmup=warmup, repeats=repeats)
        dense = _zero_cuda_timing(warmup=warmup, repeats=repeats)
    else:
        gather = benchmark_cuda_call(
            lambda: operator.gather_neighbors_into(x, operator.x_extended),
            warmup=warmup,
            repeats=repeats,
            device_id=operator.device_id,
        )
        # Ensure the dense stage sees a valid gathered vector before its warmup.
        operator.gather_neighbors_into(x, operator.x_extended)
        dense = benchmark_cuda_call(
            lambda: _operator_dense_product_into(operator, out),
            warmup=warmup,
            repeats=repeats,
            device_id=operator.device_id,
        )
    total = benchmark_cuda_call(
        lambda: operator.matvec_into(x, out),
        warmup=warmup,
        repeats=repeats,
        device_id=operator.device_id,
    )

    itemsize = int(np.dtype(operator.dtype.name).itemsize)
    flops = int(2 * operator.num_rows * operator.block_size * operator.extended_size)
    matrix_bytes = int(operator.matrix_batches.size * itemsize)
    if operator.implementation == "raw_fused":
        vector_bytes = int(2 * operator.num_dofs * itemsize)
    else:
        vector_bytes = int(
            (operator.num_dofs + operator.x_extended.size + operator.num_dofs)
            * itemsize
        )
    return FaceDenseOperatorProfile(
        total=total,
        gather=gather,
        dense_product=dense,
        estimated_flops=flops,
        matrix_bytes=matrix_bytes,
        vector_bytes=vector_bytes,
    )


def profile_block_jacobi(
    preconditioner: Any,
    x: Any,
    out: Any,
    *,
    warmup: int = 10,
    repeats: int = 100,
) -> CudaTimingStats:
    """Profile one complete block-Jacobi application."""

    return benchmark_cuda_call(
        lambda: preconditioner.apply_into(x, out),
        warmup=warmup,
        repeats=repeats,
        device_id=preconditioner.device_id,
    )


def _asm_local_solve_into(preconditioner: Any) -> None:
    """Execute the ``_asm_local_solve_into`` numerical helper."""
    cp = preconditioner._cp
    if preconditioner.application == "fused":
        raise RuntimeError("fused ASM local application requires the global input")
    if preconditioner._element_rhs is None:
        raise RuntimeError("ASM restricted workspace is unavailable")
    if preconditioner.local_solver == "gpu_solve":
        solved = solve_batched_vectors(
            cp,
            preconditioner.local_matrices,
            preconditioner._element_rhs,
        )
        preconditioner._local_output[...] = solved
        return
    if preconditioner.application == "raw":
        from .cupy_preconditionners import _launch_raw_batched_mv

        if preconditioner._raw_apply_kernel is None:
            raise RuntimeError("raw ASM application kernel is unavailable")
        _launch_raw_batched_mv(
            preconditioner._raw_apply_kernel,
            preconditioner.inverse_matrices,
            preconditioner._element_rhs,
            preconditioner._local_output,
            matrix_size=preconditioner.local_size,
        )
        return
    cp.matmul(
        preconditioner.inverse_matrices,
        preconditioner._element_rhs.reshape(
            preconditioner.num_elements,
            preconditioner.local_size,
            1,
        ),
        out=preconditioner._local_output.reshape(
            preconditioner.num_elements,
            preconditioner.local_size,
            1,
        ),
    )


def profile_additive_schwarz(
    preconditioner: Any,
    x: Any,
    out: Any,
    *,
    warmup: int = 10,
    repeats: int = 100,
) -> AdditiveSchwarzProfile:
    """Profile restriction, local dense operation, prolongation, and total ASM."""

    if preconditioner.application == "fused":
        restriction = _zero_cuda_timing(warmup=warmup, repeats=repeats)
        local_solve = benchmark_cuda_call(
            lambda: preconditioner.fused_local_into(
                x,
                preconditioner.local_solution_buffer,
            ),
            warmup=warmup,
            repeats=repeats,
            device_id=preconditioner.device_id,
        )
        preconditioner.fused_local_into(
            x,
            preconditioner.local_solution_buffer,
        )
    else:
        restricted = preconditioner.restricted_buffer
        if restricted is None:
            raise RuntimeError("ASM restricted workspace is unavailable")
        restriction = benchmark_cuda_call(
            lambda: preconditioner.restrict_into(x, restricted),
            warmup=warmup,
            repeats=repeats,
            device_id=preconditioner.device_id,
        )
        preconditioner.restrict_into(x, restricted)
        local_solve = benchmark_cuda_call(
            lambda: _asm_local_solve_into(preconditioner),
            warmup=warmup,
            repeats=repeats,
            device_id=preconditioner.device_id,
        )
        _asm_local_solve_into(preconditioner)
    prolongation = benchmark_cuda_call(
        lambda: preconditioner.prolong_into(
            preconditioner.local_solution_buffer,
            out,
        ),
        warmup=warmup,
        repeats=repeats,
        device_id=preconditioner.device_id,
    )
    total = benchmark_cuda_call(
        lambda: preconditioner.apply_into(x, out),
        warmup=warmup,
        repeats=repeats,
        device_id=preconditioner.device_id,
    )
    return AdditiveSchwarzProfile(
        total=total,
        restriction=restriction,
        local_solve=local_solve,
        prolongation=prolongation,
    )


__all__ = [
    "AdditiveSchwarzProfile",
    "CudaTimingStats",
    "CuPyGMRESProfiler",
    "FaceDenseOperatorProfile",
    "GMRESOperationTiming",
    "GMRESProfilingSummary",
    "SetupTiming",
    "benchmark_cuda_call",
    "profile_additive_schwarz",
    "profile_block_jacobi",
    "profile_face_dense_operator",
    "time_synchronized_setup",
]