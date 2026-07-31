"""Architecture-specific autotuning and persistent selection cache.

The face-dense CUDA kernels have different performance crossovers depending on
GPU architecture, polynomial order, dtype, and problem size.  This module
benchmarks numerically equivalent candidates with interleaved CUDA-event
measurements, selects the lowest median time, and can persist that decision in
an atomic JSON cache for later runs on the same device/problem configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .cupy import require_cupy_device
from .cupy_face_dense import CuPyFaceDenseOperator
from .cupy_preconditionners import CuPyFaceAdditiveSchwarzPreconditioner
from .cupy_profiling import CudaTimingStats

AUTOTUNE_CACHE_SCHEMA_VERSION = 1
AUTOTUNE_KERNEL_ABI_VERSION = 2


@dataclass(frozen=True)
class KernelCandidateTiming:
    name: str
    median_ms: float
    minimum_ms: float
    workspace_bytes: int
    relative_error: float
    mean_ms: float | None = None
    standard_deviation_ms: float | None = None
    p90_ms: float | None = None
    repeats: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KernelCandidateTiming":
        return cls(
            name=str(payload["name"]),
            median_ms=float(payload["median_ms"]),
            minimum_ms=float(payload["minimum_ms"]),
            workspace_bytes=int(payload["workspace_bytes"]),
            relative_error=float(payload["relative_error"]),
            mean_ms=(
                None if payload.get("mean_ms") is None else float(payload["mean_ms"])
            ),
            standard_deviation_ms=(
                None
                if payload.get("standard_deviation_ms") is None
                else float(payload["standard_deviation_ms"])
            ),
            p90_ms=(
                None if payload.get("p90_ms") is None else float(payload["p90_ms"])
            ),
            repeats=(
                None if payload.get("repeats") is None else int(payload["repeats"])
            ),
        )


@dataclass(frozen=True)
class FaceDenseAutotuneResult:
    device_name: str
    dtype: str
    num_dofs: int
    block_size: int
    operator_choice: str
    asm_choice: str | None
    operator_candidates: tuple[KernelCandidateTiming, ...]
    asm_candidates: tuple[KernelCandidateTiming, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_name": self.device_name,
            "dtype": self.dtype,
            "num_dofs": self.num_dofs,
            "block_size": self.block_size,
            "operator_choice": self.operator_choice,
            "asm_choice": self.asm_choice,
            "operator_candidates": [x.to_dict() for x in self.operator_candidates],
            "asm_candidates": [x.to_dict() for x in self.asm_candidates],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FaceDenseAutotuneResult":
        return cls(
            device_name=str(payload["device_name"]),
            dtype=str(payload["dtype"]),
            num_dofs=int(payload["num_dofs"]),
            block_size=int(payload["block_size"]),
            operator_choice=str(payload["operator_choice"]),
            asm_choice=(
                None if payload.get("asm_choice") is None else str(payload["asm_choice"])
            ),
            operator_candidates=tuple(
                KernelCandidateTiming.from_dict(item)
                for item in payload.get("operator_candidates", ())
            ),
            asm_candidates=tuple(
                KernelCandidateTiming.from_dict(item)
                for item in payload.get("asm_candidates", ())
            ),
        )


@dataclass(frozen=True)
class CUDADeviceFingerprint:
    device_name: str
    compute_capability: str
    total_global_memory: int
    driver_version: int
    runtime_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaceDenseAutotuneKey:
    kernel_abi_version: int
    device: CUDADeviceFingerprint
    dtype: str
    num_rows: int
    num_slots: int
    block_size: int
    boundary_mode: str
    polynomial_order: int | None
    local_solver: str
    operator_implementations: tuple[str, ...]
    asm_applications: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_abi_version": self.kernel_abi_version,
            "device": self.device.to_dict(),
            "dtype": self.dtype,
            "num_rows": self.num_rows,
            "num_slots": self.num_slots,
            "block_size": self.block_size,
            "boundary_mode": self.boundary_mode,
            "polynomial_order": self.polynomial_order,
            "local_solver": self.local_solver,
            "operator_implementations": list(self.operator_implementations),
            "asm_applications": list(self.asm_applications),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FaceDenseAutotuneKey":
        device_payload = payload["device"]
        device = CUDADeviceFingerprint(
            device_name=str(device_payload["device_name"]),
            compute_capability=str(device_payload["compute_capability"]),
            total_global_memory=int(device_payload["total_global_memory"]),
            driver_version=int(device_payload["driver_version"]),
            runtime_version=int(device_payload["runtime_version"]),
        )
        return cls(
            kernel_abi_version=int(payload["kernel_abi_version"]),
            device=device,
            dtype=str(payload["dtype"]),
            num_rows=int(payload["num_rows"]),
            num_slots=int(payload["num_slots"]),
            block_size=int(payload["block_size"]),
            boundary_mode=str(payload["boundary_mode"]),
            polynomial_order=(
                None
                if payload.get("polynomial_order") is None
                else int(payload["polynomial_order"])
            ),
            local_solver=str(payload["local_solver"]),
            operator_implementations=tuple(
                str(item) for item in payload.get("operator_implementations", ())
            ),
            asm_applications=tuple(
                str(item) for item in payload.get("asm_applications", ())
            ),
        )

    @property
    def cache_id(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class CachedFaceDenseAutotuneResult:
    key: FaceDenseAutotuneKey
    result: FaceDenseAutotuneResult
    cache_hit: bool
    cache_path: Path | None


def default_autotune_cache_path() -> Path:
    configured = os.environ.get("HDGFEM_AUTOTUNE_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "hdgfem" / "face_dense_autotune.json"


class PersistentFaceDenseAutotuneCache:
    """Small atomic JSON cache for architecture-specific kernel decisions."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = (
            default_autotune_cache_path()
            if path is None
            else Path(path).expanduser()
        )

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "schema_version": AUTOTUNE_CACHE_SCHEMA_VERSION,
            "entries": {},
        }

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_payload()
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return self._empty_payload()
        if payload.get("schema_version") != AUTOTUNE_CACHE_SCHEMA_VERSION:
            return self._empty_payload()
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return self._empty_payload()
        return payload

    def get(self, key: FaceDenseAutotuneKey) -> FaceDenseAutotuneResult | None:
        payload = self._read_payload()
        entry = payload["entries"].get(key.cache_id)
        if not isinstance(entry, dict):
            return None
        try:
            stored_key = FaceDenseAutotuneKey.from_dict(entry["key"])
            if stored_key != key:
                return None
            result = FaceDenseAutotuneResult.from_dict(entry["result"])
        except (KeyError, TypeError, ValueError):
            return None
        requested_operators = set(key.operator_implementations)
        requested_asm = set(key.asm_applications)
        if result.operator_choice not in requested_operators:
            return None
        if result.asm_choice is not None and result.asm_choice not in requested_asm:
            return None
        return result

    def put(
        self,
        key: FaceDenseAutotuneKey,
        result: FaceDenseAutotuneResult,
    ) -> None:
        payload = self._read_payload()
        payload["entries"][key.cache_id] = {
            "key": key.to_dict(),
            "result": result.to_dict(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.tmp-{os.getpid()}"
        )
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.path)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def entry_count(self) -> int:
        return len(self._read_payload()["entries"])


def _relative_device_error(cp: Any, actual: Any, reference: Any) -> float:
    difference = cp.linalg.norm(actual - reference)
    denominator = cp.linalg.norm(reference)
    eps = cp.finfo(reference.dtype).eps
    return float((difference / cp.maximum(denominator, eps)).item())


def _device_name(cp: Any, device_id: int) -> str:
    name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def _device_fingerprint(cp: Any, device_id: int) -> CUDADeviceFingerprint:
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    return CUDADeviceFingerprint(
        device_name=_device_name(cp, device_id),
        compute_capability=f"{int(properties['major'])}.{int(properties['minor'])}",
        total_global_memory=int(properties["totalGlobalMem"]),
        driver_version=int(cp.cuda.runtime.driverGetVersion()),
        runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
    )


def build_face_dense_autotune_key(
    system: Any,
    *,
    dtype: Any,
    device_id: int,
    polynomial_order: int | None,
    local_solver: str,
    operator_implementations: Iterable[str],
    asm_applications: Iterable[str],
) -> FaceDenseAutotuneKey:
    cp = require_cupy_device()
    dtype = cp.dtype(dtype)
    operators = tuple(str(item) for item in operator_implementations)
    asm = tuple(str(item) for item in asm_applications)
    return FaceDenseAutotuneKey(
        kernel_abi_version=AUTOTUNE_KERNEL_ABI_VERSION,
        device=_device_fingerprint(cp, int(device_id)),
        dtype=dtype.name,
        num_rows=int(system.num_rows),
        num_slots=int(system.num_slots),
        block_size=int(system.block_size),
        boundary_mode=str(getattr(system, "mode", "unknown")),
        polynomial_order=(
            None if polynomial_order is None else int(polynomial_order)
        ),
        local_solver=str(local_solver),
        operator_implementations=operators,
        asm_applications=asm,
    )


def _benchmark_interleaved_cuda_calls(
    calls: Mapping[str, Callable[[], Any]],
    *,
    warmup: int,
    repeats: int,
    device_id: int,
) -> dict[str, CudaTimingStats]:
    """Benchmark candidates in alternating order to reduce clock/order bias."""

    if not calls:
        return {}
    if isinstance(warmup, bool) or int(warmup) != warmup or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    warmup = int(warmup)
    repeats = int(repeats)

    cp = require_cupy_device()
    names = tuple(calls)
    with cp.cuda.Device(int(device_id)):
        stream = cp.cuda.get_current_stream()
        for index in range(warmup):
            order = names if index % 2 == 0 else tuple(reversed(names))
            for name in order:
                calls[name]()
        stream.synchronize()

        events: dict[str, list[tuple[Any, Any]]] = {
            name: [] for name in names
        }
        for index in range(repeats):
            order = names if index % 2 == 0 else tuple(reversed(names))
            for name in order:
                start = cp.cuda.Event()
                stop = cp.cuda.Event()
                start.record()
                calls[name]()
                stop.record()
                events[name].append((start, stop))
        stream.synchronize()

        result: dict[str, CudaTimingStats] = {}
        for name in names:
            samples = np.asarray(
                [
                    cp.cuda.get_elapsed_time(start, stop)
                    for start, stop in events[name]
                ],
                dtype=np.float64,
            )
            result[name] = CudaTimingStats(
                samples_ms=samples,
                warmup=warmup,
                repeats=repeats,
            )
        return result


def _timing_row(
    *,
    name: str,
    timing: CudaTimingStats,
    workspace_bytes: int,
    relative_error: float,
) -> KernelCandidateTiming:
    return KernelCandidateTiming(
        name=name,
        median_ms=timing.median_ms,
        minimum_ms=timing.minimum_ms,
        workspace_bytes=int(workspace_bytes),
        relative_error=float(relative_error),
        mean_ms=timing.mean_ms,
        standard_deviation_ms=timing.standard_deviation_ms,
        p90_ms=timing.p90_ms,
        repeats=timing.repeats,
    )


def autotune_face_dense_gpu(
    system: Any,
    *,
    element_blocks: np.ndarray | None = None,
    loc2glob_face: np.ndarray | None = None,
    dtype: Any = np.float64,
    device_id: int = 0,
    operator_implementations: Iterable[str] = ("raw", "raw_fused"),
    asm_applications: Iterable[str] = ("raw", "fused"),
    local_solver: str = "cublas_inverse",
    warmup: int = 10,
    repeats: int = 50,
    validation_tolerance: float | None = None,
) -> FaceDenseAutotuneResult:
    """Benchmark and select operator/ASM application kernels.

    Candidate calls are measured in alternating order within one event stream,
    reducing the bias caused by thermal state, GPU clocks, and always timing
    one implementation first.  Setup/factorization time remains excluded.
    """

    cp = require_cupy_device()
    dtype = cp.dtype(dtype)
    if dtype not in (cp.float32, cp.float64):
        raise TypeError("dtype must be float32 or float64")
    tolerance = (
        (5.0e-5 if dtype == cp.float32 else 5.0e-12)
        if validation_tolerance is None
        else float(validation_tolerance)
    )
    operator_names = tuple(str(item) for item in operator_implementations)
    asm_names = tuple(str(item) for item in asm_applications)
    if not operator_names:
        raise ValueError("at least one operator implementation is required")

    with cp.cuda.Device(int(device_id)):
        rng = np.random.default_rng(20260730)
        x = cp.asarray(rng.standard_normal(system.rhs.shape), dtype=dtype)

        operators: dict[str, Any] = {}
        operator_outputs: dict[str, Any] = {}
        operator_errors: dict[str, float] = {}
        reference_out = None
        for name in operator_names:
            operator = CuPyFaceDenseOperator.from_system(
                system,
                implementation=name,
                dtype=dtype,
                device_id=device_id,
            )
            operators[name] = operator
            out = cp.empty_like(x)
            operator_outputs[name] = out
            operator.matvec_into(x, out)
            cp.cuda.get_current_stream().synchronize()
            if reference_out is None:
                reference_out = out.copy()
                error = 0.0
            else:
                error = _relative_device_error(cp, out, reference_out)
                if error > tolerance:
                    raise AssertionError(
                        f"operator candidate {name!r} differs from reference by {error:.3e}"
                    )
            operator_errors[name] = error

        operator_timings = _benchmark_interleaved_cuda_calls(
            {
                name: (
                    lambda op=operators[name], target=operator_outputs[name]:
                    op.matvec_into(x, target)
                )
                for name in operator_names
            },
            warmup=warmup,
            repeats=repeats,
            device_id=device_id,
        )
        operator_rows = [
            _timing_row(
                name=name,
                timing=operator_timings[name],
                workspace_bytes=int(operators[name].workspace_bytes),
                relative_error=operator_errors[name],
            )
            for name in operator_names
        ]
        operator_choice = min(operator_rows, key=lambda item: item.median_ms).name

        asm_rows: list[KernelCandidateTiming] = []
        asm_choice = None
        if element_blocks is not None or loc2glob_face is not None:
            if element_blocks is None or loc2glob_face is None:
                raise ValueError("element_blocks and loc2glob_face must be provided together")
            preconditioners: dict[str, Any] = {}
            asm_outputs: dict[str, Any] = {}
            asm_errors: dict[str, float] = {}
            asm_reference = None
            for name in asm_names:
                preconditioner = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                    system,
                    element_blocks,
                    loc2glob_face,
                    dtype=dtype,
                    device_id=device_id,
                    local_solver=local_solver,
                    application=name,
                )
                preconditioners[name] = preconditioner
                out = cp.empty_like(x)
                asm_outputs[name] = out
                preconditioner.apply_into(x, out)
                cp.cuda.get_current_stream().synchronize()
                if asm_reference is None:
                    asm_reference = out.copy()
                    error = 0.0
                else:
                    error = _relative_device_error(cp, out, asm_reference)
                    if error > tolerance:
                        raise AssertionError(
                            f"ASM candidate {name!r} differs from reference by {error:.3e}"
                        )
                asm_errors[name] = error

            asm_timings = _benchmark_interleaved_cuda_calls(
                {
                    name: (
                        lambda pc=preconditioners[name], target=asm_outputs[name]:
                        pc.apply_into(x, target)
                    )
                    for name in asm_names
                },
                warmup=warmup,
                repeats=repeats,
                device_id=device_id,
            )
            asm_rows = [
                _timing_row(
                    name=name,
                    timing=asm_timings[name],
                    workspace_bytes=int(preconditioners[name].workspace_bytes),
                    relative_error=asm_errors[name],
                )
                for name in asm_names
            ]
            if asm_rows:
                asm_choice = min(asm_rows, key=lambda item: item.median_ms).name

        return FaceDenseAutotuneResult(
            device_name=_device_name(cp, int(device_id)),
            dtype=dtype.name,
            num_dofs=int(system.num_dofs),
            block_size=int(system.block_size),
            operator_choice=operator_choice,
            asm_choice=asm_choice,
            operator_candidates=tuple(operator_rows),
            asm_candidates=tuple(asm_rows),
        )


def autotune_face_dense_gpu_cached(
    system: Any,
    *,
    element_blocks: np.ndarray | None = None,
    loc2glob_face: np.ndarray | None = None,
    dtype: Any = np.float64,
    device_id: int = 0,
    polynomial_order: int | None = None,
    operator_implementations: Iterable[str] = ("raw", "raw_fused"),
    asm_applications: Iterable[str] = ("raw", "fused"),
    local_solver: str = "cublas_inverse",
    warmup: int = 10,
    repeats: int = 50,
    validation_tolerance: float | None = None,
    cache_path: str | os.PathLike[str] | None = None,
    use_cache: bool = True,
    force_retune: bool = False,
) -> CachedFaceDenseAutotuneResult:
    """Load a compatible tuning result or benchmark and persist a new one."""

    operators = tuple(str(item) for item in operator_implementations)
    asm = tuple(str(item) for item in asm_applications)
    key = build_face_dense_autotune_key(
        system,
        dtype=dtype,
        device_id=device_id,
        polynomial_order=polynomial_order,
        local_solver=local_solver,
        operator_implementations=operators,
        asm_applications=asm,
    )
    cache = PersistentFaceDenseAutotuneCache(cache_path) if use_cache else None
    if cache is not None and not force_retune:
        cached = cache.get(key)
        if cached is not None:
            return CachedFaceDenseAutotuneResult(
                key=key,
                result=cached,
                cache_hit=True,
                cache_path=cache.path,
            )

    result = autotune_face_dense_gpu(
        system,
        element_blocks=element_blocks,
        loc2glob_face=loc2glob_face,
        dtype=dtype,
        device_id=device_id,
        operator_implementations=operators,
        asm_applications=asm,
        local_solver=local_solver,
        warmup=warmup,
        repeats=repeats,
        validation_tolerance=validation_tolerance,
    )
    if cache is not None:
        cache.put(key, result)
    return CachedFaceDenseAutotuneResult(
        key=key,
        result=result,
        cache_hit=False,
        cache_path=None if cache is None else cache.path,
    )


__all__ = [
    "AUTOTUNE_CACHE_SCHEMA_VERSION",
    "AUTOTUNE_KERNEL_ABI_VERSION",
    "CUDADeviceFingerprint",
    "CachedFaceDenseAutotuneResult",
    "FaceDenseAutotuneKey",
    "FaceDenseAutotuneResult",
    "KernelCandidateTiming",
    "PersistentFaceDenseAutotuneCache",
    "autotune_face_dense_gpu",
    "autotune_face_dense_gpu_cached",
    "build_face_dense_autotune_key",
    "default_autotune_cache_path",
]