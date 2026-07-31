"""Lightweight architecture-specific autotuning for face-dense CUDA kernels.

The T600 measurements show that neither gather+matvec versus fused matvec nor
three-stage versus fused ASM wins for every polynomial order and problem size.
This module benchmarks numerically equivalent candidates on the current CUDA
device and selects the lowest median application time.  The same short tuning
run can be repeated on a V100/P100 before expensive production jobs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .cupy import require_cupy_device
from .cupy_face_dense import CuPyFaceDenseOperator
from .cupy_preconditionners import CuPyFaceAdditiveSchwarzPreconditioner
from .cupy_profiling import benchmark_cuda_call


@dataclass(frozen=True)
class KernelCandidateTiming:
    name: str
    median_ms: float
    minimum_ms: float
    workspace_bytes: int
    relative_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _relative_device_error(cp: Any, actual: Any, reference: Any) -> float:
    difference = cp.linalg.norm(actual - reference)
    denominator = cp.linalg.norm(reference)
    eps = cp.finfo(reference.dtype).eps
    return float((difference / cp.maximum(denominator, eps)).item())


def _device_name(cp: Any, device_id: int) -> str:
    name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


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

    Setup/factorization time is deliberately excluded: the selected candidates
    implement the same operator and preconditioner and are normally reused for
    many Krylov iterations.  Every candidate is compared with the first one
    before it can be selected.
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

    with cp.cuda.Device(int(device_id)):
        rng = np.random.default_rng(20260730)
        x = cp.asarray(rng.standard_normal(system.rhs.shape), dtype=dtype)
        operator_rows: list[KernelCandidateTiming] = []
        operators: dict[str, Any] = {}
        reference_out = None
        for name in tuple(operator_implementations):
            operator = CuPyFaceDenseOperator.from_system(
                system,
                implementation=name,
                dtype=dtype,
                device_id=device_id,
            )
            operators[name] = operator
            out = cp.empty_like(x)
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
            timing = benchmark_cuda_call(
                lambda op=operator, target=out: op.matvec_into(x, target),
                warmup=warmup,
                repeats=repeats,
                device_id=device_id,
            )
            operator_rows.append(
                KernelCandidateTiming(
                    name=name,
                    median_ms=timing.median_ms,
                    minimum_ms=timing.minimum_ms,
                    workspace_bytes=int(operator.workspace_bytes),
                    relative_error=error,
                )
            )
        if not operator_rows:
            raise ValueError("at least one operator implementation is required")
        operator_choice = min(operator_rows, key=lambda item: item.median_ms).name

        asm_rows: list[KernelCandidateTiming] = []
        asm_choice = None
        if element_blocks is not None or loc2glob_face is not None:
            if element_blocks is None or loc2glob_face is None:
                raise ValueError("element_blocks and loc2glob_face must be provided together")
            asm_reference = None
            for name in tuple(asm_applications):
                preconditioner = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                    system,
                    element_blocks,
                    loc2glob_face,
                    dtype=dtype,
                    device_id=device_id,
                    local_solver=local_solver,
                    application=name,
                )
                out = cp.empty_like(x)
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
                timing = benchmark_cuda_call(
                    lambda pc=preconditioner, target=out: pc.apply_into(x, target),
                    warmup=warmup,
                    repeats=repeats,
                    device_id=device_id,
                )
                asm_rows.append(
                    KernelCandidateTiming(
                        name=name,
                        median_ms=timing.median_ms,
                        minimum_ms=timing.minimum_ms,
                        workspace_bytes=int(preconditioner.workspace_bytes),
                        relative_error=error,
                    )
                )
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


__all__ = [
    "FaceDenseAutotuneResult",
    "KernelCandidateTiming",
    "autotune_face_dense_gpu",
]
