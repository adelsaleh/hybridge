"""Shared launch-policy helpers for raw-CUDA element kernels."""

from __future__ import annotations

from numbers import Integral
from typing import Literal, TypeAlias

import time
from hdgfem.runtime.precision import real_raw_kernel



RawCudaEquation = Literal["advection-reaction", "diffusion-reaction"]
RawCudaBlockSize: TypeAlias = int | Literal["auto"]

RAW_CUDA_EXPLICIT_BLOCK_SIZES = (1, 32, 64, 128)


def triangle_element_dof(order: int) -> int:
    """Return the scalar total-degree DOF count on a triangle."""
    order = int(order)
    if order < 0:
        raise ValueError("order must be nonnegative")
    return (order + 1) * (order + 2) // 2


def recommended_raw_cuda_block_size(equation: RawCudaEquation, order: int) -> int:
    """Return the initial production launch recommendation for an element kernel.

    Advection uses the smallest warp-multiple that covers one scalar local row.
    Diffusion has more dense Schur-complement work and shared state, so it uses
    conservative degree tiers pending a recorded device sweep.
    """
    nel = triangle_element_dof(order)
    if equation == "advection-reaction":
        # The spill-free p=8--9 fused BSR qualification on SM75 measured 128
        # threads as the best legal launch.  These orders already consume one
        # shared-memory-resident CTA per SM, so additional row coverage improves
        # throughput without reducing resident-block occupancy.
        if 8 <= order <= 9:
            return 128
        if nel <= 32:
            return 32
        if nel <= 64:
            return 64
        if nel <= 128:
            return 128
    elif equation == "diffusion-reaction":
        if order <= 2:
            return 32
        if order <= 4:
            return 64
        if order <= 6:
            return 128
    else:
        raise ValueError(
            "equation must be 'advection-reaction' or 'diffusion-reaction'"
        )
    raise ValueError(
        f"no raw-CUDA block-size recommendation for {equation} at p={order} "
        f"(element DOFs={nel}); pass an explicit supported size only after "
        "qualifying that kernel configuration"
    )


def resolve_raw_cuda_block_size(
        requested: RawCudaBlockSize | None,
        *,
        equation: RawCudaEquation,
        order: int,
) -> int:
    """Resolve ``None``/``'auto'`` or validate an explicit launch size."""
    if requested is None or (isinstance(requested, str) and requested.lower() == "auto"):
        return recommended_raw_cuda_block_size(equation, order)
    if isinstance(requested, bool):
        raise ValueError("raw-CUDA block size must be 'auto' or one of 1, 32, 64, 128")
    if isinstance(requested, str):
        try:
            block_size = int(requested)
        except ValueError as exc:
            raise ValueError(
                "raw-CUDA block size must be 'auto' or one of 1, 32, 64, 128"
            ) from exc
    elif isinstance(requested, Integral):
        block_size = int(requested)
    else:
        raise ValueError("raw-CUDA block size must be 'auto' or one of 1, 32, 64, 128")
    if block_size not in RAW_CUDA_EXPLICIT_BLOCK_SIZES:
        raise ValueError("raw-CUDA block size must be 'auto' or one of 1, 32, 64, 128")
    return block_size


def _compile_kernel(cupy, source: str, name: str, shared_bytes: int):
    """Compile a raw CUDA kernel and request its dynamic shared-memory budget."""
    kernel = real_raw_kernel(source, name, options=('--std=c++11',))
    try:
        kernel.max_dynamic_shared_size_bytes = int(shared_bytes)
    except Exception:
        pass
    return kernel


def _compile_kernel_timed(cupy, source: str, name: str, shared_bytes: int):
    """Eagerly compile a raw kernel and return its host-side JIT/load time."""
    start = time.perf_counter()
    kernel = _compile_kernel(cupy, source, name, shared_bytes)
    kernel.compile()
    return kernel, time.perf_counter() - start
