"""Initial CUDA preconditioners for face-dense HDG systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..assembly.face_dense import FaceDenseSystem
from ..linalg.block_jacobi import build_face_block_jacobi_preconditioner
from .cupy import require_cupy_device

@dataclass
class CuPyFaceBlockJacobiPreconditioner:
    """Device-resident inverse face blocks with batched application.

    This first correctness implementation computes the small block inverses on
    the CPU using the already validated reference builder and transfers them
    once.  The solve phase is fully device-resident.  A later setup kernel can
    replace the host inversion with batched cuBLAS/cuSOLVER factorization
    without changing the application interface.
    """

    inverse_blocks: Any
    device_id: int

    def __post_init__(self) -> None:
        cp = require_cupy_device()
        if not isinstance(self.inverse_blocks, cp.ndarray):
            raise TypeError("inverse_blocks must be a CuPy array")
        if self.inverse_blocks.ndim != 3:
            raise ValueError("inverse_blocks must have shape (NF, b, b)")
        if self.inverse_blocks.shape[1] != self.inverse_blocks.shape[2]:
            raise ValueError("inverse face blocks must be square")
        if self.inverse_blocks.dtype not in (cp.float32, cp.float64):
            raise TypeError("inverse_blocks must use float32 or float64")
        if int(self.inverse_blocks.device.id) != int(self.device_id):
            raise ValueError("inverse_blocks is on the wrong CUDA device")
        if not self.inverse_blocks.flags.c_contiguous:
            raise ValueError("inverse_blocks must be C-contiguous")

        self.device_id = int(self.device_id)
        self._cp = cp
        self._matmul_output = cp.empty(
            (self.num_faces, self.block_size, 1),
            dtype=self.dtype,
        )

    @classmethod
    def from_system(
        cls,
        system: FaceDenseSystem,
        *,
        device_id: int | None = None,
        inverse_residual_tolerance: float | None = None,
    ) -> "CuPyFaceBlockJacobiPreconditioner":
        cp = require_cupy_device()
        if not isinstance(system, FaceDenseSystem):
            raise TypeError("system must be a FaceDenseSystem")
        selected_device = (
            int(cp.cuda.Device().id) if device_id is None else int(device_id)
        )
        cpu_preconditioner = build_face_block_jacobi_preconditioner(
            system,
            inverse_residual_tolerance=inverse_residual_tolerance,
        )
        with cp.cuda.Device(selected_device):
            inverse_blocks = cp.asarray(cpu_preconditioner.inverse_blocks)
        return cls(inverse_blocks=inverse_blocks, device_id=selected_device)

    @property
    def num_faces(self) -> int:
        return int(self.inverse_blocks.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.inverse_blocks.shape[1])

    @property
    def num_dofs(self) -> int:
        return self.num_faces * self.block_size

    @property
    def dtype(self) -> Any:
        return self.inverse_blocks.dtype

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
        if vector.ndim == 2 and vector.shape != (
            self.num_faces,
            self.block_size,
        ):
            raise ValueError(f"{name} has an incompatible face-major shape")
        if not vector.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        return vector.reshape(self.num_faces, self.block_size)

    def apply_into(self, x: Any, out: Any) -> None:
        """Compute ``out = P^{-1} x`` without iteration-time allocation."""

        cp = self._cp
        x_faces = self._validate_vector(x, name="x")
        out_faces = self._validate_vector(out, name="out")
        if cp.shares_memory(x, out):
            raise ValueError("x and out must not alias")
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


__all__ = ["CuPyFaceBlockJacobiPreconditioner"]
