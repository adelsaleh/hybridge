"""Validate the first CuPy face-dense GPU operator layer.

Run from the repository root with

    PYTHONPATH=. python scripts/validate_face_dense_gpu_operator.py

The script exits successfully with a clear SKIPPED message when CuPy or a CUDA
device is unavailable.  On a GPU machine it compares both the CuPy/cuBLAS
batched-matmul path and the independent raw-kernel path against the NumPy
face-dense operator.
"""

from __future__ import annotations

import numpy as np

from hdgfem.assembly.face_dense import face_dense_matvec
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def build_system(boundary_mode: str):
    space = DGSpace(
        rectangle_mesh(4, 4),
        3,
        basis_type="dub_orth",
    )
    diffusion, reaction, source, exact = quadratic_poisson_case()
    return solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode=boundary_mode,
        boundary_penalty=1.0e6,
    ).system


def main() -> None:
    try:
        cp = require_cupy_device()
    except RuntimeError as error:
        print(f"SKIPPED: {error}")
        return

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    print(f"CUDA device: {device.id} ({device_name})")
    rng = np.random.default_rng(20260720)

    for boundary_mode in ("eliminate", "penalty"):
        system = build_system(boundary_mode)
        x_host = rng.standard_normal(system.rhs.shape)
        expected = face_dense_matvec(
            system.blocks,
            system.neighbors,
            x_host,
        )

        print(
            f"\n{boundary_mode}: rows={system.num_rows}, "
            f"block={system.block_size}, slots={system.num_slots}, "
            f"dofs={system.num_dofs}"
        )

        results = {}
        for implementation in ("matmul", "raw"):
            operator = CuPyFaceDenseOperator.from_system(
                system,
                implementation=implementation,
            )
            x_device = operator.to_device(x_host)
            y_device = operator.matvec(x_device)
            operator.synchronize()
            y_host = operator.to_host(y_device)

            absolute = float(np.linalg.norm(y_host - expected))
            relative = absolute / max(
                float(np.linalg.norm(expected)),
                np.finfo(np.float64).eps,
            )
            results[implementation] = y_host
            print(
                f"  {implementation:6s}: absolute error={absolute:.3e}, "
                f"relative error={relative:.3e}"
            )

        cross = float(
            np.linalg.norm(results["matmul"] - results["raw"])
            / max(
                float(np.linalg.norm(results["matmul"])),
                np.finfo(np.float64).eps,
            )
        )
        print(f"  cross-path relative difference: {cross:.3e}")


if __name__ == "__main__":
    main()
