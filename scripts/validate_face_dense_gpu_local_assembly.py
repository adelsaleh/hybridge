"""Validate and time the GPU element-local diffusion assembly pipeline."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_diffusion_assembly import CuPyDiffusionLocalAssembler
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import (
    _diffusion_is_identity,
    diffusion_element_boundary_mats,
    local_solvers,
)
from hdgfem.solvers.diff_rea_face_dense import (
    assemble_diffusion_face_dense_components,
)
from scripts.diff_rea_cases import quadratic_poisson_case


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.linalg.norm(expected.ravel())
    numerator = np.linalg.norm((actual - expected).ravel())
    return float(numerator if denominator == 0.0 else numerator / denominator)


def _cpu_reference(space, reaction, source, exact, tau):
    start = time.perf_counter()
    local_solver = local_solvers(reaction, tau, space, diffusion=1.0)
    boundary = diffusion_element_boundary_mats(tau, space)
    source_rhs = hdg_assembly.block_source_moments(
        source, space, num_blocks=3, source_block=0
    )
    assembly = assemble_diffusion_face_dense_components(
        local_solver,
        boundary,
        source_rhs,
        exact,
        tau,
        space,
    )
    return local_solver, source_rhs, assembly, 1.0e3 * (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--inverse-backend",
        choices=("cublas_inverse", "gpu_inverse"),
        default="cublas_inverse",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    cp = require_cupy_device()
    dtype = np.float32 if args.dtype == "float32" else np.float64
    space = DGSpace(
        rectangle_mesh(args.mesh, args.mesh),
        args.order,
        basis_type="dub_orth",
    )
    expected_element_dofs = (args.order + 1) * (args.order + 2) // 2
    expected_trace_dofs = args.order + 1
    if space.order != args.order:
        raise RuntimeError(
            f"requested p={args.order}, but DGSpace constructed p={space.order}"
        )
    if space.el_dof != expected_element_dofs:
        raise RuntimeError(
            "DGSpace element-dof mismatch: "
            f"expected {expected_element_dofs}, got {space.el_dof}"
        )
    if space.reference.edg_dof != expected_trace_dofs:
        raise RuntimeError(
            "DGSpace trace-dof mismatch: "
            f"expected {expected_trace_dofs}, got {space.reference.edg_dof}"
        )
    diffusion, reaction, source, exact = quadratic_poisson_case()
    if not _diffusion_is_identity(diffusion):
        raise RuntimeError("this validation currently targets identity diffusion")
    tau = 1.3

    local_cpu, source_rhs, assembly_cpu, cpu_ms = _cpu_reference(
        space, reaction, source, exact, tau
    )
    assembler = CuPyDiffusionLocalAssembler.from_space(
        reaction,
        tau,
        space,
        dtype=dtype,
        inverse_backend=args.inverse_backend,
    )

    for _ in range(args.warmup):
        assembler.assemble(retain_intermediates=True)
    cp.cuda.get_current_stream().synchronize()

    times = []
    result = None
    for _ in range(args.repeats):
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        result = assembler.assemble(retain_intermediates=True)
        cp.cuda.get_current_stream().synchronize()
        times.append(1.0e3 * (time.perf_counter() - start))
    assert result is not None

    cp.cuda.get_current_stream().synchronize()
    start = time.perf_counter()
    global_gpu = assembler.assemble_global_blocks(
        result,
        loc2glob_face=space.mesh.loc2glob_edge,
        topology=assembly_cpu.topology,
        active_row_faces=space.mesh.interior_face_mask,
    )
    rhs_gpu = assembler.assemble_interior_rhs(
        source_rhs,
        result,
        topology=assembly_cpu.topology,
    )
    cp.cuda.get_current_stream().synchronize()
    global_rhs_ms = 1.0e3 * (time.perf_counter() - start)

    element_gpu = cp.asnumpy(result.element_blocks)
    local_gpu = cp.asnumpy(result.local_solver)
    global_host = cp.asnumpy(global_gpu)
    rhs_host = cp.asnumpy(rhs_gpu)

    device_id = int(cp.cuda.Device().id)
    device_name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    print("GPU element-local HDG assembly validation")
    print("=" * 72)
    print(f"Device / elements : {device_name} / {space.mesh.num_tri}")
    print(f"Mesh / order      : {args.mesh}x{args.mesh} / p={space.order}")
    print(
        f"Element / trace dofs: {space.el_dof} / {space.reference.edg_dof}"
    )
    print(f"dtype / inverse   : {args.dtype} / {args.inverse_backend}")
    print()
    print(f"CPU local+face assembly       : {cpu_ms:10.3f} ms")
    print(f"GPU local assembly median     : {statistics.median(times):10.3f} ms")
    print(f"GPU local assembly minimum    : {min(times):10.3f} ms")
    print(f"GPU global blocks + RHS       : {global_rhs_ms:10.3f} ms")
    print(f"ping-pong workspace           : {assembler.workspace_bytes / 2**20:10.3f} MiB")
    print(f"scalar inverse residual       : {result.maximum_scalar_inverse_residual:10.3e}")
    print()
    print(f"local solver relative error   : {_relative_error(local_gpu, local_cpu.astype(dtype)):10.3e}")
    print(f"element blocks relative error : {_relative_error(element_gpu, assembly_cpu.element_blocks.astype(dtype)):10.3e}")
    print(f"global blocks relative error  : {_relative_error(global_host, assembly_cpu.interior_row_blocks.astype(dtype)):10.3e}")
    print(f"interior RHS relative error   : {_relative_error(rhs_host, assembly_cpu.interior_rhs.astype(dtype)):10.3e}")


if __name__ == "__main__":
    main()