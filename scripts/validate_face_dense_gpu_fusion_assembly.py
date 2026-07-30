"""Validate fused face matvec and the first GPU global-assembly layer."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.assembly.face_dense import assemble_global_face_blocks, face_dense_matvec
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_assembly import CuPyGlobalFaceAssembler
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_profiling import benchmark_cuda_call
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import diffusion_element_boundary_mats, local_solvers
from hdgfem.solvers.diff_rea_face_dense import assemble_diffusion_face_dense_components
from scripts.diff_rea_cases import quadratic_poisson_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cp = require_cupy_device()
    dtype = np.dtype(args.dtype)
    selected_device = int(cp.cuda.Device().id) if args.device is None else args.device

    diffusion, reaction, source, boundary = quadratic_poisson_case()
    space = DGSpace(
        rectangle_mesh(args.mesh, args.mesh),
        args.order,
        basis_type="dub_orth",
    )
    local_solver = local_solvers(
        reaction,
        1.3,
        space,
        backend="numpy",
        diffusion=diffusion,
    )
    boundary_mats = diffusion_element_boundary_mats(1.3, space)
    source_rhs = hdg_assembly.block_source_moments(
        source,
        space,
        num_blocks=3,
        source_block=0,
    )
    assembly = assemble_diffusion_face_dense_components(
        local_solver,
        boundary_mats,
        source_rhs,
        boundary,
        1.3,
        space,
    )

    start = perf_counter()
    cpu_blocks = assemble_global_face_blocks(
        assembly.element_blocks,
        space.mesh.loc2glob_edge,
        assembly.topology,
        active_row_faces=space.mesh.interior_face_mask,
    )
    cpu_assembly_ms = 1.0e3 * (perf_counter() - start)

    with cp.cuda.Device(selected_device):
        element_device = cp.asarray(assembly.element_blocks, dtype=dtype)
        assembler = CuPyGlobalFaceAssembler.from_topology(
            space.mesh.loc2glob_edge,
            assembly.topology,
            block_size=assembly.element_blocks.shape[-1],
            dtype=dtype,
            active_row_faces=space.mesh.interior_face_mask,
            device_id=selected_device,
        )
        global_device = cp.empty(assembler.output_shape, dtype=dtype)
        assembly_timing = benchmark_cuda_call(
            lambda: assembler.assemble_into(element_device, global_device),
            warmup=args.warmup,
            repeats=args.repeats,
            device_id=selected_device,
        )
        assembler.assemble_into(element_device, global_device)
        cp.cuda.get_current_stream().synchronize()
        gpu_blocks_host = cp.asnumpy(global_device)

        system = assembly.eliminated_system
        raw = CuPyFaceDenseOperator.from_system(
            system,
            implementation="raw",
            dtype=dtype,
            device_id=selected_device,
        )
        fused = CuPyFaceDenseOperator.from_system(
            system,
            implementation="raw_fused",
            dtype=dtype,
            device_id=selected_device,
        )
        rng = np.random.default_rng(1907)
        x_host = rng.standard_normal(system.num_dofs).astype(dtype)
        x = fused.to_device(x_host)
        y_raw = cp.empty_like(x)
        y_fused = cp.empty_like(x)
        raw_timing = benchmark_cuda_call(
            lambda: raw.matvec_into(x, y_raw),
            warmup=args.warmup,
            repeats=args.repeats,
            device_id=selected_device,
        )
        fused_timing = benchmark_cuda_call(
            lambda: fused.matvec_into(x, y_fused),
            warmup=args.warmup,
            repeats=args.repeats,
            device_id=selected_device,
        )
        raw.matvec_into(x, y_raw)
        fused.matvec_into(x, y_fused)
        cp.cuda.get_current_stream().synchronize()
        raw_host = cp.asnumpy(y_raw)
        fused_host = cp.asnumpy(y_fused)

    expected = face_dense_matvec(system.blocks.astype(dtype), system.neighbors, x_host)
    assembly_error = float(
        np.linalg.norm(gpu_blocks_host - cpu_blocks.astype(dtype))
        / max(np.linalg.norm(cpu_blocks.astype(dtype)), np.finfo(dtype).eps)
    )
    fused_error = float(
        np.linalg.norm(fused_host - expected)
        / max(np.linalg.norm(expected), np.finfo(dtype).eps)
    )
    raw_fused_difference = float(
        np.linalg.norm(fused_host - raw_host)
        / max(np.linalg.norm(raw_host), np.finfo(dtype).eps)
    )

    device_name = cp.cuda.runtime.getDeviceProperties(selected_device)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    print("GPU face-dense fusion and assembly validation")
    print("=" * 64)
    print(f"Device / dofs       : {device_name} / {system.num_dofs}")
    print(f"Mesh / order        : {args.mesh}x{args.mesh} / p={args.order}")
    print(f"dtype               : {dtype.name}")
    print()
    print(f"CPU global assembly : {cpu_assembly_ms:10.3f} ms")
    print(f"GPU global assembly : {assembly_timing.median_ms:10.4f} ms median")
    print(f"GPU assembly relerr : {assembly_error:10.3e}")
    print(f"assembly map storage: {assembler.mapping_bytes / 2**20:10.3f} MiB")
    print()
    print(f"raw matvec          : {raw_timing.median_ms:10.4f} ms")
    print(f"raw_fused matvec    : {fused_timing.median_ms:10.4f} ms")
    speedup = raw_timing.median_ms / fused_timing.median_ms
    print(f"fusion speedup      : {speedup:10.3f} x")
    print(f"raw workspace       : {raw.workspace_bytes / 2**20:10.3f} MiB")
    print(f"fused workspace     : {fused.workspace_bytes / 2**20:10.3f} MiB")
    print(f"fused CPU relerr    : {fused_error:10.3e}")
    print(f"fused-vs-raw relerr : {raw_fused_difference:10.3e}")


if __name__ == "__main__":
    main()
