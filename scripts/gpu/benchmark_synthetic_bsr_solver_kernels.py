#!/usr/bin/env python3
"""Profile HYBRIDGE's FP64 BSR solver paths on a synthetic sparse operator.

The operator has a periodic 2-D five-point block stencil and a strictly
dominant diagonal. Its values and RHS are deterministic; no mesh, PDE assembly, or
manufactured solution is involved.  Native candidates time one preconditioner
application.  AMGX candidates time a short solve, because pyamgx does not
expose its preconditioner application.  These are kernel probes, not solver
convergence or cross-family runtime comparisons.

Run under Nsight Compute with ``--nvtx --nvtx-include hybridge_synthetic/`` to
collect counters only inside the timed operation.  Warmup and hierarchy setup
are outside the NVTX range.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import bsr_matrix


ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = (
    "bsr_matvec", "asm_pp", "bj_pp", "pmg_amg_standard", "pmg_amg_robust",
    "amgx_block_jacobi_fgmres", "amgx_block_jacobi_pbicgstab",
    "amgx_dilu_fgmres", "amgx_dilu_pbicgstab",
    "amgx_multicolor_dilu_fgmres", "amgx_multicolor_dilu_pbicgstab",
)


def synthetic_bsr(faces: int, block_size: int, seed: int):
    """Make a reproducible SPD five-point BSR matrix on a periodic 2-D grid.

    Order-one nearest-neighbor coupling gives AMG a spatial graph; tiny random
    couplings on a ring can leave an enormous dense coarse grid.
    """
    rng = np.random.default_rng(seed)
    width = int(np.sqrt(faces))
    while faces % width:
        width -= 1
    height = faces // width
    if width < 3 or height < 3:
        raise ValueError("synthetic grid needs at least three rows and columns")
    row = np.arange(faces, dtype=np.int32)
    grid_y, grid_x = np.divmod(row, width)
    west = grid_y * width + (grid_x - 1) % width
    east = grid_y * width + (grid_x + 1) % width
    south = ((grid_y - 1) % height) * width + grid_x
    north = ((grid_y + 1) % height) * width + grid_x
    neighbors = np.stack((south, west, row, east, north), axis=1)
    blocks = np.empty((faces, 5, block_size, block_size), dtype=np.float64)
    identity = np.eye(block_size)
    horizontal = -0.45 * identity + rng.standard_normal((block_size, block_size)) * (0.005 / block_size)
    vertical = -0.45 * identity + rng.standard_normal((block_size, block_size)) * (0.005 / block_size)
    horizontal_weight = rng.uniform(0.95, 1.05, size=faces)
    vertical_weight = rng.uniform(0.95, 1.05, size=faces)
    blocks[:, 0] = vertical_weight[south, None, None] * vertical.T
    blocks[:, 1] = horizontal_weight[west, None, None] * horizontal.T
    blocks[:, 2] = rng.uniform(2.2, 2.4, size=faces)[:, None, None] * identity
    blocks[:, 3] = horizontal_weight[:, None, None] * horizontal
    blocks[:, 4] = vertical_weight[:, None, None] * vertical
    ordering = np.argsort(neighbors, axis=1)
    neighbors = np.ascontiguousarray(np.take_along_axis(neighbors, ordering, axis=1))
    blocks = np.ascontiguousarray(np.take_along_axis(blocks, ordering[:, :, None, None], axis=1))
    indptr = np.arange(0, 5 * (faces + 1), 5, dtype=np.int32)
    matrix = bsr_matrix((blocks.reshape(-1, block_size, block_size),
                         neighbors.reshape(-1), indptr),
                        shape=(faces * block_size, faces * block_size))
    return matrix, blocks, neighbors, rng.standard_normal(faces * block_size)


def polynomial_roots(degree: int, center: float) -> np.ndarray:
    """Fixed mixed real/complex roots to exercise both production update kernels."""
    real_count = degree // 2
    pair_count = (degree - real_count) // 2
    real_values = np.linspace(0.85 * center, 1.15 * center, real_count)
    pair_values = np.linspace(0.9 * center, 1.1 * center, pair_count)
    roots = []
    for index in range(max(real_count, pair_count)):
        if index < real_count:
            roots.append(real_values[index])
        if index < pair_count:
            value = pair_values[index]
            roots.extend((complex(value, 0.05 * center), complex(value, -0.05 * center)))
    while len(roots) < degree:
        roots.append(center)
    return np.asarray(roots, dtype=np.complex128)


def synthetic_element_faces(cp, faces: int):
    """Three faces/element, two elements/face, as on a periodic triangulation."""
    if faces % 3:
        raise ValueError("--faces must be divisible by 3 for ASM incidence")
    elements = 2 * faces // 3
    element = cp.arange(elements, dtype=cp.int32)
    return cp.stack((element, (element - 1) % elements,
                     elements + element % (elements // 2)), axis=1)


def native_operation(cp, candidate, matrix, blocks, neighbors, rhs, pp_degree):
    from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
    from hybridge.linalg.gpu.preconditioners import (
            CuPyFaceAdditiveSchwarzPreconditioner,
            CuPyFaceBlockJacobiPreconditioner,
        )
    from hybridge.linalg.gpu.polynomial import CuPyPolynomialPreconditioner
    from hybridge.linalg.multigrid.krylov import BernsteinHpSymmetricPartPreconditioner

    faces, _, block_size, _ = blocks.shape
    output = cp.empty_like(rhs)
    if candidate == "bsr_matvec":
        from hybridge.linalg.gpu.legendre_face_bsr import LegendreFaceBsrOperator
        operator = LegendreFaceBsrOperator(
            cp.asarray(matrix.indptr, dtype=cp.int32),
            cp.asarray(matrix.indices, dtype=cp.int32),
            cp.asarray(matrix.data),
            backend="auto",
        )
        return (lambda: operator.matvec(rhs, out=output), operator.close,
                f"BSR SpMV ({operator.backend_used})")
    if candidate.startswith("pmg_amg_"):
        policy = candidate.removeprefix("pmg_amg_")
        pre = BernsteinHpSymmetricPartPreconditioner(
            matrix, degree=block_size - 1, policy=policy,
        )
        return lambda: pre.apply_into(rhs, output), pre.close, "preconditioner apply"

    operator = CuPyFaceDenseOperator.from_device_blocks(
        cp.asarray(blocks), cp.asarray(neighbors), implementation="raw_fused",
    )
    if candidate == "bj_pp":
        inverse = cp.broadcast_to(cp.eye(block_size, dtype=cp.float64)[None] / 4.0,
                                  (faces, block_size, block_size)).copy()
        base = CuPyFaceBlockJacobiPreconditioner(
            inverse_blocks=inverse, device_id=int(cp.cuda.runtime.getDevice()),
            local_solver="external_inverse", application="raw",
        )
        center = 1.0
    else:
        element_faces = synthetic_element_faces(cp, faces)
        size = 3 * block_size
        inverse = cp.broadcast_to(cp.eye(size, dtype=cp.float64)[None] / 8.0,
                                  (2 * faces // 3, size, size)).copy()
        base = CuPyFaceAdditiveSchwarzPreconditioner(
            inverse_matrices=inverse, element_system_faces=element_faces,
            block_size=block_size, num_system_faces=faces,
            device_id=int(cp.cuda.runtime.getDevice()),
            local_solver="external_inverse", application="fused",
        )
        center = 1.0
    pre = CuPyPolynomialPreconditioner(
        operator, roots=polynomial_roots(pp_degree, center), base_preconditioner=base,
    )
    return lambda: pre.apply_into(rhs, output), lambda: None, "preconditioner apply"


def amgx_configuration(candidate: str, iterations: int):
    if "_dilu_" in candidate and "multicolor" not in candidate:
        path = ROOT / "configs/amgx/adv_rea_gpu4_hdg_pbicgstab_dilu_bsr_p1_p3.json"
    else:
        path = ROOT / "configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_block_graph_dense_bsr.json"
    config = deepcopy(json.loads(path.read_text()))
    solver = config["solver"]
    solver.update(solver="FGMRES" if candidate.endswith("fgmres") else "PBICGSTAB",
                  max_iters=iterations, convergence="ABSOLUTE", tolerance=1e-30,
                  monitor_residual=1, norm="L2", use_scalar_norm=1,
                  bsr_spmv_backend="cusparse_generic", print_solve_stats=0)
    if candidate.endswith("fgmres"):
        solver["gmres_n_restart"] = max(4, iterations)
    pre = solver["preconditioner"]
    if "_dilu_" in candidate and "multicolor" not in candidate:
        pre.update(max_iters=2, relaxation_factor=0.7)
    else:
        pre.update(classical_bsr_hierarchy="block_graph_dense", cycle="W",
                   presweeps=2, postsweeps=2,
                   smoother=dict(solver="MULTICOLOR_DILU" if "multicolor" in candidate
                                 else "BLOCK_JACOBI", max_iters=1, relaxation_factor=1.0))
    return config


def amgx_operation(candidate, matrix, rhs_host, iterations):
    import pyamgx

    pyamgx.initialize()
    pyamgx.register_print_callback(lambda message: None)
    objects = []
    try:
        config = pyamgx.Config().create_from_dict(amgx_configuration(candidate, iterations))
        objects.append(config)
        resources = pyamgx.Resources().create_simple(config)
        objects.append(resources)
        operator = pyamgx.Matrix().create(resources)
        objects.append(operator)
        block_size = matrix.blocksize[0]
        faces = matrix.shape[0] // block_size
        operator.upload(matrix.indptr, matrix.indices, np.ascontiguousarray(matrix.data),
                        block_dims=[block_size, block_size], shape=[faces, faces])
        rhs = pyamgx.Vector().create(resources)
        objects.append(rhs)
        rhs.upload_raw(rhs_host.ctypes.data, faces, block_size)
        solution = pyamgx.Vector().create(resources)
        objects.append(solution)
        solver = pyamgx.Solver().create(resources, config)
        objects.append(solver)
        solver.setup(operator)
    except Exception:
        for obj in reversed(objects):
            obj.destroy()
        pyamgx.finalize()
        raise

    def apply():
        solution.set_zero(n=faces, block_dim=block_size)
        solver.solve(rhs, solution, zero_initial_guess=True)

    def close():
        for obj in reversed(objects):
            obj.destroy()
        pyamgx.finalize()

    return apply, close, "short AMGX solve"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--faces", type=int, default=12000)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--pp-degree", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.faces < 1000 or args.faces % 3 or args.block_size < 2:
        parser.error("--faces must be divisible by 3 and >=1000; block size >=2")
    if min(args.pp_degree, args.iterations, args.repeats) < 1 or args.warmup < 0:
        parser.error("degree, iterations, and repeats must be positive; warmup nonnegative")

    import cupy as cp
    cp.cuda.Device(args.device).use()
    matrix, blocks, neighbors, rhs_host = synthetic_bsr(
        args.faces, args.block_size, args.seed,
    )
    rhs = cp.asarray(rhs_host)
    if args.candidate.startswith("amgx_"):
        apply, close, operation = amgx_operation(args.candidate, matrix, rhs_host, args.iterations)
    else:
        apply, close, operation = native_operation(
            cp, args.candidate, matrix, blocks, neighbors, rhs, args.pp_degree,
        )
    sync = cp.cuda.get_current_stream().synchronize
    try:
        for _ in range(args.warmup):
            apply()
            sync()
        times = []
        for _ in range(args.repeats):
            sync()
            cp.cuda.nvtx.RangePush("hybridge_synthetic")
            start = perf_counter()
            try:
                apply()
                sync()
            finally:
                cp.cuda.nvtx.RangePop()
            times.append(1000 * (perf_counter() - start))
        free, total = cp.cuda.runtime.memGetInfo()
        result = dict(candidate=args.candidate, operation=operation,
                      matrix_kind="periodic_2d_spd_five_point",
                      faces=args.faces, block_size=args.block_size,
                      dofs=args.faces * args.block_size, bsr_blocks=matrix.indices.size,
                      dtype="float64", seed=args.seed, iterations=args.iterations,
                      pp_degree=args.pp_degree, samples_ms=times,
                      median_ms=float(np.median(times)), gpu_free_bytes=int(free),
                      gpu_total_bytes=int(total),
                      note="Synthetic kernel probe; no convergence or PDE accuracy claim.")
        print(json.dumps(result, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        close()


if __name__ == "__main__":
    main()
