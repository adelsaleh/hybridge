#!/usr/bin/env python3
"""Profile raw-CUDA transport and Poisson assembly and cached RHS kernels.

This is assembly-only: no global solve, simulation, or accuracy claim.  Both
operators use the same structured triangular mesh and DG order. Supports fused
COO/CSR/BSR, transport TSLE BSR, and diffusion Schur-LU construction/RHS reuse.  Run without
Nsight for representative CUDA-event timings; under Nsight Compute, filter on
the named assembly kernel and the ``hdgfem_raw_assembly`` NVTX range.  Nsight
replay makes the wall/device timings printed during profiling meaningless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.backends.advection_cuda import assemble_reduced_system_cuda, as_cupy_trace_space
from hdgfem.backends.cupy import as_cupy_space, as_cupy_vector_coefficients, require_cupy
from hdgfem.backends.diffusion_cupy import (
    assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
    assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy,
    assemble_projected_diffusion_trace_rhs_cached_cupy,
    attach_schur_cholesky_cache_cupy,
    build_trace_reference,
    reconstruct_compact_diffusion_field_cupy,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace, VectorDGField


KERNEL_NAMES = {
    "transport": "assemble_advection_raw_fused_bsr",
    "poisson": "assemble_diffusion_raw_coop_bsr",
    "adr": "assemble_adr_tensor",
}


def _zero(x, y):
    return 0.0 * (x + y)


def _make_assembler(case: str, space: DGSpace, args):
    """Prepare one supported stage without a global solve or time stepping."""
    variable = getattr(args, "coefficient_case", "constant") == "variable"
    source = (space.project_callable(lambda x, y: 1.0 + x * y) if variable
              else space.constant(1.0, name="source"))
    reaction = space.zeros(name="reaction")
    if variable and case != "poisson":
        reaction = space.project_callable(lambda x, y: 0.2 + 0.1 * x * x)
    if case == "poisson":
        def assemble():
            return assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
                source, reaction, _zero, 1.0, space,
                trace_basis=args.trace_basis, matrix_format=args.matrix_format,
                block_size=args.block_size, cache_local_factors=args.cache_policy == "schur-lu",
            )
        if args.cache_policy == "schur-cholesky":
            cspace = as_cupy_space(space)
            trace_ref = build_trace_reference(cspace, args.trace_basis)
            raw_assembled = assemble()

            def build_cache():
                result = attach_schur_cholesky_cache_cupy(
                    raw_assembled, reaction, cspace, trace_ref, 1.0)
                # The retained raw assembly predates this measured operation.
                return replace(result, timings={
                    key: value for key, value in result.timings.items()
                    if key.startswith("cupy.local_cache.")})

            if args.phase == "cache":
                return build_cache
            cached = build_cache()
            if args.phase == "reconstruction":
                cp = require_cupy()
                trace = cp.sin(cp.arange(space.mesh.num_edg * (space.order + 1), dtype=cp.float64))

                def reconstruct_cholesky():
                    field, unknowns, seconds = reconstruct_compact_diffusion_field_cupy(
                        trace, cached.schur_cholesky_cache, cspace)
                    return SimpleNamespace(field=field, unknowns=unknowns,
                                           timings={"reconstruction.kernel.device": seconds})

                return reconstruct_cholesky

            def assemble_cholesky_rhs():
                return assemble_projected_diffusion_trace_rhs_cached_cupy(
                    source, _zero, cspace, trace_ref, cached)

            return assemble_cholesky_rhs
        if args.phase == "reconstruction":
            from hdgfem.backends.diffusion_raw_cuda import reconstruct_projected_diffusion_field_raw_cuda
            cp = require_cupy()
            cached = assemble()
            raw = cached.raw_assembly
            cspace = as_cupy_space(space)
            trace_ref = build_trace_reference(cspace, args.trace_basis)
            trace = cp.sin(cp.arange(space.mesh.num_edg * (space.order + 1), dtype=cp.float64))

            def reconstruct_raw():
                field, unknowns, seconds = reconstruct_projected_diffusion_field_raw_cuda(
                    trace=trace, source_rhs=cached.source_rhs, cspace=cspace, trace_ref=trace_ref,
                    d0_reference=raw.d0_reference, d1_reference=raw.d1_reference,
                    face_element_mass=raw.face_element_mass, tau=1.0,
                    block_size=args.block_size, return_local_unknowns=True,
                    cached_factors=raw if args.cache_policy == "schur-lu" else None)
                return SimpleNamespace(field=field, unknowns=unknowns,
                                       timings={"reconstruction.wrapper.wall": seconds})

            return reconstruct_raw
        if args.phase == "rhs":
            cached = assemble()
            def assemble_rhs():
                return assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy(
                    source, reaction, _zero, 1.0, space,
                    cached_raw=cached.raw_assembly, trace_basis=args.trace_basis,
                    block_size=args.block_size)
            return assemble_rhs
        return assemble

    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(space.trace_space(args.trace_basis), device=cspace.device_id)
    beta = VectorDGField(
        (space.project_callable(lambda x, y: 1.0 + 0.1 * y),
         space.project_callable(lambda x, y: 0.3 - 0.1 * x)) if variable else
        (space.constant(1.0, name="beta_x"), space.constant(0.3, name="beta_y")),
        space, name="beta",
    )
    beta_coeffs = as_cupy_vector_coefficients(beta, cspace)

    if case == "adr":
        from hdgfem.assembly.advection_diffusion_reaction import prepare_adr_data
        from hdgfem.backends.advection_diffusion_reaction_raw_cuda import (
            assemble_projected_adr_trace_operator_raw_cuda,
        )
        host_trace = space.trace_space(args.trace_basis)
        prepared = prepare_adr_data(
            source, reaction, beta, space, diffusion=1.0, trace_space=host_trace,
            dense_local_matrices=args.block_size == 1)

        def assemble_adr():
            return assemble_projected_adr_trace_operator_raw_cuda(
                prepared, _zero, space, diffusion=1.0, trace_space=host_trace,
                matrix_format=args.matrix_format, block_size=args.block_size).assembly

        return assemble_adr

    def assemble():
        return assemble_reduced_system_cuda(
            source, reaction, _zero, beta_coeffs, cspace, trace_ref,
            backend="raw-cuda", raw_local_assembly=args.local_assembly, raw_lu_mode="coop",
            raw_matrix_format=args.matrix_format, raw_block_size=args.block_size,
            raw_cache_local_response=False,
        )
    return assemble


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(KERNEL_NAMES), required=True)
    parser.add_argument("--nx", type=int, default=128, help="2*nx*nx triangles")
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--matrix-format", choices=("coo", "csr", "bsr"), default="bsr")
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal"), default="legendre-modal")
    parser.add_argument("--block-size", type=int, choices=(1, 32, 64, 128), default=128)
    parser.add_argument("--cache-policy", choices=("none", "schur-lu", "schur-cholesky"), default="none")
    parser.add_argument("--phase", choices=("assembly", "rhs", "cache", "reconstruction"), default="assembly")
    parser.add_argument("--local-assembly", choices=("fused", "split3"), default="fused")
    parser.add_argument("--coefficient-case", choices=("constant", "variable"), default="constant")
    args = parser.parse_args(argv)
    if args.case != "adr" and args.block_size == 1:
        parser.error("this harness profiles cooperative Poisson/transport at 32/64/128 threads")
    if args.case != "poisson" and (args.phase != "assembly" or args.cache_policy != "none"):
        parser.error("RHS and factor-cache options currently apply only to Poisson")
    if args.phase == "rhs" and args.matrix_format == "coo":
        parser.error("cached RHS requires CSR or BSR")
    if args.cache_policy == "schur-cholesky" and (args.phase == "assembly" or args.matrix_format == "coo"):
        parser.error("Schur-Cholesky profiles the separate CuPy cache/RHS path attached to raw CSR/BSR")
    if args.phase == "cache" and args.cache_policy != "schur-cholesky":
        parser.error("separate cache construction applies to CuPy Schur-Cholesky; raw LU builds during assembly")
    if args.local_assembly == "split3" and (args.case != "transport" or args.matrix_format != "bsr"):
        parser.error("split3 requires transport BSR")
    max_order = 9 if args.case == "transport" else 6
    if args.nx < 1 or not 1 <= args.order <= max_order or args.warmup < 1 or args.repeats < 1:
        parser.error(f"require nx >= 1, order in 1..{max_order} for {args.case}, warmup >= 1, repeats >= 1")

    cp = require_cupy()
    mesh = rectangle_mesh(args.nx, args.nx)
    space = DGSpace(mesh, args.order, basis_type="dub_orth")
    setup_start = time.perf_counter()
    assemble = _make_assembler(args.case, space, args)
    cp.cuda.get_current_stream().synchronize()
    setup_seconds = time.perf_counter() - setup_start

    for _ in range(args.warmup):
        result = assemble()
        del result
    cp.cuda.get_current_stream().synchronize()

    samples_ms = []
    wall_samples_ms = []
    phase_samples = []
    wrapper_event_samples_ms = []
    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
    for _ in range(args.repeats):
        started = time.perf_counter()
        start_event.record()
        cp.cuda.nvtx.RangePush("hdgfem_raw_assembly")
        try:
            result = assemble()
        finally:
            cp.cuda.nvtx.RangePop()
        end_event.record()
        cp.cuda.get_current_stream().synchronize()
        wrapper_event_samples_ms.append(float(cp.cuda.get_elapsed_time(start_event, end_event)))
        wall_samples_ms.append(1000.0 * (time.perf_counter() - started))
        timings = result.timings
        phase_samples.append(dict(timings))
        measured_kernel = timings.get("raw.kernel.device", timings.get(
            "cached_rhs.fused_solve_flux_scatter", timings.get("reconstruction.kernel.device")))
        if measured_kernel is not None:
            samples_ms.append(1000.0 * float(measured_kernel))
        del result
    print(json.dumps({
        "case": args.case,
        "configuration": vars(args),
        "emission": "coo-to-csr" if args.case == "adr" and args.block_size == 1 and args.matrix_format == "csr" else args.matrix_format,
        "preparation_wall_seconds": setup_seconds,
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cupy_version": cp.__version__,
        "wall_samples_ms": wall_samples_ms,
        "wrapper_event_samples_ms": wrapper_event_samples_ms,
        "wrapper_event_scope": "whole wrapper stream interval, including GPU idle gaps during host work; not summed kernel time",
        "phase_samples_seconds_and_metadata": phase_samples,
        "triangles": int(mesh.num_tri),
        "order": args.order,
        "trace_dofs": int(mesh.int_edges_inds.size * (args.order + 1)),
        "samples_device_ms": samples_ms,
        "median_device_ms": statistics.median(samples_ms) if samples_ms else None,
        "stdev_device_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "warning": "Timing under Nsight Compute includes replay; use an unprofiled run for timing.",
    }, indent=2))


if __name__ == "__main__":
    main()
