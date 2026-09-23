#!/usr/bin/env python3
"""Profile the fully fused raw-CUDA transport and Poisson assembly kernels.

This is assembly-only: no global solve, simulation, or accuracy claim.  Both
operators use the same structured triangular mesh and DG order.  Run without
Nsight for representative CUDA-event timings; under Nsight Compute, filter on
the named assembly kernel and the ``hdgfem_raw_assembly`` NVTX range.  Nsight
replay makes the wall/device timings printed during profiling meaningless.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.backends.advection_cuda import assemble_reduced_system_cuda, as_cupy_trace_space
from hdgfem.backends.cupy import as_cupy_space, as_cupy_vector_coefficients, require_cupy
from hdgfem.backends.diffusion_cupy import assemble_projected_diffusion_trace_system_eliminated_raw_cupy
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace, VectorDGField


KERNEL_NAMES = {
    "transport": "assemble_advection_raw_fused_bsr",
    "poisson": "assemble_diffusion_raw_coop_bsr",
}


def _zero(x, y):
    return 0.0 * (x + y)


def _make_assembler(case: str, space: DGSpace):
    source = space.constant(1.0, name="source")
    reaction = space.zeros(name="reaction")
    if case == "poisson":
        def assemble():
            return assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
                source, reaction, _zero, 1.0, space,
                trace_basis="legendre-modal", matrix_format="bsr",
                block_size=128, cache_local_factors=False,
            )
        return assemble

    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(space.trace_space("legendre-modal"), device=cspace.device_id)
    beta = VectorDGField(
        (space.constant(1.0, name="beta_x"), space.constant(0.3, name="beta_y")),
        space, name="beta",
    )
    beta_coeffs = as_cupy_vector_coefficients(beta, cspace)

    def assemble():
        return assemble_reduced_system_cuda(
            source, reaction, _zero, beta_coeffs, cspace, trace_ref,
            backend="raw-cuda", raw_local_assembly="fused", raw_lu_mode="coop",
            raw_matrix_format="bsr", raw_block_size=128,
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
    args = parser.parse_args(argv)
    if args.nx < 1 or not 1 <= args.order <= 6 or args.warmup < 1 or args.repeats < 1:
        parser.error("require nx >= 1, order in 1..6, warmup >= 1, repeats >= 1")

    cp = require_cupy()
    mesh = rectangle_mesh(args.nx, args.nx)
    space = DGSpace(mesh, args.order, basis_type="dub_orth")
    assemble = _make_assembler(args.case, space)

    for _ in range(args.warmup):
        result = assemble()
        del result
    cp.cuda.get_current_stream().synchronize()

    samples_ms = []
    for _ in range(args.repeats):
        cp.cuda.nvtx.RangePush("hdgfem_raw_assembly")
        try:
            result = assemble()
        finally:
            cp.cuda.nvtx.RangePop()
        timings = result.timings
        samples_ms.append(1000.0 * float(timings["raw.kernel.device"]))
        del result
    print(json.dumps({
        "case": args.case,
        "kernel": KERNEL_NAMES[args.case],
        "triangles": int(mesh.num_tri),
        "order": args.order,
        "trace_dofs": int(mesh.int_edges_inds.size * (args.order + 1)),
        "samples_device_ms": samples_ms,
        "median_device_ms": statistics.median(samples_ms),
        "warning": "Timing under Nsight Compute includes replay; use an unprofiled run for timing.",
    }, indent=2))


if __name__ == "__main__":
    main()
