#!/usr/bin/env python3
"""Benchmark fused versus tri-stage local-elimination face-BSR assembly.

Use ``raw_local_assembly="split3"`` (TSLE-BSR) in solver options to select the
three-stage build, cooperative-LU/solve, and Schur/scatter implementation.
This driver is assembly-only: AMGX setup and global iteration time are excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hdgfem.transport import raw_cuda as fused_cuda
from hdgfem.transport import tsle_bsr as tsle_cuda
from hdgfem.transport.cuda import assemble_reduced_system_cuda
from hdgfem.core.device import as_cupy_trace_space
from hdgfem.transport.tsle_bsr import RawAdvectionTsleWorkspace
from hdgfem.core.device import (
    as_cupy_space,
    as_cupy_vector_coefficients,
    clear_cupy_space_cache,
)
from hdgfem.runtime.optional import require_cupy
from hdgfem.hdg.cuda.launch import resolve_raw_cuda_block_size
from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_rectangle_mesh
from hdgfem.core.space import DGSpace, VectorDGField
from scripts.advection_reaction.cases import case_definition_by_key


def _csv_values(text: str, convert) -> list[Any]:
    values = [convert(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _median(samples: list[float]) -> float:
    return float(statistics.median(samples))


def _device_name(cp) -> str:
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = properties.get("name", "unknown")
    return name.decode() if isinstance(name, bytes) else str(name)


def _build_mesh(args):
    if args.case == "disk-tangent":
        return gmsh_disc_mesh(
            args.mesh_size,
            center=(0.0, 0.0),
            radius=1.0,
            verbosity=args.gmsh_verbosity,
            num_threads=args.gmsh_num_threads,
        )
    return gmsh_rectangle_mesh(
        args.mesh_size,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        verbosity=args.gmsh_verbosity,
        num_threads=args.gmsh_num_threads,
    )


def _problem(case_name: str):
    case_key = "disk_tangent" if case_name == "disk-tangent" else "test2"
    beta_x, beta_y, reaction, source, exact = case_definition_by_key(case_key).build()
    return beta_x, beta_y, reaction, source, None if case_name == "disk-tangent" else exact


def _max_abs(cp, lhs, rhs) -> float:
    if lhs.size == 0:
        return 0.0
    return float(cp.max(cp.abs(lhs - rhs)).get())


def _parity(cp, fused, split3) -> dict[str, float]:
    cp.testing.assert_array_equal(split3.indptr, fused.indptr)
    cp.testing.assert_array_equal(split3.indices, fused.indices)
    errors = {
        "matrix_max_abs": _max_abs(cp, split3.data, fused.data),
        "rhs_max_abs": _max_abs(cp, split3.rhs, fused.rhs),
        "response_max_abs": _max_abs(
            cp,
            split3.raw.local_response,
            fused.raw.local_response,
        ),
    }
    scales = {
        "matrix_max_abs": max(1.0, float(cp.max(cp.abs(fused.data)).get())),
        "rhs_max_abs": max(1.0, float(cp.max(cp.abs(fused.rhs)).get())),
        "response_max_abs": max(
            1.0,
            float(cp.max(cp.abs(fused.raw.local_response)).get()),
        ),
    }
    errors["max_relative"] = max(errors[key] / scales[key] for key in scales)
    return errors


def _record_timings(result) -> dict[str, float]:
    timings = result.timings
    return {
        "wrapper": float(timings["total"]),
        "device": float(timings.get("raw.bsr_kernel", timings.get("raw.kernel.device", np.nan))),
        "build": float(timings.get("raw.tsle.build", np.nan)),
        "solve": float(timings.get("raw.tsle.solve", np.nan)),
        "scatter": float(timings.get("raw.tsle.scatter", np.nan)),
    }


def _kernel_attributes(kernel, *, dynamic_shared_bytes: int, block_size: int) -> dict[str, int]:
    """Normalize CUDA function resource attributes for JSON output."""
    attributes = kernel.attributes
    return {
        "block_size": int(block_size),
        "dynamic_shared_bytes": int(dynamic_shared_bytes),
        "static_shared_bytes": int(attributes["shared_size_bytes"]),
        "local_size_bytes": int(attributes["local_size_bytes"]),
        "registers": int(attributes["num_regs"]),
        "max_threads_per_block": int(attributes["max_threads_per_block"]),
    }


def _kernel_profiles(cp, cspace, trace_ref, fused_block: int, split_blocks) -> dict[str, Any]:
    """Return compiler resource metadata for the exact benchmark kernels."""
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    nqf = int(trace_ref.weights.size)
    orientation = fused_cuda._raw_trace_orientation_mode(trace_ref)
    fused_shared, _ = fused_cuda._fused_shared_sizes(
        nel, ntr, nqf, lu_mode="coop"
    )
    source = fused_cuda._kernel_source(
        fused_cuda._raw_fused_bsr_template(),
        nel=nel,
        ntr=ntr,
        ncols=3 * ntr + 1,
        nqf=nqf,
        lu_mode="coop",
        trace_orientation=orientation,
    )
    source = fused_cuda._apply_fused_assembly_launch_bounds(
        source,
        kernel_name="assemble_advection_raw_fused_bsr",
        nel=nel,
        block_size=fused_block,
    )
    fused_kernel, _ = fused_cuda._compile_kernel_timed(
        cp,
        source,
        "assemble_advection_raw_fused_bsr",
        fused_shared,
    )
    split_kernels = tsle_cuda._compile_kernels(
        cp,
        nel=nel,
        ntr=ntr,
        nqf=nqf,
        trace_orientation=orientation,
    )
    return {
        "fused": _kernel_attributes(
            fused_kernel,
            dynamic_shared_bytes=fused_shared,
            block_size=fused_block,
        ),
        "split3": {
            "build": _kernel_attributes(
                split_kernels.build,
                dynamic_shared_bytes=tsle_cuda._shared_bytes_build(
                    nel, 3 * ntr + 1, nqf
                ),
                block_size=split_blocks["build"],
            ),
            "solve": _kernel_attributes(
                split_kernels.solve,
                dynamic_shared_bytes=tsle_cuda._shared_bytes_solve(
                    nel, 3 * ntr + 1, split_blocks["solve"]
                ),
                block_size=split_blocks["solve"],
            ),
            "scatter": _kernel_attributes(
                split_kernels.scatter,
                dynamic_shared_bytes=tsle_cuda._shared_bytes_scatter(nel, ntr),
                block_size=split_blocks["scatter"],
            ),
        },
    }


def _summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    keys = samples[0]
    return {
        f"{key}_median": _median([sample[key] for sample in samples])
        for key in keys
        if np.isfinite(samples[0][key])
    } | {
        f"{key}_min": min(sample[key] for sample in samples)
        for key in keys
        if np.isfinite(samples[0][key])
    }


def _benchmark_configuration(cp, mesh, args, order: int, trace_basis: str) -> dict[str, Any]:
    beta_x, beta_y, reaction, source, boundary = _problem(args.case)
    space = DGSpace(
        mesh,
        order,
        basis_type="dub_orth",
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    source_h = space.project_callable(source, name="source_h")
    reaction_h = space.project_callable(reaction, name="reaction_h")
    beta_h = VectorDGField((beta_x, beta_y), space, name="beta_h")
    cspace = as_cupy_space(space)
    trace_ref = as_cupy_trace_space(
        space.trace_space(trace_basis),
        device=cspace.device_id,
    )
    beta_coeffs = as_cupy_vector_coefficients(beta_h, cspace)
    zero_boundary_flux = args.case == "disk-tangent"
    fused_block = resolve_raw_cuda_block_size(
        args.fused_block_size,
        equation="advection-reaction",
        order=order,
    )
    fused_response = None
    split_workspace = RawAdvectionTsleWorkspace()

    def assemble(path: str):
        nonlocal fused_response
        result = assemble_reduced_system_cuda(
            source_h,
            reaction_h,
            boundary,
            beta_coeffs,
            cspace,
            trace_ref,
            backend="raw-cuda",
            raw_local_assembly=path,
            raw_lu_mode="coop",
            raw_block_size="auto" if path == "split3" else fused_block,
            raw_matrix_format="bsr",
            zero_boundary_flux=zero_boundary_flux,
            raw_response_workspace=fused_response if path == "fused" else None,
            raw_tsle_workspace=split_workspace if path == "split3" else None,
            raw_cache_local_response=True,
        )
        if path == "fused":
            fused_response = result.raw.local_response
        return result

    cold_start = time.perf_counter()
    fused = assemble("fused")
    fused_cold_wall = time.perf_counter() - cold_start
    cold_start = time.perf_counter()
    split3 = assemble("split3")
    split3_cold_wall = time.perf_counter() - cold_start
    cold = {
        "fused_wall": fused_cold_wall,
        "fused_device": _record_timings(fused)["device"],
        "split3_wall": split3_cold_wall,
        "split3_device": _record_timings(split3)["device"],
        "split3_autotune": float(split3.timings.get("raw.tsle.autotune.wall", 0.0)),
    }

    parity = _parity(cp, fused, split3)
    selected_blocks = {
        stage: int(split3.timings[f"raw.tsle.{stage}.block_size"])
        for stage in ("build", "solve", "scatter")
    }
    kernel_profiles = _kernel_profiles(
        cp, cspace, trace_ref, fused_block, selected_blocks
    )
    if order >= 8:
        local_bytes = [kernel_profiles["fused"]["local_size_bytes"]]
        local_bytes.extend(
            kernel_profiles["split3"][stage]["local_size_bytes"]
            for stage in ("build", "solve", "scatter")
        )
        if any(local_bytes):
            raise RuntimeError(
                f"p={order} high-order assembly kernel spilled to local memory: "
                f"{local_bytes} bytes/thread"
            )
    del fused, split3

    for _ in range(max(0, args.warmups - 1)):
        for path in ("split3", "fused"):
            result = assemble(path)
            del result
    cp.cuda.get_current_stream().synchronize()

    samples: dict[str, list[dict[str, float]]] = {"fused": [], "split3": []}
    for repeat in range(args.repeats):
        paths = ("fused", "split3") if repeat % 2 == 0 else ("split3", "fused")
        for path in paths:
            result = assemble(path)
            samples[path].append(_record_timings(result))
            del result

    fused_summary = _summarize(samples["fused"])
    split_summary = _summarize(samples["split3"])
    result = {
        "case": args.case,
        "order": order,
        "trace_basis": trace_basis,
        "elements": int(mesh.num_tri),
        "element_dof": int(space.el_dof),
        "trace_block_size": int(space.quad_data.edg_dof),
        "fused_block_size": int(fused_block),
        "split3_blocks": selected_blocks,
        "kernel_profiles": kernel_profiles,
        "split3_workspace_gib": split_workspace.nbytes / (1024.0 ** 3),
        "cold": cold,
        "fused": fused_summary,
        "split3": split_summary,
        "device_speedup": fused_summary["device_median"] / split_summary["device_median"],
        "wrapper_speedup": fused_summary["wrapper_median"] / split_summary["wrapper_median"],
        "parity": parity,
        "p8_experimental": order == 8,
        "high_order_experimental": order >= 8,
    }

    del beta_coeffs, cspace, trace_ref, beta_h, source_h, reaction_h, fused_response, split_workspace
    clear_cupy_space_cache(space)
    del space
    gc.collect()
    cp.cuda.get_current_stream().synchronize()
    return result


def _print_result(result: dict[str, Any]) -> None:
    blocks = result["split3_blocks"]
    print(
        f"p={result['order']} {result['trace_basis']:<17} "
        f"fused={1.0e3 * result['fused']['device_median']:8.3f} ms  "
        f"split3={1.0e3 * result['split3']['device_median']:8.3f} ms  "
        f"speedup={result['device_speedup']:6.3f}x  "
        f"stages={1.0e3 * result['split3']['build_median']:.3f}/"
        f"{1.0e3 * result['split3']['solve_median']:.3f}/"
        f"{1.0e3 * result['split3']['scatter_median']:.3f} ms  "
        f"blocks={blocks['build']}/{blocks['solve']}/{blocks['scatter']}  "
        f"parity={result['parity']['max_relative']:.2e}"
        + (
            "  local=0/0/0/0 B"
            if result["high_order_experimental"]
            else ""
        ),
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("disk-tangent", "test2"), default="disk-tangent")
    parser.add_argument("--mesh-size", type=float, default=0.0068)
    parser.add_argument("--orders", default="1,2,3,4,5,6,7")
    parser.add_argument("--trace-bases", default="legacy-lagrange")
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--fused-block-size", choices=("auto", "32", "64", "128"), default="auto")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-num-threads", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    orders = _csv_values(args.orders, int)
    trace_bases = _csv_values(args.trace_bases, str)
    if any(order < 1 or order > 9 for order in orders):
        raise ValueError(
            "TSLE qualification supports p=1 through p=7; p=8 and p=9 are experimental"
        )
    invalid_bases = set(trace_bases) - {"legacy-lagrange", "legendre-modal"}
    if invalid_bases:
        raise ValueError(f"unsupported trace bases: {sorted(invalid_bases)!r}")
    if args.warmups < 1 or args.repeats < 1:
        raise ValueError("warmups and repeats must be positive")

    cp = require_cupy()
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    mesh_start = time.perf_counter()
    mesh = _build_mesh(args)
    mesh_seconds = time.perf_counter() - mesh_start
    metadata = {
        "device": _device_name(cp),
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "nvrtc": list(cp.cuda.nvrtc.getVersion()),
        "case": args.case,
        "mesh_size": args.mesh_size,
        "elements": int(mesh.num_tri),
        "mesh_seconds": mesh_seconds,
        "warmups": args.warmups,
        "repeats": args.repeats,
    }
    print(
        f"TSLE-BSR assembly benchmark | {metadata['device']} | "
        f"elements={mesh.num_tri:,} | CUDA runtime={metadata['cuda_runtime']} | "
        f"NVRTC={tuple(metadata['nvrtc'])}",
        flush=True,
    )

    results = []
    for order in orders:
        for trace_basis in trace_bases:
            result = _benchmark_configuration(cp, mesh, args, order, trace_basis)
            results.append(result)
            _print_result(result)

    payload = {"metadata": metadata, "results": results}
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
