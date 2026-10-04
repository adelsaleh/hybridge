#!/usr/bin/env python3
"""Benchmark tensorized TSLE (experimental) against split3 TSLE-BSR on identical inputs.

The problem is the guiding-center showcase transport step: the star mesh from
``scripts/reports/gpu_showcase_setup.showcase_mesh``, a BDF2 transport operator,
zero-flux boundaries, and conflict-averaged upwinding. The split3 inputs are
captured from the public solver, and every configuration then assembles the same
reduced face-BSR system. Rows report CUDA-event stage times (median of
``--repeats`` after one warm-up), workspace size, and the max-norm and Frobenius
differences of ``data``, ``rhs``, and the local response against FP64 split3.

A configuration is ``precision/contraction/compute/local_solver``, for example
``float32/cublas/default/coop``. On FP64-limited GPUs, FP32 rows serve as a
throughput proxy for FP64 on FP64-capable hardware; FP64 rows check the
tensorized algebra. MAGMA rows need ``HDGFEM_MAGMA_ROOT`` or
``HDGFEM_MAGMA_LIBRARY``; cuTENSOR rows need the ``cutensor-cu13`` wheel.
Assembly only: no AMGX setup, global solve, or time stepping.

``--native fused,split3`` also times the native raw-CUDA paths through the
public solver in the process precision. Native FP32 rows therefore need
``HDGFEM_PRECISION=float32``, a separate process in which tensor configurations
are refused (use ``--configs none``). The JSON output records the GPU, driver,
CUDA, and library versions so that runs on different machines can be compared;
see ``docs/research/solver_studies/advection_assembly_baseline_2026_10_03.md``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hdgfem as hdg
import hdgfem.transport.cuda as transport_cuda
from hdgfem.core.space import VectorDGField
from hdgfem.runtime.optional import require_cupy
from hdgfem.runtime.precision import PRECISION
from hdgfem.transport import tsle_bsr
from hdgfem.transport.tsle_tensor import (
    RawAdvectionTsleTensorWorkspace,
    assemble_projected_advection_trace_system_eliminated_tsle_tensor,
)
from scripts.reports.gpu_showcase_setup import showcase_mesh, transport_solver
from scripts.reports.record_gpu_showcase import positive_transport_options

DEFAULT_CONFIGS = (
    "float64/cutensor/default/coop,float64/cutensor/default/magma,"
    "float32/cutensor/default/coop,float32/cutensor/default/cublas,float32/cutensor/default/magma,"
    "float32/cublas/default/coop,float32/cublas/default/magma"
)
TENSOR_STAGES = ("prepare", "build", "solve", "condense", "scatter", "device")
NATIVE_PATHS = ("fused", "split3")
SPLIT3_STAGES = ("build", "solve", "scatter", "device")


def _transport_step(cp, mesh, order: int):
    """Return the space and BDF2 transport data of the showcase step."""
    space = hdg.DGSpace(mesh, order, basis_type="dub_orth")
    rho = hdg.project_callable(lambda x, y: 1.0 + np.exp(-20 * ((x - 0.3) ** 2 + y ** 2)), space, backend="device")
    velocity = VectorDGField((
        hdg.project_callable(lambda x, y: -2 * np.cos(3 * x) * np.sin(2 * y), space, backend="device"),
        hdg.project_callable(lambda x, y: 3 * np.sin(3 * x) * np.cos(2 * y), space, backend="device"),
    ))
    source, beta, _ = hdg.bdf2_transport_data(
        rho, velocity, 0.000390625, previous_field=rho, previous_velocity=velocity)
    return space, source, beta


def _split3_inputs(space, source, beta) -> dict:
    """Capture the keyword inputs the solver passes to TSLE-BSR."""
    captured: dict = {}
    original = transport_cuda.assemble_projected_advection_trace_system_eliminated_tsle_bsr

    def capture(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    transport_cuda.assemble_projected_advection_trace_system_eliminated_tsle_bsr = capture
    try:
        solver = transport_solver(space, **positive_transport_options(), raw_local_assembly="split3")
        solver.set_problem(source, beta, space.constant(1.0), None)
        solver.assemble_tangent_boundary_raw_cuda_bsr()
        solver.close()
    finally:
        transport_cuda.assemble_projected_advection_trace_system_eliminated_tsle_bsr = original
    captured.pop("workspace", None)
    return captured


def _median_ms(samples: dict[str, list[float]]) -> dict[str, float]:
    return {f"{key}_ms": round(1.0e3 * statistics.median(values), 3) for key, values in samples.items()}


def _differences(cp, result, reference) -> dict[str, float]:
    row = {}
    for name in ("data", "rhs", "local_response"):
        expected = cp.asarray(getattr(reference, name)).ravel()
        difference = cp.asarray(getattr(result, name)).ravel() - expected
        row[f"{name}_rel_max"] = float(cp.abs(difference).max() / cp.abs(expected).max())
        row[f"{name}_rel_fro"] = float(cp.linalg.norm(difference) / cp.linalg.norm(expected))
    return row


def _native_rows(cp, mesh, order: int, space, source, beta, paths, repeats: int) -> list[dict]:
    """Time native fused/split3 assembly through the public solver."""
    rows = []
    for path in paths:
        solver = transport_solver(space, **positive_transport_options(), raw_local_assembly=path)
        solver.set_problem(source, beta, space.constant(1.0), None)
        samples = {key: [] for key in ("device", "build", "solve", "scatter")}
        for repeat in range(repeats + 1):
            result = solver.assemble_tangent_boundary_raw_cuda_bsr()
            if repeat:
                samples["device"].append(result.timings["raw.kernel.device"])
                for key in ("build", "solve", "scatter"):
                    if f"raw.tsle.{key}" in result.timings:
                        samples[key].append(result.timings[f"raw.tsle.{key}"])
        rows.append(dict(order=order, config=f"{path}/{PRECISION}", elements=int(mesh.num_tri),
                         block_size=result.timings.get("raw.block_size"),
                         **_median_ms({key: values for key, values in samples.items() if values})))
        solver.close()
        del solver, result
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def _benchmark_order(cp, mesh, order: int, configs, repeats: int, native=()):
    space, source, beta = _transport_step(cp, mesh, order)
    rows = _native_rows(cp, mesh, order, space, source, beta, native, repeats)
    if not configs:
        return rows
    inputs = _split3_inputs(space, source, beta)

    workspace = tsle_bsr.RawAdvectionTsleWorkspace()
    samples = {key: [] for key in SPLIT3_STAGES}
    for repeat in range(repeats + 1):
        reference = tsle_bsr.assemble_projected_advection_trace_system_eliminated_tsle_bsr(
            **dict(inputs, workspace=workspace, cache_local_response=True))
        if repeat:
            for key in SPLIT3_STAGES:
                samples[key].append(reference.timings[f"raw.tsle.{key}"])
    reference_response = reference.local_response.copy()
    reference = type(reference)(**{**reference.__dict__, "local_response": reference_response})
    rows.append(dict(order=order, config="split3-reference/float64", elements=int(mesh.num_tri),
                     workspace_gib=workspace.nbytes / 2 ** 30, **_median_ms(samples)))
    del workspace
    cp.get_default_memory_pool().free_all_blocks()

    for precision, contraction, compute, local_solver in configs:
        workspace = RawAdvectionTsleTensorWorkspace()
        samples = {key: [] for key in TENSOR_STAGES}
        for repeat in range(repeats + 1):
            result = assemble_projected_advection_trace_system_eliminated_tsle_tensor(
                **dict(inputs, workspace=workspace, cache_local_response=True, precision=precision,
                       contraction=contraction, compute=compute, local_solver=local_solver))
            if repeat:
                for key in TENSOR_STAGES:
                    samples[key].append(result.timings[f"raw.tsle_tensor.{key}"])
        rows.append(dict(
            order=order, config=f"{precision}/{contraction}/{compute}/{local_solver}",
            elements=int(mesh.num_tri),
            workspace_gib=result.timings["raw.tsle_tensor.workspace.bytes"] / 2 ** 30,
            coop_block=result.timings.get("raw.tsle_tensor.solve.block_size"),
            **_median_ms(samples), **_differences(cp, result, reference)))
        del workspace, result
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def _print_row(row: dict) -> None:
    stages = [row.get(f"{key}_ms") for key in TENSOR_STAGES[:-1] if f"{key}_ms" in row]
    error = f"  data/rhs/response max {row['data_rel_max']:.1e}/{row['rhs_rel_max']:.1e}/{row['local_response_rel_max']:.1e}" \
        if "data_rel_max" in row else ""
    workspace = f"  workspace={row['workspace_gib']:.2f} GiB" if "workspace_gib" in row else ""
    print(f"p={row['order']} {row['config']:<32} device={row['device_ms']:9.3f} ms  stages={stages}"
          f"{workspace}{error}", flush=True)


def environment_metadata(cp) -> dict:
    """Return the GPU, driver, CUDA, and library versions of this process."""
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())

    def prop(name):
        value = properties.get(name, properties.get(name.encode()))
        return value.decode() if isinstance(value, bytes) else value

    metadata = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
        "argv": sys.argv,
        "precision": PRECISION,
        "gpu": {key: prop(key) for key in (
            "name", "major", "minor", "multiProcessorCount", "totalGlobalMem", "memoryBusWidth",
            "l2CacheSize", "sharedMemPerBlockOptin", "regsPerMultiprocessor")},
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "nvrtc": list(cp.cuda.nvrtc.getVersion()),
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    try:
        metadata["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        metadata["nvidia_driver"] = None
    try:
        from cupy_backends.cuda.libs import cublas

        metadata["cublas"] = cublas.getVersion(cp.cuda.device.get_cublas_handle())
    except Exception:
        metadata["cublas"] = None
    try:
        from hdgfem.runtime.optional import require_cutensor

        require_cutensor()
        from cupy_backends.cuda.libs import cutensor

        metadata["cutensor"] = cutensor.get_version()
    except Exception:
        metadata["cutensor"] = None
    try:
        import ctypes
        from hdgfem.linalg.gpu.magma_batched import load_magma

        library = load_magma()
        version = [ctypes.c_int(), ctypes.c_int(), ctypes.c_int()]
        library.magma_version(*(ctypes.byref(v) for v in version))
        metadata["magma"] = {"version": ".".join(str(v.value) for v in version), "library": library._name}
    except Exception:
        metadata["magma"] = None
    try:
        root = Path(__file__).resolve().parents[2]
        metadata["git_commit"] = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                                text=True, check=True).stdout.strip()
        metadata["git_dirty"] = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                                    capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        metadata["git_commit"] = None
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh-size", type=float, default=0.0095, help="showcase star mesh size (0.0095: about 117k triangles)")
    parser.add_argument("--orders", default="4,5,6,7,8,9")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, help="tensor configurations, or 'none'")
    parser.add_argument("--native", default="", help="native paths to time in the process precision: fused,split3")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    orders = [int(value) for value in args.orders.split(",") if value.strip()]
    configs = [] if args.configs.strip().lower() == "none" else [
        tuple(value.split("/")) for value in args.configs.split(",") if value.strip()]
    if any(len(config) != 4 for config in configs):
        raise ValueError("configs must be precision/contraction/compute/local_solver")
    native = [value.strip() for value in args.native.split(",") if value.strip()]
    if set(native) - set(NATIVE_PATHS):
        raise ValueError(f"--native accepts {NATIVE_PATHS}")
    if configs and PRECISION != "float64":
        raise ValueError("tensor configurations need the FP64 package mode; use --configs none with HDGFEM_PRECISION=float32")
    cp = require_cupy()
    metadata = environment_metadata(cp)
    mesh = showcase_mesh(args.mesh_size)
    print(f"tensorized TSLE benchmark | {metadata['gpu']['name']} | precision={PRECISION} | "
          f"elements={mesh.num_tri:,} | repeats={args.repeats}", flush=True)
    results = []
    for order in orders:
        for row in _benchmark_order(cp, mesh, order, configs, args.repeats, native):
            results.append(row)
            _print_row(row)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": metadata, "mesh_size": args.mesh_size, "elements": int(mesh.num_tri),
                   "repeats": args.repeats, "results": results}
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
