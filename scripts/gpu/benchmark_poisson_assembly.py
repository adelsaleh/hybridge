#!/usr/bin/env python3
"""Compare raw Poisson assembly/recovery with the arbitrary-order CuPy path.

This stationary benchmark does not solve a global system. It reserves the GPU
before CUDA initialization and saves JSON samples, a log and a source archive.
The default five raw variants match the October 2026 100k-triangle study.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from hybridge.runtime.benchmarking import measure_synchronized, reserve_gpu
from scripts.reports.run_provenance import (
    changed_snapshot_files, code_state, package_versions, source_snapshot,
)

RAW_CASES = (("schur-cholesky", "coo"), ("schur-cholesky", "csr"),
             ("schur-cholesky", "bsr"), ("none", "csr"), ("schur-lu", "csr"))


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="7,8,9", help="comma-separated orders, 0..9")
    parser.add_argument("--nx", type=int, default=250, help="rectangle subdivisions in x")
    parser.add_argument("--ny", type=int, default=200, help="rectangle subdivisions in y; triangles = 2*nx*ny")
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal"), default="legendre-modal")
    parser.add_argument("--element-basis", choices=("dub_orth", "bernstein", "hier_C0"), default="dub_orth")
    parser.add_argument("--tau", type=float, default=1.3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/poisson_raw_cupy.json"))
    parser.add_argument("--gpu-lock", type=Path, default=ROOT / ".cache/gpu-campaign.lock")
    parser.add_argument("--wait-timeout", type=float, help="seconds to wait for reservation/idle GPU; default unlimited")
    parser.add_argument("--dry-run", action="store_true", help="print cases without reserving or initializing CUDA")
    parser.add_argument("-v", "--verbosity", type=int, choices=range(4), default=1)
    args = parser.parse_args(argv)
    try:
        args.orders = list(dict.fromkeys(int(value.strip()) for value in args.orders.split(",")))
    except ValueError:
        parser.error("orders must be comma-separated integers")
    if not args.orders or any(p < 0 or p > 9 for p in args.orders):
        parser.error("orders must be 0..9")
    if min(args.nx, args.ny, args.warmups, args.repeats) < 1:
        parser.error("mesh subdivisions, warmups and repeats must be positive")
    if not math.isfinite(args.tau) or args.tau <= 0:
        parser.error("tau must be finite and positive")
    if args.wait_timeout is not None and (not math.isfinite(args.wait_timeout) or args.wait_timeout < 0):
        parser.error("wait timeout must be finite and nonnegative")
    return args


def check_errors(item, *, tolerance, label):
    for key, error in item.items():
        if key.endswith("relative_error") and (not math.isfinite(error) or error > tolerance):
            raise ArithmeticError(f"{label}: {key}={error}, tolerance={tolerance}")


def run_comparison(args, report, say, save):
    # Import the device code only after the reservation has been acquired.
    import cupy as cp
    import cupyx.scipy.sparse as sparse
    import numpy as np

    from hybridge import DGSpace, rectangle_mesh
    from hybridge.core.device import as_cupy_space
    from hybridge.hdg.cuda.tuning import device_signature
    from hybridge.linalg.gpu.sparse import _assembly_device_csr_matrix, _device_compressed_matvec
    from hybridge.mixed.cupy import (
        assemble_projected_diffusion_trace_system_eliminated_cupy as cupy_assemble,
        assemble_projected_diffusion_trace_system_eliminated_raw_cupy as raw_assemble,
        assemble_projected_diffusion_trace_rhs_cached_cupy as cupy_refresh,
        assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy as raw_refresh,
        build_trace_reference, solve_mixed_from_scalar_cholesky_cupy,
    )
    from hybridge.mixed.raw_cuda import poisson
    from hybridge.runtime.precision import REAL_DTYPE

    stream = cp.cuda.get_current_stream()
    tolerance = max(2.e-10, 512 * np.finfo(REAL_DTYPE).eps)
    report.update(device=device_signature(cp), precision=np.dtype(REAL_DTYPE).name,
                  kernel_source=poisson.SOURCE_VERSION, parity_tolerance=tolerance)
    save()

    def measure(label, function):
        say(f"  {label}: warming and measuring", level=2)

        def sample(index, elapsed):
            phase = "warmup" if index < 0 else f"sample {index + 1}"
            say(f"  {label}: {phase} {elapsed:.6f} s", level=3)

        result, timing = measure_synchronized(function, synchronize=stream.synchronize,
            warmups=args.warmups, repeats=args.repeats, on_sample=sample)
        say(f"  {label}: {timing['median_seconds'] * 1000:.3f} ms")
        return result, timing

    def relative(left, right):
        error = float((cp.linalg.norm(left - right) / cp.maximum(cp.linalg.norm(right), 1.e-30)).item())
        if not math.isfinite(error):
            raise ArithmeticError(f"nonfinite CuPy parity error: {error}")
        return error

    def matvec(assembled, vector):
        matrix = _assembly_device_csr_matrix(assembled, cp, sparse)
        return _device_compressed_matvec(matrix, vector, sparse, cp)

    def boundary(x, y):
        return 0. * x

    boundary._hybridge_constant_value = 0.

    def run_order(p):
        say(f"p={p}: prepare {report['triangles']} triangles")
        space = DGSpace(rectangle_mesh(args.nx, args.ny), p, basis_type=args.element_basis)
        device = as_cupy_space(space)
        tr = build_trace_reference(device, args.trace_basis)
        source, reaction = space.constant(1.), space.zeros()
        reduced = int(device.mesh.int_edges_inds.size) * device.edg_dof
        vector = cp.sin(.731 * cp.arange(reduced, dtype=REAL_DTYPE))
        trace = cp.sin(.273 * cp.arange(device.mesh.num_edg * device.edg_dof, dtype=REAL_DTYPE))
        common = dict(trace_basis=args.trace_basis, trace_ref=tr)
        row = dict(order=p, triangles=space.mesh.num_tri, trace_unknowns=reduced, raw=[])
        report["cases"].append(row)

        oracle, row["cupy_schur_cholesky_assembly"] = measure("CuPy Schur-Cholesky CSR assembly",
            lambda: cupy_assemble(source, reaction, boundary, args.tau, space,
                                  **common, use_schur_cholesky=True))
        expected_mv, expected_rhs = matvec(oracle, vector), oracle.rhs.copy()
        row["cupy_factor_bytes"] = oracle.schur_cholesky_cache.local_factor_bytes
        row["cupy_retained_device_bytes"] = cp.get_default_memory_pool().used_bytes()
        refreshed, row["cupy_rhs"] = measure("CuPy cached RHS",
            lambda: cupy_refresh(source, boundary, device, tr, oracle))
        row["cupy_rhs_relative_error"] = relative(refreshed.rhs, expected_rhs)
        del refreshed

        def cupy_reconstruct():
            traces = trace.reshape(device.mesh.num_edg, device.edg_dof)[device.mesh.loc2glob_edge]
            traces = traces.reshape(device.mesh.num_tri, 3 * device.edg_dof)
            rhs = oracle.source_rhs[..., None] + oracle.element_boundary_mats @ traces[..., None]
            return solve_mixed_from_scalar_cholesky_cupy(oracle.schur_cholesky_cache, rhs).squeeze(-1)

        expected_local, row["cupy_reconstruction"] = measure("CuPy mixed reconstruction", cupy_reconstruct)
        check_errors(row, tolerance=tolerance, label=f"p={p} CuPy")
        del oracle
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        save()

        for policy, fmt in RAW_CASES:
            label = f"raw {policy} {fmt}"
            say(f"p={p}: {label}")
            options = dict(matrix_format=fmt, block_size="auto", cache_local_factors=policy != "none",
                           local_factor_kind=policy if policy != "none" else "schur-lu")
            assembled, timing = measure(label + " assembly",
                lambda: raw_assemble(source, reaction, boundary, args.tau, space, **common, **options))
            raw = assembled.raw_assembly
            context = raw.poisson_context
            item = dict(policy=policy, format=fmt, assembly=timing,
                matrix_relative_error=relative(matvec(assembled, vector), expected_mv),
                rhs_relative_error=relative(assembled.rhs, expected_rhs),
                factor_bytes=context.local_factor_bytes,
                assembly_kernel_seconds=raw.timings["raw.kernel.device"], tuning=context.reports)
            # Match the original study: assembly in all formats; RHS and mixed
            # reconstruction for the CSR variant of each factor policy.
            if fmt == "csr":
                refreshed, item["rhs"] = measure(label + " RHS",
                    lambda: raw_refresh(source, reaction, boundary, args.tau, space, cached_raw=raw,
                                        **common, block_size="auto"))
                item["cached_rhs_relative_error"] = relative(refreshed.rhs, expected_rhs)
                del refreshed
                recovered, item["reconstruction"] = measure(label + " mixed reconstruction",
                    lambda: poisson.reconstruct(trace=trace, source_rhs=assembled.source_rhs,
                        cspace=device, trace_ref=tr, d0_reference=raw.d0_reference,
                        d1_reference=raw.d1_reference, face_element_mass=raw.face_element_mass,
                        tau=args.tau, context=context, block_size="auto", return_local_unknowns=True))
                item["reconstruction_relative_error"] = relative(recovered[1], expected_local)
                del recovered
            row["raw"].append(item)
            check_errors(item, tolerance=tolerance, label=f"p={p} {label}")
            save()
            say(f"  {label}: CuPy parity passed")
            del assembled, raw, context
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        say(f"p={p}: complete")

    for p in args.orders:
        run_order(p)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


def main(argv=None):
    args = parse_arguments(argv)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = dict(schema_version=1, arguments=arguments, triangles=2 * args.nx * args.ny,
        subdivisions=[args.nx, args.ny], trace_basis=args.trace_basis, element_basis=args.element_basis,
        tau=args.tau, source=1., dirichlet=0., warmups=args.warmups, repeats=args.repeats,
        timing="synchronized wall seconds; complete assembly wrappers; mesh/JIT/autotuning excluded",
        scope="stationary assembly/RHS/reconstruction only; no global solve, no NumPy assembly",
        raw_variants=[dict(policy=policy, format=fmt) for policy, fmt in RAW_CASES], cases=[], complete=False)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".log").open("w", buffering=1) as log:
        def say(message, *, level=1):
            print(message, file=log, flush=True)
            if args.verbosity >= level:
                print(message, flush=True)

        def save():
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            temporary.replace(args.output)

        with reserve_gpu(args.gpu_lock, timeout_seconds=args.wait_timeout, notify=say):
            paths = [path.relative_to(ROOT) for path in (ROOT / "hybridge").rglob("*.py")]
            paths += [Path(__file__).resolve().relative_to(ROOT), Path("scripts/reports/run_provenance.py"),
                      Path("pyproject.toml")]
            provenance = dict(recorded=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                python=platform.python_version(), packages=package_versions(), code=code_state(ROOT),
                source=source_snapshot(ROOT, paths, args.output.with_suffix(".source.tar.gz")))
            report["provenance"] = provenance
            save()
            try:
                run_comparison(args, report, say, save)
                report["complete"] = True
            except BaseException as exc:
                report["error"] = repr(exc)
                raise
            finally:
                changed = changed_snapshot_files(ROOT, provenance["source"])
                provenance["source_changes_during_run"] = changed
                if changed:
                    say("Source changes during run: " + ", ".join(changed))
                save()
            say(f"Done: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
