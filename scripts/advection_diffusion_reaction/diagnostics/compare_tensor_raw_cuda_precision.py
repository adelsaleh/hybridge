#!/usr/bin/env python3
"""Compare raw-CUDA tensor ADR assembly and reconstruction in FP32 and FP64.

Precision is fixed per process (``HDGFEM_PRECISION`` is read at import), so the
parent runs one worker per precision in a fresh process. Each worker samples
the case's coefficients on the device, assembles the reduced trace operator in
every requested format (COO, CSR, face-BSR) and reconstructs the local fields
from the exact solution's trace. No global solve or time integration is run.

The parent reports, per order and format:

* FP32 against FP64: relative Frobenius and max-entry differences of the
  assembled matrix, the right-hand side, and the reconstructed u, qx, qy;
* agreement of the three formats within each precision;
* device kernel times of both precisions.

It exits with status 1 when a difference exceeds its tolerance. Example::

    .venv/bin/python -m scripts.advection_diffusion_reaction.diagnostics.compare_tensor_raw_cuda_precision \\
        --case raw_tensor_sine --domain disk --mesh-size 0.05 --orders 1 3 6
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMATS = ("coo", "csr", "bsr")
PRECISIONS = ("float64", "float32")
FIELDS = ("u", "qx", "qy")


def _canonical_csr(system):
    """Return the assembled reduced matrix as a sorted, duplicate-free FP64 SciPy CSR."""
    import cupy as cp
    from scipy.sparse import bsr_matrix, coo_matrix, csr_matrix

    shape = (system.rhs.size,) * 2
    data = cp.asnumpy(system.data).astype(np.float64)
    if system.matrix_format == "coo":
        matrix = coo_matrix((data, (cp.asnumpy(system.rows), cp.asnumpy(system.cols))), shape=shape).tocsr()
    else:
        constructor = bsr_matrix if system.matrix_format == "bsr" else csr_matrix
        matrix = constructor((data, cp.asnumpy(system.indices), cp.asnumpy(system.indptr)), shape=shape).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def run_worker(args) -> None:
    """Assemble and reconstruct in this process's precision; write an ``.npz``."""
    import cupy as cp
    from hdgfem import DGSpace
    from hdgfem.hdg import condensation as hdg
    from hdgfem.backends.adr_coefficients_cupy import prepare_adr_data_cupy
    from hdgfem.backends.advection_diffusion_reaction_raw_cuda import (
        assemble_projected_adr_trace_operator_raw_cuda, reconstruct_projected_adr_local_unknowns_raw_cuda)
    from hdgfem.runtime.precision import PRECISION
    from scripts.advection_diffusion_reaction.cases import CASE_DEFINITIONS
    from scripts.advection_diffusion_reaction.run_cases import _build_mesh

    problem = CASE_DEFINITIONS[args.case].build()
    if problem.exact is None:
        raise ValueError(f"case {args.case!r} has no exact solution to reconstruct from")
    mesh = _build_mesh(SimpleNamespace(domain=args.domain, nx=args.nx, ny=args.nx, mesh_size=args.mesh_size,
                                       gmsh_verbosity=0, verbosity=0), problem)
    out = {"precision": np.array(PRECISION), "num_tri": np.array(mesh.num_tri)}
    for order in args.orders:
        space = DGSpace(mesh, order, basis_type="dub_orth", volume_quad_1d=args.volume_quad_1d)
        trace = space.trace_space(args.trace_basis)
        # The public ADR API takes callable or field velocity components, not numbers.
        beta = tuple(value if callable(value) else space.constant(value) for value in problem.beta)
        prepared = prepare_adr_data_cupy(problem.source, problem.reaction, beta, space,
                                         diffusion=problem.diffusion, trace_space=trace)
        full_trace = hdg.boundary_trace_coefficients(problem.exact, space, trace_space=trace)
        for matrix_format in args.formats:
            key = f"p{order}_{matrix_format}"
            try:
                kernel_times, reconstruction_times = [], []
                for _ in range(args.repeats):
                    operator = assemble_projected_adr_trace_operator_raw_cuda(
                        prepared, problem.boundary_condition, space, diffusion=problem.diffusion,
                        trace_space=trace, matrix_format=matrix_format, cache_local_factors=args.cache)
                    unknowns, timings = reconstruct_projected_adr_local_unknowns_raw_cuda(operator, full_trace)
                    kernel_times.append(operator.assembly.timings["raw.kernel.device"])
                    reconstruction_times.append(timings["raw.reconstruction.device"])
            except np.linalg.LinAlgError as exc:
                out[f"{key}_error"] = np.array(str(exc))
                continue
            matrix = _canonical_csr(operator.assembly)
            out.update({
                f"{key}_indptr": matrix.indptr, f"{key}_indices": matrix.indices, f"{key}_data": matrix.data,
                f"{key}_rhs": cp.asnumpy(operator.assembly.rhs).astype(np.float64),
                f"{key}_unknowns": cp.asnumpy(unknowns).astype(np.float64),
                f"{key}_nel": np.array(space.el_dof),
                f"{key}_kinds": np.array(json.dumps({k: v for k, v in operator.diffusion_structure.items() if v})),
                f"{key}_kernel": np.array(min(kernel_times)),
                f"{key}_reconstruction": np.array(min(reconstruction_times)),
            })
    np.savez(args.out, **out)


def _relative(a, b):
    """Relative Frobenius and max-entry differences of ``a`` against reference ``b``."""
    scale_2, scale_max = np.linalg.norm(b), np.max(np.abs(b), initial=0.)
    delta = a - b
    return (float(np.linalg.norm(delta) / scale_2) if scale_2 else 0.,
            float(np.max(np.abs(delta), initial=0.) / scale_max) if scale_max else 0.)


def _matrix(results, key):
    """Rebuild a worker's canonical CSR matrix."""
    from scipy.sparse import csr_matrix

    n = results[f"{key}_rhs"].size
    return csr_matrix((results[f"{key}_data"], results[f"{key}_indices"], results[f"{key}_indptr"]), shape=(n, n))


def _matrix_difference(a, b):
    """Relative Frobenius and max-entry differences of sparse ``a`` against ``b``."""
    delta = (a - b).tocsr()
    scale_2, scale_max = np.linalg.norm(b.data), np.max(np.abs(b.data), initial=0.)
    return (float(np.linalg.norm(delta.data) / scale_2), float(np.max(np.abs(delta.data), initial=0.) / scale_max))


def compare(results, args):
    """Print the comparison tables; return a list of tolerance violations."""
    r64, r32 = results["float64"], results["float32"]
    failures = []
    print(f"case={args.case} domain={args.domain} triangles={int(r64['num_tri'])} "
          f"trace={args.trace_basis} cache={args.cache}")
    header = (f"{'p':>2} {'fmt':>4} {'kappa kinds':<28} {'A fro':>9} {'A max':>9} {'rhs fro':>9} "
              f"{'u fro':>9} {'qx fro':>9} {'qy fro':>9} {'kernel64':>9} {'kernel32':>9} {'recon64':>9} {'recon32':>9}")
    print("\nFP32 against FP64 (relative differences; kernel/reconstruction device seconds)")
    print(header)
    print("-" * len(header))
    for order in args.orders:
        for matrix_format in args.formats:
            key = f"p{order}_{matrix_format}"
            for name, results_p in (("float64", r64), ("float32", r32)):
                if f"{key}_error" in results_p:
                    failures.append(f"{key} {name}: {results_p[f'{key}_error']}")
            if f"{key}_rhs" not in r64 or f"{key}_rhs" not in r32:
                print(f"{order:>2} {matrix_format:>4} local factorization failed (see failures)")
                continue
            a_fro, a_max = _matrix_difference(_matrix(r32, key), _matrix(r64, key))
            rhs_fro, _ = _relative(r32[f"{key}_rhs"], r64[f"{key}_rhs"])
            nel = int(r64[f"{key}_nel"])
            fields = [_relative(r32[f"{key}_unknowns"][:, j*nel:(j+1)*nel], r64[f"{key}_unknowns"][:, j*nel:(j+1)*nel])[0]
                      for j in range(3)]
            kinds = str(r64[f"{key}_kinds"])
            print(f"{order:>2} {matrix_format:>4} {kinds:<28.28} {a_fro:9.2e} {a_max:9.2e} {rhs_fro:9.2e} "
                  f"{fields[0]:9.2e} {fields[1]:9.2e} {fields[2]:9.2e} "
                  f"{float(r64[f'{key}_kernel']):9.4f} {float(r32[f'{key}_kernel']):9.4f} "
                  f"{float(r64[f'{key}_reconstruction']):9.4f} {float(r32[f'{key}_reconstruction']):9.4f}")
            worst = max(a_fro, rhs_fro, *fields)
            if not np.isfinite(worst) or worst > args.fp32_rtol:
                failures.append(f"{key}: FP32 vs FP64 relative difference {worst:.2e} > {args.fp32_rtol:.1e}")
    print("\nFormat agreement within each precision (relative Frobenius vs COO: matrix, rhs, unknowns)")
    for name, results_p, tolerance in (("float64", r64, args.fp64_format_rtol), ("float32", r32, args.fp32_format_rtol)):
        for order in args.orders:
            reference = f"p{order}_{args.formats[0]}"
            if f"{reference}_rhs" not in results_p:
                continue
            for matrix_format in args.formats[1:]:
                key = f"p{order}_{matrix_format}"
                if f"{key}_rhs" not in results_p:
                    continue
                values = (_matrix_difference(_matrix(results_p, key), _matrix(results_p, reference))[0],
                          _relative(results_p[f"{key}_rhs"], results_p[f"{reference}_rhs"])[0],
                          _relative(results_p[f"{key}_unknowns"], results_p[f"{reference}_unknowns"])[0])
                print(f"  {name} p={order} {matrix_format} vs {args.formats[0]}: "
                      + "  ".join(f"{label} {value:.2e}" for label, value in zip(("A", "rhs", "unknowns"), values)))
                if max(values) > tolerance:
                    failures.append(f"{name} p={order} {matrix_format} vs {args.formats[0]}: "
                                    f"{max(values):.2e} > {tolerance:.1e}")
    return failures


def parse_args(argv=None):
    """Command-line options shared by the parent and its workers."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="raw_tensor_sine", help="ADR case with an exact solution")
    parser.add_argument("--domain", default="disk", choices=("disk", "square", "unit-square"))
    parser.add_argument("--mesh-size", type=float, default=0.05, help="Gmsh target size (disk)")
    parser.add_argument("--nx", type=int, default=16, help="cells per side (square domains)")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    parser.add_argument("--trace-basis", default="legendre-modal", choices=("legacy-lagrange", "legendre-modal"))
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--cache", default="schur-lu+mass", choices=("none", "schur-lu", "schur-lu+mass"),
                        help="cache_local_factors used by assembly and reconstruction")
    parser.add_argument("--repeats", type=int, default=2, help="timed repetitions; the minimum is reported")
    parser.add_argument("--fp32-rtol", type=float, default=1e-3,
                        help="max FP32-vs-FP64 relative Frobenius difference of matrix, rhs and fields")
    parser.add_argument("--fp32-format-rtol", type=float, default=1e-5)
    parser.add_argument("--fp64-format-rtol", type=float, default=1e-12)
    parser.add_argument("--json", type=Path, help="also write the failure list and settings as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run one worker per precision, then compare their results."""
    args = parse_args(argv)
    if args.worker:
        run_worker(args)
        return 0
    passthrough = [arg for arg in (argv if argv is not None else sys.argv[1:])]
    results = {}
    with tempfile.TemporaryDirectory(prefix="adr_precision_") as scratch:
        for precision in PRECISIONS:
            out = Path(scratch) / f"{precision}.npz"
            env = dict(os.environ, HDGFEM_PRECISION=precision)
            command = [sys.executable, "-m", "scripts.advection_diffusion_reaction.diagnostics."
                       "compare_tensor_raw_cuda_precision", *passthrough, "--worker", "--out", str(out)]
            print(f"running {precision} worker ...", flush=True)
            subprocess.run(command, env=env, cwd=ROOT, check=True)
            with np.load(out) as data:
                results[precision] = {key: data[key] for key in data.files}
    failures = compare(results, args)
    print("\n" + ("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures)))
    if args.json:
        args.json.write_text(json.dumps({"settings": {k: str(v) for k, v in vars(args).items()},
                                         "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
