"""Check AMGX MULTICOLOR_DILU at every block size on the GPU.

Three checks:

* ``--mode sweep`` applies DILU (as a stand-alone solver, a fixed number of sweeps) and FGMRES+DILU
  to a nonsymmetric block grid matrix at each block size, and saves the iterates to ``--save``.
* Embedding parity, in the same run: a block-``b`` matrix embedded in block ``b+1`` with a decoupled
  identity component has the same sparsity, hence the same colouring, and in exact arithmetic the
  same DILU factor. Its first ``b`` components must match the block-``b`` result to round-off.
  This checks a new block size against one already in production use.
* ``--mode compare OLD NEW`` checks that two saved sweeps (for example the production library and
  a rebuilt one) agree bitwise at every block size they share.

Run with ``scripts/gpu/run_cuda13.sh``; ``HYBRIDGE_AMGX_BUILD_ROOT`` selects the library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

SWEEPS = 3
FGMRES = {
    "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
    "solver": {"solver": "FGMRES", "gmres_n_restart": 50, "max_iters": 500, "convergence": "RELATIVE_INI_CORE",
               "tolerance": 1e-10, "norm": "L2", "use_scalar_norm": 1,
               "preconditioner": {"solver": "MULTICOLOR_DILU", "max_iters": 1, "relaxation_factor": 1.0}},
}
DILU = {
    "config_version": 2, "determinism_flag": 1, "exception_handling": 1,
    "solver": {"solver": "MULTICOLOR_DILU", "max_iters": SWEEPS, "relaxation_factor": 1.0,
               "convergence": "ABSOLUTE", "tolerance": 1e-300, "norm": "L2", "use_scalar_norm": 1},
}


def grid_matrix(b: int, nx: int, ny: int, seed: int):
    """Nonsymmetric, block diagonally dominant 5-point block matrix: (indptr, indices, blocks)."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for j in range(ny):
        for i in range(nx):
            r = j * nx + i
            for di, dj in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= i + di < nx and 0 <= j + dj < ny:
                    rows.append(r)
                    cols.append((j + dj) * nx + i + di)
    pattern = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(nx * ny, nx * ny))
    pattern.sort_indices()
    indptr, indices = pattern.indptr.astype(np.int32), pattern.indices.astype(np.int32)
    blocks = rng.uniform(-1.0, 1.0, size=(indices.size, b, b)) / (4.0 * b)
    for r in range(nx * ny):
        for k in range(indptr[r], indptr[r + 1]):
            if indices[k] == r:
                blocks[k] = rng.uniform(-0.5, 0.5, size=(b, b)) / b + 2.0 * np.eye(b)
    return indptr, indices, blocks


def embed(indptr, indices, blocks, extra: int):
    """Append ``extra`` decoupled identity components to every block."""
    b = blocks.shape[1]
    out = np.zeros((blocks.shape[0], b + extra, b + extra))
    out[:, :b, :b] = blocks
    for r in range(indptr.size - 1):
        for k in range(indptr[r], indptr[r + 1]):
            if indices[k] == r:
                out[k, b:, b:] = np.eye(extra)
    return out


def bsr(indptr, indices, blocks):
    n = indptr.size - 1
    b = blocks.shape[1]
    return sp.bsr_matrix((blocks, indices, indptr), shape=(n * b, n * b))


class DeviceMatrix:
    """The attributes ``PyAMGXCsrDeviceSolver.setup`` reads."""

    def __init__(self, cp, indptr, indices, blocks):
        self.indptr, self.indices = cp.asarray(indptr), cp.asarray(indices)
        self.data = cp.asarray(np.ascontiguousarray(blocks))
        self.block_size = int(blocks.shape[1])
        n = (indptr.size - 1) * self.block_size
        self.shape = (n, n)


def run(config, matrix, rhs, cp):
    from hybridge.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver

    solver = PyAMGXCsrDeviceSolver(config=config)
    try:
        started = time.perf_counter()
        solver.setup(matrix)
        setup = time.perf_counter() - started
        x, info = solver.solve(cp.asarray(rhs))
        return cp.asnumpy(x), setup, info
    finally:
        solver.close(suppress_errors=True)


def loaded_amgx() -> str:
    """Path of the libamgxsh.so mapped into this process."""
    with open("/proc/self/maps") as maps:
        return next((line.split()[-1] for line in maps if "libamgxsh" in line), "not loaded")


def sweep(args) -> int:
    import cupy as cp
    import pyamgx  # noqa: F401  (maps libamgxsh)

    library = loaded_amgx()
    print(f"amgx library {library} (HYBRIDGE_AMGX_BUILD_ROOT={os.environ.get('HYBRIDGE_AMGX_BUILD_ROOT')})", flush=True)
    saved, failures, rows = {}, 0, []
    for b in args.blocks:
        # Per-block generators: a block's matrix and right-hand side do not depend on which other
        # blocks, or embeddings, the run includes, so two runs compare bitwise.
        indptr, indices, blocks = grid_matrix(b, args.nx, args.ny, args.seed + b)
        rhs = np.random.default_rng((args.seed, b)).uniform(-1.0, 1.0, size=(indptr.size - 1) * b)
        host = bsr(indptr, indices, blocks).tocsr()
        matrix = DeviceMatrix(cp, indptr, indices, blocks)
        x_dilu, setup, info = run(DILU, matrix, rhs, cp)
        x_gm, gm_setup, gm_info = run(FGMRES, matrix, rhs, cp)
        true_rel = float(np.linalg.norm(rhs - host @ x_gm) / np.linalg.norm(rhs))
        saved[f"dilu_{b}"], saved[f"fgmres_{b}"] = x_dilu, x_gm
        row = {"block": b, "dilu_sweeps": SWEEPS, "dilu_setup_s": setup, "dilu_finite": bool(np.isfinite(x_dilu).all()),
               "fgmres_iterations": gm_info.get("amgx_iterations"), "fgmres_status": gm_info.get("amgx_status"),
               "fgmres_true_relative_residual": true_rel, "fgmres_setup_s": gm_setup,
               "fgmres_solve_s": gm_info.get("amgx_solve_elapsed_seconds")}
        ok = row["dilu_finite"] and true_rel <= 1e-9
        if b + 1 <= args.max_embed:
            big = embed(indptr, indices, blocks, 1)
            extra = np.random.default_rng((args.seed, b, 1)).uniform(-1.0, 1.0, size=(indptr.size - 1, 1))
            rhs_big = np.concatenate([rhs.reshape(-1, b), extra], axis=1).ravel()
            x_big, _, _ = run(DILU, DeviceMatrix(cp, indptr, indices, big), rhs_big, cp)
            diff = float(np.abs(x_big.reshape(-1, b + 1)[:, :b].ravel() - x_dilu).max() / np.abs(x_dilu).max())
            row["embedding_into"] = b + 1
            row["embedding_relative_max_diff"] = diff
            ok = ok and diff <= 1e-12
        row["ok"] = bool(ok)
        failures += not ok
        rows.append(row)
        print(f"timings dilu block={b}: " + " ".join(f"{k}={v}" for k, v in row.items() if k != "block"), flush=True)
    if args.save:
        np.savez(args.save, **saved)
        with open(args.save + ".json", "w") as handle:
            json.dump({"library": library, "nx": args.nx, "ny": args.ny, "seed": args.seed, "rows": rows}, handle, indent=1)
    print(f"dilu sweep: {len(rows) - failures}/{len(rows)} ok", flush=True)
    return 1 if failures else 0


def compare(old: str, new: str) -> int:
    a, b = np.load(old), np.load(new)
    shared = sorted(set(a.files) & set(b.files))
    mismatched = [key for key in shared if not np.array_equal(a[key], b[key])]
    for key in shared:
        print(f"compare {key}: {'bitwise' if key not in mismatched else 'DIFFERS max %.3e' % np.abs(a[key] - b[key]).max()}")
    print(f"dilu compare: {len(shared) - len(mismatched)}/{len(shared)} bitwise identical", flush=True)
    return 1 if mismatched or not shared else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("sweep", "compare"), default="sweep")
    parser.add_argument("--blocks", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--max-embed", type=int, default=10, help="largest block size an embedding may target")
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--save", default=None, help="npz path for the iterates (plus .json with the rows)")
    parser.add_argument("files", nargs="*", help="--mode compare: OLD.npz NEW.npz")
    args = parser.parse_args()
    if args.mode == "compare":
        return compare(*args.files)
    args.blocks = [int(value) for value in args.blocks.split(",")]
    return sweep(args)


if __name__ == "__main__":
    sys.exit(main())
