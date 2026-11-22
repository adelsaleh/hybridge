"""Required Q1 dependencies: host linkage, then real cuTENSOR and MAGMA operations.

Missing libraries are failures, never skips. Host checks do not initialize CUDA.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def check(*, host_only=False):
    report = {"status": "failed", "scope": "host" if host_only else "gpu", "checks": {}, "errors": {}}

    def attempt(name, operation):
        try:
            report["checks"][name] = operation()
        except Exception as error:
            report["errors"][name] = f"{type(error).__name__}: {error}"

    def cutensor_host():
        from hybridge.runtime.optional import require_cutensor
        require_cutensor()  # Includes the libcutensorMg preload required by CuPy 14.
        from cupy_backends.cuda.libs import cutensor
        return {"package": importlib.metadata.version("cutensor-cu13"), "runtime": cutensor.get_version()}

    def magma_host():
        from hybridge.linalg.gpu.magma_batched import _declare, _library_candidates
        candidates = _library_candidates()
        if not candidates:
            raise RuntimeError("MAGMA shared library not installed")
        library = ctypes.CDLL(candidates[0], mode=ctypes.RTLD_GLOBAL)
        _declare(library)  # Require every symbol used by the application, without magma_init.
        library.magma_version.argtypes = [ctypes.POINTER(ctypes.c_int)] * 3
        library.magma_version.restype = None
        version = [ctypes.c_int() for _ in range(3)]
        library.magma_version(*(ctypes.byref(part) for part in version))
        result = {"library": candidates[0], "version": ".".join(str(part.value) for part in version),
                  "binding_integer_bits": 32}
        if Path(candidates[0]).is_file():
            with Path(candidates[0]).open("rb") as stream:
                result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result

    attempt("cutensor", cutensor_host)
    attempt("magma", magma_host)
    if not host_only and not report["errors"]:
        def operations():
            import cupy as cp
            import numpy as np
            from cupy_backends.cuda.libs import cutensor
            from hybridge.transport.tsle_tensor import _CutensorContraction
            from hybridge.linalg.gpu.magma_batched import lu_solve_batched_magma

            device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
            if device["major"] != 9:
                raise RuntimeError(f"Expected Hopper device (sm_90), got {device['name']}")
            result = {"device": device["name"].decode(), "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                      "cuda_driver": cp.cuda.runtime.driverGetVersion(), "operations": []}
            rng = np.random.default_rng(26)
            for dtype, compute, tolerance in ((np.float64, cutensor.COMPUTE_DESC_64F, 1e-12),
                                               (np.float32, cutensor.COMPUTE_DESC_32F, 1e-4)):
                a = rng.standard_normal((8, 8)) + 8 * np.eye(8)
                b = rng.standard_normal((8, 3))
                ca, cb = cp.asarray(a, dtype=dtype), cp.asarray(b, dtype=dtype)
                out = cp.zeros((8, 3), dtype=dtype)
                contraction = _CutensorContraction(cp, ca, "ik", cb, "kj", out, "ij", compute)
                contraction(ca, cb, out)
                cp.cuda.get_current_stream().synchronize()
                np.testing.assert_allclose(cp.asnumpy(out), a.astype(dtype) @ b.astype(dtype),
                                           rtol=tolerance, atol=tolerance)
                for fused in (False, True):
                    matrices = np.repeat(a[None], 4, axis=0)
                    rhs = np.repeat(b[None], 4, axis=0)
                    # MAGMA expects contiguous, column-major matrices and RHSs.
                    da = cp.asarray(np.ascontiguousarray(matrices.transpose(0, 2, 1)), dtype=dtype)
                    db = cp.asarray(np.ascontiguousarray(rhs.transpose(0, 2, 1)), dtype=dtype)
                    lu_solve_batched_magma(da, db, fused=fused)
                    cp.cuda.get_current_stream().synchronize()
                    expected = np.linalg.solve(matrices, rhs).transpose(0, 2, 1)
                    np.testing.assert_allclose(cp.asnumpy(db), expected, rtol=tolerance, atol=tolerance)
                result["operations"].append({"dtype": np.dtype(dtype).name, "cutensor": "passed",
                                             "magma_getrf_getrs": "passed", "magma_gesv": "passed"})
            return result

        attempt("gpu_operations", operations)
    if not report["errors"]:
        report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(host_only=args.host_only)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, args.output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
