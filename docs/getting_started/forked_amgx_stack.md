# Forked AMGX/PyAMGX Stack

## Required Post-Release Dependency Set

The `v0.1.0a1` tag is the standalone early-alpha release boundary. Every HDG
commit after that tag is developed and qualified as part of this three-repository
stack:

| Component | Required branch | Exact revision qualified here |
|---|---|---|
| HDG | post-`v0.1.0a1` `master` | this checkout |
| AMGX | [`adelsaleh/AMGX@quality-of-life`](https://github.com/adelsaleh/AMGX/tree/quality-of-life) | `6699fa4` |
| PyAMGX | [`adelsaleh/pyamgx@quality-of-life`](https://github.com/adelsaleh/pyamgx/tree/quality-of-life) | `6b26b12` |

The AMGX pin contains the earlier diagnostics/memory-reporting commit and the
block-aware classical-AMG/BSR work. The PyAMGX pin contains the earlier device
error/memory bindings and the AMGX print-callback signature fix. The upstream
`NVIDIA/AMGX` and `shwina/pyamgx` `main` branches do not currently provide this
combined interface and are not interchangeable with the pins above.

Host-only NumPy, Numba, SciPy, PARDISO, and PETSc paths keep optional CUDA
imports lazy and can still run without AMGX installed. That does not change the
repository qualification rule: post-release HDG validation, CUDA evidence, and
performance claims must use the forked dependency set.

## Clone And Verify The Fork Branches

From a common source directory:

```bash
git clone --branch quality-of-life https://github.com/adelsaleh/AMGX.git
git clone --branch quality-of-life https://github.com/adelsaleh/pyamgx.git

git -C AMGX merge-base --is-ancestor 6699fa4 HEAD
git -C pyamgx merge-base --is-ancestor 6b26b12 HEAD
```

Those two checks must exit successfully. For exact reproduction rather than
continued branch development, detach both checkouts at the qualified pins:

```bash
git -C AMGX checkout --detach 6699fa4
git -C pyamgx checkout --detach 6b26b12
```

## Build AMGX

Use a CUDA toolkit compatible with the target driver. The current fork was
built and tested with CUDA 13.0.88. Set `CMAKE_CUDA_ARCHITECTURES` for the
actual GPU; `75` below is the value used on the benchmark machine and is only
an example.

```bash
cmake -S AMGX -B AMGX/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_NO_MPI=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build AMGX/build --target amgxsh --parallel 48
```

The CUDA-13 generic cuSPARSE BSR backend is the preferred AMGX BSR SpMV path.
Other supported toolkits may use the retained legacy/custom fallback, but those
runs are a different performance configuration and must be labeled as such.

## Build And Install PyAMGX Against That Library

Activate the Python environment used by HDG, ensure its normal build
dependencies are present, and point PyAMGX at both the AMGX source and build
directories:

```bash
python -m pip install --upgrade cython setuptools wheel

AMGX_DIR="$PWD/AMGX" \
AMGX_BUILD_DIR="$PWD/AMGX/build" \
python -m pip install \
  --no-build-isolation \
  --no-deps \
  --force-reinstall \
  ./pyamgx
```

Reinstall PyAMGX whenever the AMGX C API, public headers, shared library ABI,
or PyAMGX Cython declarations change. Rebuilding only `libamgxsh.so` is enough
for implementation-only AMGX changes that preserve those interfaces, although
rerunning the import and solve checks below is still required.

## Runtime Library And Revision Checks

PyAMGX records a runtime library directory during the extension build. If the
loader cannot find `libamgxsh.so`, expose the same AMGX build directory used
above:

```bash
export LD_LIBRARY_PATH="$PWD/AMGX/build:${LD_LIBRARY_PATH:-}"
```

Verify the source pins and imported extension before running HDG evidence:

```bash
git -C AMGX rev-parse HEAD
git -C pyamgx rev-parse HEAD
python -c "import pyamgx; print(pyamgx.__file__)"
```

For an exact reproduction, the first two commands must print commits `6699fa4`
and `6b26b12`. For branch-tip development, each `HEAD` must contain its pin and
the additional commits must be recorded with the HDG result. Finally, run the
smallest relevant HDG AMGX parity case before any heavy benchmark; a successful
import alone does not qualify the CUDA solver stack.

## Ownership Of The BSR Paths

HDG owns direct face-BSR assembly, matrix/RHS caching, basis transforms,
physical-residual acceptance, and benchmark orchestration. PyAMGX owns the
Python/Cython upload and handle interface. AMGX owns uploaded matrices,
hierarchy construction, block smoothers, generic-BSR SpMV selection, and the
iterative solve. The detailed call and data-ownership map is in
[`../backends/bsr_amgx_dependency_map.md`](../backends/bsr_amgx_dependency_map.md).
