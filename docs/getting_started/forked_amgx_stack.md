# Forked AMGX/PyAMGX Stack

## Required Post-Release Dependency Set

The `v0.1.0a1` tag is the standalone early-alpha release boundary. Every HDG
commit after that tag is developed and qualified as part of this three-repository
stack:

| Component | Required branch | Exact revision qualified here |
|---|---|---|
| HDG | post-`v0.1.0a1` `master` | this checkout |
| AMGX | [`adelsaleh/AMGX@hdg-cuda13-integration`](https://github.com/adelsaleh/AMGX/tree/hdg-cuda13-integration) | `583084b` |
| PyAMGX | [`adelsaleh/pyamgx@quality-of-life`](https://github.com/adelsaleh/pyamgx/tree/quality-of-life) | `81efd1e` |

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
git clone --branch hdg-cuda13-integration https://github.com/adelsaleh/AMGX.git
git clone --branch quality-of-life https://github.com/adelsaleh/pyamgx.git

git -C AMGX merge-base --is-ancestor 583084b HEAD
git -C pyamgx merge-base --is-ancestor 81efd1e HEAD
```

Those two checks must exit successfully. For exact reproduction rather than
continued branch development, detach both checkouts at the qualified pins:

```bash
git -C AMGX checkout --detach 583084b
git -C pyamgx checkout --detach 81efd1e
```

## Build AMGX

Use a CUDA toolkit compatible with the target driver. The current fork was
built and tested with CUDA 13.0.88. Set `CMAKE_CUDA_ARCHITECTURES` for the
actual GPU; `75` below is the value used on the benchmark machine and is only
an example.

```bash
cmake -S AMGX -B /tmp/AMGX-build-cuda13.0.1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/tmp/cuda-13.0.1/bin/nvcc \
  -DCUDAToolkit_ROOT=/tmp/cuda-13.0.1 \
  -DCUDA_TOOLKIT_ROOT_DIR=/tmp/cuda-13.0.1 \
  -DCMAKE_NO_MPI=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DAMGX_INCLUDE_EXTERNAL=ON \
  -DAMGX_NO_RPATH=OFF \
  -DCMAKE_INSTALL_PREFIX=/tmp/AMGX-install-cuda13.0.1
cmake --build /tmp/AMGX-build-cuda13.0.1 --target amgxsh --parallel 48
cmake --install /tmp/AMGX-build-cuda13.0.1
```

Do not reuse a cache first configured through `/usr/local/cuda`: on the current
machine that symlink names CUDA 12.8, and CMake retains that compiler even if a
CUDA 13 runtime directory is later prepended to `LD_LIBRARY_PATH`. Verify a
cache before building with:

```bash
rg 'CMAKE_CUDA_COMPILER:STRING' /tmp/AMGX-build-cuda13.0.1/CMakeCache.txt
```

It must report `/tmp/cuda-13.0.1/bin/nvcc`.

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
AMGX_BUILD_DIR=/tmp/AMGX-build-cuda13.0.1 \
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
export LD_LIBRARY_PATH="/tmp/AMGX-build-cuda13.0.1:/tmp/AMGX-install-cuda13.0.1/lib:/tmp/cuda-13.0.1/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For HDG development on the qualified machine, keep the unsuffixed `/tmp`
aliases pointed at those exact CUDA-13 trees:

```bash
ln -s /tmp/cuda-13.0.1 /tmp/cuda
ln -s /tmp/AMGX-build-cuda13.0.1 /tmp/AMGX-build
ln -s /tmp/AMGX-install-cuda13.0.1 /tmp/AMGX-install
```

Inspect or move any pre-existing target before creating an alias; never reuse
an unsuffixed CMake build directory first configured with CUDA 12.8. The checked launcher resolves all three aliases, requires an NVCC 13.x
report, verifies that the AMGX CMake compiler belongs to the same toolkit, and
checks that AMGX's cuBLAS/nvJitLink dependencies resolve there. It exports
`CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `NVCC`, `PATH`, and `LD_LIBRARY_PATH`
together. Audit the stack without launching a program, then run through the
same guard:

```bash
scripts/gpu/run_cuda13.sh --check
scripts/gpu/run_cuda13.sh .venv/bin/python -c \
  "import cupy, pyamgx; print(cupy.cuda.runtime.runtimeGetVersion(), cupy.cuda.nvrtc.getVersion(), pyamgx.__file__)"
```

Use this launcher for GPU tests, benchmarks, and guiding-center runs instead of
hand-writing a partial `LD_LIBRARY_PATH`. Advanced installations can override
`HDGFEM_CUDA13_ROOT`, `HDGFEM_AMGX_BUILD_ROOT`, and
`HDGFEM_AMGX_INSTALL_ROOT`.

Verify the source pins and imported extension before running HDG evidence:

```bash
git -C AMGX rev-parse HEAD
git -C pyamgx rev-parse HEAD
python -c "import pyamgx; print(pyamgx.__file__)"
```

For an exact reproduction, the first two commands must print commits `583084b`
and `81efd1e`. For branch-tip development, each `HEAD` must contain its pin and
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
