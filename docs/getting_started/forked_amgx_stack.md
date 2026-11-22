# Forked AMGX/PyAMGX Stack

## Required Post-Release Dependency Set

The `v0.1.0a1` tag is the standalone early-alpha release boundary. Every HDG
commit after that tag is developed and qualified as part of this three-repository
stack:

| Component | Required branch | Historical release qualification |
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

## H200 Campaign Snapshot (2026-10-31)

Hybridge `master` includes the H200 campaign work. Its combined CUDA 13.1/H200
qualification uses:

| Fork | Branch | Qualified commit |
|---|---|---|
| AMGX | `hdg-cuda13-integration` | `2a6625b840f03e2264e0ab3bb79d4733d8f197bd` |
| PyAMGX | `quality-of-life` | `a95e878b1cced80705237dccd5a9c22448d3787b` |

Each is the single published commit that carries the qualified source: its
`src` and `include` trees are those that were built and qualified. The older
pins above reproduce the historical release
qualification, without the new borrowed-BSR and supplied-order ILU(k) additions.
For the H200 work, build both forks at these newer commits. See the
primary runtime and validation record (`docs/campaigns/fork_integration/primary_promotion.md`, local, untracked).
The original combined qualification used the isolated build
`builds/AMGX/cuda13.1-integration`. The AMGX commit above includes the
qualified native ILU synchronization and truncation corrections. The
native ILU record (`docs/campaigns/matk/ilu_synchronization_qualification.md`, local, untracked) and
truncation record (`docs/campaigns/fork_integration/records/truncation-29209/README.md`, local, untracked)
retain their separate candidate identities. The grouped
build layout (`docs/campaigns/fork_integration/build_layout.md`, local, untracked) records runtime
promotion and retirement of the obsolete isolated builds.

## Build AMGX

The new supplied-order ILU(k) extension is **experimental** and requires an
explicit configuration. Its autotuning is deferred; qualification of the native
ILU synchronization fixes does not promote ILU(k) into an automatic solver
choice. See the [tuning policy](../development/architecture_tuning.md).

Choose a CUDA 13 toolkit compatible with the driver and an architecture for
this GPU. From the common source directory, configure explicit locations:

```bash
export HYBRIDGE_CUDA13_ROOT="${CUDA_HOME:?Set CUDA_HOME to the selected CUDA 13 toolkit}"
export HYBRIDGE_AMGX_BUILD_ROOT="$PWD/builds/AMGX/cuda13"
export HYBRIDGE_AMGX_INSTALL_ROOT="$PWD/builds/AMGX/install-cuda13"
export CUDA_ARCHITECTURES=120  # example: replace with the target GPU architecture

cmake -S AMGX -B "$HYBRIDGE_AMGX_BUILD_ROOT" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$HYBRIDGE_CUDA13_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$HYBRIDGE_CUDA13_ROOT" \
  -DCUDA_TOOLKIT_ROOT_DIR="$HYBRIDGE_CUDA13_ROOT" \
  -DCMAKE_NO_MPI=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DAMGX_INCLUDE_EXTERNAL=ON \
  -DAMGX_NO_RPATH=OFF \
  -DCMAKE_INSTALL_PREFIX="$HYBRIDGE_AMGX_INSTALL_ROOT"
cmake --build "$HYBRIDGE_AMGX_BUILD_ROOT" --target amgx amgxsh amgx_capi --parallel
cmake --install "$HYBRIDGE_AMGX_BUILD_ROOT"
```

A CMake cache retains the compiler selected when it was created. Before
reusing a build directory, verify that its compiler belongs to the selected
toolkit; changing `LD_LIBRARY_PATH` does not change the cached compiler:

```bash
rg 'CMAKE_CUDA_COMPILER:' "$HYBRIDGE_AMGX_BUILD_ROOT/CMakeCache.txt"
"$HYBRIDGE_CUDA13_ROOT/bin/nvcc" --version
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
AMGX_BUILD_DIR="$HYBRIDGE_AMGX_BUILD_ROOT" \
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

PyAMGX records a runtime library directory during the extension build. Use the
same source/build locations when building the extension and running HYBRIDGE.
From the HYBRIDGE checkout, the checked launcher validates the selected CUDA
13 toolkit, AMGX compiler and runtime dependencies before running a command:

```bash
scripts/gpu/run_cuda13.sh --check
scripts/gpu/run_cuda13.sh .venv/bin/python -c \
  "import cupy, pyamgx; print(cupy.cuda.runtime.runtimeGetVersion(), cupy.cuda.nvrtc.getVersion(), pyamgx.__file__)"
```

`HYBRIDGE_CUDA13_ROOT` selects the toolkit, falling back to `CUDA_HOME`,
`CUDA_PATH`, then the conventional `/usr/local/cuda` location. The AMGX build
and install directories default to the neighboring `AMGX-build-cuda13` and
`AMGX-install-cuda13` directories; set `HYBRIDGE_AMGX_BUILD_ROOT` and
`HYBRIDGE_AMGX_INSTALL_ROOT` for a different layout. `AMGX_BUILD_DIR` is also
accepted as the build-directory fallback. The launcher exports the compiler,
toolkit and library search paths together. Python GPU sweep/inspection tools
inherit that environment instead of injecting paths from another machine.

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
