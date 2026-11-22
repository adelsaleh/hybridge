#!/usr/bin/env bash
# Install the Q1 optional libraries, build only MAGMA's shared library for Hopper.
# Offline: set HYBRIDGE_MAGMA_SOURCE and HYBRIDGE_CUTENSOR_WHEEL to staged files.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$repo/campaigns/common/hopper_env.sh"
cd "$repo"
"$repo/scripts/gpu/run_cuda13.sh" --check
if [[ $# -eq 1 && $1 == --check ]]; then
    exec "$repo/scripts/gpu/run_cuda13.sh" "$HYBRIDGE_PYTHON" -m campaigns.common.hopper_stack --host-only
elif [[ $# -ne 0 ]]; then
    echo 'usage: bash campaigns/common/setup_hopper.sh [--check]' >&2
    exit 2
fi

mkdir -p "$repo/.cache/hopper-setup"
if [[ -n ${HYBRIDGE_CUTENSOR_WHEEL:-} ]]; then
    "$HYBRIDGE_PYTHON" -m pip install --no-index --no-deps "$HYBRIDGE_CUTENSOR_WHEEL"
else
    "$HYBRIDGE_PYTHON" -m pip install --cache-dir "$repo/.cache/pip" --retries 1 --timeout 15 'cutensor-cu13==2.8.1'
fi

magma_source=${HYBRIDGE_MAGMA_SOURCE:-$repo/../magma-2.10.0}
if [[ ! -f $magma_source/CMakeLists.txt ]]; then
    if [[ -n ${HYBRIDGE_MAGMA_SOURCE:-} ]]; then
        echo "MAGMA source not found at $magma_source" >&2
        exit 2
    fi
    archive="$repo/.cache/hopper-setup/magma-2.10.0.tar.gz"
    if [[ ! -f $archive ]]; then
        curl --fail --location --connect-timeout 10 --max-time 180 \
            https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.10.0.tar.gz \
            --output "$archive.part"
        mv "$archive.part" "$archive"
    fi
    sha256sum "$archive" > "$archive.sha256"
    "$HYBRIDGE_PYTHON" - "$archive" "$repo/.." <<'PY'
import sys, tarfile
with tarfile.open(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2], filter="data")
PY
fi

# LP64 is required by the ctypes binding (32-bit magma_int_t). Do not choose
# libopenblas64.so or an ILP64 MKL build automatically.
lapack=${HYBRIDGE_MAGMA_LAPACK:-/usr/lib64/libopenblas.so}
test -f "$lapack"
cmake -S "$magma_source" -B "$HYBRIDGE_MAGMA_ROOT" \
    -DMAGMA_ENABLE_CUDA=ON -DGPU_TARGET=sm_90 \
    -DCMAKE_CUDA_COMPILER="$HYBRIDGE_CUDA13_ROOT/bin/nvcc" \
    -DCUDAToolkit_ROOT="$HYBRIDGE_CUDA13_ROOT" \
    -DUSE_FORTRAN=OFF -DFORTRAN_CONVENTION=-DADD_ \
    -DLAPACK_LIBRARIES="$lapack" -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    2>&1 | tee "$repo/.cache/hopper-setup/magma-configure.log"
cmake --build "$HYBRIDGE_MAGMA_ROOT" --target magma --parallel "${HYBRIDGE_BUILD_JOBS:-4}" \
    2>&1 | tee "$repo/.cache/hopper-setup/magma-build.log"
"$HYBRIDGE_CUDA13_ROOT/bin/cuobjdump" --list-elf "$HYBRIDGE_MAGMA_LIBRARY" \
    > "$repo/.cache/hopper-setup/magma-cubins.txt"
if ! grep -q sm_90 "$repo/.cache/hopper-setup/magma-cubins.txt"; then
    echo 'MAGMA library does not contain sm_90 cubins' >&2
    exit 1
fi
"$repo/scripts/gpu/run_cuda13.sh" "$HYBRIDGE_PYTHON" -m campaigns.common.hopper_stack --host-only
echo 'Host setup passed. GPU contraction/LU checks must still pass in the smoke job.'
