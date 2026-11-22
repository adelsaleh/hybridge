#!/usr/bin/env bash
# CPU-only primary AMGX build/install and a fresh, isolated PyAMGX wheel.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
workspace=${HYBRIDGE_WORKSPACE_ROOT:-$(dirname "$repo")}
builds=${HYBRIDGE_BUILDS_ROOT:-$workspace/builds}
build=${HYBRIDGE_INTEGRATION_BUILD:-$workspace/AMGX-build-cuda13.1}
install=${HYBRIDGE_AMGX_INSTALL_ROOT:-$builds/AMGX/install-cuda13.1}
out=${HYBRIDGE_STACK_OUT:-$workspace/hybridge/run_outputs/fork-integration-build}
if [[ ${1:-} == --plan ]]; then
    printf '%s\n' "CPU build: $workspace/AMGX -> $build; install: $install" \
        "Targets: amgx amgxsh amgx_capi amgx_tests_launcher; Release, sm90, CUDA13.1, non-MPI" \
        "Fresh committed PyAMGX archive -> wheel and isolated test install in $out" \
        "No GPU allocation. Resume with unchanged source and HYBRIDGE_STACK_OUT."
    exit 0
fi
export HYBRIDGE_BUILDS_ROOT=$builds
source "$repo/campaigns/common/hopper_env.sh"
out=$(realpath -m "$out")
install=$(realpath -m "$install")
state=("$HYBRIDGE_PYTHON" -B "$repo/campaigns/fork_integration/stack_state.py")
identity=(--workspace "$workspace" --stack "$out" --build "$build" --install "$install" --cuda "$HYBRIDGE_CUDA13_ROOT")
"${state[@]}" prepare "${identity[@]}"
if [[ -f "$out/BUILD.json" ]]; then
    "${state[@]}" verify-build "${identity[@]}"
    printf 'Verified completed CPU build: %s\n' "$out/BUILD.json"
    exit 0
fi
stage() { "${state[@]}" stage --output "$out" --name "$1" -- "${@:2}"; }
stage configure cmake -S "$workspace/AMGX" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$HYBRIDGE_CUDA13_ROOT/bin/nvcc" \
    -DCUDAToolkit_ROOT="$HYBRIDGE_CUDA13_ROOT" -DCMAKE_NO_MPI=ON \
    -DCMAKE_CUDA_ARCHITECTURES=90 -DAMGX_INCLUDE_EXTERNAL=ON -DAMGX_NO_RPATH=OFF \
    -DCMAKE_INSTALL_PREFIX="$install"
stage build cmake --build "$build" --target amgx amgxsh amgx_capi amgx_tests_launcher \
    --parallel "${SLURM_CPUS_PER_TASK:-16}"
stage install cmake --install "$build" --prefix "$install"
stage pyamgx-archive "${state[@]}" archive "${identity[@]}"
mkdir -p "$out/wheelhouse"
stage pyamgx-wheel env AMGX_DIR="$install" AMGX_BUILD_DIR="$(realpath "$build")" \
    LDFLAGS=-Wl,--enable-new-dtags "$HYBRIDGE_PYTHON" -m pip wheel \
    --no-build-isolation --no-deps --no-cache-dir --wheel-dir "$out/wheelhouse" "$out/pyamgx-source"
wheels=("$out/wheelhouse/"*.whl)
[[ ${#wheels[@]} == 1 && -f ${wheels[0]} ]] || { printf 'Expected exactly one wheel.\n' >&2; exit 1; }
stage pyamgx-isolated "$HYBRIDGE_PYTHON" -m pip install --no-index --no-deps --no-compile \
    --target "$out/python" "${wheels[0]}"
"${state[@]}" finish-build "${identity[@]}"
printf 'CPU build complete: %s\n' "$out/BUILD.json"
