#!/usr/bin/env bash
# Bounded regression checks of a verified CPU-built stack on one H200.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
workspace=${HYBRIDGE_WORKSPACE_ROOT:-$(dirname "$repo")}
out=${HYBRIDGE_CAMPAIGN_OUT:-$repo/run_outputs/fork-integration-${SLURM_JOB_ID:-local}}
builds=${HYBRIDGE_BUILDS_ROOT:-$workspace/builds}
build=${HYBRIDGE_INTEGRATION_BUILD:-$workspace/AMGX-build-cuda13.1}
install=${HYBRIDGE_AMGX_INSTALL_ROOT:-$builds/AMGX/install-cuda13.1}
stack=${HYBRIDGE_STACK_OUT:-$workspace/hybridge/run_outputs/fork-integration-build}
baselines=${HYBRIDGE_AMGX_BASELINE_ROOT:-$workspace/hybridge/run_outputs/fork-integration-baselines/20261031}
baseline_main=$baselines/main
baseline_bsr=$baselines/bsr
if [[ ${1:-} == --plan ]]; then
    printf '%s\n' "Verify completed CPU build $stack/BUILD.json; qualify AMGX at $build." \
        "Use its isolated PyAMGX wheel install; no compilation or production Python installation." \
        "Historical runtime fixtures: $baselines" \
        "Outputs: $out; one allocated H200. Resume with unchanged source/build and HYBRIDGE_CAMPAIGN_OUT."
    exit 0
fi
export HYBRIDGE_BUILDS_ROOT=$builds
source "$repo/campaigns/common/hopper_env.sh"
out=$(realpath -m "$out")
stack=$(realpath -m "$stack")
export HYBRIDGE_AMGX_INSTALL_ROOT=$install
export HYBRIDGE_AMGX_BUILD_ROOT=$build
export PYTHONPATH="$stack/python:$repo"
state=("$HYBRIDGE_PYTHON" -B "$repo/campaigns/fork_integration/stack_state.py")
identity=(--workspace "$workspace" --stack "$stack" --build "$build" --install "$install" \
    --cuda "$HYBRIDGE_CUDA13_ROOT" --baselines "$baselines" --output "$out")
cd "$repo"
"${state[@]}" qualify "${identity[@]}"
stage() {
    "${state[@]}" stage --output "$out" --name "$1" -- "${@:2}"
}
gpu=(bash "$repo/scripts/gpu/run_cuda13.sh")
mkdir -p "$out/native"
cd "$out/native"
stage native "${gpu[@]}" "$build/src/amgx_tests_launcher" \
    BorrowedBSR BorrowedCSR BorrowedVectors DeviceMemoryStats \
    GMRESReliableResidual BiCGStabResidual KrylovBsrSpmvBackend BlockILUBackend
stage truncation "${gpu[@]}" "$build/src/amgx_tests_launcher" --mode dDDI --mode dDFI --mode dFFI TruncateMaxElements
stage smoothers "${gpu[@]}" "$build/src/amgx_tests_launcher" --mode dDDI --mode dFFI SmootherBlocksizes
# The dDFI smoother path predates this change and rejects mixed precision.
# Keep its baseline evidence separate from the supported-mode acceptance gate.
mkdir -p "$out/baseline-mixed"
cd "$out/baseline-mixed"
if env HYBRIDGE_AMGX_BUILD_ROOT="$baseline_bsr" \
    "${gpu[@]}" "$baseline_bsr/src/amgx_tests_launcher" \
    --mode dDFI SmootherBlocksizes > "$out/baseline-mixed.log" 2>&1; then
    printf 'Baseline unexpectedly supports mixed-precision smoothing; investigate.\n' >&2
    exit 1
fi
rg -q 'Mixed precision modes not currently supported' "$out/baseline-mixed.log"
cd "$repo"
stage pyamgx "${gpu[@]}" "$HYBRIDGE_PYTHON" -m pytest -q -p no:cacheprovider "$stack/pyamgx-source/tests"
stage hybridge-borrow "${gpu[@]}" "$HYBRIDGE_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_amgx_borrowed_csr.py tests/test_amgx_block_config_validation.py \
    tests/test_amgx_fixed_cycles.py tests/test_hp_mg_packed_kernels.py
stage natural-iluk "${gpu[@]}" "$HYBRIDGE_PYTHON" campaigns/matk/check_natural_iluk.py \
    --library "$build/libamgxsh.so" --output "$out/natural-iluk.json"
stage natural-iluk-high "${gpu[@]}" "$HYBRIDGE_PYTHON" campaigns/matk/check_natural_iluk.py \
    --library "$build/libamgxsh.so" --blocks 8 --fill-levels 5 10 20 --fill-chain --output "$out/natural-iluk-high.json"
stage native-capacity "${gpu[@]}" "$HYBRIDGE_PYTHON" campaigns/matk/check_ilu_capacity.py \
    --expected-library "$build/libamgxsh.so" --capacity 1024 --output "$out/native-capacity.json"
stage dilu "${gpu[@]}" "$HYBRIDGE_PYTHON" scripts/gpu/check_amgx_dilu_blocks.py --save "$out/dilu.npz"
stage baseline-dilu env HYBRIDGE_AMGX_BUILD_ROOT="$baseline_main" \
    "${gpu[@]}" "$HYBRIDGE_PYTHON" scripts/gpu/check_amgx_dilu_blocks.py \
    --blocks 1,2,3,4,5,7,8,10 --max-embed 0 --save "$out/baseline-dilu.npz"
stage dilu-parity "$HYBRIDGE_PYTHON" scripts/gpu/check_amgx_dilu_blocks.py \
    --mode compare "$out/baseline-dilu.npz" "$out/dilu.npz"
stage calibration-residency "${gpu[@]}" "$HYBRIDGE_PYTHON" -m pytest -q -ra -p no:cacheprovider \
    tests/test_adr_tuning_cuda.py::test_cold_adr_calibration_keeps_sample_arrays_on_device \
    tests/test_split3_tuning.py
stage gpu-smoke "${gpu[@]}" "$HYBRIDGE_PYTHON" scripts/dev/alpha_test_matrix.py gpu-smoke
"${state[@]}" verify-build "${identity[@]}"
sha256sum "$build/libamgxsh.so" > "$out/library.sha256"
printf 'All integration stages passed.\n' > "$out/COMPLETE"
