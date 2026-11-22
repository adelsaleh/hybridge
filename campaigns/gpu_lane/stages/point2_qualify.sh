#!/usr/bin/env bash
# Check maintained borrowed-BSR/CSR/vector support on the selected H200 stack.
# Usage: point2_qualify.sh OUTPUT
# HYBRIDGE_REPO, HYBRIDGE_AMGX_BUILD_ROOT and HYBRIDGE_PYTHON may select an
# explicit source/runtime snapshot. Historical commands and records are unchanged.
set -uo pipefail
out=${1:?usage: point2_qualify.sh OUTPUT}
repo=${HYBRIDGE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
source "$repo/campaigns/common/hopper_env.sh" || exit 2
build=$HYBRIDGE_AMGX_BUILD_ROOT
native_launcher=${HYBRIDGE_NATIVE_TEST_LAUNCHER:-$build/src/amgx_tests_launcher}
if [[ ! -f "$build/libamgxsh.so" || ! -x "$native_launcher" || ! -x "$HYBRIDGE_PYTHON" ]]; then
    printf 'Build/install the selected AMGX and PyAMGX stack before qualification: %s\n' "$build" >&2
    exit 2
fi
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd)
log=$out/qualify.log
cd "$repo" || exit 2
echo "point-2 qualification on $(hostname) at $(date -Is), library $build" | tee -a "$log"
nm -D "$build/libamgxsh.so" | grep -c AMGX_matrix_attach_bsr | sed 's/^/attach_bsr symbols: /' | tee -a "$log"
status=0
# The native launcher writes .launched/.utestsnums into its working directory.
( cd "$out" && bash "$repo/scripts/gpu/run_cuda13.sh" "$native_launcher" BorrowedBSR BorrowedCSR BorrowedVectors ) 2>&1 | tee -a "$log"
native_status=${PIPESTATUS[0]}
echo "native suites exited $native_status" | tee -a "$log"
[[ $native_status == 0 ]] || status=1
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" bash scripts/gpu/run_cuda13.sh "$HYBRIDGE_PYTHON" -u -B -m pytest -q -p no:cacheprovider \
    tests/test_amgx_borrowed_csr.py tests/test_amgx_block_config_validation.py \
    tests/test_amgx_fixed_cycles.py tests/test_hp_mg_packed_kernels.py 2>&1 | tee -a "$log"
python_status=${PIPESTATUS[0]}
echo "python tests exited $python_status" | tee -a "$log"
[[ $python_status == 0 ]] || status=1
echo "$(date -Is) point-2 qualification exited $status" | tee -a "$log"
exit "$status"
