#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${HYBRIDGE_PYTHON:-"${repo_root}/../hdg-guiding-center/.venv/bin/python"}
amgx_lib=${AMGX_LIB_DIR:-"${HOME}/.local/amgx/lib"}
export LD_LIBRARY_PATH="${amgx_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_root}"
exec "${python_bin}" -u "${repo_root}/scripts/guiding_center/run_guiding_center_cases.py" "$@"
