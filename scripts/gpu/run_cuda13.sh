#!/usr/bin/env bash
set -euo pipefail

cuda13_root="${HYBRIDGE_CUDA13_ROOT:-/tmp/cuda}"
amgx_build_root="${HYBRIDGE_AMGX_BUILD_ROOT:-/tmp/AMGX-build}"
amgx_install_root="${HYBRIDGE_AMGX_INSTALL_ROOT:-/tmp/AMGX-install}"
cuda13_lib_root="${cuda13_root}/targets/x86_64-linux/lib"

if [[ ! -x "${cuda13_root}/bin/nvcc" ]]; then
    echo "error: CUDA 13 nvcc is unavailable at ${cuda13_root}/bin/nvcc" >&2
    exit 2
fi
if [[ ! -f "${cuda13_lib_root}/libcudart.so.13" ]]; then
    echo "error: CUDA 13 runtime is unavailable under ${cuda13_lib_root}" >&2
    exit 2
fi
if [[ ! -f "${amgx_build_root}/libamgxsh.so" ]]; then
    echo "error: CUDA 13 AMGX library is unavailable at ${amgx_build_root}/libamgxsh.so" >&2
    exit 2
fi
if [[ ! -d "${amgx_install_root}/lib" ]]; then
    echo "error: CUDA 13 AMGX install is unavailable at ${amgx_install_root}/lib" >&2
    exit 2
fi

cuda13_real="$(readlink -f "${cuda13_root}")"
amgx_build_real="$(readlink -f "${amgx_build_root}")"
amgx_install_real="$(readlink -f "${amgx_install_root}")"
nvcc_report="$("${cuda13_root}/bin/nvcc" --version)"
if [[ "${nvcc_report}" != *"release 13."* ]]; then
    echo "error: ${cuda13_root}/bin/nvcc is not a CUDA 13 compiler" >&2
    exit 2
fi

amgx_cache="${amgx_build_root}/CMakeCache.txt"
if [[ ! -f "${amgx_cache}" ]]; then
    echo "error: AMGX CMake cache is unavailable at ${amgx_cache}" >&2
    exit 2
fi
amgx_nvcc="$(sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' "${amgx_cache}" | tail -n 1)"
amgx_nvcc_real="$(readlink -f "${amgx_nvcc}")"
if [[ "${amgx_nvcc_real}" != "${cuda13_real}"/* ]]; then
    echo "error: AMGX was configured with ${amgx_nvcc_real}, not ${cuda13_real}" >&2
    exit 2
fi

export CUDA_HOME="${cuda13_root}"
export CUDA_PATH="${cuda13_root}"
export CUDACXX="${cuda13_root}/bin/nvcc"
export NVCC="${cuda13_root}/bin/nvcc"
export PATH="${cuda13_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${amgx_build_root}:${amgx_install_root}/lib:${cuda13_lib_root}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

amgx_dependencies="$(ldd "${amgx_build_root}/libamgxsh.so")"
if [[ "${amgx_dependencies}" == *"not found"* ]]; then
    echo "error: AMGX has unresolved runtime dependencies" >&2
    exit 2
fi
for library in libcublas.so.13 libnvJitLink.so.13; do
    dependency_line="$(printf '%s\n' "${amgx_dependencies}" | grep "${library}" | head -n 1 || true)"
    if [[ -z "${dependency_line}" ]] || {
        [[ "${dependency_line}" != *"${cuda13_root}"* ]] &&
        [[ "${dependency_line}" != *"${cuda13_real}"* ]];
    }; then
        echo "error: ${library} is not resolving from the selected CUDA 13 toolkit" >&2
        exit 2
    fi
done

if [[ $# -eq 1 && "$1" == "--check" ]]; then
    echo "CUDA 13 toolkit: ${cuda13_real}"
    echo "AMGX build: ${amgx_build_real}"
    echo "AMGX install: ${amgx_install_real}"
    echo "AMGX compiler: ${amgx_nvcc_real}"
    exit 0
fi
if [[ $# -eq 0 ]]; then
    echo "usage: $0 --check | COMMAND [ARG ...]" >&2
    exit 2
fi

exec "$@"
