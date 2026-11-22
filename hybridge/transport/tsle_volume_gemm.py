"""Element-batched volume operator for the TSLE (split3) build (opt-in).

The split3 build kernel (one block per element) forms the dense-tensor volume
operator of element ``e`` as

    A_e[i, j] = J_e (r M[i, j] - sum_k b0[e, k] T0[k, i, j] + b1[e, k] T1[k, i, j])

(``b0``/``b1`` the advection field in reference coordinates, a field reaction
contracting the triple products instead of ``M``). That is a ``1 x K`` by
``K x NEL^2`` product per block, so every block streams the whole reference
tensor (2.6 MB at p=9) and nothing is reused across elements: on an H200 the
p=8-9 build runs at L2 bandwidth, not FP64 throughput. This module computes all
elements at once as one GEMM,

    C (E x NEL^2) = X (E x K) @ Y (K x NEL^2),
    X[e] = [-J b0[e] | -J b1[e] | J r[e]],   Y = [T0; T1; M or R],

so each ``Y`` tile is reused by a block of elements, and writes ``C`` into the
TSLE element operator. The build kernel is then compiled with
``RAW_VOLUME_PRELOADED`` and adds the face terms on top. Two GEMM engines:

* ``dmma``: a raw kernel on the FP64 tensor cores (``mma.sync.m8n8k4`` f64,
  compute capability 8.0+): 64 elements x 32 columns per 128-thread block, the
  block's ``X`` rows resident in shared memory, ``Y`` streamed in 32-column tiles.
* ``cublas``: ``cupy.matmul`` (cuBLAS DGEMM).

``HYBRIDGE_RAW_SPLIT3_VOLUME`` selects ``inline`` (default: the unchanged
per-element loop), ``dmma`` or ``cublas``. The pre-pass applies only to the
dense-tensor operator (no quadrature volume table, no quadrature-point
coefficients); other cases stay inline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

VOLUME_MODES = ("inline", "dmma", "cublas")
_BM = 64   # elements per block (M tile)
_BN = 32   # operator columns per tile (N tile)
_THREADS = 128
_ROWS_BLOCK = 256
_MODULE_CACHE: dict[tuple[Any, ...], Any] = {}

_SOURCE = r"""
// Coefficient rows X[e] = [-J b0 | -J b1 | J r | 0 ...] (VG_KP columns).
extern "C" __global__ void advection_volume_rows(
        double* __restrict__ rows,
        const double* __restrict__ beta_coeffs,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ inv_aff_mats_t,
        const double* __restrict__ reaction_coeffs,
        const double reaction_scalar,
        const int reaction_is_scalar,
        const long long num_elements,
        const long long count)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * VG_KP) return;
    const long long e = idx / VG_KP;
    const int k = (int)(idx - e * VG_KP);
    const double jac = aff_jacs[e];
    double value = 0.0;
    if (k < 2 * VG_NB) {
        const int mode = k < VG_NB ? k : k - VG_NB;
        const int column = k < VG_NB ? 0 : 1;
        const double bx = beta_coeffs[(0 * num_elements + e) * VG_NB + mode];
        const double by = beta_coeffs[(1 * num_elements + e) * VG_NB + mode];
        const double t0 = inv_aff_mats_t[(e * 2 + 0) * 2 + column];
        const double t1 = inv_aff_mats_t[(e * 2 + 1) * 2 + column];
        value = -jac * (bx * t0 + by * t1);
    } else if (k < 2 * VG_NB + VG_NR) {
        value = reaction_is_scalar ? jac * reaction_scalar
                                   : jac * reaction_coeffs[e * VG_NEL + (k - 2 * VG_NB)];
    }
    rows[idx] = value;
}

#define XS_STRIDE (VG_KP + 4)
#define YS_STRIDE (VG_BN + 8)

__device__ __forceinline__ void dmma_m8n8k4(double& d0, double& d1, const double a, const double b) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+d"(d0), "+d"(d1) : "d"(a), "d"(b));
}

// C[e, n] = sum_k X[e, k] Y[k, n] for e < count, n < VG_NN, written to out[e * VG_NN + n].
// Four warps in a 2 x 2 grid; warp tile 32 rows x 16 columns = 4 x 2 m8n8 fragments.
// Fragment layout of mma.m8n8k4 f64 (lane = 4 g + t): A[g][t], B[t][g],
// C[g][2 t + i] for i in {0, 1}.
extern "C" __global__ void __launch_bounds__(128) advection_volume_gemm_dmma(
        const double* __restrict__ x,
        const double* __restrict__ y,
        double* __restrict__ out,
        const long long count)
{
    extern __shared__ double smem[];
    double* xs = smem;
    double* ys = smem + VG_BM * XS_STRIDE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const long long e0 = (long long)blockIdx.x * VG_BM;
    for (int idx = tid; idx < VG_BM * VG_KP; idx += blockDim.x) {
        const int r = idx / VG_KP, k = idx - r * VG_KP;
        xs[r * XS_STRIDE + k] = e0 + r < count ? x[(e0 + r) * VG_KP + k] : 0.0;
    }
    for (int n0 = 0; n0 < VG_NP; n0 += VG_BN) {
        __syncthreads();
        for (int idx = tid; idx < VG_KP * VG_BN; idx += blockDim.x) {
            const int k = idx / VG_BN, c = idx - k * VG_BN;
            ys[k * YS_STRIDE + c] = y[(long long)k * VG_NP + n0 + c];
        }
        __syncthreads();
        double acc[4][2][2];
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                acc[mt][nt][0] = acc[mt][nt][1] = 0.0;
#pragma unroll 4
        for (int k0 = 0; k0 < VG_KP; k0 += 4) {
            double a[4], b[2];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
                a[mt] = xs[(wm * 32 + mt * 8 + g) * XS_STRIDE + k0 + t];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                b[nt] = ys[(k0 + t) * YS_STRIDE + wn * 16 + nt * 8 + g];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    dmma_m8n8k4(acc[mt][nt][0], acc[mt][nt][1], a[mt], b[nt]);
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const long long e = e0 + wm * 32 + mt * 8 + g;
            if (e >= count) continue;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int n = n0 + wn * 16 + nt * 8 + 2 * t;
                if (n < VG_NN) out[e * VG_NN + n] = acc[mt][nt][0];
                if (n + 1 < VG_NN) out[e * VG_NN + n + 1] = acc[mt][nt][1];
            }
        }
    }
}
"""


def volume_mode() -> str:
    """Return ``HYBRIDGE_RAW_SPLIT3_VOLUME`` (``inline`` by default)."""
    mode = os.environ.get("HYBRIDGE_RAW_SPLIT3_VOLUME", "inline").strip().lower() or "inline"
    if mode not in VOLUME_MODES:
        raise ValueError(f"HYBRIDGE_RAW_SPLIT3_VOLUME must be one of {VOLUME_MODES}")
    return mode


def volume_prepass_applies(*, nqv, pointwise_beta: bool, pointwise_reaction: bool) -> bool:
    """Return whether the dense-tensor volume operator (the pre-pass target) is in use."""
    return nqv is None and not pointwise_beta and not pointwise_reaction


def _dims(nel: int, nbeta: int, reaction_is_scalar: bool) -> tuple[int, int, int, int]:
    """Return ``(K, KP, NN, NP)``: rows, rows padded to 4, columns NEL^2, columns padded to the N tile."""
    k = 2 * nbeta + (1 if reaction_is_scalar else nel)
    nn = nel * nel
    return k, -(-k // 4) * 4, nn, -(-nn // _BN) * _BN


def _compile(cupy, *, nel: int, nbeta: int, reaction_is_scalar: bool):
    """Compile the rows and DMMA kernels for one signature (cached per device)."""
    k, kp, nn, np_ = _dims(nel, nbeta, reaction_is_scalar)
    key = (int(cupy.cuda.runtime.getDevice()), nel, nbeta, bool(reaction_is_scalar))
    cached = _MODULE_CACHE.get(key)
    if cached is not None:
        return cached
    defines = (f"#define VG_NEL {nel}\n#define VG_NB {nbeta}\n#define VG_NR {1 if reaction_is_scalar else nel}\n"
               f"#define VG_KP {kp}\n#define VG_NN {nn}\n#define VG_NP {np_}\n#define VG_BM {_BM}\n#define VG_BN {_BN}\n")
    module = cupy.RawModule(code=defines + _SOURCE, options=("--std=c++11",),
                            name_expressions=("advection_volume_rows", "advection_volume_gemm_dmma"))
    kernels = dict(rows=module.get_function("advection_volume_rows"),
                   dmma=module.get_function("advection_volume_gemm_dmma"))
    shared = (_BM * (kp + 4) + kp * (_BN + 8)) * 8
    kernels["dmma"].max_dynamic_shared_size_bytes = int(shared)
    kernels["shared_bytes"] = int(shared)
    _MODULE_CACHE[key] = kernels
    return kernels


@dataclass
class VolumeGemm:
    """One split3 assembly's volume pre-pass: buffers, tables, and launches.

    ``__call__(count)`` writes the volume operator of the first ``count``
    elements into ``operator`` (``(E, NEL, NEL)``, the TSLE element operator).
    """

    mode: str
    operator: Any
    rows: Any
    table: Any
    launch_args: dict = field(default_factory=dict)
    signature: tuple | None = None

    @property
    def nbytes(self) -> int:
        """Bytes of the coefficient rows and the padded reference table."""
        return int(getattr(self.rows, "nbytes", 0)) + int(getattr(self.table, "nbytes", 0))

    def __call__(self, count: int) -> None:
        """Write the volume operator of the first ``count`` elements (rows kernel, then the GEMM)."""
        cupy = self.launch_args["cupy"]
        kernels, kp, k, nn = (self.launch_args[name] for name in ("kernels", "kp", "k", "nn"))
        count = int(count)
        total = count * kp
        kernels["rows"](((total + _ROWS_BLOCK - 1) // _ROWS_BLOCK,), (_ROWS_BLOCK,),
                        self.launch_args["rows_args"] + (np.int64(count),))
        if self.mode == "dmma":
            kernels["dmma"](((count + _BM - 1) // _BM,), (_THREADS,),
                            (self.rows, self.table, self.operator, np.int64(count)),
                            shared_mem=kernels["shared_bytes"])
        else:
            out = self.operator.reshape(-1, nn)[:count]
            cupy.matmul(self.rows[:count, :k], self.launch_args["table_dense"], out=out)


def prepare_volume_gemm(cupy, *, mode: str, operator, existing: VolumeGemm | None, num_elements: int, nel: int,
                        nbeta: int, advection_tensor, mass_matrix, reaction_triples, mass_is_diagonal: bool,
                        beta_coeffs, aff_jacs, inv_aff_mats_t, reaction_coeffs, reaction_scalar: float,
                        reaction_is_scalar: bool, timings: dict) -> VolumeGemm:
    """Build (or refresh) the pre-pass for one assembly; reuses buffers of an equal signature."""
    if mode == "dmma":
        major = int(cupy.cuda.Device().compute_capability[:-1] or 0)
        if major < 8:
            raise ValueError("HYBRIDGE_RAW_SPLIT3_VOLUME=dmma needs FP64 tensor cores (compute capability 8.0+)")
    started = time.perf_counter()
    kernels = _compile(cupy, nel=nel, nbeta=nbeta, reaction_is_scalar=bool(reaction_is_scalar))
    k, kp, nn, np_ = _dims(nel, nbeta, bool(reaction_is_scalar))
    signature = (int(cupy.cuda.runtime.getDevice()), mode, num_elements, nel, nbeta, bool(reaction_is_scalar),
                 bool(mass_is_diagonal), id(advection_tensor))
    gemm = existing if existing is not None and existing.signature == signature else None
    if gemm is None:
        rows = cupy.zeros((num_elements, kp), dtype=np.float64)
        table = cupy.zeros((kp, np_), dtype=np.float64)
        table[:2 * nbeta, :nn] = cupy.asarray(advection_tensor, dtype=np.float64).reshape(2 * nbeta, nn)
        if reaction_is_scalar:
            mass = cupy.asarray(mass_matrix, dtype=np.float64).reshape(nel, nel)
            if mass_is_diagonal:
                mass = cupy.diag(cupy.diag(mass))
            table[2 * nbeta, :nn] = mass.reshape(-1)
        else:
            table[2 * nbeta:2 * nbeta + nel, :nn] = cupy.asarray(reaction_triples, dtype=np.float64).reshape(nel, nn)
        gemm = VolumeGemm(mode=mode, operator=operator, rows=rows, table=table, signature=signature)
    gemm.operator = operator
    gemm.launch_args = dict(
        cupy=cupy, kernels=kernels, k=k, kp=kp, nn=nn,
        table_dense=cupy.ascontiguousarray(gemm.table[:k, :nn]) if mode == "cublas" else None,
        rows_args=(gemm.rows, beta_coeffs, aff_jacs, inv_aff_mats_t, reaction_coeffs, np.float64(reaction_scalar),
                   np.int32(1 if reaction_is_scalar else 0), np.int64(num_elements)),
    )
    timings["raw.tsle.volume.prepare"] = time.perf_counter() - started
    timings["raw.tsle.volume.mode"] = float(VOLUME_MODES.index(mode))
    timings["raw.tsle.volume.bytes"] = float(gemm.nbytes)
    return gemm


class PreloadedBuild:
    """Split3 build stage = volume pre-pass for the launched elements, then the build kernel.

    Called like a CuPy kernel, so the autotuner's repeated build launches stay
    idempotent (each one rewrites the volume operator first). ``kernel`` is the
    compiled build kernel, for resource records.
    """

    def __init__(self, kernel, volume: VolumeGemm):
        """Wrap the compiled build ``kernel`` with its volume pre-pass."""
        self.kernel = kernel
        self.volume = volume

    @property
    def attributes(self):
        """Return the build kernel's CUDA attributes."""
        return self.kernel.attributes

    def __call__(self, grid, block, args, **kwargs):
        """Run the pre-pass for ``grid[0]`` elements, then launch the build kernel."""
        self.volume(int(grid[0]))
        return self.kernel(grid, block, args, **kwargs)


__all__ = ["PreloadedBuild", "VOLUME_MODES", "VolumeGemm", "prepare_volume_gemm", "volume_mode",
           "volume_prepass_applies"]
