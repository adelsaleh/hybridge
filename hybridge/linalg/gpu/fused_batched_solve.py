"""Fused batched dense solve ``A_b X_b = B_b`` for small FP64 systems (experimental).

One thread block solves one system. The augmented matrix ``[A_b | B_b]``
(``N x (N + NRHS)``) lives in registers, distributed over a ``TR x TC`` thread
grid: thread ``(tr, tc)`` owns rows ``tr + TR i`` and columns ``tc + TC j``
(cyclic, so the shrinking active part stays balanced). Elimination is
Gauss-Jordan with partial pivoting and implicit row exchange: rows never move,
the pivot of column ``k`` is the largest ``|a_ik|`` among rows not yet used as a
pivot, and step ``k`` subtracts multiples of the pivot row from every other row
(columns ``> k``). After ``N`` steps, row ``p_k`` holds ``a_{p_k k} x_k`` in its
right-hand-side columns, so ``X[k] = B'[p_k] / a_{p_k k}`` is written straight
to the output with no triangular solve and no transposition. ``A_b`` is read
only (not overwritten).

Variants (``config[3]``): ``1`` per step lets the ``TR`` owners of column
``k`` (contiguous lanes of one warp) find the pivot with a masked shuffle
reduction and publish the multipliers, then the owners of the pivot row publish
it, then every thread applies a register-blocked rank-1 update: two block
barriers per step. ``2`` reduces a packed key (``|a_ik|`` with ``63 - row`` in
the low mantissa bits, so ties go to the first row as in ``idamax``) with
full-warp shuffles. ``3`` (default) needs one barrier per step: right after its
update, the owner warp of column ``k + 1`` publishes that column and its pivot,
and the owners of row ``k + 1`` publish that row as the speculative pivot row;
when the pivot is another row (rare for HDG element operators), its owners
republish behind one extra barrier. Rows with NaN entries still give a valid
pivot order; a zero or non-finite pivot is reported through ``info``.

Layout matches :func:`hybridge.linalg.gpu.cublas_batched.lu_solve_batched_cublas`
with ``trans=False``: ``matrices[b]`` ``(N, N)`` holds ``A_b`` column-major (the
bytes of its C-order transpose) and ``rhs[b]`` ``(NRHS, N)`` holds ``B_b``
column-major; ``rhs`` is overwritten with ``X_b``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybridge.runtime.optional import require_cupy_device

# (TR, TC, min blocks per SM, variant) per system size N; TR must divide 32.
# Fastest per N on the H200 (stage lu-04, AR p=6-9 systems, 117k elements).
DEFAULT_CONFIGS: dict[int, tuple[int, int, int, int]] = {
    28: (8, 4, 8, 3),
    36: (8, 8, 4, 2),
    45: (8, 8, 3, 3),
    55: (8, 8, 2, 3),
}

_SOURCE = r"""
#define GJ_M (GJ_N + GJ_NRHS)
#define GJ_R ((GJ_N + GJ_TR - 1) / GJ_TR)
#define GJ_C ((GJ_M + GJ_TC - 1) / GJ_TC)
#define GJ_CA ((GJ_N + GJ_TC - 1) / GJ_TC)
#define GJ_THREADS (GJ_TR * GJ_TC)

#if GJ_VARIANT == 3
// Publish column kn (slot ja or ja + 1) with its pivot, and row kn as the speculative
// pivot row (slots >= ja), into the step-kn buffers.
__device__ __forceinline__ void gj_publish(
        const double (&a)[GJ_R][GJ_C], const int kn, const int ja, const unsigned used,
        const int tid, const int tr, const int tc, double (*s_l)[GJ_R * GJ_TR],
        double (*s_u)[GJ_C * GJ_TC], int* s_p, int* s_step)
{
    const int nb = kn & 1;
    const int tcn = kn % GJ_TC;
    const bool next_slot = kn / GJ_TC != ja;
    const int jb = ja + 1 < GJ_C ? ja + 1 : ja;
    if ((tid >> 5) == ((tcn * GJ_TR) >> 5)) {
        double key = -1.0;
        double col[GJ_R];
#pragma unroll
        for (int i = 0; i < GJ_R; ++i) {
            const int row = tr + GJ_TR * i;
            col[i] = next_slot ? a[i][jb] : a[i][ja];
            double av = fabs(col[i]);
            av = av == av ? av : 0.0;  // NaN entries still yield a valid (unused) pivot row
            const double cand = __longlong_as_double((__double_as_longlong(av) & ~63LL) | (long long)(63 - row));
            if (row < GJ_N && !((used >> i) & 1u)) key = fmax(key, cand);
        }
#pragma unroll
        for (int off = 1; off < GJ_TR; off <<= 1)
            key = fmax(key, __shfl_xor_sync(0xffffffffu, key, off, GJ_TR));
        if (tc == tcn) {
            const int prow = 63 - (int)(__double_as_longlong(key) & 63LL);
#pragma unroll
            for (int i = 0; i < GJ_R; ++i) {
                const int row = tr + GJ_TR * i;
                s_l[nb][row] = row == prow ? 0.0 : col[i];
            }
            if (tr == 0) {
                s_p[nb] = prow;
                s_step[prow] = kn;
            }
        }
    }
    if (tr == kn % GJ_TR) {
        const int ik = kn / GJ_TR;
#pragma unroll
        for (int i = 0; i < GJ_R; ++i) {
            if (i == ik) {
#pragma unroll
                for (int j = ja; j < GJ_C; ++j) s_u[nb][tc + GJ_TC * j] = a[i][j];
            }
        }
    }
}
#endif

extern "C" __global__ void __launch_bounds__(GJ_THREADS, GJ_MINB)
fused_gj_solve(const double* __restrict__ A, double* __restrict__ B, int* __restrict__ info,
               const long long batch)
{
    __shared__ double s_u[2][GJ_C * GJ_TC];
    __shared__ double s_l[2][GJ_R * GJ_TR];
    __shared__ double s_inv[GJ_N];
    __shared__ int s_step[GJ_R * GJ_TR];
    __shared__ int s_p[2];

    const long long e = blockIdx.x;
    if (e >= batch) return;
    const int tid = threadIdx.x;
    const int tr = tid % GJ_TR;
    const int tc = tid / GJ_TR;
#if GJ_VARIANT == 1
    const unsigned lane = tid & 31;
    const unsigned seg_mask = GJ_TR >= 32 ? 0xffffffffu
                              : (((1u << (GJ_TR & 31)) - 1u) << (lane & ~(unsigned)(GJ_TR - 1)));
#endif
    const double* __restrict__ Ae = A + e * (long long)(GJ_N * GJ_N);
    double* __restrict__ Be = B + e * (long long)(GJ_N * GJ_NRHS);

    double a[GJ_R][GJ_C];
#pragma unroll
    for (int i = 0; i < GJ_R; ++i) {
        const int row = tr + GJ_TR * i;
#pragma unroll
        for (int j = 0; j < GJ_C; ++j) {
            const int col = tc + GJ_TC * j;
            double v = 0.0;
            if (row < GJ_N) {
                if (col < GJ_N) v = __ldg(Ae + col * GJ_N + row);
                else if (col < GJ_M) v = Be[(col - GJ_N) * GJ_N + row];
            }
            a[i][j] = v;
        }
    }

#if GJ_VARIANT < 3
    unsigned used = 0u;
#pragma unroll
    for (int jk = 0; jk < GJ_CA; ++jk) {
#pragma unroll 1
        for (int tck = 0; tck < GJ_TC; ++tck) {
            const int k = jk * GJ_TC + tck;
            if (k >= GJ_N) break;
            const int buf = k & 1;
#if GJ_VARIANT == 1
            if (tc == tck) {
                // Pivot: largest |a_ik| over unused rows (first index on ties, as idamax).
                double best = -1.0, value = 0.0;
                int prow = 0x7fffffff;
#pragma unroll
                for (int i = 0; i < GJ_R; ++i) {
                    const int row = tr + GJ_TR * i;
                    const double v = a[i][jk];
                    double av = fabs(v);
                    av = av == av ? av : 0.0;  // NaN entries still yield a valid (unused) pivot row
                    if (row < GJ_N && !((used >> i) & 1u) && av > best) { best = av; value = v; prow = row; }
                }
#pragma unroll
                for (int off = 1; off < GJ_TR; off <<= 1) {
                    const double ob = __shfl_xor_sync(seg_mask, best, off, GJ_TR);
                    const double ov = __shfl_xor_sync(seg_mask, value, off, GJ_TR);
                    const int orow = __shfl_xor_sync(seg_mask, prow, off, GJ_TR);
                    if (ob > best || (ob == best && orow < prow)) { best = ob; value = ov; prow = orow; }
                }
                const double inv = 1.0 / value;
#pragma unroll
                for (int i = 0; i < GJ_R; ++i) {
                    const int row = tr + GJ_TR * i;
                    s_l[buf][row] = row == prow ? 0.0 : a[i][jk] * inv;
                }
                if (tr == 0) {
                    s_p[buf] = prow;
                    s_inv[k] = inv;
                    s_step[prow] = k;
                }
            }
            __syncthreads();
            const int p = s_p[buf];
            if (tr == p % GJ_TR) {
                const int ip = p / GJ_TR;
#pragma unroll
                for (int i = 0; i < GJ_R; ++i) {
                    if (i == ip) {
#pragma unroll
                        for (int j = jk; j < GJ_C; ++j) s_u[buf][tc + GJ_TC * j] = a[i][j];
                        used |= 1u << i;
                    }
                }
            }
            __syncthreads();
            double l[GJ_R];
#pragma unroll
            for (int i = 0; i < GJ_R; ++i) l[i] = s_l[buf][tr + GJ_TR * i];
#else
            // Owner warp of column k: publish the raw column, then a full-mask shuffle
            // reduction of the packed key (|a_ik| with 63 - row in the low mantissa bits,
            // so ties pick the first row as idamax). Other segments of the warp reduce
            // their own (unused) keys.
            if ((tid >> 5) == ((tck * GJ_TR) >> 5)) {
                double key = -1.0;
#pragma unroll
                for (int i = 0; i < GJ_R; ++i) {
                    const int row = tr + GJ_TR * i;
                    const double v = a[i][jk];
                    if (tc == tck) s_l[buf][row] = v;
                    double av = fabs(v);
                    av = av == av ? av : 0.0;  // NaN entries still yield a valid (unused) pivot row
                    const double cand = __longlong_as_double(
                        (__double_as_longlong(av) & ~63LL) | (long long)(63 - row));
                    if (row < GJ_N && !((used >> i) & 1u)) key = fmax(key, cand);
                }
#pragma unroll
                for (int off = 1; off < GJ_TR; off <<= 1)
                    key = fmax(key, __shfl_xor_sync(0xffffffffu, key, off, GJ_TR));
                if (tc == tck && tr == 0) {
                    const int prow = 63 - (int)(__double_as_longlong(key) & 63LL);
                    s_p[buf] = prow;
                    s_step[prow] = k;
                }
            }
            __syncthreads();
            const int p = s_p[buf];
            if (tr == p % GJ_TR) {
                const int ip = p / GJ_TR;
#pragma unroll
                for (int i = 0; i < GJ_R; ++i) {
                    if (i == ip) {
#pragma unroll
                        for (int j = jk; j < GJ_C; ++j) s_u[buf][tc + GJ_TC * j] = a[i][j];
                        used |= 1u << i;
                    }
                }
                if (tc == tck) s_l[buf][p] = 0.0;
            }
            __syncthreads();
            const double inv = 1.0 / s_u[buf][k];
            if (tid == 0) s_inv[k] = inv;
            double l[GJ_R];
#pragma unroll
            for (int i = 0; i < GJ_R; ++i) l[i] = s_l[buf][tr + GJ_TR * i] * inv;
#endif
#pragma unroll
            for (int j = jk; j < GJ_C; ++j) {
                const double u = s_u[buf][tc + GJ_TC * j];
#pragma unroll
                for (int i = 0; i < GJ_R; ++i) a[i][j] = fma(-l[i], u, a[i][j]);
            }
        }
    }

#else
    // One barrier per step: before it, the owners of column k+1 publish that column
    // (pivot row entry zeroed) and the chosen pivot, and the owners of row k+1 publish
    // row k+1 as the speculative pivot row. A different pivot (rare) costs one more
    // barrier while its owners republish.
    unsigned used = 0u;
    gj_publish(a, 0, 0, used, tid, tr, tc, s_l, s_u, s_p, s_step);
    __syncthreads();
#pragma unroll
    for (int jk = 0; jk < GJ_CA; ++jk) {
#pragma unroll 1
        for (int tck = 0; tck < GJ_TC; ++tck) {
            const int k = jk * GJ_TC + tck;
            if (k >= GJ_N) break;
            const int buf = k & 1;
            const int p = s_p[buf];
            if (p != k) {
                if (tr == p % GJ_TR) {
                    const int ip = p / GJ_TR;
#pragma unroll
                    for (int i = 0; i < GJ_R; ++i) {
                        if (i == ip) {
#pragma unroll
                            for (int j = jk; j < GJ_C; ++j) s_u[buf][tc + GJ_TC * j] = a[i][j];
                        }
                    }
                }
                __syncthreads();
            }
            if (tr == p % GJ_TR) used |= 1u << (p / GJ_TR);
            const double inv = 1.0 / s_u[buf][k];
            if (tid == 0) s_inv[k] = inv;
            double l[GJ_R];
#pragma unroll
            for (int i = 0; i < GJ_R; ++i) l[i] = s_l[buf][tr + GJ_TR * i] * inv;
#pragma unroll
            for (int j = jk; j < GJ_C; ++j) {
                const double u = s_u[buf][tc + GJ_TC * j];
#pragma unroll
                for (int i = 0; i < GJ_R; ++i) a[i][j] = fma(-l[i], u, a[i][j]);
            }
            if (k + 1 < GJ_N) gj_publish(a, k + 1, jk, used, tid, tr, tc, s_l, s_u, s_p, s_step);
            __syncthreads();
        }
    }
#endif

    // Row p_k holds a_{p_k k} x_k in its right-hand-side columns.
#pragma unroll
    for (int i = 0; i < GJ_R; ++i) {
        const int row = tr + GJ_TR * i;
        if (row < GJ_N) {
            const int k = s_step[row];
            const double inv = s_inv[k];
#pragma unroll
            for (int j = 0; j < GJ_C; ++j) {
                if (GJ_TC * (j + 1) <= GJ_N) continue;
                const int col = tc + GJ_TC * j;
                if (col >= GJ_N && col < GJ_M) Be[(col - GJ_N) * GJ_N + k] = a[i][j] * inv;
            }
        }
    }
    if (info != nullptr && tid == 0) {
        int flag = 0;
        for (int k = 0; k < GJ_N; ++k) {
            if (!(fabs(s_inv[k]) <= 1.7976931348623157e308)) { flag = k + 1; break; }
        }
        info[e] = flag;
    }
}
"""

_MODULE_CACHE: dict[tuple[int, ...], Any] = {}


def kernel_source(n: int, nrhs: int, tr: int, tc: int, min_blocks: int, variant: int = 3) -> str:
    """Return the CUDA source of the fused solve for one size, thread grid, and variant."""
    if tr not in (1, 2, 4, 8, 16, 32):
        raise ValueError("TR must divide 32")
    if (tr * tc) % 32 or tr * tc > 1024:
        raise ValueError("TR * TC must be a multiple of 32 and at most 1024")
    if -(-n // tr) > 32:
        raise ValueError("at most 32 rows per thread (row mask width)")
    if variant not in (1, 2, 3):
        raise ValueError("variant must be 1, 2 or 3")
    if variant >= 2 and n > 64:
        raise ValueError("variants 2 and 3 pack the row index in 6 bits (n <= 64)")
    return (f"#define GJ_N {int(n)}\n#define GJ_NRHS {int(nrhs)}\n#define GJ_TR {int(tr)}\n"
            f"#define GJ_TC {int(tc)}\n#define GJ_MINB {int(min_blocks)}\n#define GJ_VARIANT {int(variant)}\n"
            + _SOURCE)


def _kernel(cp: Any, n: int, nrhs: int, config: tuple[int, ...]):
    """Return a cached device-specific fused solve kernel for this launch configuration."""
    tr, tc, min_blocks, variant = (tuple(int(value) for value in config) + (3,))[:4]
    key = (int(cp.cuda.runtime.getDevice()), n, nrhs, tr, tc, min_blocks, variant)
    kernel = _MODULE_CACHE.get(key)
    if kernel is None:
        module = cp.RawModule(code=kernel_source(n, nrhs, tr, tc, min_blocks, variant), options=("--std=c++11",),
                              name_expressions=("fused_gj_solve",))
        kernel = module.get_function("fused_gj_solve")
        _MODULE_CACHE[key] = kernel
    return kernel


def default_config(n: int) -> tuple[int, int, int, int]:
    """Return the default ``(TR, TC, min_blocks, variant)`` for systems of size ``n``."""
    if n in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[n]
    return (8, 16, 2, 3 if n <= 64 else 1) if n > 32 else (8, 8, 4, 3)


def solve_batched_fused(matrices: Any, rhs: Any, *, config: tuple[int, ...] | None = None,
                        info: Any | None = None) -> None:
    """Solve ``A_b X_b = B_b`` in place of ``rhs`` with the fused Gauss-Jordan kernel.

    ``matrices`` ``(batch, n, n)`` and ``rhs`` ``(batch, nrhs, n)`` are FP64,
    C-contiguous, and column-major per system (see the module docstring).
    ``matrices`` is not modified. ``info`` (int32 ``(batch,)``), when given,
    receives ``k + 1`` for the first zero pivot of a system and ``0`` otherwise.
    """
    cp = require_cupy_device()
    if matrices.dtype != cp.float64 or rhs.dtype != cp.float64:
        raise TypeError("the fused solve is FP64 only")
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("matrices must have shape (batch, n, n)")
    if rhs.ndim != 3 or rhs.shape[0] != matrices.shape[0] or rhs.shape[2] != matrices.shape[1]:
        raise ValueError("rhs must have shape (batch, nrhs, n)")
    if not (matrices.flags.c_contiguous and rhs.flags.c_contiguous):
        raise ValueError("matrices and rhs must be C-contiguous")
    batch, n, nrhs = int(matrices.shape[0]), int(matrices.shape[1]), int(rhs.shape[1])
    if batch == 0:
        return
    config = default_config(n) if config is None else config
    kernel = _kernel(cp, n, nrhs, config)
    threads = int(config[0]) * int(config[1])
    info_arg = info if info is not None else np.uint64(0)
    kernel((batch,), (threads,), (matrices, rhs, info_arg, cp.int64(batch)))


__all__ = ["DEFAULT_CONFIGS", "default_config", "kernel_source", "solve_batched_fused"]
