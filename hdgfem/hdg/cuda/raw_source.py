"""Shared raw CUDA trace orientation and cooperative local solve source.

Diffusion consumes the mechanically extracted source verbatim. ADR uses the
warp-parallel LU and column solves below, which add failure reporting and do
not change diffusion code.
"""

RAW_TRACE_ORIENTATION_HELPERS = r"""
__device__ __forceinline__ int raw_trace_local_dof(
        const bool positive,
        const int dof,
        const int edge_dof)
{
#if TRACE_ORIENTATION_MODE == 1
    return dof;
#else
    return positive ? dof : (edge_dof - 1 - dof);
#endif
}

__device__ __forceinline__ double raw_trace_orientation_sign(
        const bool positive,
        const int dof)
{
#if TRACE_ORIENTATION_MODE == 1
    return ((!positive) && ((dof & 1) == 1)) ? -1.0 : 1.0;
#else
    return 1.0;
#endif
}
"""

RAW_COOP_LU_FACTOR = r"""__device__ __forceinline__ void factor_local_lu_coop_raw(
        double* __restrict__ schur_lu,
        int* __restrict__ pivots)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < NEL; ++k) {
        if (tid == 0) {
            int pivot = k;
            double max_value = fabs(schur_lu[k * NEL + k]);
            for (int i = k + 1; i < NEL; ++i) {
                const double value = fabs(schur_lu[i * NEL + k]);
                if (value > max_value) {
                    max_value = value;
                    pivot = i;
                }
            }
            pivots[k] = pivot;
            if (pivot != k) {
                for (int j = 0; j < NEL; ++j) {
                    const double tmp = schur_lu[k * NEL + j];
                    schur_lu[k * NEL + j] = schur_lu[pivot * NEL + j];
                    schur_lu[pivot * NEL + j] = tmp;
                }
            }
            double diagonal = schur_lu[k * NEL + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                schur_lu[k * NEL + k] = diagonal;
            }
            for (int i = k + 1; i < NEL; ++i) {
                schur_lu[i * NEL + k] /= diagonal;
            }
        }
        __syncthreads();

        const int width = NEL - k - 1;
        for (int idx = tid; idx < width * width; idx += blockDim.x) {
            const int i = k + 1 + idx / width;
            const int j = k + 1 + idx - (idx / width) * width;
            schur_lu[i * NEL + j] -= schur_lu[i * NEL + k] * schur_lu[k * NEL + j];
        }
        __syncthreads();
    }
}

"""

# Cooperative multi-column triangular solves (need NCOLS / RAW_BATCH_COLS).
RAW_COOP_COLUMN_SOLVES = r"""__device__ __forceinline__ void solve_diffusion_all_columns_coop_raw(
        const double* __restrict__ schur_lu,
        const int* __restrict__ pivots,
        double* __restrict__ columns)
{
    const int tid = threadIdx.x;
    // Columns are independent after the shared LU factorization.  Let each
    // thread carry its columns through pivoting and both triangular solves so
    // dependencies remain thread-local.  The previous row-wise formulation
    // imposed 3 * NEL block barriers even though no column consumed another
    // column's values.
    for (int col = tid; col < NCOLS; col += blockDim.x) {
        for (int k = 0; k < NEL; ++k) {
            const int pivot = pivots[k];
            if (pivot != k) {
                const double tmp = columns[k * NCOLS + col];
                columns[k * NCOLS + col] = columns[pivot * NCOLS + col];
                columns[pivot * NCOLS + col] = tmp;
            }
        }

        for (int i = 0; i < NEL; ++i) {
            double value = columns[i * NCOLS + col];
            for (int j = 0; j < i; ++j) {
                value -= schur_lu[i * NEL + j] * columns[j * NCOLS + col];
            }
            columns[i * NCOLS + col] = value;
        }
        for (int i = NEL - 1; i >= 0; --i) {
            double value = columns[i * NCOLS + col];
            for (int j = i + 1; j < NEL; ++j) {
                value -= schur_lu[i * NEL + j] * columns[j * NCOLS + col];
            }
            columns[i * NCOLS + col] = value / schur_lu[i * NEL + i];
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void solve_diffusion_column_batch_coop_raw(
        const double* __restrict__ schur_lu,
        const int* __restrict__ pivots,
        double* __restrict__ columns,
        const int batch_cols)
{
    const int tid = threadIdx.x;
    for (int col = tid; col < batch_cols; col += blockDim.x) {
        for (int k = 0; k < NEL; ++k) {
            const int pivot = pivots[k];
            if (pivot != k) {
                const double tmp = columns[k * RAW_BATCH_COLS + col];
                columns[k * RAW_BATCH_COLS + col] = columns[pivot * RAW_BATCH_COLS + col];
                columns[pivot * RAW_BATCH_COLS + col] = tmp;
            }
        }
        for (int i = 0; i < NEL; ++i) {
            double value = columns[i * RAW_BATCH_COLS + col];
            for (int j = 0; j < i; ++j) {
                value -= schur_lu[i * NEL + j] * columns[j * RAW_BATCH_COLS + col];
            }
            columns[i * RAW_BATCH_COLS + col] = value;
        }
        for (int i = NEL - 1; i >= 0; --i) {
            double value = columns[i * RAW_BATCH_COLS + col];
            for (int j = i + 1; j < NEL; ++j) {
                value -= schur_lu[i * NEL + j] * columns[j * RAW_BATCH_COLS + col];
            }
            columns[i * RAW_BATCH_COLS + col] = value / schur_lu[i * NEL + i];
        }
    }
    __syncthreads();
}

"""

# Factor plus column solves, for kernels that define NCOLS and RAW_BATCH_COLS.
RAW_COOPERATIVE_SOLVES = RAW_COOP_LU_FACTOR + RAW_COOP_COLUMN_SOLVES


_WARP_LU_TEMPLATE = r"""
// Pivoted LU of a row-major LU_SIZE x LU_SIZE shared matrix with an explicit
// status (k + 1 on a nonfinite or tiny pivot). Warp 0 performs each step's
// serial work in parallel: a shuffle arg-max pivot search (ties keep the lowest
// row and a NaN on the diagonal keeps row k, exactly like the serial search),
// the row interchange, and the pivot-column scaling. The whole block then
// applies the trailing rank-one update, as in the cooperative diffusion LU.
__device__ __forceinline__ void LU_NAME(
        double* __restrict__ lu,
        int* __restrict__ pivots,
        int* status)
{
    const int n = LU_SIZE;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int lanes = blockDim.x < 32 ? blockDim.x : 32;
    for (int k = 0; k < n; ++k) {
        if (tid < 32) {
            const unsigned mask = __activemask();
            double best = -1.0;
            int row = n;
            for (int i = k + lane; i < n; i += lanes) {
                double value = fabs(lu[i * n + k]);
                if (value != value) value = -1.0;
                if (value > best) {
                    best = value;
                    row = i;
                }
            }
            for (int offset = lanes / 2; offset > 0; offset >>= 1) {
                const double other = __shfl_xor_sync(mask, best, offset);
                const int other_row = __shfl_xor_sync(mask, row, offset);
                if (other > best || (other == best && other_row < row)) {
                    best = other;
                    row = other_row;
                }
            }
            const double current = lu[k * n + k];
            const int pivot = (current != current || row >= n) ? k : row;
            if (pivot != k) {
                for (int j = lane; j < n; j += lanes) {
                    const double tmp = lu[k * n + j];
                    lu[k * n + j] = lu[pivot * n + j];
                    lu[pivot * n + j] = tmp;
                }
            }
            __syncwarp(mask);
            double diagonal = lu[k * n + k];
            if (!isfinite(diagonal) || fabs(diagonal) < 1.0e-30) {
                if (lane == 0) *status = k + 1;
                diagonal = 1.0;
            }
            if (lane == 0) pivots[k] = pivot;
            for (int i = k + 1 + lane; i < n; i += lanes) {
                lu[i * n + k] /= diagonal;
            }
        }
        __syncthreads();
        if (*status) return;
        const int width = n - k - 1;
        for (int idx = tid; idx < width * width; idx += blockDim.x) {
            const int i = k + 1 + idx / width;
            const int j = k + 1 + idx - (idx / width) * width;
            lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
        }
        __syncthreads();
    }
}
"""


def checked_warp_lu_source(name, size):
    """Status-returning pivoted LU for ADR (pivots match the cooperative diffusion LU).

    Supports the public 1/32/64/128-thread launches. Rows stride active lanes.
    """
    return _WARP_LU_TEMPLATE.replace("LU_NAME", name).replace("LU_SIZE", size)


RAW_WARP_COLUMN_SOLVES = r"""
#if NEL <= 32
// Per-element setup for the warp column solves: fold the LU row interchanges
// into one permutation (rows[i] is the original row that ends up in position
// i), and take the pivot reciprocals once. On GPUs with few FP64 units every
// warp-wide FP64 instruction is expensive, so the solves only multiply.
__device__ __forceinline__ void lu_solve_setup(
        const double* __restrict__ lu,
        const int* __restrict__ pivots,
        int* __restrict__ rows,
        double* __restrict__ inverse_diagonal)
{
    if (threadIdx.x == 0) {
        for (int i = 0; i < NEL; ++i) rows[i] = i;
        for (int k = 0; k < NEL; ++k) {
            const int p = pivots[k], tmp = rows[k];
            rows[k] = rows[p];
            rows[p] = tmp;
        }
    }
    for (int i = threadIdx.x; i < NEL; i += blockDim.x)
        inverse_diagonal[i] = 1.0 / lu[i * (NEL + 1)];
    __syncthreads();
}

// Solve a batch of right-hand sides with a pivoted NEL x NEL LU: one warp per
// column, row r in lane r's register, so both triangular sweeps are shuffles
// with one FP64 FMA per step. Forward sums keep the serial order; backward sums
// run in reverse column order.
__device__ __forceinline__ void solve_lu_columns_warp(
        const double* __restrict__ lu,
        const int* __restrict__ rows,
        const double* __restrict__ inverse_diagonal,
        double* __restrict__ columns,
        const int count)
{
    // A sub-warp launch cannot hold every row in a lane register.
    if (blockDim.x < 32) {
        for (int c = threadIdx.x; c < count; c += blockDim.x) {
            double x[NEL];
            for (int i = 0; i < NEL; ++i)
                x[i] = columns[rows[i] * RAW_BATCH_COLS + c];
            for (int i = 0; i < NEL; ++i)
                for (int j = 0; j < i; ++j) x[i] -= lu[i * NEL + j] * x[j];
            for (int i = NEL - 1; i >= 0; --i) {
                for (int j = NEL - 1; j > i; --j) x[i] -= lu[i * NEL + j] * x[j];
                x[i] *= inverse_diagonal[i];
            }
            for (int i = 0; i < NEL; ++i) columns[i * RAW_BATCH_COLS + c] = x[i];
        }
        __syncthreads();
        return;
    }
    const int lane = threadIdx.x & 31;
    const bool active = lane < NEL;
    const int source_row = active ? rows[lane] : 0;
    for (int c = threadIdx.x >> 5; c < count; c += blockDim.x >> 5) {
        double x = active ? columns[source_row * RAW_BATCH_COLS + c] : 0.0;
#pragma unroll
        for (int j = 0; j < NEL - 1; ++j) {
            const double l = (lane > j && active) ? lu[lane * NEL + j] : 0.0;
            x -= l * __shfl_sync(0xffffffffu, x, j);
        }
#pragma unroll
        for (int j = NEL - 1; j >= 0; --j) {
            if (lane == j) x *= inverse_diagonal[j];
            const double u = lane < j ? lu[lane * NEL + j] : 0.0;
            x -= u * __shfl_sync(0xffffffffu, x, j);
        }
        if (active) columns[lane * RAW_BATCH_COLS + c] = x;
    }
    __syncthreads();
}
#else
#error "warp LU column solves hold one row per lane and need NEL <= 32 (p <= 6)"
#endif
"""
