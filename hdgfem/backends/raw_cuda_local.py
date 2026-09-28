"""Shared raw CUDA trace orientation and cooperative local solve source.

Diffusion consumes the mechanically extracted source verbatim. ADR specializes
matrix dimensions and adds failure reporting without changing diffusion code.
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

RAW_COOPERATIVE_SOLVES = r"""__device__ __forceinline__ void factor_diffusion_schur_lu_coop_raw(
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

__device__ __forceinline__ void solve_diffusion_all_columns_coop_raw(
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


def checked_lu_source(name, size):
    """Specialize diffusion's cooperative pivoted LU with explicit status."""
    source = RAW_COOPERATIVE_SOLVES.split(
        "__device__ __forceinline__ void solve_diffusion_all_columns_coop_raw", 1)[0]
    source = source.replace("factor_diffusion_schur_lu_coop_raw", name).replace("NEL", size)
    source = source.replace("int* __restrict__ pivots)", "int* __restrict__ pivots, int* status)")
    source = source.replace("if (fabs(diagonal) < 1.0e-30)",
                            "if (!isfinite(diagonal) || fabs(diagonal) < 1.0e-30)")
    source = source.replace("diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;",
                            "*status = k + 1; diagonal = 1.0;")
    source = source.replace("        const int width =", "        if (*status) return;\n        const int width =")
    return source
