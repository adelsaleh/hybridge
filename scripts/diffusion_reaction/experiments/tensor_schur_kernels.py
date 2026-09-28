"""Experimental WMMA substitution for two fused diffusion Schur products.

The caller specializes the existing production source; all LU, triangular
solves, condensation and scatter remain unchanged FP64 CUDA-core operations.
"""

WMMA_SOURCE = r'''
#include <mma.h>
__device__ __forceinline__ void tensor_schur_add(
    const double* a, const double* b, double* c, const double scale,
    float* scratch, const int n)
{
    using namespace nvcuda;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int tiles = (n + 15) / 16;
    const int active_warps = min((int)blockDim.x / 32, tiles * tiles);
    if (warp >= active_warps) return;
    float* tile_a = scratch + warp * 256;
    float* tile_b = tile_a + 128;
    for (int tile = warp; tile < tiles * tiles; tile += active_warps) {
        const int row_base = (tile / tiles) * 16;
        const int col_base = (tile % tiles) * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> accum[TENSOR_TERMS];
        #pragma unroll
        for (int term = 0; term < TENSOR_TERMS; ++term) wmma::fill_fragment(accum[term], 0.0f);
        for (int kbase = 0; kbase < n; kbase += 8) {
            #pragma unroll
            for (int term = 0; term < TENSOR_TERMS; ++term) {
                for (int q = lane; q < 128; q += 32) {
                    const int ar = row_base + q / 8, ak = kbase + q % 8;
                    const int bk = kbase + q / 16, bc = col_base + q % 16;
                    const double av = (ar < n && ak < n) ? a[ar*n+ak] : 0.0;
                    const double bv = (bk < n && bc < n) ? b[bk*n+bc] : 0.0;
                    const float ah = wmma::__float_to_tf32((float)av);
                    const float bh = wmma::__float_to_tf32((float)bv);
                    tile_a[q] = (term == 2) ? wmma::__float_to_tf32((float)(av-(double)ah)) : ah;
                    tile_b[q] = (term == 1) ? wmma::__float_to_tf32((float)(bv-(double)bh)) : bh;
                }
                __syncwarp();
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bf;
                wmma::load_matrix_sync(af, tile_a, 8);
                wmma::load_matrix_sync(bf, tile_b, 16);
                wmma::mma_sync(accum[term], af, bf, accum[term]);
                __syncwarp();
            }
        }
        double values[8] = {0.0};
        #pragma unroll
        for (int term = 0; term < TENSOR_TERMS; ++term) {
            wmma::store_matrix_sync(tile_a, accum[term], 16, wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int q = 0; q < 8; ++q) values[q] += (double)tile_a[lane + q*32];
            __syncwarp();
        }
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int index = lane + q*32;
            const int row = row_base + index/16, col = col_base + index%16;
            if (row < n && col < n) c[row*n+col] += scale*values[q];
        }
        __syncwarp();
    }
}
'''


def specialize_tensor_schur(source, *, nel, ntr, batch_cols, block_size, mode):
    """Replace only the two pre-LU matrix products; reject template drift."""
    if mode not in {"tf32", "tf32x3"} or block_size not in {32, 64, 128}:
        raise ValueError("unsupported tensor prototype configuration")
    warps = min(block_size // 32, ((nel + 15) // 16) ** 2)
    required = warps * 256 * 4
    available = (2 * nel * batch_cols + 3 * ntr * batch_cols) * 8
    # The column workspaces are dead during Schur construction. Reuse them
    # without touching the adjacent pivot storage. Their base is 32B aligned.
    scratch = "reinterpret_cast<float*>(local_rhs)"
    if required > available:
        scratch = "tensor_scratch"
        declaration = f"__shared__ __align__(32) float tensor_scratch[{warps*256}];\n"
        needle = "    const int source_col = "
        assert source.count(needle) == 1
        source = source.replace(needle, declaration + needle)
    start = source.index("#elif RAW_BATCHED_FULL\n        // Build the factor")
    stop = source.index("#else\n        for (int idx = tid;", start)
    body = source[start:stop]
    for direction in (0, 1):
        old = f'''        for (int idx = tid; idx < {nel} * {nel}; idx += blockDim.x) {{
            const int i = idx / {nel};
            const int j = idx - i * {nel};
            double value = 0.0;
            for (int k = 0; k < {nel}; ++k) {{
                value += mn{direction}[i * {nel} + k] * k_d0[k * {nel} + j];
            }}
            schur_matrix[idx] += jac_inv * value;
        }}'''
        if body.count(old) != 1:
            raise ValueError("production Schur source changed; inspect before applying prototype")
        body = body.replace(old, f"        tensor_schur_add(mn{direction}, k_d0, schur_matrix, jac_inv, {scratch}, {nel});")
    terms = 1 if mode == "tf32" else 3
    result = f"#define TENSOR_TERMS {terms}\n" + WMMA_SOURCE + source[:start] + body + source[stop:]
    return result, {"scratch_bytes": required, "scratch_reuses_column_workspace": required <= available,
                    "additional_static_shared_bytes": max(0, required if required > available else 0),
                    "tensor_products_per_element": 2, "terms_per_product": terms}
