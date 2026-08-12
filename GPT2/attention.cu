// Self-attention (FlashAttention-style, one Q tile per block, streaming K/V).
//
// Layout: Q/K/V/out are [plane][seq][d_head], where a "plane" is one (batch, head)
// pair. grid.x walks planes, grid.y walks Q row tiles.
//
// The kernel is issue-bound rather than compute- or DRAM-bound, so the structure is
// built around two ideas: keep loop-invariant operands in registers, and always have
// several ldmatrix in flight before the dependent mma issues.

#include "kernels.cuh"
#include <cuda_profiler_api.h>
#include <cfloat>


// Width-4 butterflies: the four lanes sharing a C-fragment row hold the eight columns of
// one n-tile group, so the reduction stops there rather than spanning the warp.
__device__ __forceinline__ float max_reduction(float val) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int i = 2; i > 0; i >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, i, 4));
    }
    return val;
}

__device__ __forceinline__ float sum_reduction(float sum) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int i = 2; i > 0; i >>= 1) {
        sum += __shfl_xor_sync(mask, sum, i, 4);
    }
    return sum;
}


// Tile sizes and the shared-memory budget live in kernels.cuh, since the driver needs
// them to size the launch. BK here is 64; gemm_cfg::BK is 32, which is exactly why they
// are namespaced rather than macros.
using namespace attn;

// Byte deltas used to walk the precomputed shared-memory addresses. Advancing by
// whole 8-row groups leaves row % 8 unchanged, so the swizzle is invariant and the
// delta is a plain constant. Purely internal, so they stay here.
constexpr uint32_t KV_BUF_BYTES = BK * BN * sizeof(half);   // one double-buffer stage
constexpr uint32_t K_N_BYTES    = 8 * BN * sizeof(half);    // one K n-tile = 8 key rows
constexpr uint32_t ROW16_BYTES  = 16 * BN * sizeof(half);   // one cp_async row group


__global__ void flash_attention(const half* __restrict__ Q, const half* __restrict__ K,
                                const half* __restrict__ V, half* __restrict__ out,
                                int d_M, int d_K, int d_N,
                                int in_row_stride, int out_row_stride,
                                int in_plane_stride, int out_plane_stride) {
    extern __shared__ half smem[];
    half* sQ = smem;
    half* sK = sQ + Q_ELES;
    half* sV = sK + K_ELES;

    // The row stride is separate from the head dim so the kernel can read straight out
    // of a packed [T, 3*d_model] QKV buffer (in_row_stride = 3*d_model, plane stride =
    // d_head, caller offsets K and V by d_model and 2*d_model) as well as out of
    // per-plane [plane][seq][d_head] tensors (in_row_stride = d_head, plane stride =
    // seq*d_head). That is what removes the split and merge passes.
    Q += (size_t)blockIdx.x * in_plane_stride;
    K += (size_t)blockIdx.x * in_plane_stride;
    V += (size_t)blockIdx.x * in_plane_stride;
    out += (size_t)blockIdx.x * out_plane_stride;

    const int tid      = threadIdx.x;
    const int warp_id  = tid / 32;
    const int lane     = tid % 32;
    const int warp_row = warp_id * WARP_M;   // first Q row this warp owns, block-relative

    float acc[PV_N_TILES * 4] = {0.0f};
    float m_row[2] = {-FLT_MAX, -FLT_MAX};
    float l_row[2] = {0.0f, 0.0f};

    // exp2f is one SFU op; expf is EX2 plus a range-reduction multiply. Folding log2(e)
    // into the QK scale runs the entire softmax in base 2 at no cost.
    const float attn_scale = rsqrtf((float)d_N) * 1.4426950408889634f;

    const int load_row = tid / 16;
    const int load_col = (tid % 16) * 8;

    #pragma unroll
    for (int i = 0; i < BM / 16; ++i) {
        cp_async_128(swizzled_ptr(sQ, load_row + i * 16, load_col, BN),
                     &Q[(size_t)(BM * blockIdx.y + load_row + i * 16) * in_row_stride + load_col]);
    }

    // Precomputing part of the smem swizzle so we don't have to calculate it a billion times (stall wait was an issue in prev profiling)
    // Every tile is read with row % 8 == lane % 8, so the XOR swizzle depends only on
    // the column. That splits the two operands:
    //   K: n walks key *rows*, which the swizzle never touches, so advancing n is a
    //      constant byte delta and only the 8 k-steps need a base.
    //   V: n walks head-dim *columns*, which is exactly the swizzled axis, so n stays
    //      inside the XOR and only the row base can be hoisted.
    // The store side is loop invariant apart from the buffer toggle.
    const uint32_t sK_base = cvta_generic_to_shared(sK);
    const uint32_t sV_base = cvta_generic_to_shared(sV);
    const int lane_s = lane % 8;

    const uint32_t dst_k0 = swizzled_ptr(sK, load_row, load_col, BN);
    const uint32_t dst_v0 = swizzled_ptr(sV, load_row, load_col, BN);

    uint32_t k_addr[QK_K_STEPS];
    #pragma unroll
    for (int k = 0; k < QK_K_STEPS; ++k) {
        const int chunk = k * 2 + ((lane / 8) % 2);
        k_addr[k] = sK_base + (uint32_t)(lane_s * BN + (chunk ^ lane_s) * 8) * sizeof(half);
    }

    uint32_t v_addr[PV_K_STEPS];
    #pragma unroll
    for (int k = 0; k < PV_K_STEPS; ++k) {
        v_addr[k] = sV_base + (uint32_t)((k * 16 + (lane % 16)) * BN) * sizeof(half);
    }

    // The V column index is the swizzled axis, so its byte delta is a per-lane XOR
    // rather than a constant. Precomputing all 16 turns each address into one add.
    uint32_t v_off[PV_N_TILES];
    #pragma unroll
    for (int n = 0; n < PV_N_TILES; ++n) {
        v_off[n] = (uint32_t)((n ^ lane_s) << 4);
    }

    // Global side: the k-tile advances by a fixed row stride, so walk pointers
    // instead of recomputing a full index every iteration.
    const half* k_src = K + (size_t)load_row * in_row_stride + load_col;
    const half* v_src = V + (size_t)load_row * in_row_stride + load_col;
    const int   src_row16 = 16 * in_row_stride;
    const int   src_tile  = BK * in_row_stride;

    #pragma unroll
    for (int i = 0; i < BK / 16; ++i) {
        cp_async_128(dst_k0 + i * ROW16_BYTES, k_src + i * src_row16);
        cp_async_128(dst_v0 + i * ROW16_BYTES, v_src + i * src_row16);
    }
    cp_async_commit_group(); cp_async_wait_group<0>(); __syncthreads();

    // loading Q into regs to keep forever (we have enough regs to do this)
    uint32_t rQ[QK_K_STEPS][4];
    #pragma unroll
    for (int k = 0; k < QK_K_STEPS; ++k) {
        ld_matrix_x4(rQ[k], swizzled_ptr(sQ, warp_row + (lane % 16),
                                         k * 16 + (lane / 16) * 8, BN));
    }

    int write = 0, read = 0;

    // ---- causal bounds ---------------------------------------------------------
    // This block owns queries [q_tile_lo, q_tile_lo + BM), so every key from
    // q_tile_lo + BM onward is in the future for all of them. Those tiles are not
    // masked, they are never streamed: the loop simply stops. That is where the
    // triangle actually pays -- roughly half the K/V traffic and half the mmas.
    const int q_tile_lo = BM * (int)blockIdx.y;
    const int k_end     = (q_tile_lo + BM < d_K) ? (q_tile_lo + BM) : d_K;

    // Per-element masking is only needed on tiles that reach past this warp's first
    // query row; anything wholly below the diagonal is fully visible. The test is
    // warp-uniform, so the branch costs nothing on an issue-bound kernel.
    const int q_warp_lo = q_tile_lo + warp_row;

    for (int tile_k = 0; tile_k < k_end; tile_k += BK) {
        const int next = tile_k + BK;
        const bool has_next = next < k_end;

        if (has_next) {
            write ^= 1;
            const uint32_t buf_w = write * KV_BUF_BYTES;
            k_src += src_tile; v_src += src_tile;
            #pragma unroll
            for (int i = 0; i < BK / 16; ++i) {
                cp_async_128(dst_k0 + buf_w + i * ROW16_BYTES, k_src + i * src_row16);
                cp_async_128(dst_v0 + buf_w + i * ROW16_BYTES, v_src + i * src_row16);
            }
            cp_async_commit_group();
        }

        const uint32_t buf_r = read * KV_BUF_BYTES;

        float scores[QK_N_TILES * 4] = {0.0f};

        #pragma unroll
        for (int k = 0; k < QK_K_STEPS; ++k) {
            // Every B fragment is issued before the first dependent mma so the MIO
            // pipe stays busy; a 1:1 ldmatrix/mma interleave stalls on every step.
            uint32_t rK[QK_N_TILES][2];
            #pragma unroll
            for (int n = 0; n < QK_N_TILES; ++n) {
                // K is row-major [key][head]; mma wants B column-major over (head, key),
                // which is the same bytes, so no transpose is needed here.
                ld_matrix_x2(rK[n], k_addr[k] + n * K_N_BYTES + buf_r);
            }
            #pragma unroll
            for (int n = 0; n < QK_N_TILES; ++n) {
                mma(rQ[k], rK[n], &scores[n * 4]);
            }
        }

        // ---- causal mask -----------------------------------------------------
        // Same C-fragment layout the P repack below relies on: for n-tile n,
        //   regs 0,1 -> query row (lane/4),     keys 8n + 2*(lane%4) + {0,1}
        //   regs 2,3 -> query row (lane/4) + 8, the same two keys
        // A key strictly greater than its query is unseen, so it is driven to
        // -FLT_MAX. The max reduction then ignores it and the exp2f below returns
        // exactly 0, which leaves l_row and acc untouched -- no separate bookkeeping
        // for "this row saw nothing in this tile" is needed.
        //
        // A row can never end up with an empty softmax: tile 0 always contains key 0,
        // and every query can see key 0, so l_row is strictly positive by the time the
        // epilogue divides by it.
        if (tile_k + BK - 1 > q_warp_lo) {
            const int q_lo = q_warp_lo + (lane / 4);
            const int k_lo = tile_k + (lane % 4) * 2;
            #pragma unroll
            for (int n = 0; n < QK_N_TILES; ++n) {
                const int k = k_lo + n * 8;
                if (k     > q_lo)     scores[n * 4 + 0] = -FLT_MAX;
                if (k + 1 > q_lo)     scores[n * 4 + 1] = -FLT_MAX;
                if (k     > q_lo + 8) scores[n * 4 + 2] = -FLT_MAX;
                if (k + 1 > q_lo + 8) scores[n * 4 + 3] = -FLT_MAX;
            }
        }

        // ---- online softmax --------------------------------------------------
        // Lane L owns C-fragment rows L/4 and L/4 + 8; regs 0,1 belong to the first
        // row and regs 2,3 to the second, hence the two independent running stats.
        // The scale is never applied to S directly. Since attn_scale > 0,
        // max(s*c) == c*max(s), so the max is taken on raw scores and scaled once;
        // the per-element scaling then folds into the exponent's FFMA below. That
        // turns 2*QK_N_TILES*4 separate multiplies and subtracts into FFMAs.
        float local_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int n = 0; n < QK_N_TILES; ++n) {
            local_max[0] = fmaxf(local_max[0], fmaxf(scores[n * 4 + 0], scores[n * 4 + 1]));
            local_max[1] = fmaxf(local_max[1], fmaxf(scores[n * 4 + 2], scores[n * 4 + 3]));
        }
        local_max[0] = max_reduction(local_max[0]) * attn_scale;
        local_max[1] = max_reduction(local_max[1]) * attn_scale;

        const float new_m0 = fmaxf(m_row[0], local_max[0]);
        const float new_m1 = fmaxf(m_row[1], local_max[1]);
        const float corr0  = exp2f(m_row[0] - new_m0);
        const float corr1  = exp2f(m_row[1] - new_m1);
        m_row[0] = new_m0; m_row[1] = new_m1;

        #pragma unroll
        for (int j = 0; j < PV_N_TILES; ++j) {
            acc[j * 4 + 0] *= corr0; acc[j * 4 + 1] *= corr0;
            acc[j * 4 + 2] *= corr1; acc[j * 4 + 3] *= corr1;
        }

        // P is written in place over S; keeping both live would cost 16 registers.
        float tile_sum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int n = 0; n < QK_N_TILES; ++n) {
            scores[n * 4 + 0] = exp2f(fmaf(scores[n * 4 + 0], attn_scale, -m_row[0]));
            scores[n * 4 + 1] = exp2f(fmaf(scores[n * 4 + 1], attn_scale, -m_row[0]));
            scores[n * 4 + 2] = exp2f(fmaf(scores[n * 4 + 2], attn_scale, -m_row[1]));
            scores[n * 4 + 3] = exp2f(fmaf(scores[n * 4 + 3], attn_scale, -m_row[1]));
            tile_sum[0] += scores[n * 4 + 0] + scores[n * 4 + 1];
            tile_sum[1] += scores[n * 4 + 2] + scores[n * 4 + 3];
        }
        l_row[0] = l_row[0] * corr0 + sum_reduction(tile_sum[0]);
        l_row[1] = l_row[1] * corr1 + sum_reduction(tile_sum[1]);

        // ---- repack P from accumulator layout into A-operand layout -----------
        // For m16n8k16 the C fragment of n-tile t holds
        //   c0,c1 -> (row = L/4,     col = 8t + 2*(L%4) + {0,1})
        //   c2,c3 -> (row = L/4 + 8, col = 8t + 2*(L%4) + {0,1})
        // and the A fragment of a 16x16 tile wants
        //   a0 -> (row,   col + {0,1})   a1 -> (row + 8, col + {0,1})
        //   a2 -> (row,   col + {8,9})   a3 -> (row + 8, col + {8,9})
        // so two adjacent n-tiles packed to half2 ARE one A fragment. No transpose,
        // no shuffle, and no shared memory round trip for P.
        uint32_t rP[PV_K_STEPS][4];
        #pragma unroll
        for (int k = 0; k < PV_K_STEPS; ++k) {
            const int t = k * 2;
            rP[k][0] = pack_half2(scores[(t + 0) * 4 + 0], scores[(t + 0) * 4 + 1]);
            rP[k][1] = pack_half2(scores[(t + 0) * 4 + 2], scores[(t + 0) * 4 + 3]);
            rP[k][2] = pack_half2(scores[(t + 1) * 4 + 0], scores[(t + 1) * 4 + 1]);
            rP[k][3] = pack_half2(scores[(t + 1) * 4 + 2], scores[(t + 1) * 4 + 3]);
        }

        // ---- O += P * V -------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < PV_K_STEPS; ++k) {
            #pragma unroll
            for (int nb = 0; nb < PV_N_TILES / 4; ++nb) {
                uint32_t rV[4][2];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // V is row-major [key][head] but mma wants B column-major over
                    // (key, head); the transposing ldmatrix does that for free.
                    ld_matrix_x2_trans(rV[j], v_addr[k] + v_off[nb * 4 + j] + buf_r);
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma(rP[k], rV[j], &acc[(nb * 4 + j) * 4]);
                }
            }
        }

        if (has_next) {
            read ^= 1;
            cp_async_wait_group<0>();
            __syncthreads();
        }
    }

    // ---- epilogue -------------------------------------------------------------
    // One reciprocal instead of 64 divides, and 4-byte stores instead of 2-byte ones.
    const float inv_l0 = 1.0f / l_row[0];
    const float inv_l1 = 1.0f / l_row[1];

    const int out_row = BM * blockIdx.y + warp_row + (lane / 4);
    const int col_base = (lane % 4) * 2;

    #pragma unroll
    for (int n = 0; n < PV_N_TILES; ++n) {
        const int col = n * 8 + col_base;
        const __half2 lo = __floats2half2_rn(acc[n * 4 + 0] * inv_l0, acc[n * 4 + 1] * inv_l0);
        const __half2 hi = __floats2half2_rn(acc[n * 4 + 2] * inv_l1, acc[n * 4 + 3] * inv_l1);
        *reinterpret_cast<__half2*>(&out[(size_t)out_row * out_row_stride + col]) = lo;
        *reinterpret_cast<__half2*>(&out[(size_t)(out_row + 8) * out_row_stride + col]) = hi;
    }
}
