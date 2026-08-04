// Self-attention (FlashAttention-style, one Q tile per block, streaming K/V).
//
// Layout: Q/K/V/out are [plane][seq][d_head], where a "plane" is one (batch, head)
// pair. grid.x walks planes, grid.y walks Q row tiles.
//
// The kernel is issue-bound rather than compute- or DRAM-bound, so the structure is
// built around two ideas: keep loop-invariant operands in registers, and always have
// several ldmatrix in flight before the dependent mma issues.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>


__device__ __forceinline__ uint32_t cvta_generic_to_shared(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(ptr)));
}


__device__ __forceinline__ uint32_t swizzled_ptr(const void* smem_ptr, int row, int col, int stride) {
    int eles_per_vec = 16 / (int)sizeof(half);
    int vecs_per_row = stride / eles_per_vec;

    int chunk_idx = col / eles_per_vec;
    int offset = col % eles_per_vec;

    int swizzled_chunk = chunk_idx ^ ((row % 8) % vecs_per_row);
    int flat_idx = (row * stride) + (swizzled_chunk * eles_per_vec) + offset;
    return cvta_generic_to_shared(static_cast<const half*>(smem_ptr) + flat_idx);
}

__device__ __forceinline__ void cp_async_128(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile ("cp.async.commit_group;\n");
}

template <int N> // availability at runtime
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n" :: "n"(N));
}

// volatile is kept on every ldmatrix: without it the compiler may CSE a load across
// __syncthreads, and the double-buffered K/V addresses repeat every two iterations
// while the data underneath them does not. Operand reuse is done structurally instead.
__device__ __forceinline__ void ld_matrix_x4(uint32_t rA[4], uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(rA[0]), "=r"(rA[1]), "=r"(rA[2]), "=r"(rA[3])
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2(uint32_t rB[2], uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rB[0]), "=r"(rB[1])
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2_trans(uint32_t rB[2], uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rB[0]), "=r"(rB[1])
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void mma(const uint32_t rA[4], const uint32_t rB[2], float rC[4]) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(rC[0]), "+f"(rC[1]), "+f"(rC[2]), "+f"(rC[3])
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]),
        "r"(rB[0]), "r"(rB[1])
    );
}

// Two floats -> one packed half2 register. F2FP.PACK_AB, single instruction on sm_80+.
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}


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


#define BM 128   // Q rows per block tile
#define BN 128   // head dim, also the smem row stride
#define BK 64    // K/V rows (keys) per streamed tile

#define WARPS   8
#define WARP_M  (BM / WARPS)      // 16 Q rows per warp -> one m16 mma tile

#define QK_K_STEPS  (BN / 16)     // 8 reduction steps over head dim
#define QK_N_TILES  (BK / 8)      // n-tiles of S, 8 keys each
#define PV_K_STEPS  (BK / 16)     // reduction steps over keys
#define PV_N_TILES  (BN / 8)      // 16 n-tiles of O, 8 head-dim cols each

constexpr size_t Q_ELES = BM * BN;
constexpr size_t K_ELES = 2 * BK * BN;
constexpr size_t V_ELES = 2 * BK * BN;

// Byte deltas used to walk the precomputed shared-memory addresses. Advancing by
// whole 8-row groups leaves row % 8 unchanged, so the swizzle is invariant and the
// delta is a plain constant.
constexpr uint32_t KV_BUF_BYTES = BK * BN * sizeof(half);   // one double-buffer stage
constexpr uint32_t K_N_BYTES    = 8 * BN * sizeof(half);    // one K n-tile = 8 key rows
constexpr uint32_t ROW16_BYTES  = 16 * BN * sizeof(half);   // one cp_async row group

// P never reaches shared memory, so there is no sP staging buffer here.
constexpr size_t SMEM_BYTES = (Q_ELES + K_ELES + V_ELES) * sizeof(half);


__global__ void flash_attention(const half* __restrict__ Q, const half* __restrict__ K,
                                const half* __restrict__ V, half* __restrict__ out,
                                int d_M, int d_K, int d_N) {
    extern __shared__ half smem[];
    half* sQ = smem;
    half* sK = sQ + Q_ELES;
    half* sV = sK + K_ELES;

    const size_t q_plane  = (size_t)blockIdx.x * d_M * d_N;
    const size_t kv_plane = (size_t)blockIdx.x * d_K * d_N;
    Q += q_plane; out += q_plane;
    K += kv_plane; V += kv_plane;

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
                     &Q[(size_t)(BM * blockIdx.y + load_row + i * 16) * d_N + load_col]);
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
    const half* k_src = K + (size_t)load_row * d_N + load_col;
    const half* v_src = V + (size_t)load_row * d_N + load_col;
    const int   src_row16 = 16 * d_N;
    const int   src_tile  = BK * d_N;

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

    for (int tile_k = 0; tile_k < d_K; tile_k += BK) {
        const int next = tile_k + BK;
        const bool has_next = next < d_K;

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
        *reinterpret_cast<__half2*>(&out[(size_t)out_row * d_N + col]) = lo;
        *reinterpret_cast<__half2*>(&out[(size_t)(out_row + 8) * d_N + col]) = hi;
    }
}

#define CUDA_CHECK(call) do{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)


// Reference attention for a single output row, used to spot-check the kernel.
static void reference_row(const half* Q, const half* K, const half* V, float* o_row,
                          int plane, int row, int seq, int d_head) {
    const size_t off = (size_t)plane * seq * d_head;
    const half* q = Q + off + (size_t)row * d_head;
    const half* k = K + off;
    const half* v = V + off;

    const float scale = 1.0f / sqrtf((float)d_head);
    float* s = (float*)malloc(sizeof(float) * seq);

    float m = -FLT_MAX;
    for (int j = 0; j < seq; ++j) {
        float dot = 0.0f;
        for (int c = 0; c < d_head; ++c) {
            dot += __half2float(q[c]) * __half2float(k[(size_t)j * d_head + c]);
        }
        s[j] = dot * scale;
        m = fmaxf(m, s[j]);
    }

    float l = 0.0f;
    for (int c = 0; c < d_head; ++c) o_row[c] = 0.0f;
    for (int j = 0; j < seq; ++j) {
        const float p = expf(s[j] - m);
        l += p;
        for (int c = 0; c < d_head; ++c) {
            o_row[c] += p * __half2float(v[(size_t)j * d_head + c]);
        }
    }
    for (int c = 0; c < d_head; ++c) o_row[c] /= l;

    free(s);
}


int main() {
    // No boundary masking in the kernel, so the shapes must tile exactly:
    //   seq % BM == 0 (queries), seq % BK == 0 (keys), d_head == BN.
    // 14 * (2048/128) = 224 blocks = exactly 4 waves on 56 SMs, so no partial-wave
    // tail contaminates the measurement.
    const int planes = 14;     // batch * heads -> grid.x
    const int seq    = 2048;   // queries and keys/values
    const int d_head = BN;     // 128

    const size_t plane_eles = (size_t)seq * d_head;
    const size_t total_eles = (size_t)planes * plane_eles;
    const size_t bytes = sizeof(half) * total_eles;

    half *h_Q, *h_K, *h_V, *h_O;
    CUDA_CHECK(cudaMallocHost(&h_Q, bytes));
    CUDA_CHECK(cudaMallocHost(&h_K, bytes));
    CUDA_CHECK(cudaMallocHost(&h_V, bytes));
    CUDA_CHECK(cudaMallocHost(&h_O, bytes));

    half *d_Q, *d_Kk, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_Kk, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    for (size_t i = 0; i < total_eles; ++i) {
        h_Q[i] = static_cast<half>(((i * 7 + 1) % 17) / 19.0f);
        h_K[i] = static_cast<half>(((i * 5 + 3) % 17) / 21.0f);
        h_V[i] = static_cast<half>(((i * 3 + 2) % 13) / 23.0f);
    }

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Kk, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    // 64 KB of dynamic smem is over the 48 KB default cap, so it must be opted into.
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)SMEM_BYTES));

    dim3 tpb(WARPS * 32);              // 8 warps, one 16-row Q slice each
    dim3 bpg(planes, seq / BM);        // x -> (batch, head) planes, y -> Q row tiles

    for (int i = 0; i < 50; ++i) {
        flash_attention<<<bpg, tpb, SMEM_BYTES>>>(d_Q, d_Kk, d_V, d_O, seq, seq, d_head); // warm up
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStart());


    flash_attention<<<bpg, tpb, SMEM_BYTES>>>(d_Q, d_Kk, d_V, d_O, seq, seq, d_head);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaProfilerStop());


    CUDA_CHECK(cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Spot-check a spread of rows against the CPU reference. fp16 operands with fp32
    // accumulation over 2048 keys puts the expected relative error around 1e-3.
    float* ref = (float*)malloc(sizeof(float) * d_head);
    float max_rel = 0.0f;
    const int probe_rows[] = {0, 1, 7, 8, 127, 128, 1023, 2047};
    for (int p = 0; p < planes; p += 5) {
        for (int r = 0; r < (int)(sizeof(probe_rows) / sizeof(int)); ++r) {
            const int row = probe_rows[r];
            reference_row(h_Q, h_K, h_V, ref, p, row, seq, d_head);
            for (int c = 0; c < d_head; ++c) {
                const float got = __half2float(h_O[(size_t)p * plane_eles + (size_t)row * d_head + c]);
                const float rel = fabsf(got - ref[c]) / fmaxf(fabsf(ref[c]), 1e-6f);
                max_rel = fmaxf(max_rel, rel);
            }
        }
    }
    printf("max relative error vs reference: %.5f  (%s)\n",
           max_rel, max_rel < 2e-2f ? "PASS" : "FAIL");
    free(ref);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_Kk));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    CUDA_CHECK(cudaFreeHost(h_Q));
    CUDA_CHECK(cudaFreeHost(h_K));
    CUDA_CHECK(cudaFreeHost(h_V));
    CUDA_CHECK(cudaFreeHost(h_O));

    return 0;
}
