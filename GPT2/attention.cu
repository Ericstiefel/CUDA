// This file will contain the attention mechanism (Self-Attention).
// After profiling, this kernel is not close to SOTA as it's extremely heavy on both register count
// And smem, despite only double buffering (I've seen triple done in adjacent configurations). 


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>


__device__ __forceinline__ uint32_t cvta_generic_to_shared(void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t swizzled_ptr(const void* smem_ptr, int row, int col, int stride) {
    int eles_per_vec = 16 / sizeof(half);
    int vecs_per_row = stride / eles_per_vec;

    int chunk_idx = col / eles_per_vec;
    int offset = col % eles_per_vec;

    int swizzled_col = chunk_idx ^ ((row % 8) % vecs_per_row);
    int flat_idx = (row * stride) + (col * eles_per_vec) + offset;
    return cvta_generic_to_shared(smem_ptr + flat_idx);
}

__device__ __forceinline__ void cp_async_128(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        "r"(smem_ptr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile ("cp.async.commit_group;\n");
}

template <int N> // availability at runtime
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ld_matrix_x4(const uint32_t smem_ptr, uint32_t* rA) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(rA[0]), "=r"(rA[1]), "=r"(rA[2]), "=r"(rA[3]),
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2(const uint32_t smem_ptr, uint32_t* rB) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rB[0]), "=r"(rB[1]),
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2_trans(const uint32_t smem_ptr, uint32_t* rB) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rB[0]), "=r"(rB[1]),
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void mma(uint32_t* rA, uint32_t* rB, uint32_t* rC) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(rC[0]), "+f"(rC[1]), "+f"(rC[2]), "+f"(rC[3]),
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]), 
        "r"(rB[0]), "r"(rB[1])
    );
}

// Stores 16 bits (2 bytes) as deliberate store instruction with smem ptr
__device__ __forceinline__ void st_shared_b16(uint32_t smem_ptr, half val) {
    unsigned int bits = __half_as_ushort(val);
    asm volatile (
        "st.shared.b16 [%0], %1;\n"
        :: "r"(smem_ptr), "h"(bits)
    );
}

__device__ __forceinline__ void max_reduction(float val) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int i = 2; i > 0; i >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, i, 4));
    }
    return val;
}

__device__ __forceinline__ void sum_reduction(float sum) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int i = 2; i > 0; i >>= 1) {
        sum += __shfl_xor_sync(mask, sum, i, 4);
    }
    return sum;
}



#define BM 128
#define BN 128
#define BK 32


constexpr size_t Q_ELES = BM * BN;
constexpr size_t K_ELES = 2 * BK * BN;
constexpr size_t V_ELES = 2 * BK * BN;
constexpr size_t P_ELES = BM * BN;

constexpr size_t SMEM_BYTES = (Q_ELES + K_ELES + V_ELES + P_ELES) * sizeof(half);


__global__ void flash_attention(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ out, int d_M, int d_K, int d_N) {
    extern __shared__ half smem[];
    half* sQ = smem;
    half* sK = sQ + Q_ELES;
    half* sV = SK + K_ELES;
    half* SP = sV + V_ELES;

    uint32_t rA[4]; uint32_t rB[2]; float acc[64] = {0.0f};

    int tid = threadIdx.x; 
    int warp_id = tid / 32; int warp_lane = tid % 32;
    int q_base_row = warp_id * 16; // each warp handles 16 rows of Q

    float m_row[2] = {-FLT_MAX, -FLT_MAX};
    float l_row[2] = {0.0f, 0.0f};

    float attn_scale = rsqrtf((float)d_N);

    int load_row = tid / 16; int load_col = (tid % 16) * 8;

    // load full Q into SMEM
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t smem_Q_loc = swizzled_ptr(sQ, load_row + i * 16, load_col, BN);
        cp_async_128(smem_Q_loc, &Q[((load_row + i * 16) + BM * blockIdx.y) * d_N + load_col]);
    }
    int write = 0; int read = 0;

    uint32_t dst_k_1 = swizzled_ptr(sK, load_row, load_col, BN); uint32_t dst_k_2 = swizzled_ptr(sK, load_row + 16, load_col, BN);
    uint32_t dst_v_1 = swizzled_ptr(sV, load_row, load_col, BN); uint32_t dst_v_2 = swizzled_ptr(sV, load_row + 16, load_col, BN);

    cp_async_128(dst_k_1, &K[load_row * d_N + load_col]); cp_async_128(dst_k_2, &K[(load_row + 16) * d_N + load_col]);
    cp_async_128(dst_v_1, &V[load_row * d_N + load_col]); cp_async_128(dst_v_2, &V[(load_row + 16) * d_N + load_col]);
    cp_async_commit_group(); cp_async_wait_group<0>(); __syncthreads();

    for (int tile_k = 0; tile_k < d_K; tile_k += BK) {
        int next_tile_k = tile_k + BK; bool has_next = next_tile_k < d_K;
        if (has_next) {
            write ^= 1;
            half* sK_w = sK + write * BK * BN; half* sV_w = sV + write * BK * BN;
            dst_k_1 = swizzled_ptr(sK_w, load_row, load_col, BN); dst_k_2 = swizzled_ptr(sK_w, load_row + 16, load_col, BN);
            dst_v_1 = swizzled_ptr(sV_w, load_row, load_col, BN); dst_v_2 = swizzled_ptr(sV_w, load_row + 16, load_col, BN);
            cp_async_128(dst_k_1, &K[load_row * d_N + load_col + next_tile_k * d_N]); cp_async_128(dst_k_2, &K[(load_row + 16) * d_N + next_tile_k * d_N]);
            cp_async_128(dst_k_2, &K[load_row * d_N + load_col + next_tile_k * d_N]); cp_async_128(dst_v_2, &K[(load_row + 16) * d_N + next_tile_k * d_N]);
            cp_async_commit_group();
        }

        float scores[16] = {0.0f};

        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            #pragma unroll
            for (int k_dim = 0; k_dim < 8; ++k_dim) {
                uint32_t q_addr = swizzled_ptr(sQ, (warp_id * 16) + (warp_lane % 16), k_dim * 16 + (warp_lane / 16) * 8, BN);
                ld_matrix_x4(rA, q_addr);

                uint32_t k_addr = swizzled_ptr(sK + read * BK * BN, n_dim * 8 + (warp_lane % 8), k_dim * 16 + ((warp_lane / 8) % 2) * 8, BN);
                ld_matrix_x2(rB, k_addr);

                mma(rA, rB, &scores[n_dim * 4]);
            }
        }

        // Online Softmax Segment

        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            scores[n_dim * 4] *= attn_scale; scores[n_dim * 4 + 1] *= attn_scale;
            scores[n_dim * 4 + 2] *= attn_scale; scores[n_dim * 4 + 3] *= attn_scale;
        }

        // Simpler computationally to reduce a block of maxes and use that for the exponentiation online update instead of using a bunch more expf (SFU's) than required.

        float local_max[2] = {-FLT_MAX, -FLT_MAX};
        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            local_max[0] = fmaxf(local_max[0], fmaxf(scores[n_dim * 4 + 0], scores[n_dim * 4 + 1]));
            local_max[1] = fmaxf(local_max[1], fmaxf(scores[n_dim * 4 + 2], scores[n_dim * 4 + 3]));
        }

        local_max[0] = frag_row_max(local_max[0]);
        local_max[1] = frag_row_max(local_max[1]);

        float new_maxes[2] = {fmaxf(m_row[0], local_max[0]), fmaxf(m_row[1], local_max[1])};

        float corrections[2] = {expf(m_row[0] - new_maxes[0]), expf(m_row[1], new_maxes[1])};
        m_row[0] = new_maxes[0]; m_row[1] = new_maxes[1];

        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j * 4] = corrections[0]; acc[j * 4 + 1] = corrections[0];
            acc[j * 4 + 2] = corrections[1]; acc[j * 4 + 3] = corrections[1];
        }

        float P[16];

        float tile_sum[2] = { 0.0f, 0.0f};
        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
           P[n_dim * 4] = expf(scores[n_dim * 4] - m_row[0]);
           P[n_dim * 4 + 1] = expf(scores[n_dim * 4 + 1] - m_row[0]);
           P[n_dim * 4 + 2] = expf(scores[n_dim * 4 + 2] - m_row[1]);
           P[n_dim * 4 + 3] = expf(scores[n_dim * 4 + 3] - m_row[1]);

           tile_sum[0] += P[n_dim * 4] + P[n_dim * 4 + 1];
           tile_sum[1] += P[n_dim * 4 + 2] + P[n_dim * 4 + 3];

        }

        l_row[0] = l_row[0] * corrections[0] + sum_reduction(tile_sum[0]);
        l_row[1] = l_row[1] * corrections[1] + sum_reduction(tile_sum[1]);

        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            uint32_t p_addr0 = swizzled_ptr(sP, warp_id * 16 + tid / 4, n_dim * 8 + (tid % 4) * 2, BK);
            st_shared_b16(p_addr0, __float2half(P[n_dim * 4]));
            uint32_t p_addr1 = swizzled_ptr(sP, warp_id * 16 + tid / 4, n_dim * 8 + (tid % 4) * 2 + 1, BK);
            st_shared_b16(p_addr1, __float2half(P[n_dim * 4 + 1]));
            uint32_t p_addr2 = swizzled_ptr(sP, warp_id * 16 + tid / 4, n_dim * 8 + (tid % 4) * 2 + 2, BK);
            st_shared_b16(p_addr2, __float2half(P[n_dim * 4 + 2]));
            uint32_t p_addr3 = swizzled_ptr(sP, warp_id * 16 + tid / 4, n_dim * 8 + (tid % 4) * 2 + 3, BK);
            st_shared_b16(p_addr3, __float2half(P[n_dim * 4 + 3]));
        }
        __syncwarp();

        #pragma unroll
        for (int n_dim2 = 0; n_dim2 < 16; ++n_dim2) {
            #pragma unroll
            for (int k_dim2 = 0; k_dim2 < 2; ++k_dim2) {
                uint32_t p_addr = swizzled_ptr(sP, warp_id * 16 + warp_lane % 16, k_dim2 * 16 + (warp_lane / 16) * 8, BK);
                ld_matrix_x4(rA, p_addr);

                uint32_t v_addr = swizzled_ptr(sV + BK * BN * read, k_dim2 * 16 + (warp_lane % 16), n_dim2 * 8, BN);
                ld_matrix_x2(rB, v_addr);

                mma(rA, rB, &acc[n_dim2 * 4]);
            }
        }

        if (has_next) {
            read ^= 1; cp_async_wait_group<0>(); __syncthreads();
        }
    }

    int out_row = BM * blockIdx.y + (warp_id  * 16) + warp_lane / 4;
    
    #pragma unroll
    for (int n_dim2 = 0; n_dim2 < 16; ++n_dim2) {
        acc[n_dim2 * 4] /= l_row[0]; acc[n_dim2 * 4 + 1] /= l_row[0];
        acc[n_dim2 * 4 + 2] /= l_row[1]; acc[n_dim2 * 4 + 3] /= l_row[1];

        int out_col = n_dim2 * 8 + (tid % 4) * 2;

        out[out_row * d_N + out_col] = __float2half(acc[n_dim2 * 4]);
        out[out_row * d_N + out_col + 1] = __float2half(acc[n_dim2 * 4 + 1]);
        out[(out_row + 8) * d_N + out_col] = __float2half(acc[n_dim2 * 4 + 2]);
        out[(out_row + 8) * d_N + out_col + 1] = __float2half(acc[n_dim2 * 4 + 3]);
    }
    

