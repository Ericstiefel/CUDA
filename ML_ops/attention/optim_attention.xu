#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __forceinline__ uint32_t gts(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}


__device__ __forceinline__ uint32_t sw_ptr(const half* base_ptr, int row, int col, int stride) {
    int vals_per_vec = 16 / sizeof(half); // 8 elements per 16 bytes
    int chunks_per_row = stride / vals_per_vec;
    int chunk_idx = col / vals_per_vec;
    int offset = col % vals_per_vec;

    int swizzled_col = chunk_idx ^ ((row % 8) % chunks_per_row);
    int flat_idx = (row * stride) + (swizzled_col * vals_per_vec) + offset;

    return gts(base_ptr + flat_idx);
}

__device__ __forceinline__ void cp_async_128(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile (
        "cp.async.commit_group;\n"
    );
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile (
        "cp.async.wait_group %0;\n"
        ::"n"(N)
    );
}

__device__ __forceinline__ void ld_matrix_x4(uint32_t* regs, const uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_ptr) 
    );
}


__device__ __forceinline__ void ld_matrix_x2(uint32_t* regs, const uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2_trans(uint32_t* regs, const uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void st_shared_b16(uint32_t smem_ptr, half val) {
    unsigned short bits = __half_as_ushort(val);
    asm volatile (
        "st.shared.b16 [%0], %1;\n"
        :: "r"(smem_ptr), "h"(bits)
    );
}

__device__ __forceinline__ void mma(const uint32_t* rA, const uint32_t* rB, float* rC) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(rC[0]), "+f"(rC[1]), "+f"(rC[2]), "+f"(rC[3])
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]),
        "r"(rB[0]), "r"(rB[1])   
    );
}

template <int WIDTH>
__device__ __forceinline__ void softmax_warp_call(float& l_max, float& l_sum) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset /= 2) {
        float neighbor_max = __shfl_down_sync(mask, l_max, offset, WIDTH);
        float neighbor_sum = __shfl_down_sync(mask, l_sum, offset, WIDTH);

        float joint_max = fmaxf(l_max, neighbor_max);
        l_sum = l_sum * expf(l_max - joint_max) + neighbor_sum * expf(neighbor_max - joint_max);
        l_max = joint_max;
    }
}

#define BM 128
#define BN 128
#define BK 32

// Per-region element counts (in halfs), hardcoded separately so each buffer's size is explicit:
//   Q: BM x BN, single-buffered (loaded once, reused for every K/V tile).
//   K, V: BK x BN, DOUBLE-buffered (one half being read while the other prefetches the next tile).
//   P: BM x BK, scratch to round-trip the softmax output through smem for the P@V mma's A-load.
constexpr size_t Q_ELEMS = BM * BN;
constexpr size_t K_ELEMS = 2 * BK * BN;
constexpr size_t V_ELEMS = 2 * BK * BN;
constexpr size_t P_ELEMS = BM * BK;

// This GPU (RTX A4500, Ampere/sm_86) has 100KB of shared memory per SM, and the 72KB below
// fits easily -- BUT plain `__shared__` (static) arrays are capped at 48KB on every current CUDA
// architecture regardless of the hardware's real per-SM capacity, so we just dynamically allocate. 
constexpr size_t SMEM_BYTES = (Q_ELEMS + K_ELEMS + V_ELEMS + P_ELEMS) * sizeof(half);

__global__ void flash_attention(const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V, half* __restrict__ out, int d_M, int d_K, int d_N) {
    extern __shared__ half smem[];
    half* sQ = smem;
    half* sK = sQ + Q_ELEMS;
    half* sV = sK + K_ELEMS;
    half* sP = sV + V_ELEMS; 

    // Output is of size 128 x 128 per block, 128 * 128 / 256 = 64 output registers per thread.
    uint32_t rA[4]; uint32_t rB[2]; float acc[64] = {0.0f};

    int tid = threadIdx.x; int bx = blockIdx.x; int by = blockIdx.y;
    int warp_id = tid / 32; int warp_lane = tid % 32;
    int q_row_base = warp_id * 16; // this warp's 16-row band of Q, reused for every K-tile
    int tig = warp_lane % 4; int groupID = warp_lane / 4; // mma C-fragment coords: row = groupID / groupID+8

    float m_row[2] = { -FLT_MAX, -FLT_MAX };
    float l_row[2] = { 0.0f, 0.0f };
    float attn_scale = rsqrtf((float)d_N);

    // Load full 128 x 128 tile of Q into smem
    int load_Q_row = tid / 16; int load_Q_col = (tid % 16) * 8;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // Each warp can load a 2x128 (16 bytes per load), so block will load 16 x 128 (add 16 to row on each call)
        uint32_t smem_Q_location = sw_ptr(sQ, load_Q_row + i * 16, load_Q_col, BN);
        cp_async_128(smem_Q_location, &Q[(load_Q_row + BM * by + i * 16 ) * d_N + load_Q_col]);
    }

    int write = 0; int read = 0;

    /*
    K Loading:
        Of size 32 x 128, need 2 loads per thread, at that index and add 16 rows and load again. We will transpose this fragment before the matmul portion, load in row-major
        Indexing:
            load_k_row = tid / 16
            load_k_col = bx * BN + (tid % 16) * 8

            load this and load_k_row + 16

    V Loading:
        of same size, exact same loading conventions. 
    */


    int load_row = tid / 16; int load_col = (tid % 16) * 8;
    uint32_t dst_k_1 = sw_ptr(sK, tid / 16, (tid % 16) * 8, BN); uint32_t dst_k_2 = sw_ptr(sK, tid / 16 + 16, (tid % 16) * 8, BN);
    uint32_t dst_v_1 = sw_ptr(sV, tid / 16, (tid % 16) * 8, BN); uint32_t dst_v_2 = sw_ptr(sV, tid / 16 + 16, (tid % 16) * 8, BN);

    cp_async_128(dst_k_1, &K[load_row * d_N + load_col]); cp_async_128(dst_k_2, &K[(load_row + 16) * d_N + load_col]);
    cp_async_128(dst_v_1, &V[load_row * d_N + load_col]); cp_async_128(dst_v_2, &V[(load_row + 16) * d_N + load_col]);
    cp_async_commit_group(); cp_async_wait_group<0>(); __syncthreads();

    int lane_group = tid % 8; int row_in_group = tid / 8;


    for (int tile_k = 0; tile_k < d_K; tile_k += BK) {
        int next_tile_k = tile_k + BK;
        bool has_next = next_tile_k < d_K;

        if (has_next) {
            write ^= 1;

            // Async load next K & V into smem
            half* sK_w = sK + write * BK * BN; half* sV_w = sV + write * BK * BN;
            dst_k_1 = sw_ptr(sK_w, tid / 16, (tid % 16) * 8, BN); dst_k_2 = sw_ptr(sK_w, tid / 16 + 16, (tid % 16) * 8, BN);
            dst_v_1 = sw_ptr(sV_w, tid / 16, (tid % 16) * 8, BN); dst_v_2 = sw_ptr(sV_w, tid / 16 + 16, (tid % 16) * 8, BN);

            cp_async_128(dst_k_1, &K[load_row * d_N + load_col + next_tile_k * d_N]); cp_async_128(dst_k_2, &K[(load_row + 16) * d_N + load_col + next_tile_k * d_N]);
            cp_async_128(dst_v_1, &V[load_row * d_N + load_col + next_tile_k * d_N]); cp_async_128(dst_v_2, &V[(load_row + 16) * d_N + load_col + next_tile_k * d_N]);

            cp_async_commit_group();
        }

        float scores[16] = {0.0f}; // 4 n-chunks (8 cols each) x 4 mma output regs

        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            #pragma unroll
            for (int k_dim = 0; k_dim < 8; ++k_dim) {
                uint32_t q_addr = sw_ptr(sQ,
                    q_row_base + (warp_lane % 16),
                    k_dim * 16 + (warp_lane / 16) * 8,
                    BN);
                ld_matrix_x4(rA, q_addr);

                uint32_t k_addr = sw_ptr(sK + read * BK * BN,
                    n_dim * 8 + (warp_lane % 8),
                    k_dim * 16 + ((warp_lane / 8) % 2) * 8,
                    BN);
                ld_matrix_x2(rB, k_addr);

                mma(rA, rB, &scores[n_dim * 4]);
            }
        }

        // Online Softmax
        //
        // scores[n_dim*4 + i] follows the mma C-fragment layout: for i in {0,1} the element
        // belongs to local row `groupID`, for i in {2,3} it belongs to local row `groupID+8`.
        // Each lane holds 8 of the 32 BK columns per row; the other 24 live in the other 3
        // lanes that share this lane's groupID (tig = 0..3) -- so the reduction has to be
        // 4-wide, not 32-wide.

        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            scores[n_dim * 4 + 0] *= attn_scale; scores[n_dim * 4 + 1] *= attn_scale;
            scores[n_dim * 4 + 2] *= attn_scale; scores[n_dim * 4 + 3] *= attn_scale;
        }

        float local_max[2] = { -FLT_MAX, -FLT_MAX };
        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            local_max[0] = fmaxf(local_max[0], fmaxf(scores[n_dim * 4 + 0], scores[n_dim * 4 + 1]));
            local_max[1] = fmaxf(local_max[1], fmaxf(scores[n_dim * 4 + 2], scores[n_dim * 4 + 3]));
        }

        float local_sum[2] = { 0.0f, 0.0f };
        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            local_sum[0] += expf(scores[n_dim * 4 + 0] - local_max[0]) + expf(scores[n_dim * 4 + 1] - local_max[0]);
            local_sum[1] += expf(scores[n_dim * 4 + 2] - local_max[1]) + expf(scores[n_dim * 4 + 3] - local_max[1]);
        }

        // Reduce each row's (max, sum) across its 4-lane group, then broadcast the result
        // (shfl_down only leaves the correct answer in the group's lane 0) back to all 4 lanes.
        softmax_warp_call<4>(local_max[0], local_sum[0]);
        softmax_warp_call<4>(local_max[1], local_sum[1]);
        float tile_max0 = __shfl_sync(0xffffffff, local_max[0], 0, 4);
        float tile_sum0 = __shfl_sync(0xffffffff, local_sum[0], 0, 4);
        float tile_max1 = __shfl_sync(0xffffffff, local_max[1], 0, 4);
        float tile_sum1 = __shfl_sync(0xffffffff, local_sum[1], 0, 4);

        float new_max0 = fmaxf(m_row[0], tile_max0);
        float new_max1 = fmaxf(m_row[1], tile_max1);
        float correction0 = expf(m_row[0] - new_max0);
        float correction1 = expf(m_row[1] - new_max1);

        l_row[0] = l_row[0] * correction0 + tile_sum0 * expf(tile_max0 - new_max0);
        l_row[1] = l_row[1] * correction1 + tile_sum1 * expf(tile_max1 - new_max1);
        m_row[0] = new_max0;
        m_row[1] = new_max1;

        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc[j * 4 + 0] *= correction0; acc[j * 4 + 1] *= correction0;
            acc[j * 4 + 2] *= correction1; acc[j * 4 + 3] *= correction1;
        }

        half P[16];
        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            P[n_dim * 4 + 0] = __float2half(expf(scores[n_dim * 4 + 0] - m_row[0]));
            P[n_dim * 4 + 1] = __float2half(expf(scores[n_dim * 4 + 1] - m_row[0]));
            P[n_dim * 4 + 2] = __float2half(expf(scores[n_dim * 4 + 2] - m_row[1]));
            P[n_dim * 4 + 3] = __float2half(expf(scores[n_dim * 4 + 3] - m_row[1]));
        }

        #pragma unroll
        for (int n_dim = 0; n_dim < 4; ++n_dim) {
            uint32_t p_addr0 = sw_ptr(sP, q_row_base + groupID, n_dim * 8 + tig * 2, BK);
            st_shared_b16(p_addr0, P[n_dim * 4 + 0]);
            uint32_t p_addr1 = sw_ptr(sP, q_row_base + groupID, n_dim * 8 + tig * 2 + 1, BK);
            st_shared_b16(p_addr1, P[n_dim * 4 + 1]);
            uint32_t p_addr2 = sw_ptr(sP, q_row_base + groupID + 8, n_dim * 8 + tig * 2, BK);
            st_shared_b16(p_addr2, P[n_dim * 4 + 2]);
            uint32_t p_addr3 = sw_ptr(sP, q_row_base + groupID + 8, n_dim * 8 + tig * 2 + 1, BK);
            st_shared_b16(p_addr3, P[n_dim * 4 + 3]);
        }
        __syncwarp();

        #pragma unroll
        for (int n_dim2 = 0; n_dim2 < 16; ++n_dim2) {
            #pragma unroll
            for (int k_dim2 = 0; k_dim2 < 2; ++k_dim2) {
                uint32_t p_addr = sw_ptr(sP,
                    q_row_base + (warp_lane % 16),
                    k_dim2 * 16 + (warp_lane / 16) * 8,
                    BK);
                ld_matrix_x4(rA, p_addr);

                uint32_t v_addr = sw_ptr(sV + read * BK * BN,
                    k_dim2 * 16 + (warp_lane % 16),
                    n_dim2 * 8,
                    BN);
                ld_matrix_x2_trans(rB, v_addr);

                mma(rA, rB, &acc[n_dim2 * 4]);
            }
        }

        if (has_next) {
            read ^= 1; cp_async_wait_group<0>(); __syncthreads();
        }
    }

    int out_row0 = BM * by + q_row_base + groupID;
    int out_row1 = out_row0 + 8;

    #pragma unroll
    for (int n_dim2 = 0; n_dim2 < 16; ++n_dim2) {
        acc[n_dim2 * 4 + 0] /= l_row[0]; acc[n_dim2 * 4 + 1] /= l_row[0];
        acc[n_dim2 * 4 + 2] /= l_row[1]; acc[n_dim2 * 4 + 3] /= l_row[1];

        int out_col0 = n_dim2 * 8 + tig * 2;
        int out_col1 = out_col0 + 1;

        out[out_row0 * d_N + out_col0] = __float2half(acc[n_dim2 * 4 + 0]);
        out[out_row0 * d_N + out_col1] = __float2half(acc[n_dim2 * 4 + 1]);
        out[out_row1 * d_N + out_col0] = __float2half(acc[n_dim2 * 4 + 2]);
        out[out_row1 * d_N + out_col1] = __float2half(acc[n_dim2 * 4 + 3]);
    }
}

// FOLLOWING KERNELS ARE PLACED HERE TO CHECK THE RESULTS


__global__ void basic_gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, const int M, const int K, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = sum;
}




__global__ void online_kernel(const float* __restrict__ inp, half* __restrict__ out, const int N) {
    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];

    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int warp_lane = tid % 32;

    int num_warps = blockDim.x / 32;

    const float4* row_in_vec = reinterpret_cast<const float4*>(inp + row_idx * N);
    half* row_out = out + row_idx * N;

    int N_vec = N / 4;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = tid; i < N_vec; i += blockDim.x) {
        float4 vec = row_in_vec[i];
        float vals[4] = {vec.x, vec.y, vec.z, vec.w};

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float x_i = vals[j];
            float new_max = fmaxf(local_max, x_i);
            local_sum = local_sum * expf(local_max - new_max) + expf(x_i - new_max);
            local_max = new_max;
        }
    }

    softmax_warp_call<32>(local_max, local_sum);

    if (warp_lane == 0) {
        warp_max[warp_id] = local_max;
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();
    float block_max = (tid < num_warps) ? warp_max[tid] : -FLT_MAX;
    float block_sum = (tid < num_warps) ? warp_sum[tid] : 0.0f;

    if (warp_id == 0) { softmax_warp_call<32>(block_max, block_sum); }

    if (tid  == 0) {
        warp_max[0] = block_max;
        warp_sum[0] = block_sum;
    }
    __syncthreads();

    float global_max = warp_max[0];
    float global_sum = warp_sum[0];


    for (int i = tid; i < N_vec; i += blockDim.x) {
        float4 vec = row_in_vec[i];

        row_out[i * 4 + 0] = __float2half(expf(vec.x - global_max) / global_sum);
        row_out[i * 4 + 1] = __float2half(expf(vec.y - global_max) / global_sum);
        row_out[i * 4 + 2] = __float2half(expf(vec.z - global_max) / global_sum);
        row_out[i * 4 + 3] = __float2half(expf(vec.w - global_max) / global_sum);
    }
}


__global__ void check_identical(const half* __restrict__ C1, const float* __restrict__ C2, int M, int N, int* correct) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int idx = row * N + col;
    int val = fabsf(__half2float(C1[idx]) - C2[idx]) < 1e-3f ? 1 : 0;
    atomicAdd(correct, val);
}

__global__ void scale_kernel(float* __restrict__ data, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= s;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

int main() {
    const int M = BM, Kseq = 4 * BK, Nh = BN;
    srand(42);

    std::vector<half> hQ(M * Nh), hK(Kseq * Nh), hV(Kseq * Nh), hKT(Nh * Kseq);
    for (int i = 0; i < M * Nh; ++i) hQ[i] = __float2half((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
    for (int i = 0; i < Kseq * Nh; ++i) hK[i] = __float2half((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
    for (int i = 0; i < Kseq * Nh; ++i) hV[i] = __float2half((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
    for (int s = 0; s < Kseq; ++s)
        for (int d = 0; d < Nh; ++d)
            hKT[d * Kseq + s] = hK[s * Nh + d]; // K^T, needed because basic_gemm has no transpose support

    half *dQ, *dK, *dV, *dKT, *dOut, *dPh;
    float *dS, *dOref;
    int *dCorrect;

    CUDA_CHECK(cudaMalloc(&dQ, M * Nh * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dK, Kseq * Nh * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dV, Kseq * Nh * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dKT, Nh * Kseq * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOut, M * Nh * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dS, M * Kseq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPh, M * Kseq * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOref, M * Nh * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCorrect, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), M * Nh * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), Kseq * Nh * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), Kseq * Nh * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKT, hKT.data(), Nh * Kseq * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dCorrect, 0, sizeof(int)));

    CUDA_CHECK(cudaFuncSetAttribute(flash_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));

    dim3 flashBlock(256);
    dim3 flashGrid(1, M / BM);
    dim3 gemmBlock(16, 16);
    dim3 gemmGridS((Kseq + 15) / 16, (M + 15) / 16);
    dim3 gemmGridO((Nh + 15) / 16, (M + 15) / 16);
    float attn_scale = 1.0f / sqrtf((float)Nh);
    int totalS = M * Kseq;

    // ---- timing: our kernel vs the reference path, event-clocked over several iterations.
    // Inputs are static across iterations, so the buffers after the loops hold the same result
    // a single run would -- the looping is purely to average out launch-to-launch noise. This
    // matters more at the matrix sizes this is meant to scale to than at M=Kseq=Nh=128. ----
    const int numIters = 100;

    flash_attention<<<flashGrid, flashBlock, SMEM_BYTES>>>(dQ, dK, dV, dOut, M, Kseq, Nh); // warmup
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t startEvt, stopEvt;
    CUDA_CHECK(cudaEventCreate(&startEvt));
    CUDA_CHECK(cudaEventCreate(&stopEvt));

    CUDA_CHECK(cudaEventRecord(startEvt));
    for (int it = 0; it < numIters; ++it) {
        flash_attention<<<flashGrid, flashBlock, SMEM_BYTES>>>(dQ, dK, dV, dOut, M, Kseq, Nh);
    }
    CUDA_CHECK(cudaEventRecord(stopEvt));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stopEvt));
    float oursMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&oursMs, startEvt, stopEvt));
    oursMs /= numIters;

    // warmup
    basic_gemm<<<gemmGridS, gemmBlock>>>(dQ, dKT, dS, M, Nh, Kseq);
    scale_kernel<<<(totalS + 255) / 256, 256>>>(dS, attn_scale, totalS);
    online_kernel<<<M, 32>>>(dS, dPh, Kseq);
    basic_gemm<<<gemmGridO, gemmBlock>>>(dPh, dV, dOref, M, Kseq, Nh);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(startEvt));
    for (int it = 0; it < numIters; ++it) {
        basic_gemm<<<gemmGridS, gemmBlock>>>(dQ, dKT, dS, M, Nh, Kseq);
        scale_kernel<<<(totalS + 255) / 256, 256>>>(dS, attn_scale, totalS);
        online_kernel<<<M, 32>>>(dS, dPh, Kseq);
        basic_gemm<<<gemmGridO, gemmBlock>>>(dPh, dV, dOref, M, Kseq, Nh);
    }
    CUDA_CHECK(cudaEventRecord(stopEvt));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stopEvt));
    float refMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&refMs, startEvt, stopEvt));
    refMs /= numIters;

    CUDA_CHECK(cudaEventDestroy(startEvt));
    CUDA_CHECK(cudaEventDestroy(stopEvt));

    int totalO = M * Nh;
    dim3 cmpGrid((Nh + 15) / 16, (M + 15) / 16);
    check_identical<<<cmpGrid, gemmBlock>>>(dOut, dOref, M, Nh, dCorrect);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int correct = 0;
    CUDA_CHECK(cudaMemcpy(&correct, dCorrect, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<half> hOut(totalO);
    std::vector<float> hOref(totalO);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, totalO * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hOref.data(), dOref, totalO * sizeof(float), cudaMemcpyDeviceToHost));

    float maxAbsErr = 0.0f, sumAbsErr = 0.0f;
    for (int i = 0; i < totalO; ++i) {
        float e = fabsf(__half2float(hOut[i]) - hOref[i]);
        maxAbsErr = fmaxf(maxAbsErr, e);
        sumAbsErr += e;
    }

    printf("M=%d Kseq=%d Nh=%d\n", M, Kseq, Nh);
    printf("check_identical (tol 1e-3): %d / %d matched (%.2f%%)\n", correct, totalO, 100.0f * correct / totalO);
    printf("max abs error: %f, mean abs error: %f\n", maxAbsErr, sumAbsErr / totalO);

    printf("flash_attention: %.4f ms/iter (avg over %d iters)\n", oursMs, numIters);
    printf("reference path:  %.4f ms/iter (avg over %d iters)\n", refMs, numIters);
    if (oursMs < refMs)
        printf("flash_attention is %.2fx faster than the reference path\n", refMs / oursMs);
    else
        printf("reference path is %.2fx faster than flash_attention\n", oursMs / refMs);

    return 0;
}
