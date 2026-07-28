#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t swizzle_addr(const half* base_smem, int row, int col, int stride) {
    int vals_per_vec = 16 / sizeof(half); // 8 2 byte half values fit inside 1 transfer 16 byte load
    int vecs_per_row = stride / vals_per_vec;

    int chunk_idx = col / vals_per_vec;
    int offset = col % vals_per_vec;

    int swizzled_col = chunk_idx ^ ((row % 8) % vecs_per_row);
    int flat_idx = row * stride + swizzled_col * vals_per_vec + offset;
    return smem_ptr_to_uint(base_smem + flat_idx);
}


__device__ __forceinline__ void cp_async_128(uint32_t smem_addr, const void* gmem_ptr) { // gmem -> smem async
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n" 
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile (
        "cp.async.commit_group;\n"
    );
}

template <int N> // must be available at runtime
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n"
    :: "n"(N) );
}

// b16 for 16 bits per element
__device__ __forceinline__ void ld_matrix_x4(uint32_t regs[4], uint32_t smem_addr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_addr) 
    );
}

__device__ __forceinline__ void ld_matrix_x2(uint32_t regs[2], uint32_t smem_addr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void mma(const uint32_t rA[4], const uint32_t rB[2], float rC[4]) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"

        :"+f"(rC[0]), "+f"(rC[1]), "+f"(rC[2]), "+f"(rC[3])
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]),
        "r"(rB[0]), "r"(rB[1])
    );
}

#define BM 128
#define BN 128 
#define BK 32

// Tune these, for max arithmetic intensity on matmul, M & N >> K (K Doesn't contribute).

// A MxK, B KxN.
__global__ void gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, const int M, const int K, const int N) {
    __shared__ half sA[2][BM][BK];
    __shared__ half sB[2][BK][BN];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // A has width 32, so we're loading 8 eles at a time, 32 / 8 = 4.
    int load_A_row = blockIdx.y * BM + (tid / 4);
    int load_A_col = (tid % 4) * 8;
    
    // B tile has width 128, so we use 16 here
    int load_B_row = tid / 16;
    int load_B_col = blockIdx.x * BN + (tid % 16) * 8;

    uint32_t rA[4]; uint32_t rB[2];
    float rC[4][4][4] = {0.0f};

    int write_stage = 0; int read_stage = 0;

    uint32_t smem_A_p1 = swizzle_addr(&sA[0][0][0], tid / 4, (tid / 4) * 8, BK);
    uint32_t smem_A_p2 = swizzle_addr(&sA[0][0][0], tid / 4 + 64, (tid / 4) * 8, BK);
    cp_async_128(smem_A_p1, &A[load_A_row * K + load_A_col]);
    cp_async_128(smem_A_p2, &A[(load_A_row + 64) * K + load_A_col]);

    uint32_t smem_B_p1 = swizzle_addr(&sB[0][0][0], tid / 16, (tid % 16) * 8, BN);
    uint32_t smem_B_p2 = swizzle_addr(&sB[0][0][0], tid / 16 + 16, (tid % 16) * 8, BN);
    cp_async_128(smem_B_p1, &B[load_B_row * N + load_B_col]);
    cp_async_128(smem_B_p2, &B[(load_B_row + 16) * N + load_B_col]);

    cp_async_commit_group(); cp_async_wait_group<0>(); __syncthreads();

    int warp_row = warp_id / 4; int warp_col = warp_id % 4; // where that warp is inside of the 2x4 warp grid.

    int warp_row_offset = warp_row * 64; int warp_col_offset = warp_col * 32;

    for (int tile_k = BK; tile_k < K; tile_k += BK) {
        // preload next set of tiles
        write_stage ^= 1;

        uint32_t smem_A_p1 = swizzle_addr(&sA[write_stage][0][0], tid / 4, (tid / 4) * 8, BK);
        uint32_t smem_A_p2 = swizzle_addr(&sA[write_stage][0][0], tid / 4 + 64, (tid / 4) * 8, BK);
        cp_async_128(smem_A_p1, &A[load_A_row * K + load_A_col + tile_k]);
        cp_async_128(smem_A_p2, &A[(load_A_row + 64) * K + load_A_col + tile_k]);

        uint32_t smem_B_p1 = swizzle_addr(&sB[write_stage][0][0], tid / 16, (tid % 16) * 8, BN);
        uint32_t smem_B_p2 = swizzle_addr(&sB[write_stage][0][0], tid / 16 + 16, (tid % 16) * 8, BN);
        cp_async_128(smem_B_p1, &B[load_B_row * N + load_B_col + tile_k * N]);
        cp_async_128(smem_B_p2, &B[(load_B_row + 16) * N + load_B_col + tile_k * N]);

        cp_async_commit_group(); 

        for (int k_dim = 0; k_dim < 2; ++k_dim) {
            for (int m_dim = 0; m_dim < 4; ++m_dim) {
                int lane_group = lane / 8; 
                int row_in_group = lane % 8;
                

                // tensor core ldmatrix register pattern has it as 4 8x8 tiles
                int a_row, a_col;
                a_row = warp_row_offset + m_dim * 16 + row_in_group;
                a_row = (lane_group == 0 || lane_group == 2) ? a_row : a_row + 8;

                a_col = (lane_group == 0 || lane_group == 1) ? k_dim * 16 : k_dim * 16 + 8;

                uint32_t smem_read_A = swizzle_addr(&sA[read_stage][0][0], a_row, a_col, BK);
                ld_matrix_x4(rA, smem_read_A);

                for (int n_dim = 0; n_dim < 4; ++n_dim) {
                    int b_row, b_col;
                    b_row = k_dim * 16 + row_in_group;
                    b_row = (lane_group == 0 || lane_group == 2) ? b_row : b_row + 8;

                    b_col = warp_col_offset + n_dim * 8;

                    uint32_t smem_read_B = swizzle_addr(&sB[read_stage][0][0], b_row, b_col, BN);
                    ld_matrix_x2(rB, smem_read_B);
                    mma(rA, rB, rC[m_dim][n_dim]);
                }
            }
        }

        read_stage ^= 1;
        cp_async_wait_group<1>();
        __syncthreads();

    }

    // last tile
    for (int k_dim = 0; k_dim < 2; ++k_dim) {
        for (int m_dim = 0; m_dim < 4; ++m_dim) {
            int lane_group = lane / 8; 
            int row_in_group = lane % 8;
            

            // tensor core ldmatrix register pattern has it as 4 8x8 tiles
            int a_row, a_col;
            a_row = warp_row_offset + m_dim * 16 + row_in_group;
            a_row = (lane_group == 0 || lane_group == 2) ? a_row : a_row + 8;

            a_col = (lane_group == 0 || lane_group == 1) ? k_dim * 16 : k_dim * 16 + 8;

            uint32_t smem_read_A = swizzle_addr(&sA[read_stage][0][0], a_row, a_col, BK);
            ld_matrix_x4(rA, smem_read_A);

            for (int n_dim = 0; n_dim < 4; ++n_dim) {
                int b_row, b_col;
                b_row = k_dim * 16 + row_in_group;
                b_row = (lane_group == 0 || lane_group == 2) ? b_row : b_row + 8;

                b_col = warp_col_offset + n_dim * 8;

                uint32_t smem_read_B = swizzle_addr(&sB[read_stage][0][0], b_row, b_col, BN);
                ld_matrix_x2(rB, smem_read_B);
                mma(rA, rB, rC[m_dim][n_dim]);
            }
        }
    }

    // output is 16 x 8 per tile, each thread has 2 consecutive items in each of its 2 output rowws.
    int frag_row = lane / 4;
    int frag_col = (lane % 4) * 2;

    int warp_global_row = blockIdx.y * BM + warp_row_offset;
    int warp_global_col = blockIdx.x * BN + warp_col_offset;

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int block_global_row = warp_global_row + (m * 16);
            int block_global_col = warp_global_col + (n * 8);

            int global_row = block_global_row + frag_row;
            int global_col = block_global_col + frag_col;

            C[global_row * N + global_col] = rC[m][n][0];
            C[global_row * N + global_col + 1] = rC[m][n][1];
            C[(global_row + 8) * N + global_col] = rC[m][n][2];
            C[(global_row + 8) * N + global_col + 1] = rC[m][n][3];
        }
    }


}

#define CUDA_CHECK(call) do{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)



int main() {
    // Grab at least one of the dims of the matmuls we're going to use the matmul for.
    // Must be exact multiples of the block tiles (no boundary masking in the kernel):
    //   M % BM(128) == 0,  N % BN(128) == 0,  K % BK(32) == 0
    int M = 4096;
    int N = 4096;
    int K = 1024;

    half *h_A, *h_B; float* h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, sizeof(half) * M * K));
    CUDA_CHECK(cudaMallocHost(&h_B, sizeof(half) * K * N));
    CUDA_CHECK(cudaMallocHost(&h_C, sizeof(float) * M * N));

    half *d_A, *d_B; float* d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(half) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(half) * K * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M * N));


    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < K; ++n) {
            h_A[m * K + n] = static_cast<half>(((m + 1 + n * 2) % 17) / 19.0f);
        }
    }

    for (int m = 0; m < K; ++m) {
        for (int n = 0; n < N; ++n) {
            h_B[m * N + n] = static_cast<half>(((m + 1 + n * 2) % 17) / 21.0f);
        }
    }


    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(half) * K * N, cudaMemcpyHostToDevice));

    dim3 tpb(256);                    // 8 warps (2 x 4 warp grid) -> fixed by the kernel
    dim3 bpg(N / BN, M / BM);         // x -> N tiles, y -> M tiles

    for (int i = 0; i < 50; ++i) {
        gemm<<<bpg, tpb>>>(d_A, d_B, d_C, M, K, N); // warming up  (grid, block)
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStart());


    gemm<<<bpg, tpb>>>(d_A, d_B, d_C, M, K, N);   // grid, block
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaProfilerStop());


    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    



    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}