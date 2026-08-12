#include "common.cuh"
#include <cuda_profiler_api.h>
#include <cmath>
#include <cfloat>

// Anonymous namespace, not macros: BK is 64 in attention.cu and these names would
// otherwise collide the moment a driver pulls both in.
namespace {
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
}

// Tune these, for max arithmetic intensity on matmul, M & N >> K (K Doesn't contribute).

// A MxK, B KxN.
__global__ void gemm(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, const int M, const int K, const int N) {
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

    uint32_t smem_A_p1 = swizzled_ptr(&sA[0][0][0], tid / 4, (tid / 4) * 8, BK);
    uint32_t smem_A_p2 = swizzled_ptr(&sA[0][0][0], tid / 4 + 64, (tid / 4) * 8, BK);
    cp_async_128(smem_A_p1, &A[load_A_row * K + load_A_col]);
    cp_async_128(smem_A_p2, &A[(load_A_row + 64) * K + load_A_col]);

    uint32_t smem_B_p1 = swizzled_ptr(&sB[0][0][0], tid / 16, (tid % 16) * 8, BN);
    uint32_t smem_B_p2 = swizzled_ptr(&sB[0][0][0], tid / 16 + 16, (tid % 16) * 8, BN);
    cp_async_128(smem_B_p1, &B[load_B_row * N + load_B_col]);
    cp_async_128(smem_B_p2, &B[(load_B_row + 16) * N + load_B_col]);

    cp_async_commit_group(); cp_async_wait_group<0>(); __syncthreads();

    int warp_row = warp_id / 4; int warp_col = warp_id % 4; // where that warp is inside of the 2x4 warp grid.

    int warp_row_offset = warp_row * 64; int warp_col_offset = warp_col * 32;

    for (int tile_k = BK; tile_k < K; tile_k += BK) {
        // preload next set of tiles
        write_stage ^= 1;

        uint32_t smem_A_p1 = swizzled_ptr(&sA[write_stage][0][0], tid / 4, (tid / 4) * 8, BK);
        uint32_t smem_A_p2 = swizzled_ptr(&sA[write_stage][0][0], tid / 4 + 64, (tid / 4) * 8, BK);
        cp_async_128(smem_A_p1, &A[load_A_row * K + load_A_col + tile_k]);
        cp_async_128(smem_A_p2, &A[(load_A_row + 64) * K + load_A_col + tile_k]);

        uint32_t smem_B_p1 = swizzled_ptr(&sB[write_stage][0][0], tid / 16, (tid % 16) * 8, BN);
        uint32_t smem_B_p2 = swizzled_ptr(&sB[write_stage][0][0], tid / 16 + 16, (tid % 16) * 8, BN);
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

                uint32_t smem_read_A = swizzled_ptr(&sA[read_stage][0][0], a_row, a_col, BK);
                ld_matrix_x4(rA, smem_read_A);

                for (int n_dim = 0; n_dim < 4; ++n_dim) {
                    int b_row, b_col;
                    b_row = k_dim * 16 + row_in_group;
                    b_row = (lane_group == 0 || lane_group == 2) ? b_row : b_row + 8;

                    b_col = warp_col_offset + n_dim * 8;

                    uint32_t smem_read_B = swizzled_ptr(&sB[read_stage][0][0], b_row, b_col, BN);
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

            uint32_t smem_read_A = swizzled_ptr(&sA[read_stage][0][0], a_row, a_col, BK);
            ld_matrix_x4(rA, smem_read_A);

            for (int n_dim = 0; n_dim < 4; ++n_dim) {
                int b_row, b_col;
                b_row = k_dim * 16 + row_in_group;
                b_row = (lane_group == 0 || lane_group == 2) ? b_row : b_row + 8;

                b_col = warp_col_offset + n_dim * 8;

                uint32_t smem_read_B = swizzled_ptr(&sB[read_stage][0][0], b_row, b_col, BN);
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

            // Accumulation stays fp32 the whole way down; the narrowing happens once,
            // here, so the next kernel in the chain can consume C as half directly.
            C[global_row * N + global_col] = static_cast<half>(rC[m][n][0]);
            C[global_row * N + global_col + 1] = static_cast<half>(rC[m][n][1]);
            C[(global_row + 8) * N + global_col] = static_cast<half>(rC[m][n][2]);
            C[(global_row + 8) * N + global_col + 1] = static_cast<half>(rC[m][n][3]);
        }
    }


}

#ifndef GPT2_NO_MAIN

int main() {
    // Grab at least one of the dims of the matmuls we're going to use the matmul for.
    // Must be exact multiples of the block tiles (no boundary masking in the kernel):
    //   M % BM(128) == 0,  N % BN(128) == 0,  K % BK(32) == 0
    int M = 4096;
    int N = 4096;
    int K = 1024;

    half *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, sizeof(half) * M * K));
    CUDA_CHECK(cudaMallocHost(&h_B, sizeof(half) * K * N));
    CUDA_CHECK(cudaMallocHost(&h_C, sizeof(half) * M * N));

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(half) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(half) * K * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(half) * M * N));


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


    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(half) * M * N, cudaMemcpyDeviceToHost));

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

#endif // GPT2_NO_MAIN