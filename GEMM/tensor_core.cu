#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;


/*
"Simultaneous" Tensor Core pipeline: 
    (1) Global Memory Tile N transfer to the dedicated Write Shared Memory buffer at the time 
        - ~500 clock cycles per access (if coalesced, accesses / 8)
    (2) Transfer Tile N - 1 from dedicated Read Shared Memory buffer to Tensor Core Registers (max 16x16 fp32)
        - ~5 clock cycles per access (if coalesced, accesses / 8)
    (3) Compute Tile N - 2 already placed in fragments / tensor core registers
        - ~15 clock cycle per tile using the power of tensor cores and ~1 clock cycle register access

    (4) Store result of tile N - 3 in C

*/
#define TILE_WIDTH 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
__global__ void tc_gemm(const half* A, const half* B, float* C, const int M, const int K, const int N) {
    // Identify the warp and its position in the thread block.
    // In this simple example, we assume one warp per block for clarity.
    // A more complex implementation would use threadIdx to map threads to the warp.
    int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) / WMMA_M;
    int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / WMMA_N;


    __shared__ half A_tile[2][WMMA_M * WMMA_K];
    __shared__ half B_tile[2][WMMA_K * WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

    int buffer_idx = 0;

    // --- Stage 1: Initial Prefetch ---
    // Before the main loop, we load the very first tile from global memory
    // into the first shared memory buffer. This "primes" the pipeline.
    // Each thread in the warp loads a piece of the tile.
    for (int i = threadIdx.y * 8 + threadIdx.x; i < WMMA_M * WMMA_K; i += 256) { // 16*16=256
        A_tile[buffer_idx][i] = A[warp_m * WMMA_M * K + i];
        B_tile[buffer_idx][i] = B[warp_n * WMMA_N + (i % WMMA_K) * N + (i / WMMA_K)];
    }
    __syncthreads(); 

    // --- Stage 2: Load first tile into fragments ---
    wmma::load_matrix_sync(A_frag[buffer_idx], A_tile[buffer_idx], WMMA_K);
    wmma::load_matrix_sync(B_frag[buffer_idx], B_tile[buffer_idx], WMMA_K);

    // --- Pipelined Main Loop ---
    // This loop iterates over the tiles in the K dimension.
    for (int k_tile_idx = 1; k_tile_idx < K / WMMA_K; ++k_tile_idx) {
        int next_buffer_idx = 1 - buffer_idx; // Switch to the other buffer

        // --- Stage 1 (for next iteration): Load Global -> Shared ---
        // Asynchronously load the *next* tile (k_tile_idx) into the *other* shared memory buffer.
        // This overlaps with the computation of the current tile.
        for (int i = threadIdx.y * 8 + threadIdx.x; i < WMMA_M * WMMA_K; i += 256) {
             A_tile[next_buffer_idx][i] = __float2half(A[warp_m * WMMA_M * K + k_tile_idx * WMMA_K + i]);
             B_tile[next_buffer_idx][i] = __float2half(B[warp_n * WMMA_N + (k_tile_idx * WMMA_K + (i % WMMA_K)) * N + (i / WMMA_K)]);
        }

        // --- Stage 2 (for this iteration): Load Shared -> Fragments ---
        // Load the *current* tile from shared memory into the *other* set of fragments.
        wmma::load_matrix_sync(A_frag[next_buffer_idx], A_tile[next_buffer_idx], WMMA_K);
        wmma::load_matrix_sync(B_frag[next_buffer_idx], B_tile[next_buffer_idx], WMMA_N);
        
        // --- Stage 3 (for previous iteration): Compute ---
        // Perform the matrix-multiply-accumulate operation on the tile loaded in the *previous* iteration.
        // D = A * B + C
        wmma::mma_sync(C_frag, A_frag[buffer_idx], B_frag[buffer_idx], C_frag);

        buffer_idx = next_buffer_idx; // Move to the next buffer for the next iteration.
        __syncthreads(); // Synchronize to ensure loads are complete before the next compute.
    }

    // --- Final MMA Operation ---
    // The loop finishes one MMA operation short. Perform the final one here.
    wmma::mma_sync(C_frag, A_frag[buffer_idx], B_frag[buffer_idx], C_frag);

    // --- Stage 4: Store Result ---
    // Store the final result from the accumulator fragment back to global memory.
    int out_row = warp_m * WMMA_M;
    int out_col = warp_n * WMMA_N;
    wmma::store_matrix_sync(C + out_row * N + out_col, C_frag, N, wmma::mem_row_major);
}

void tc_launcher(const half* d_A, const half* d_B, float* d_C, const int M, const int K, const int N) {
    // Each block computes a 16x16 tile using Tensor Cores.
    // We use a 2D block of 32x8=256 threads.
    dim3 threadsPerBlock(32, 8); 

    // The grid is sized to cover the entire output matrix C.
    // Each block handles a 16x16 output tile (WMMA_M x WMMA_N).
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);

    tc_gemm<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
}