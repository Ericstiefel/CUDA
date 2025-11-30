#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define TILE_WIDTH 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void basic_tc_gemm(const half* A, const half* B, float* C, const int M, const int K, const int N) {
    int row = blockIdx.y * WMMA_M;
    int col = blockIdx.x * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fill_fragment(C_frag, 0.0f);

    for (int i = 0; i < K; i += WMMA_K) {
        const half* A_ptr = A + row * K + i;
        const half* B_ptr = B + i * N + col;

        wmma::load_matrix_sync(A_frag, A_ptr, K);
        wmma::load_matrix_sync(B_frag, B_ptr, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    float* C_ptr = C + row * N + col;

    wmma::store_matrix_sync(C_ptr, C_frag, N, wmma::mem_row_major);
}



void basic_tc_launcher(const half* d_A, const half* d_B, float* d_C, int M, int K, int N) {
    dim3 threadsPerBlock(32);

    dim3 numBlocks((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

    basic_tc_gemm<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

}