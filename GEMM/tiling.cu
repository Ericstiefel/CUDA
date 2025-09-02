#include <cuda_runtime.h>

template <int TILE_WIDTH>
__global__ void tiling_cm(const float* A, const float* B, float* C, const int M, const int K, const int N) {
    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x;

    int global_row = threadIdx.y + TILE_WIDTH * blockIdx.y;
    int global_col = threadIdx.x + TILE_WIDTH * blockIdx.x;

    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
        int a_row = global_row;
        int a_col = tile * TILE_WIDTH + tile_col;

        int b_row = tile * TILE_WIDTH + tile_row;
        int b_col = global_col;
        
        A_tile[tile_row][tile_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        B_tile[tile_row][tile_col] = (b_row < M && b_col < K) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += A_tile[tile_row][k] * B_tile[k][tile_col];
        }

        __syncthreads();


    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}
template <int TILE_WIDTH>
void tile_cm_launcher(const float* d_A, const float* d_B, float* d_C, const int M, const int K, const int N) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);

    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH);

    tiling_cm<TILE_WIDTH><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
}

// This line tells the compiler to create the code for the TILE_WIDTH = 32 version.
template void tile_cm_launcher<32>(const float*, const float*, float*, int, int, int);