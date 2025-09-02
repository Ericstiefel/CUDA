#include <cuda_runtime.h>

/* 
Each thread will compute 1 value in the resulting C matrix
*/

__global__ void gpu_naive_gemm(const float* A, const float* B, float* C, const int M, const int K, const int N) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
            
        }

        C[row * N + col] = sum;
    }
}

void naive_launcher(const float* d_A, const float* d_B, float* d_C, const int M, const int K, const int N) {
    dim3 threadsPerBlock(32, 32);

    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gpu_naive_gemm<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
}
