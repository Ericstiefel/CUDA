#include <cuda_runtime.h>
#include <cmath>



__global__ void naive_kernel(const float* __restrict__ input, float* __restrict__ output, const int __restrict__ M, const int __restrict__ N) {
    int global_row = blockIdx.x * blockIdx.x + threadIdx.x;
    if (global_row < M) {
        int i = N * global_row;

        float sum = 0.0f;
        float max_i = input[i];

        for (int k = 1; k < N; ++k) {
            if (input[i + k] > max_i) { max_i = input[i + k]; }
        }
        for (int k = 0; k < N; ++k) {
            sum += exp(input[i + k] - max_i);
        }
        for (int k = 0; k < N; ++k) {
            output[i + k] = exp(output[i + k] - max_i) / sum;
        }

    }
    
}


void naive_launcher(const float* input, float* output, int M, int N) {
    dim3 blockSize(1024);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, )

    naive_kernel<<<gridSize, blockSize>>>(input, output, M, N);
    cudaDeviceSynchronize();

}