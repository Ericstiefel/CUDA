#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }

}

void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
