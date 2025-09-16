/*
Very simple file just to implement a timer with events. Timing a small kernel.
*/
#include <cuda_runtime.h>
#include <studio.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetStringError(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


__global__ void Kernel(float* input, float* output, int N) {
    if (threadIdx.x < N) {
        output[threadIdx.x] = input[threadIdx.x] * 5.0f;
    }
    
}

int main() {
    int N = 5000;
    float* h_inp, h_out;
    size_t size = N * sizeof(float);


    CUDA_CHECK(cudaMallocHost((void**)&h_inp, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_out, size));

    for (int i = 0; i < N; ++i) { inp[i] = i; }

    float* d_inp, d_out;

    CUDA_CHECK(cudaMalloc((void**)&d_inp, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));

    CUDA_CHECK(cudaMemcpy(d_inp, h_inp, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    dim3 threadsPerBlock(N);

    CUDA_CHECK(cudaEventRecord(start));

    Kernel<<<1, N>>>(inp, out, N);

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestory(stop));

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));

    CUDA_CHECK(cudaFreeHost(h_inp));
    CUDA_CHECK(cudaFreeHost(h_out));

    return 0;


}