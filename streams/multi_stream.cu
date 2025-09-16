/*
Stream Practice

Run Kernels A and B simultaneously, and C to wait on both
*/
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \

do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Cuda Error in file %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


__global__ void kernelA(float* input, float* output, int N) {
    if (threadIdx.x < N) {
        output[threadIdx.x] = input[threadIdx.x] * 5.0f;
    }
}
__global__ void kernelB(float* input, float* output, int N) {
    if (threadIdx.x < N) {
        output[threadIdx.x] = input[threadIdx.x] * 4.0f;
    }
}
__global__ void kernelC(float* input, float* output, int N) {
    if (threadIdx.x < N) {
        output[threadIdx.x] = input[threadIdx.x] * 10.0f;
    }
}

int main() {
    float *h_inpA, *h_outA;
    int N = 1000;

    size_t size = N * sizeof(float);

    CUDA_CHECK(cudaMallocHost((void**)&h_inpA, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_outA, size));

    for (int i = 0; i < N; ++i) { h_inpA[i] = i % 15; }

    float *h_inpB, *h_outB;

    CUDA_CHECK(cudaMallocHost((void**)&h_inpB, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_outB, size));


    float *d_inpA, *d_outA;

    CUDA_CHECK(cudaMalloc((void**)&d_inpA, size));
    CUDA_CHECK(cudaMalloc((void**)&d_outA, size));

    for (int i = 0; i < N; ++i) { h_inpA[i] = i % 15; }

    float *d_inpB, *d_outB;

    CUDA_CHECK(cudaMalloc((void**)&d_inpB, size));
    CUDA_CHECK(cudaMalloc((void**)&d_outB, size));

    for (int i = 0; i < N; ++i) { h_inpB[i] = i % 20; }

    float *h_inpC, *h_outC;

    CUDA_CHECK(cudaMallocHost((void**)&h_inpC, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_outC, size));
    

    float *d_inpC, *d_outC;

    CUDA_CHECK(cudaMalloc((void**)&d_inpC, size));
    CUDA_CHECK(cudaMalloc((void**)&d_outC, size));


    cudaEvent_t memA, memB, kernel_completion;
    
    cudaEventCreate(&memA);
    cudaEventCreate(&memB);
    cudaEventCreate(&kernel_completion);


    cudaStream_t s1, s2;

    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    CUDA_CHECK(cudaMemcpyAsynch())



}