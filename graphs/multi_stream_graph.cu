#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                 \
do {                                                                     \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "Cuda Error in file %s at line %d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));                                \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

__global__ void kernelA(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 5.0f;
    }
}

__global__ void kernelB(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * 4.0f;
    }
}

__global__ void kernelC(const float* inputA, const float* inputB, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = inputA[idx] + inputB[idx];
    }
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    float *h_ainp, *h_aout, *h_binp, *h_bout, *h_cout;
    float *d_ainp, *d_aout, *d_binp, *d_bout, *d_cout;
    
    CUDA_CHECK(cudaMallocHost((void**)&h_ainp, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_aout, size));

    CUDA_CHECK(cudaMallocHost((void**)&h_binp, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_bout, size));

    CUDA_CHECK(cudaMallocHost((void**)&h_cout, size));

    for (int i = 0; i < N; ++i) { h_ainp[i] = (float)i; }
    for (int i = 0; i < N; ++i) { h_binp[i] = (float)(i % 25); }

    CUDA_CHECK(cudaMalloc((void**)&d_ainp, size));
    CUDA_CHECK(cudaMalloc((void**)&d_aout, size));

    CUDA_CHECK(cudaMalloc((void**)&d_binp, size));
    CUDA_CHECK(cudaMalloc((void**)&d_bout, size));

    CUDA_CHECK(cudaMalloc((void**)&d_cout, size));

    cudaGraph_t graph;
    cudaGraphExec_t graph_executable;

    CUDA_CHECK(cudaGraphCreate(&graph));

    cudaStream_t streamA, streamB, streamC;

    cudaEvent_t kerA, kerB;


    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));
    CUDA_CHECK(cudaStreamCreate(&streamC));

    CUDA_CHECK(cudaEventCreate(&kerA));
    CUDA_CHECK(cudaEventCreate(&kerB));

    CUDA_CHECK(cudaStreamBeginCapture(streamA, cudaStreamCaptureModeGlobal));

    CUDA_CHECK(cudaMemcpyAsyc(d_ainp, h_ainp, size, cudaMemcpyHostToDevice, streamA));

    CUDA_CHECK(cudaMemcpyAsyc(d_binp, h_binp, size, cudaMemcpyHostToDevice, streamB));

    CUDA_CHECK(kernelA<<<1, N, 0, streamA>>>(d_ainp, d_aout, N));
    CUDA_CHECK(kernelB<<<1, N, 0, streamB>>>(d_binp, d_bout, N));

    CUDA_CHECK(cudaEventRecord(kerA, streamA));
    CUDA_CHECK(cudaEventRecord(kerB, streamB));

    CUDA_CHECK(cudaStreamWaitEvent(streamC, kerA));
    CUDA_CHECK(cudaStreamWaitEvent(streamC, kerB));

    CUDA_CHECK(kernelC<<<1, N, 0, streamC>>>(d_aout, d_bout, d_cout, N));

    CUDA_CHECK(cudaMemcpyAsync(h_cout, d_cout, size, cudaMemcpyDeviceToHost, streamC));

    CUDA_CHECK(cudaStreamEndCapture(streamA, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_executable, graph));

    CUDA_CHECK(cudaGraphLaunch(graph_executable, streamC));
    CUDA_CHECK(cudaStreamSynchronize(streamC));

    CUDA_CHECK(cudaHostFree(h_ainp));
    // Rest of freeing memory

    return 0;
}