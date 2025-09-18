/*
This file will contain using cuda api to create a basic reusable graph
*/

#include <cuda_runtime.h>
#include <studio.h>

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
__global__ void kernelC(const float* inputA, const float* inputB, float* output, int N) {
    if (threadIdx.x < N) {
        output[threadIdx.x] = inputA[threadIdx.x] + inputB[threadIdx.x];
    }
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);

    // Host memory
    float *h_inpA, *h_inpB, *h_outC;
    CUDA_CHECK(cudaMallocHost((void**)&h_inpA, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_inpB, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_outC, size));

    // Initialize host inputs
    for (int i = 0; i < N; ++i) { h_inpA[i] = (float)i; }
    for (int i = 0; i < N; ++i) { h_inpB[i] = (float)i * 2.0f; }

    // Device memory
    float *d_inpA, *d_inpB, *d_outA, *d_outB, *d_outC;
    CUDA_CHECK(cudaMalloc((void**)&d_inpA, size));
    CUDA_CHECK(cudaMalloc((void**)&d_inpB, size));
    CUDA_CHECK(cudaMalloc((void**)&d_outA, size));
    CUDA_CHECK(cudaMalloc((void**)&d_outB, size));
    CUDA_CHECK(cudaMalloc((void**)&d_outC, size)); 

    cudaStream_t streamA, streamB, streamC;
    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));
    CUDA_CHECK(cudaStreamCreate(&streamC));

    cudaEvent_t eventA_done, eventB_done;
    CUDA_CHECK(cudaEventCreate(&eventA_done));
    CUDA_CHECK(cudaEventCreate(&eventB_done));

    cudaGraph_t graph;
    cudaGraphExec_t graph_executable;

    CUDA_CHECK(cudaStreamBeginCapture(streamA, cudaStreamCaptureModeGlobal));


    CUDA_CHECK(cudaMemcpyAsync(d_inpA, h_inpA, size, cudaMemcpyHostToDevice, streamA));
    CUDA_CHECK(cudaMemcpyAsync(d_inpB, h_inpB, size, cudaMemcpyHostToDevice, streamB));


    kernelA<<<1, N, streamA>>>(d_inpA, d_outA, N);
    kernelB<<<1, N, streamB>>>(d_inpB, d_outB, N);

    CUDA_CHECK(cudaEventRecord(eventA_done, streamA));
    CUDA_CHECK(cudaEventRecord(eventB_done, streamB));

    CUDA_CHECK(cudaStreamWaitEvent(streamC, eventA_done));
    CUDA_CHECK(cudaStreamWaitEvent(streamC, eventB_done));

    kernelC<<<1, N, streamC>>>(d_outA, d_outB, d_outC, N);

    CUDA_CHECK(cudaStreamEndCapture(streamA, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_executable, graph));
    

    CUDA_CHECK(cudaGraphLaunch(graph_executable, streamA)); // Reuse later

    CUDA_CHECK(cudaMemcpyAsync(h_outC, d_outC, size, cudaMemcpyDeviceToHost, streamC));

    CUDA_CHECK(cudaStreamSynchronize(streamC)); // Result is now in h_outC


    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graph_executable));
    CUDA_CHECK(cudaFreeHost(h_inpA));
    CUDA_CHECK(cudaFreeHost(h_inpB));
    CUDA_CHECK(cudaFreeHost(h_outC));
    CUDA_CHECK(cudaFree(d_inpA));
    CUDA_CHECK(cudaFree(d_inpB));
    CUDA_CHECK(cudaFree(d_outA));
    CUDA_CHECK(cudaFree(d_outB));
    CUDA_CHECK(cudaFree(d_outC));
    CUDA_CHECK(cudaEventDestroy(eventA_done));
    CUDA_CHECK(cudaEventDestroy(eventB_done));
    CUDA_CHECK(cudaStreamDestroy(streamA));
    CUDA_CHECK(cudaStreamDestroy(streamB));
    CUDA_CHECK(cudaStreamDestroy(streamC));

    return 0;
}