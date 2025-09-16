/*
Simple Asynch call with kernel launch
*/
#include <cuda_runtime.h>
#include <studio.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void kernel(float* input, float* output, int N) {
    if (threadIdx.x < N) {
        output[threadIdx.x] = input[threadIdx.x] * 5.0f;
    }
}

int main() {
    int N = 1000;
    size_t size_bytes = N * sizeof(float);

    // 1. Allocate pinned host memory for input and output
    float *h_input, *h_output; // Correct pointer declaration
    CUDA_CHECK(cudaMallocHost((void**)&h_input, size_bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_output, size_bytes));

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_input[i] = (float)i;
    }

    // 2. Allocate device memory
    float *d_input, *d_output; // Correct pointer declaration
    CUDA_CHECK(cudaMalloc((void**)&d_input, size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_bytes));

    // 3. Create streams and an event
    cudaStream_t stream_copy, stream_compute;
    CUDA_CHECK(cudaStreamCreate(&stream_copy));
    CUDA_CHECK(cudaStreamCreate(&stream_compute));

    cudaEvent_t copy_finished_event;
    CUDA_CHECK(cudaEventCreate(&copy_finished_event));

    // 4. Execute the pipeline
    // In the copy stream, start the Host-to-Device copy
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, size_bytes, cudaMemcpyHostToDevice, stream_copy));

    // Record an event in the copy stream right after the copy is queued
    CUDA_CHECK(cudaEventRecord(copy_finished_event, stream_copy));

    // In the compute stream, wait for the copy to be finished before proceeding
    CUDA_CHECK(cudaStreamWaitEvent(stream_compute, copy_finished_event, 0));

    // Launch the kernel in the compute stream (it will only run after the wait)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock, 0, stream_compute>>>(d_input, d_output, N);

    // 5. Copy results back to host to verify
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size_bytes, cudaMemcpyDeviceToHost));
    
    // Note: A standard cudaMemcpy is synchronous and will implicitly wait for the kernel to finish.

    // 6. Clean up
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(copy_finished_event));
    CUDA_CHECK(cudaStreamDestroy(stream_copy));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));

    return 0;
}