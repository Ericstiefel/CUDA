/*
Simple Graph using Stream Capture that multiplies every element by two, single stream.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)\
do {\
    cudaError_t err = call;\

    if (err != cudaSuccess) {\
        fprintf(stderr, "Cuda Failure in File %s on line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
} while (0)

__global__ void ker(float* inp, float* out, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N) {
        out[idx] = inp[idx] * 2.0f;
    }
}

int main() {
    int N = 1000;

    size_t size = sizeof(float) * N;

    float *h_inp, *h_out;

    CUDA_CHECK(cudaMallocHost((void**)&h_inp, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_out, size));

    for (int i = 0; i < N; ++i) {
        h_inp[i] = (float)i;
    }

    float *d_inp, *d_out;

    CUDA_CHECK(cudaMalloc((void**)&d_inp, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));


    cudaGraph_t graph;
    cudaGraphExec_t graph_executable;


    cudaStream_t stream;

    cudaStreamCreate(&stream);

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    CUDA_CHECK(cudaMemcpyAsync(d_inp, h_inp, size, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(ker<<<1, N, 0, stream>>>(d_inp, d_out, N));

    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&graph_executable, graph));


    printf("Launching graph...\n");
    CUDA_CHECK(cudaGraphLaunch(graph_executable, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("Graph execution complete.\n");

    printf("Result verification for index 10: %f (Expected: 20.0)\n", h_out[10]);

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graph_executable));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_inp));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));

    return 0;

}