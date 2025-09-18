/*
Manual Graph implementation of the basic_graph.cu
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)\
do {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        fprintf(stderr, "Cuda Error in file %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
} while (0)

__global__ void ker(float* inp, float* out, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        out[idx] = inp[idx] * 2.0f;
    }
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(float);
    float *h_inp, *h_out, *d_inp, *d_out;

    CUDA_CHECK(cudaMallocHost((void**)&h_inp, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_out, size));

    CUDA_CHECK(cudaMalloc((void**)d_inp, size));
    CUDA_CHECK(cudaMalloc((void**)d_out, size));

    cudaGraph_t graph;
    cudaGraphExec_t graph_executable;

    cudaGraphCreate(&graph);

    cudaGraphNode_t to_device, from_device, ker;

    // --- Fill params for the Host-to-Device copy ---
    cudaMemcpy3DParms to_d_p = {0}; // Initialize to zero
    to_d_p.srcPtr   = make_cudaPitchedPtr(h_inp, size, N, 1);
    to_d_p.dstPtr   = make_cudaPitchedPtr(d_inp, size, N, 1);
    to_d_p.extent   = make_cudaExtent(size, 1, 1);
    to_d_p.kind     = cudaMemcpyHostToDevice;

    // --- Fill params for the Device-to-Host copy ---
    cudaMemcpy3DParms from_d_p = {0};
    from_d_p.srcPtr   = make_cudaPitchedPtr(d_out, size, N, 1);
    from_d_p.dstPtr   = make_cudaPitchedPtr(h_out, size, N, 1);
    from_d_p.extent   = make_cudaExtent(size, 1, 1);
    from_d_p.kind     = cudaMemcpyDeviceToHost;

    void* kernel_args[] = {&d_inp, &d_out, &N};

    // --- Fill params for the kernel launch ---
    cudaKernelNodeParams ker_params = {0};
    ker_params.func           = (void*)ker;
    ker_params.gridDim        = 1;
    ker_params.blockDim       = N;
    ker_params.sharedMemBytes = 0;
    ker_params.kernelParams   = kernel_args;
    ker_params.extra          = NULL;

    CUDA_CHECK(cudaGraphAddMemcpyNode(&to_device, graph, NULL, 0, &to_d_p));
    CUDA_CHECK(cudaGraphAddKernelNode(&ker, graph, to_device, 1, &ker_params));
    CUDA_CHECK(cudaGraphAddMemcpyNode(&from_device, graph, ker, 1, &from_d_p));

    CUDA_CHECK(cudaGraphInstantiate(&graph_executable, graph));


    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaGraphLaunch(graph_executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graph_executable));
    CUDA_CHECK(cudaFreeHost(h_inp));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaStreamDestroy(stream));


    return 0;
}