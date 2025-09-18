#include <cuda_runtime.h>
#include <studio.h>

#define CUDA_CHECK(call) \

do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Cuda Error in file %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE) \
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


void construct_graph() {
    cudaGraph_t graph;
    cudaGraphCreate(&graph);

    cudaGraphNode_t cpyA, cpyB, cpyC, kerA, kerB, kerC;

    cudaKernelNodeParams kerAP, kerBP, kerCP;
    cudaMemcpy3DParams memparA, memparB, memparC;


    // Assume all the params were defined here



    CUDA_CHECK(cudaGraphAddMemcpyNode(&cpyA, graph, NULL, 0, &memparA));
    CUDA_CHECK(cudaGraphAddMemcpyNode(&cpyB, graph, NULL, 0, &memparB));

    CUDA_CHECK(cudaGraphAddKernelNode(&kerA, graph, &cpyA, 1, &kerAP));
    CUDA_CHECK(cudaGraphAddKernelNode(&kerB, graph, &cpyB, 1, &kerAB));

    cudaGraphNode_t c_dep[] = {kerA, kerB};

    CUDA_CHECK(cudaGraphAddKernelNode(&kerC, graph, &c_dep, 2, &kerCP));

    cudaGraphExec_t graph_executable;

    CUDA_CHECK(cudaGraphInstantiate(&graph_executable, graph));

}