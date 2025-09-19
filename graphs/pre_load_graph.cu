

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

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
    // --- 1. Standard Setup ---
    int N = 1000;
    size_t size = N * sizeof(float);
    // Allocate all host and device memory as before
    // Create streams and events as before

    cudaGraph_t graph;
    cudaGraphExec_t graph_executable;
    const char* graph_filename = "my_graph.bin";

    std::ifstream f(graph_filename, std::ios::binary);
    if (f.good()) {
        printf("Serialized graph found. Loading from disk...\n");
        
        // Get file size and read it into a buffer
        f.seekg(0, f.end);
        size_t file_size = f.tellg();
        f.seekg(0, f.beg);
        
        std::vector<char> buffer(file_size);
        f.read(buffer.data(), file_size);
        f.close();

        // Create the executable graph directly from the buffer
        CUDA_CHECK(cudaGraphExecDeserialize(&graph_executable, buffer.data(), buffer.size()));

    } else {
        printf("No serialized graph found. Creating and saving...\n");

        CUDA_CHECK(cudaStreamBeginCapture(streamA, cudaStreamCaptureModeGlobal));

        // All the cudaMemcpyAsync, kernel launch, event calls

        CUDA_CHECK(cudaStreamEndCapture(streamA, &graph));

        CUDA_CHECK(cudaGraphInstantiate(&graph_executable, graph, NULL, NULL, 0));

        size_t buffer_size = 0;
        void* buffer = nullptr;
        CUDA_CHECK(cudaGraphExecSerialize(graph_executable, &buffer, &buffer_size));

        std::ofstream out_file(graph_filename, std::ios::binary);
        out_file.write((const char*)buffer, buffer_size);
        out_file.close();
        
        cudaFree(buffer);
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    CUDA_CHECK(cudaGraphLaunch(graph_executable, streamA));
    CUDA_CHECK(cudaStreamSynchronize(streamA));
    printf("Graph execution complete.\n");

    CUDA_CHECK(cudaGraphExecDestroy(graph_executable));
    // ... (Free all memory and destroy streams/events) ...

    return 0;
}