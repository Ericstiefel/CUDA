#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define VOCAB_LEN 1000

#define CUDA_CHECK(call)
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in file %s in line %d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Necessary to init random weight matrices & feedforward layers
__global__ void init_random_matrix(half* ptr, int rows, int cols, unsigned long seed) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        curandState localState;
        curandInit(seed, idx, 0, &localState);
        float rv = curand_uniform(&localState);
        ptr[idx] = __float2half(rv);
    }
}

/*
Place the inp tokens into constant memory, O(5) CS access, and len(sequence) * sizeof(int) will fit into constant memory using any reasonable sequence length

dim3 threadsPerBlock(M)
dim3 blocksPerGrid(N)

Each block will embed a single token.

max threadsPerBlock -> 1024, assume token input length is less than that.
max blocksPerGrid -> 2^31 - 1, we are substantially below.
*/

__global__ void encodings(half* __restrict__ inp, half* __restrict__ out, int M, int N){ // Inp is a vector of length N. Each token will be encoded by a vector of length M, generating a M x N out matrix.
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    /*
    Simple encoding function (obviously learned in a real LLM)

    sin(r * (vocab_idx)) (yes I know there will be recurring values because trig functions oscillate and reach the same values every sin(2x). This is just purely for demonstration.
    */
   
    if (row < M && col < N) {
        int idx = row * N + col;

        float comp = row * inp[row];

        out[idx] = __float2half(__sinf(__float2half(comp)));
    }

}

/*

dim3 threadsPerBlock(M);
dim3 blocksPerGrid(N);

Launched in this way to ensure all threads in a warp (assumed by them being in the same block) don't diverge with trig split causing thread divergence, a significant slowdown.
*/

__global__ void positional_encodings(
    const half* __restrict__ inp, 
    half* __restrict__ out,
    int N, 
    int M    
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // position index (pos)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // embedding dimension (dim)

    // No need for check, if launched with simple dims as above

    int idx = row * M + col; 

    float pos = static_cast<float>(row);
    float i = static_cast<float>(col / 2);  
    float denom = powf(10000.0f, (2.0f * i) / static_cast<float>(M));

    float angle = pos / denom;

    float val = (col % 2 == 0) ? __sinf(angle) : __cosf(angle);

    out[idx] = __float2half(__half2float(inp[idx]) + val);
}
