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



__device__ __forceinline__ float warpReduceSum(float val, unsigned mask = 0xffffffffu) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// dim3 tpb(M)
// dim3 bpg(N)

// Each row has its own thread block.

// N is dim(vector embedding)
// M is count(tokens_in_response), one guarenteed to be < 1024 in previous steps.

__global__ add_n_norm(const half* __restrict__ conc_heads, const half* __restrict__ mmha, half* __restrict__ out, float* mean, float* std, int M, int N) 
/*
Finalized out from concatonating heads and matmul with W0 -> conc_heads: MxN
Finalized out from Masked Multi-Head Attention -> mmha: MxN
*/
    {

        __shared__ float vals[32]; // max 1024 tpb, so 1024 / 32 warps -> max(32) vals
        int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;

        int idx = row * N + col;

        float sum = 0.0f;

        for (int i = idx; i < M * N; i += blockDim.x * gridDim.x * blockDim.y * gridDim.y) { // This will only occur once per thread because of the given tpb & bpg, but done here for general completion
            sum += conc_heads[i];
        }

        warpReduceSum(sum);

        int warp_idx = threadIdx.x / 32;
        int warp_lane = threadIdx.x % 32;

        if (warp_lane == 0) { vals[warp_idx] = sum; }

        __syncthreads();

        if (warp_idx == 0) {
            sum = vals[warp_lane];
            warpReduceSum(sum);

            if (warp_lane == 0) { *mean = sum / (M * N); }
        }

        __syncthreads(); // very necessary, especially with warp 0 being the only one scheduled for the last block

        float avg = *mean;
        sum = 0.0f;

        for (int i = idx; i < M * N; i += blockDim.x * gridDim.x * blockDim.y * gridDim.y) {
            sum += (conc_heads[i] - mean) * (conc_heads[i] - mean);
        }

        warpReduceSum(sum);

        if (warp_lane == 0) { vals[warp_idx] = sum; }

        __syncthreads();

        if (warp_idx == 0) {
            sum = vals[warp_lane];
            warpReduceSum(sum);

            if (warp_lane == 0) { *std = sum / (M * N - 1); }
        }

        __syncthreads();

        for (int i = idx; i < M * N; i += blockDim.x * gridDim.x * blockDim.y * gridDim.y) {
            out[i] = mmha[i] + (conc_heads[i] - mean) / (*std + 1e-9); 
        }


    }