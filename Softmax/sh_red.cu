/*
Gameplan:
Each block will compute one row of the input matrix.
This means we launch N blocks.

threadsPerBlock = TILE_SIZE (each thread will load one element from global to shared memory & do computations)

# of tiles iterated over = (N + TILE_SIZE - 1) / TILE_SIZE

Device called kernel: Find max element and normalized sum through reduction (O(log(n)) steps instead of O(n)). Two separate kernels.

local max and sum written to registers 

*/

#include <cuda_runtime.h>
#include <cmath>

#define TILE_SIZE 64

__device__ void block_reduce(float* s_data, float& val, bool is_max) {
    s_data[threadIdx.x] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (is_max) { // No thread divergence because all threads in a warp execute the same path
                // Max reduction
                if (s_data[threadIdx.x + stride] > s_data[threadIdx.x]) {
                    s_data[threadIdx.x] = s_data[threadIdx.x + stride];
                }
            } else {
                // Sum reduction
                s_data[threadIdx.x] += s_data[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }]
    val = s_data[0];
}

__global__ void softmax_fused_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int M, int N) {
    

    __shared__ float s_tile[TILE_SIZE];

    int row = blockIdx.x;
    int row_start_idx = row * N;
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    float row_max = -FLT_MAX;
    for (int tile = 0; tile < num_tiles; ++tile) {
        int idx = tile * TILE_SIZE + threadIdx.x;
        
        float thread_val = (idx < N) ? input[row_start_idx + idx] : -FLT_MAX; // Padding not necessary (warp threads are placed in the 32 separate Shared Memory banks)

        block_reduce(s_tile, thread_val, true);
        
        if (threadIdx.x == 0) {
            if (thread_val > row_max) {
                row_max = thread_val;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_tile[0] = row_max;
    }
    __syncthreads();
    row_max = s_tile[0];

    float row_sum = 0.0f;
    for (int tile = 0; tile < num_tiles; ++tile) {
        int idx = tile * TILE_SIZE + threadIdx.x;

        float thread_val = 0.0f;
        if (idx < N) {
            thread_val = expf(input[row_start_idx + idx] - row_max);
        }

        block_reduce(s_tile, thread_val, false);

        if (threadIdx.x == 0) {
            row_sum += thread_val;
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        s_tile[0] = row_sum;
    }
    __syncthreads();
    row_sum = s_tile[0];
    

    for (int tile = 0; tile < num_tiles; ++tile) {
        int idx = tile * TILE_SIZE + threadIdx.x;
        if (idx < N) {
            float val = expf(input[row_start_idx + idx] - row_max);
            output[row_start_idx + idx] = val / row_sum;
        }
    }
}

// Host launcher function
void softmax_launcher(float* d_input, float* d_output, int M, int N) {
    dim3 threadsPerBlock(TILE_SIZE);
    dim3 blocksPerGrid(M);

    softmax_fused_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, M, N);
}
