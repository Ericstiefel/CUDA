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

#define TILE_SIZE 16

__device__ float max_red(float* __restrict__ input) {
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            float new_val = input[threadIdx.x + stride]; // store in register instead of L1 
            float old_val = input[threadIdx.x];
            if (new_val > old_val) { input[threadIdx.x] = new_val; }
        }
        __syncthreads();

    }
    
    return input[0];

}

__global__ void find_max_and_sum_kernel(const float* __restrict__ input,
                                     float* __restrict__ maxes,
                                     float* __restrict__ sums,
                                     int M, int N) {
    // Shared memory for one tile
    __shared__ float s_tile[TILE_SIZE];
    
    // Use a register for the block's running max. Only thread 0 needs it.
    float running_max = -INFINITY;

    // --- Pass 1: Find true maximum for the row ---
    // Each block is assigned one row
    int row = blockIdx.x;
    int row_start_idx = row * N;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int idx = tile * TILE_SIZE + threadIdx.x;
        // Load data into shared memory, handling non-full tiles
        if (idx < N) {
            s_tile[threadIdx.x] = input[row_start_idx + idx];
        } else {
            s_tile[threadIdx.x] = -INFINITY; // Identity for max
        }
        __syncthreads();

        // Have one thread accumulate the max from each tile reduction
        if (threadIdx.x == 0) {
            float tile_max = max_red(s_tile);
            if (tile_max > running_max) { running_max = tile_max; }
        }
        // Sync to ensure reduction is complete before next tile
        __syncthreads();
    }

    __shared__ float final_row_max_s;
    if (threadIdx.x == 0) {
        final_row_max_s = running_max;
        maxes[row] = running_max; 
    }
    __syncthreads();
    float final_row_max_r = final_row_max_s;

    // --- Pass 2: Find sum using the true maximum ---
    float running_sum = 0.0f;
    for (int tile = 0; tile < num_tiles; ++tile) {
        int idx = tile * TILE_SIZE + threadIdx.x;
        // Load original data again and apply the expf operation
        if (idx < N) {
            s_tile[threadIdx.x] = expf(input[row_start_idx + idx] - final_row_max_r);
        } else {
            s_tile[threadIdx.x] = 0.0f; // Identity for sum
        }
        __syncthreads();

        // Have one thread accumulate the sum from each tile reduction
        if (threadIdx.x == 0) {
            float tile_sum = sum_red(s_tile);
            running_sum += tile_sum;
        }
        __syncthreads();
    }

    // One thread writes the final sum for the row
    if (threadIdx.x == 0) {
        sums[row] = running_sum;
    }
}

void sh_red_launcher(float* input, float* output, int M, int N) {
    dim3 threadsPerBlock(TILE_SIZE);
    dim3 blocksPerGrid(M);

    float* maxes[(N + TILE_SIZE - 1) / TILE_SIZE];
    float* sums[(N + TILE_SIZE - 1) / TILE_SIZE];

    sh_red<<<blocksPerGrid, threadsPerBlock>>>(input, output, maxes, aums, N);
}