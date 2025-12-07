#include <cuda_runtime.h>

/*
Estimation: 
Global Memory Read / Write is ~ 500 clock cycles
Shared Memory Read / Write is ~ 5 clock cycles
Naive Transpose: 
Global Memory Read Coalesced 500 / 8 ~= 60
Global Memory Write Uncoalesced (swapping write pattern) 500
Total 560

Tiled Transpose:
Global Memory Read Coalesced 500 / 8 ~= 60
Shared Memory Write Uncoalesced 5
Shared Memory Read Coalesced 5 / 8 ~ 1
Global Memory Write Coaleced 500 / 8 ~= 60
Total 126
*/

// Matrix A is M x N
__global__ void naive_transpose(const float* __restrict__ inp, float* __restrict__ out, const int M, const int N) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;


    if (row < M && col < N) {
        out[col * M + row] = inp[row * N + col];
    }
}

// Premise of this kernel is to launch enough blocks to be able to place full chunk in shared memory.
// We can launch 2^31 - 1 blocks in the x direction and 65535 in the y (or z directions).

// Max dims of the matrix := max elems per block * max blocks per grid

// Only potential limiting factor on elems per block is shared memory storage.
// sizeof(float32) = 32 bits = 4 bytes. 
// Let's assume max shared memory per block is 48 KB (a low estimation).
// 48,000 / 4 = 12,000.
// Since we're only assigning one thread per element, and max threads per block is 1024, there is no restriction on shared memory capacity.

// min(gridDim.x, gridDim.y) = min(2^31 - 1, 65535) = 65535.
// 65535 * 1024 elements per block = 67,107,840, I don't believe you're running any matrices with a M dimension anytime soon above that number. N can go far higher (2^31 * 1024).

// For all intents and purposes, we can adequately claim the following kernel will work on any non absurdly large matrix.
#define TILE_DIM 32

__global__ void optimized_tiled_transpose(const float* __restrict__ inp, float* __restrict__ out, int M, int N) {
    // Pad
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_in = blockIdx.y * TILE_DIM + threadIdx.y;

    if (y_in < M && x_in < N) {
        tile[threadIdx.y][threadIdx.x] = inp[y_in * N + x_in];
    }

    __syncthreads(); 


    int x_out = blockIdx.y * TILE_DIM + threadIdx.x; 
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y; 

    if (y_out < N && x_out < M) {
        out[y_out * M + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}