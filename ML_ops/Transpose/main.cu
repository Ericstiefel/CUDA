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