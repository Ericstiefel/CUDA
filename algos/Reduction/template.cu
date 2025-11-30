#include <cuda_runtime.h>
#include <algorithm> 

__device__ int warp_reduce_max(int val) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset /= 2) {
        int neighbor = __shfl_down_sync(mask, val, offset);
        val = max(val, neighbor);
    }
    return val;
}

__global__ void block_max(const int* __restrict__ inp, int* __restrict__ block_sum, const int N) {
    __shared__ int vals[32]; // 1024 / 32 = 32 at maximum

    int l_idx = threadIdx.x;
    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = l_idx / 32;
    int lane = l_idx % 32;

    int val = -2147483648; // min int

    if (g_idx < N) {
        val = inp[g_idx];
    }

    val = warp_reduce_max(val);

    if (lane == 0) {
        vals[warp_id] = val;
    }

    __syncthreads();

    if (warp_id == 0) {

        int num_warps_in_block = blockDim.x / 32;
        
        val = (lane < num_warps_in_block) ? vals[lane] : -2147483648;
        
        val = warp_reduce_max(val);

        if (lane == 0) { 
            *block_sum = val; 
        }
    }
}