#include <cuda_runtime.h>

#define BITS_PER_GROUP 8

__device__ unsigned int get_bits(unsigned int num, int shift_amount) {
    return (num >> shift_amount) & 0xFF;
}

__global__ void bitogram(const int* __restrict__ inp, int* __restrict__ global_hist, const int N, int shift) {
    __shared__ int vals[256];; // 2 ^ 8 (For 8 bits per group)

    int lid = threadIdx.x;
    int gid = lid + blockDim.x * blockIdx.x;

    for (int i = lid; i < 256; i += blockDim.x) {
        vals[i] = 0;
    }
    __syncthreads();

    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        // Cast to unsigned to ensure logical shift in get_bits
        unsigned int bin = get_bits((unsigned int)inp[i], shift);
        atomicAdd(&vals[bin], 1);
    }

    __syncthreads();

    for (int i = lid; i < 256; i += blockDim.x) {
        atomicAdd(&global_hist[i], vals[i]);
    }
}

__device__ int warp_scan(int val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = __shfl_up_sync(0xffffffffu, val, offset);
        if (threadIdx.x % 32 >= offset) { val += neighbor; }
    }
    return val;
}

// Because there are 256 items per iteration, its much simpler to do it in 1 block with 256 threads than run 2 kernels
__global__ void kogge_stone_scan(const int* __restrict__ global_hist, int* __restrict__ global_prefix) {
    __shared__ int scan[8]; // 256 / 32 = 8.

    int lid = threadIdx.x;

    int val = global_hist[lid];
    val = warp_scan(val);

    int warp_num = lid / 32;
    int warp_lane = lid % 32;

    if (warp_lane == 31) { scan[warp_num] = val; }

    __syncthreads();

    if (lid < 32) {
        int temp = (lid < 8) ? scan[lid] : 0;
        temp = warp_scan(temp);
        if (lid < 8) { scan[lid] = temp; }
    }

    __syncthreads();

    if (warp_num > 0) {
        val += scan[warp_num - 1];
    }

    global_prefix[lid] = val;
}