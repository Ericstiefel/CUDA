// This is a kogge stone addition prefix scan.
#include <cuda_runtime.h>

// The simpler, blockwide scan: 


// This presumes N <= 1024 (N elements fit in a block using 1 tpb). 
// Otherwise, simply adjust the outcome and have a separate block sums memory which you kogge stone scan (more time efficient than belloch),
// and simply add the previous block's sum onto each value to find the final. Here for simplicity (even though its not difficult),
// we remain with N < 1024 just to show the structure, but if N is really <= 1024, use the warpwide below.
__global__ void kogge_stone(const int* __restrict__ inp, int* __restrict__ out, const int N) {
    __shared__ int buff[2048]; // buffer A is first blockDim.x elements, buffer B second
    int lid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int val = 0;
    if (gid < N) {
        val = inp[gid];
    }
    buff[lid] = val;

    int tin = 0;
    int tout = blockDim.x;

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (lid >= stride) {
            buff[tout + lid] = buff[tin + lid] + buff[tin + lid - stride];
        }
        else {
            buff[tout + lid] = buff[tin + lid];
        }
        int temp = tin;
        tin = tout;
        tout = temp;

        __syncthreads();

    }

    if (gid < N) {
        out[gid] = buff[tin + lid];
    }

}

// The faster Warpwide scan:


// This will compute the prefix sum in warps, not globally, to switch over to global, create another memory
// address warp_results of size (N - 31) / 32 (ceildiv of N by 32), and if the thread is % 31, place its warp num's result.
// If N <= 1024, this can be done in a single block by using shared memory, and doing the process again for only the first warp (<= 32 
// because 1024 / 32 = 32). If N is >= 1024, use the system above (blockwide instead of warpwide).

__device__ int warp_call(int val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = __shfl_up_sync(0xffffffffu, val, offset);

        if (threadIdx.x % 32 >= offset) { val += neighbor; }
    }
    return val;
}

__global__ void warp_kogge_stone(const int* __restrict__ inp, int*__restrict__ out, const int N) {
    __shared__ int warp_results[32]; // If N <= 1024, we have <= 32 warp results.
    int val = (threadIdx.x < N) ? inp[threadIdx.x] : 0;

    val = warp_call(val);

    // no need for syncthreads here, warpwide is already snchronous. 

    int warp_num = threadIdx.x / 32;
    int warp_lane = threadIdx.x % 32;

    if (warp_lane == 31) { warp_results[warp_num] = val; }

    __syncthreads();

    if (warp_num == 0) {
        int block_val = warp_results[warp_lane];
        block_val = warp_call(block_val);
        warp_results[warp_lane] = block_val;
    }
    __syncthreads();

    if (warp_num > 0) {
        val += warp_results[warp_num - 1];
    }
    if (threadIdx.x < N) { out[threadIdx.x] = val; }
}


