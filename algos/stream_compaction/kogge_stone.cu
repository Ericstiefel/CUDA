// This is a kogge stone addition prefix scan.
#include <cuda_runtime.h>

// The simpler, blockwide scan: 



// This presumes N < 1024 (N elements fit in a block using 1 tpb). 
// Otherwise, simply adjust the outcome and have a separate block sums memory which you kogge stone scan (more time efficient than belloch),
// and simply add the previous block's sum onto each value to find the final. Here for simplicity (even though its not difficult),
// we remain with N < 1024 just to show the structure.
__global__ void kogge_stone(const int* __restrict__ inp, int* __restrict__ out, const int N) {
    __shared__ int buff[2048]; // buffer A is first blockDim.x elements, buffer B second
    int lid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    int val = 0;
    if (gid < N) {
        val = inp[gid];
    }
    buff[lid] = val;
    buff[lid + blockDim.x] = val;

    int tin = 0;
    int tout = blockDim.x;

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


