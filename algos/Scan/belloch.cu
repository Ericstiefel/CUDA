// This is a blockwise belloch scan. If you'd like to prefix scan multiple blocks worth of elements (1 thread per element),
// simply make it a 3 kernel process

// 1) (almost same as current) : blockwise scan + output memory of the end of each block
// 2) Take these block results, and run a prefix scan (belloch if desired, usually kogge stone is more time efficient with less elements).
// 3) Summing each element (other than block 0) with the previous block prefix sum result from 2.

// For simplicity, launch with N threads ( < 1024 ) (done to avoid repetitive if _ < N for educational purposes)
// NOTE: N (and blockDim.x) must be a power of 2 for this specific indexing to work without padding checks.
__global__ void belloch_block(const int* __restrict__ inp, int* __restrict__ out, const int N) {
    __shared__ int vals[1024];
    int lid = threadIdx.x;

    // Load input
    if (lid < N) {
        vals[lid] = inp[lid];
    } else {
        vals[lid] = 0;
    }
    __syncthreads();

    // Up sweep (Reduce Phase)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (lid + 1) * stride * 2 - 1;
        
        if (index < blockDim.x) {
            vals[index] += vals[index - stride];
        }
        __syncthreads();
    }

    // Set last element to 0 (identity value)
    if (lid == 0) { 
        vals[blockDim.x - 1] = 0;
    }

    __syncthreads();

    // Down sweep (Distribute Phase)
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (lid + 1) * stride * 2 - 1;
        
        if (index < blockDim.x) {
            int temp = vals[index - stride];
            
            vals[index - stride] = vals[index];
            
            vals[index] += temp;
        }
        __syncthreads();
    }

    if (lid < N) {
        out[lid] = vals[lid];
    }
}