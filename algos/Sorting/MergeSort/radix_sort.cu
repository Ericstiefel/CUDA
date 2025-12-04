#include <cuda_runtime.h>

__global__ void bitonic_local_sort(const int* __restrict__ inp, int* __restrict__ block_sorts, const int N) {
    extern __shared__ int local_vals[]; // sizeof(int) * tpb.x
    int lid = threadIdx.x;
    int gid = lid + blockDim.x * blockIdx.x;

    if (gid < N) {
        local_vals[lid] = inp[gid];
    }

    // One reason we assume N = 2^k, k integer, is because otherwise we'd be pairing some numbers that did not input
    // to shared memory in the last block, and run into some errors. Another is the pure efficiency of the loop.
    __syncthreads();

    // here we're assuming tpb.x is also 2^p, p integer.
    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int stride = k / 2; stride > 0; stride /= 2) {
            bool ascending = (lid & k) == 0;

            int partner = stride ^ lid;

            int your_val = local_vals[lid];
            int partner_val = local_vals[partner];

            int max_val = (your_val > partner_val) ? your_val : partner_val;
            int min_val = (your_val <= partner_val) ? your_val : partner_val;

            if (lid < partner) {
                if (ascending) {
                    local_vals[lid] = min_val;
                    local_vals[partner] = max_val;
                }
                else {
                    local_vals[lid] = max_val;
                    local_vals[partner] = min_val;
                }
            }

        }
        __syncthreads();
    }

    if (gid < N) {
        block_sorts[gid] = local_vals[lid];
    }
}