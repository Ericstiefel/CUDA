"""
Stream Compaction removes unwanted elements from your input.

This version will keep any number greater than p.

Effectively like Radix Sort (Prefix Sum, GLobalize, Scatter), but with removing elements.
"""

#include <cuda_runtime.h>


// Each thread is assigned to one value in the input
__global__ void block_prefix_sum(
    const int* __restrict__ inp,
    int* __restrict__ block_histograms,
    int* __restrict__ block_offsets,
    int N,
    int p)
{
    extern __shared__ int vals[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    int block_count = blockDim.x

    vals[lid] = (gid < N && inp[gid] > p) ? 1 : 0;

    __syncthreads();

    for (int stride = 1; stride < block_count; stride *= 2) {
        int idx = (lid + 1)*stride*2 - 1;
        if (idx < block_count)
            vals[idx] += vals[idx - stride];
        __syncthreads();
    }

    if (lid == block_count - 1)
        vals[lid] = 0;

    __syncthreads();

    for (int stride = block_count/2; stride > 0; stride /= 2) {
        int idx = (lid + 1)*stride*2 - 1;
        if (idx < block_count) {
            int t = vals[idx - stride];
            vals[idx - stride] = vals[idx];
            vals[idx] += t;
        }
        __syncthreads();
    }

    if (gid < N)
        block_histograms[gid] = vals[lid];

    if (lid == block_count - 1)
        block_offsets[blockIdx.x] = vals[lid] + (inp[gid] == p ? 1 : 0);
}


// Launch with 1 block, num_blocks tpb.
__global__ void block_offset_scan(
    int* __restrict__ block_offsets,
    int num_blocks)
{
    extern __shared__ int vals[]; // sizeof(int) * num_blocks

    int lid = threadIdx.x;


    vals[lid] = block_offsets[lid]; 

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int idx = (lid + 1)*stride*2 - 1;
        if (idx - stride >= 0)
            vals[idx] += vals[idx - stride];
        __syncthreads();
    }

    if (lid == blockDim.x - 1)
        vals[lid] = 0;

    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        int idx = (lid + 1)*stride*2 - 1;
        if (idx < blockDim.x) {
            int t = vals[idx - stride];
            vals[idx - stride] = vals[idx];
            vals[idx] += t;
        }
        __syncthreads();
    }

    block_offsets[lid] = vals[lid];
}



__global__ void place_coalesced(
    const int* __restrict__ inp,
    const int* __restrict__ block_local_prefix, // length N, local indices within block
    const int* __restrict__ block_offsets,      // length num_blocks
    int* __restrict__ out,
    int N,
    int p)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    if (inp[gid] > p) {
        int local_pos = block_local_prefix[gid];          
        int base = block_offsets[blockIdx.x];             
        int dest = base + local_pos;
        out[dest] = inp[gid];
    }
}
