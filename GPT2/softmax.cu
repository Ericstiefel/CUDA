// Standalone Softmax kernel

// This will not work for alternative sizes, curated specifically for our inference purposes.

#include "kernels.cuh"
#include <cuda_profiler_api.h>
#include <cfloat>

using namespace sm_cfg;

// Every lane will hold correct scanned results
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL_MASK, v, off));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL_MASK, v, off);
    return v;
}


// This is domain specific, so we know the input sizes.
__global__ void softmax(const half* __restrict__ inp, half* __restrict__ out,
                        const int N, const int V) {
    __shared__ float red_max[WARPS];
    __shared__ float red_sum[WARPS];

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    const int vecs = N / 8;
    const float4* row = reinterpret_cast<const float4*>(inp) + (size_t)blockIdx.x * vecs;

    // We have the capacity to store (based on A4500 regs / SM) roughly 2 blocks per SM. After profiling, it's 1, and is the reason bandwidth is at 52% despite perfect coalecing.
    float4 v[VPT];
    float local_max = -FLT_MAX;

    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = tid + i * BLOCK;
        if (idx < vecs) {
            v[i] = row[idx];
            const half2* h = reinterpret_cast<const half2*>(&v[i]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = idx * 8 + j * 2;
                const float a = (col     < V) ? __low2float(h[j])  : -FLT_MAX;
                const float b = (col + 1 < V) ? __high2float(h[j]) : -FLT_MAX;
                local_max = fmaxf(local_max, fmaxf(a, b));
            }
        }
    }

    local_max = warp_max(local_max);
    if (lane == 0) red_max[warp] = local_max;
    __syncthreads();
    const float row_max = warp_max(red_max[lane]);

    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = tid + i * BLOCK;
        if (idx < vecs) {
            const half2* h = reinterpret_cast<const half2*>(&v[i]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = idx * 8 + j * 2;
                if (col     < V) local_sum += expf(__low2float(h[j])  - row_max);
                if (col + 1 < V) local_sum += expf(__high2float(h[j]) - row_max);
            }
        }
    }

    local_sum = warp_sum(local_sum);
    if (lane == 0) red_sum[warp] = local_sum;
    __syncthreads();
    const float inv_sum = 1.0f / warp_sum(red_sum[lane]);


    uint4* orow = reinterpret_cast<uint4*>(out) + blockIdx.x * vecs;

    #pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = tid + i * BLOCK;
        if (idx < vecs) {
            const half2* h = reinterpret_cast<const half2*>(&v[i]);
            uint4 r;
            uint32_t* w = reinterpret_cast<uint32_t*>(&r);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = idx * 8 + j * 2;
                const float a = (col     < V) ? expf(__low2float(h[j])  - row_max) * inv_sum : 0.0f;
                const float b = (col + 1 < V) ? expf(__high2float(h[j]) - row_max) * inv_sum : 0.0f;
                w[j] = pack_half2(a, b);
            }
            orow[idx] = r;
        }
    }
}
