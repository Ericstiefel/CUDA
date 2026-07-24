// This file is for the independent softmax kernel.
// IF I wanted to (likely faster, but again this is more of an exercize than a SOL implementation),
// just using the attention kernel and taking out the final matmum would fuse the matmul with the softmax,
// but from an exercize point of view, that seems kind of pointless.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>


__device__ __forceinline__ void warp_call(float &local_sum, float &local_max) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float neighbor_max = __shfl_down_sync(mask, local_max, offset);
        float neighbor_sum = __shfl_down_sync(mask, local_sum, offset);

        float joint_max = (local_max > neighbor_max) ? local_max : neighbor_max;

        local_sum = local_sum * expf(local_max - joint_max) + neighbor_sum * expf(neighbor_max - joint_max);
        local_max = joint_max;
    }
}

/*
The consensus around SOTA softmax kernels seems to be dependnent on the row size. Perhaps, relatively independent
of row size, we can launch multiple blocks per row and combine after, but that requires a device wide threadfence,
something we can do without for smaller sizes of N (matrix MxN) by launching 1 block per row. If we're able to store
these values in 1) registers (obviously best) or 2) shared memory, we can avoid a second global read when it comes
time to write the final output.


Quick math: d_model is 768, so can we hold 768 * 2 bytes per block? register capacity is the main constraint
on warps per SM.

A4500 has 65536 4 byte registers per SM.

Register count per thread: the load is 1 float4 = 4 registers, but the second I unpack
it I'm holding 8 separate half-as-float lanes live at once = +8 registers, and the source float4 stays live while I
peel them out, so the unpack alone roughly doubles the data footprint (4 -> 12). Add local_max + local_sum (2) and
the indexing/pointer/loop junk (tid, lanes, 64-bit row pointer, temporaries ~8) and realistically ~24 regs/thread. 


Warps per SM: Max is 48. With 3 warps/block = 96 threads, register capacity now gives 65536 / (24 * 96) ~= 28 blocks
per SM. That's still above the hardware ceiling of 16 blocks / 48 warps per SM, so
we stay at full occupancy, it doesn't cost us any warps here. So assuming (hyperparam) 
3 warps per block, each SM holds the full 48 warps and be distributed among ceil(7.1111) = 8 SMs out of our available 56.

I came up with the 3 as 768 / 8 (half values per 16 byte load) / 32 = 3 warps, so we're loading 1 float4 per thread.
*/


__device__ __forceinline__ void unpack(float grouped_val, float &lower, float &upper) {
    unsigned int bits = __float_as_uint(grouped_val);
    lower = __ushort_as_half(static_cast<unsigned short>(bits & 0xffff));
    upper = __ushort_as_half(static_cast<unsigned short>(bits >> 16));
}


__device__ __forceinline__ float pack(float lower, float upper) {
    unsigned int lo = __half_as_ushort(__float2half(lower));
    unsigned int hi = __half_as_ushort(__float2half(upper));
    return __uint_as_float(lo | (hi << 16));
}


// This kernel is specifically made for our dims, do not use it for other sizes (no bounds checks and will give wrong results)
__global__ void softmax(const half* __restrict__ inp, half* __restrict__ out, const int N) {
    __shared__ float max_red[32];
    __shared__ float sum_red[32];
    int tid = threadIdx.x;
    int warp_num = tid / 32;
    int warp_lane = tid % 32;


    const float4* row_start = reinterpret_cast<const float4*>(inp + blockIdx.y * N);

    float4 val = row_start[tid];

    float val1, val2, val3, val4, val5, val6, val7, val8;
    unpack(val.x, val1, val2);
    unpack(val.y, val3, val4);
    unpack(val.z, val5, val6);
    unpack(val.w, val7, val8);


    float local_max = (val1 > val2) ? val1 : val2;
    local_max = (val3 > local_max) ? val3 : local_max; local_max = (val4 > local_max) ? val4 : local_max;
    local_max = (val5 > local_max) ? val5 : local_max; local_max = (val6 > local_max) ? val6 : local_max;
    local_max = (val7 > local_max) ? val7 : local_max; local_max = (val8 > local_max) ? val8 : local_max;

    float local_sum = expf(val1 - local_max) + expf(val2 - local_max)
                    + expf(val3 - local_max) + expf(val4 - local_max)
                    + expf(val5 - local_max) + expf(val6 - local_max)
                    + expf(val7 - local_max) + expf(val8 - local_max);

    warp_call(local_sum, local_max);

    if (warp_lane == 0) {
        max_red[warp_num] = local_max;
        sum_red[warp_num] = local_sum;   
    }
    __syncthreads(); 

    if (warp_num == 0) {

        float tmp_max = (warp_lane < 3) ? max_red[warp_lane] : -FLT_MAX;
        float tmp_sum = (warp_lane < 3) ? sum_red[warp_lane] : 0.0f;
        warp_call(tmp_sum, tmp_max);

        if (warp_lane == 0) {
            max_red[0] = tmp_max;
            sum_red[0] = tmp_sum;
        }
    }
    __syncthreads();


    float block_max = max_red[0];
    float inv_sum   = 1.0f / sum_red[0];

    float4* out_row = reinterpret_cast<float4*>(out + blockIdx.y * N);
    float4 res;
    res.x = pack(expf(val1 - block_max) * inv_sum, expf(val2 - block_max) * inv_sum);
    res.y = pack(expf(val3 - block_max) * inv_sum, expf(val4 - block_max) * inv_sum);
    res.z = pack(expf(val5 - block_max) * inv_sum, expf(val6 - block_max) * inv_sum);
    res.w = pack(expf(val7 - block_max) * inv_sum, expf(val8 - block_max) * inv_sum);
    out_row[tid] = res;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
    } \
} while (0)

int main() {
    int M = 4096; // Input tokens
    int N = 768; // length of embeddings

    half* h_inp; CUDA_CHECK(cudaMallocHost(&h_inp, sizeof(half) * M * N));
    half* h_out; CUDA_CHECK(cudaMallocHost(&h_out, sizeof(half) * M * N));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            h_inp[m * N + n] = static_cast<half>(((m + 1 + n * 2) % 17) / 19.0f);
        }
    }

    half* d_inp; CUDA_CHECK(cudaMalloc(&d_inp, sizeof(half) * M * N));
    half* d_out; CUDA_CHECK(cudaMalloc(&d_out, sizeof(half) * M * N));

    CUDA_CHECK(cudaMemcpy(d_inp, h_inp, sizeof(half) * M * N, cudaMemcpyHostToDevice));

    dim3 grid(1, M);   
    dim3 block(96);    

    for (int i = 0; i < 50; ++i) {
        softmax<<<grid, block>>>(d_inp, d_out, N); // warming up
    }
    CUDA_CHECK(cudaGetLastError());       
    CUDA_CHECK(cudaDeviceSynchronize());  

    CUDA_CHECK(cudaProfilerStart());
    softmax<<<grid, block>>>(d_inp, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(half) * M * N, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeHost(h_inp));
    CUDA_CHECK(cudaFreeHost(h_out));

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}