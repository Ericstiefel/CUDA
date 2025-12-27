#include <cuda_runtime.h>



#define EPSILLON 1e-7


__device__ __forceinline__ void warp_reduce_sum(const float val) {
    unsigned mask = 0xffffffffu;
    for (int i = 0; i < 16; i << 1) {
        val += __shfl_down_sync(mask, val, i);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float vals[32]; // 1024 / 32 = 32.
    
    int warp_lane = threadIdx.x % 32;
    int warp_num = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (warp_lane == 0) { 
        vals[warp_num] = val;
    }
    if (warp_num == 0) {
        val = vals[warp_lane];
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void layerNorm(
    const float* __restrict__ inp,
    float* __restrict__ out,
    const int M,
    const int N
) {
    __shared__ float mean; 
    __shared__ float std;

    int num_vecs = N / 4; // Assumes N is divisible by 4.

    const float* begin_row = inp + blockIdx.x * N;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    const float4* ptr = reinterpret_cast<const float4*>(begin_row) + threadIdx.x; // adds in strides of 4.

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        float4 v = ptr[i];

        sum += v.x + v.y + v.z + v.w;
        sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w + v.w;
    }

    sum = block_reduce_sum(sum);
    sq_sum = block_reduce_sum(sq_sum);

    if (threadIdx.x == 0) {
        mean = sum / N;
        float var = (sq_sum / N) - (mean * mean);
        std = rsqrtf(var + EPSILLON);
    }

    __syncthreads();

    float4* out_ptr = reinterpret_cast<float4*>(out) + (N * blockIdx.x);

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        float4 v = out_ptr + i;
        float4 out;

        float out.x = (ptr[i].x - mean) / std;
        float out.y = (ptr[i].y - mean) / std;
        float out.z = (ptr[i].z - mean) / std;
        float out.w = (ptr[i].w - mean) / std;

        put_ptr[i] = out;

    }




    
}