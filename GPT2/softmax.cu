// This file is for the independent softmax kernel.
// IF I wanted to (likely faster, but again this is more of an exercize than a SOL implementation),
// just using the attention kernel and taking out the final matmum would fuse the matmul with the softmax,
// but from an exercize point of view, that seems kind of pointless.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// accumulate sums in floats, only convert max for the final computation (outside of this kernel)
__device__ __forceinline__ void warp_call(float &local_sum, half &local_max) {
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        half neighbor_max = __shfl_down_sync(mask, local_max, offset);
        float neightbor_sum = __shfl_down_sync(mask, local_sum, offset);

        half joint_max = (local_max > neighbor_max) ? local_max : negihbor_max;

        local_sum = local_sum * hexp(local_max - joint_max) + neighbor_sum * hexp(neighbor_max - joint_max);
        local_max = joint_max; local_sum = joint_sum;
    }
}


__device__ void softmax(const half* __restrict__ inp, half* __restrict__ out, const int M, const int N) {
    
}