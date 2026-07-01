#include <cuda_runtime.h>


#define EPS 1e-7f


/*
Architecture

1 block per row. 
Reducing sums, thread 0 per block broadcasts values to operate on to block.
Use Variance = E[(X - E[X])^2] = E[X^2] - E[X]^2, so we don't require the mean before the final broadcast.
*/


__device__ __forceinline__ void warp_reductions(float& l_sum, float& l_ssum) {
    unsigned int mask = 0xffffffff;

    for (int stride = 16; stride > 0; stride /= 2) {
        float n_l_sum = __shfl_down_sync(mask, l_sum, stride);
        float n_l_dsum = __shfl_down_sync(mask, l_ssum, stride);

        l_sum += n_l_sum;
        l_ssum += n_l_dsum;
    }
} 

// Launch Config: 1 block per row, arbitrary thread count (divisible by 32), say 256 (that's what this is designed for)
// Can expand small to hold arbitrary number of float4's to avoid reading gmem again.
// Registers per sm obviously varies per GPU, so you can even have each thread storing say, 10 float4's, or 80 half values in registers.
// For this specific configuration, the N dim must be divisible by 8 and <= 256 * 8 = 2048.

__global__ void layernorm2d_small(const half* __restrict__ inp, half* __restrict__ out_ptr, const int M, const int N) {
    __shared__ float smem1[8]; // going to be reduced by warp 0
    __shared__ float smem2[8]; 

    int lid = threadIdx.x;
    int warp_id = lid / 32;
    int warp_lane = lid % 32;

    bool valid = (lid * 8 < N);

    float l_sum = 0.0f;
    float l_ssum = 0.0f;

    float4 comp_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid) {
        float4* gmem_row_start = reinterpret_cast<float4*>(inp + blockIdx.x * N);
        comp_val = gmem_row_start[lid];
    }

    float2 inp_x = __half22float2(reinterpret_cast<half2&>(comp_val.x));
    float inp_x_low = inp_x.x; float inp_x_up = inp_x.y;

    float2 inp_y = __half22float2(reinterpret_cast<half2&>(comp_val.y));
    float inp_y_low = inp_y.x; float inp_y_up = inp_y.y;

    float2 inp_z = __half22float2(reinterpret_cast<half2&>(comp_val.z));
    float inp_z_low = inp_z.x; float inp_z_up = inp_z.y;

    float2 inp_w = __half22float2(reinterpret_cast<half2&>(comp_val.w));
    float inp_w_low = inp_w.x; float inp_w_up = inp_w.y;

    l_sum += inp_x_low; l_sum += inp_x_up;
    l_ssum += inp_x_low * inp_x_low; l_ssum += inp_x_up * inp_x_up;

    l_sum += inp_y_low; l_sum += inp_y_up;
    l_ssum += inp_y_low * inp_y_low; l_ssum += inp_y_up * inp_y_up;

    l_sum += inp_z_low; l_sum += inp_z_up;
    l_ssum += inp_z_low * inp_z_low; l_ssum += inp_z_up * inp_z_up;

    l_sum += inp_w_low; l_sum += inp_w_up;
    l_ssum += inp_w_low * inp_w_low; l_ssum += inp_w_up * inp_w_up;

    warp_reductions(l_sum, l_ssum);

    if (warp_lane == 0) { smem1[warp_id] = l_sum; smem2[warp_id] = l_ssum; }

    __syncthreads();

    if (warp_id == 0) {
        float val1 = (warp_lane < 8) ? smem1[warp_lane] : 0.0f;
        float val2 = (warp_lane < 8) ? smem2[warp_lane] : 0.0f;
        warp_reductions(val1, val2);

        if (warp_lane == 0) {
            smem1[0] = val1 / N; smem2[0] = val2 / N;
        }
    }
    __syncthreads();

    float mean = smem1[0]; float var = smem2[0] - mean * mean;
    float den = 1 / sqrtf(var + EPS); // precompute this component

    // Pack the result back into a float4

    float4 out;

    half2 out_x = __floats2half2_rn((inp_x_low - mean) * den, (inp_x_up - mean) * den);
    out.x = reinterpret_cast<float&>(out_x);

    half2 out_y = __floats2half2_rn((inp_y_low - mean) * den, (inp_y_up - mean) * den);
    out.y = reinterpret_cast<float&>(out_y);

    half2 out_z = __floats2half2_rn((inp_z_low - mean) * den, (inp_z_up - mean) * den);
    out.z = reinterpret_cast<float&>(out_z);

    half2 out_w = __floats2half2_rn((inp_w_low - mean) * den, (inp_w_up - mean) * den);
    out.w = reinterpret_cast<float&>(out_w);


    if (valid) {
        float4* out_4 = reinterpret_cast<float4*>(out_ptr);
        out_4[blockIdx.x * N / 8 + lid] = out;
    }


    

}