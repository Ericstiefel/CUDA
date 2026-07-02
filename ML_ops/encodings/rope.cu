#include <cuda_runtime.h>


#define ROPE_THETA 10000.0f


/*
Architecture

Input: N x K half-precision matrix, row-major, each row is one token's embedding vector (K dims).
RoPE rotates adjacent pairs (x_2i, x_2i+1) within a row by angle m * theta_i, where m is that row's
position in the sequence and theta_i = ROPE_THETA^(-2i/K).

Each block handles a pair of consecutive rows (2*blockIdx.x and 2*blockIdx.x + 1). Each thread loads
one float4 (8 halfs = 4 pairs) from each of the 2 rows at the same column offset, rotates all 8 pairs
in registers, and writes the results back in place. No shared memory needed - rotation is per-pair,
there's no cross-thread communication.

Launch Config: gridDim.x = N / 2, blockDim.x = K / 8. Requires N divisible by 2 and K divisible by 8.
*/

__device__ __forceinline__ float2 rotate_pair(float2 v, float angle) {
    float s, c;
    sincosf(angle, &s, &c); // cuda intrinsic that computes both in one shot
    return make_float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

__global__ void rope_encode(half* __restrict__ inp, const int N, const int K) {
    int lid = threadIdx.x;
    int row0 = blockIdx.x * 2;
    int row1 = row0 + 1;

    float4* row0_ptr = reinterpret_cast<float4*>(inp + row0 * K);
    float4* row1_ptr = reinterpret_cast<float4*>(inp + row1 * K);

    float4 val0 = row0_ptr[lid];
    float4 val1 = row1_ptr[lid];

    float2 row0_x = __half22float2(reinterpret_cast<half2&>(val0.x));
    float2 row0_y = __half22float2(reinterpret_cast<half2&>(val0.y));
    float2 row0_z = __half22float2(reinterpret_cast<half2&>(val0.z));
    float2 row0_w = __half22float2(reinterpret_cast<half2&>(val0.w));

    float2 row1_x = __half22float2(reinterpret_cast<half2&>(val1.x));
    float2 row1_y = __half22float2(reinterpret_cast<half2&>(val1.y));
    float2 row1_z = __half22float2(reinterpret_cast<half2&>(val1.z));
    float2 row1_w = __half22float2(reinterpret_cast<half2&>(val1.w));

    // 4 pairs live in this thread's float4 chunk, at pair indices [lid*4, lid*4 + 3]
    int pair_base = lid * 4;

    float theta_x = powf(ROPE_THETA, -2.0f * (pair_base + 0) / K);
    float theta_y = powf(ROPE_THETA, -2.0f * (pair_base + 1) / K);
    float theta_z = powf(ROPE_THETA, -2.0f * (pair_base + 2) / K);
    float theta_w = powf(ROPE_THETA, -2.0f * (pair_base + 3) / K);

    float2 out0_x = rotate_pair(row0_x, row0 * theta_x);
    float2 out0_y = rotate_pair(row0_y, row0 * theta_y);
    float2 out0_z = rotate_pair(row0_z, row0 * theta_z);
    float2 out0_w = rotate_pair(row0_w, row0 * theta_w);

    float2 out1_x = rotate_pair(row1_x, row1 * theta_x);
    float2 out1_y = rotate_pair(row1_y, row1 * theta_y);
    float2 out1_z = rotate_pair(row1_z, row1 * theta_z);
    float2 out1_w = rotate_pair(row1_w, row1 * theta_w);

    half2 packed0_x = __floats2half2_rn(out0_x.x, out0_x.y);
    half2 packed0_y = __floats2half2_rn(out0_y.x, out0_y.y);
    half2 packed0_z = __floats2half2_rn(out0_z.x, out0_z.y);
    half2 packed0_w = __floats2half2_rn(out0_w.x, out0_w.y);

    half2 packed1_x = __floats2half2_rn(out1_x.x, out1_x.y);
    half2 packed1_y = __floats2half2_rn(out1_y.x, out1_y.y);
    half2 packed1_z = __floats2half2_rn(out1_z.x, out1_z.y);
    half2 packed1_w = __floats2half2_rn(out1_w.x, out1_w.y);

    float4 res0;
    res0.x = reinterpret_cast<float&>(packed0_x);
    res0.y = reinterpret_cast<float&>(packed0_y);
    res0.z = reinterpret_cast<float&>(packed0_z);
    res0.w = reinterpret_cast<float&>(packed0_w);

    float4 res1;
    res1.x = reinterpret_cast<float&>(packed1_x);
    res1.y = reinterpret_cast<float&>(packed1_y);
    res1.z = reinterpret_cast<float&>(packed1_z);
    res1.w = reinterpret_cast<float&>(packed1_w);

    row0_ptr[lid] = res0;
    row1_ptr[lid] = res1;
}
