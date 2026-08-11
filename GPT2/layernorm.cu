// I'm aware layernorm is an extremely memory bound kernel, as it's roughly the same computationally as a reduction kernel,
// in fact, we will use that technique in our kernel.

// It should obviously be fused with a consecutive kernel called before / after, but this is for eductational purposes,
// and that's a relatively trivial fuse.

// Layernorm takes each element and (xij - meani) / (std_dev_i + EPS), EPS for numerical stability.

// We can fast track the std_dev calc by using E[(X = E[X])^2] = var = E[X^2] - E[X]^2 (simply expanding square and combining).
// So, we will simultaneously do 2 sum reductions, xij for the mean calculation and xij^2 for the variance calculation,
// enabling us to do this all in one pass.

// For simplicity, we are launching 1 block per row, each thread loading 8 elements at a time, as otherwise we would have to cast the results globally,
// and we assume N (M x N matrix) is divisible by 8.

// d_model = N = 768, so we're going to launch 96 threads per block, each doing the full work of 8 half elements.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define MASK 0xffffffffu

#define EPS 1e-6f


// Two floats -> one packed half2 register. F2FP.PACK_AB, a single instruction on sm_80+,
// where a convert-then-shift-then-or pack costs three.
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}


__device__ __forceinline__ void warp_sum_red(float& reg_sum, float& sq_sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        reg_sum += __shfl_down_sync(MASK, reg_sum, offset);
        sq_sum += __shfl_down_sync(MASK, sq_sum, offset);
    }

}



// Launch with M blocks, 96 tpb.
__global__ void layerNorm2D(const half* __restrict__ inp, half* __restrict__ out, const int M, const int N) {
    __shared__ float smem_reg[3]; __shared__ float smem_sq[3]; // 96 / 32.
    int tid = threadIdx.x; int warp = tid / 32; int lane = tid % 32;

    const float4* inp4 = reinterpret_cast<const float4*>(inp);

    const int vecs_per_row = N / 8;   // 96, so one float4 per thread covers the row
    float4 val = inp4[blockIdx.x * vecs_per_row + tid];

    float reg_sum = 0.0f; float sq_sum = 0.0f;

    const half2* h2 = reinterpret_cast<const half2*>(&val);


    float v[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i * 2 + 0] = __low2float(h2[i]);
        v[i * 2 + 1] = __high2float(h2[i]);

        reg_sum += v[i * 2 + 0]; sq_sum = fmaf(v[i * 2 + 0], v[i * 2 + 0], sq_sum);
        reg_sum += v[i * 2 + 1]; sq_sum = fmaf(v[i * 2 + 1], v[i * 2 + 1], sq_sum);
    }

    warp_sum_red(reg_sum, sq_sum);

    if (lane == 0) { smem_reg[warp] = reg_sum; smem_sq[warp] = sq_sum; }
    __syncthreads();

    if (warp == 0) {
        reg_sum = (lane < 3) ? smem_reg[lane] : 0.0f;
        sq_sum = (lane < 3) ? smem_sq[lane] : 0.0f;

        warp_sum_red(reg_sum, sq_sum);

        if (lane == 0) {

            const float mean = reg_sum / (float)N;
            const float var = sq_sum / (float)N - mean * mean;

            smem_reg[0] = mean; smem_sq[0] = 1 / sqrtf(var + EPS);
        }
    }

    __syncthreads();


    float mean = smem_reg[0]; float rstd = smem_sq[0];


    uint4 res;
    res.x = pack_half2((v[0] - mean) * rstd, (v[1] - mean) * rstd);
    res.y = pack_half2((v[2] - mean) * rstd, (v[3] - mean) * rstd);
    res.z = pack_half2((v[4] - mean) * rstd, (v[5] - mean) * rstd);
    res.w = pack_half2((v[6] - mean) * rstd, (v[7] - mean) * rstd);

    uint4* out4 = reinterpret_cast<uint4*>(out);
    out4[blockIdx.x * vecs_per_row + tid] = res;
}
