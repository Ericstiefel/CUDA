// I'm aware layernorm is an extremely memory bound kernel, as it's roughly the same computationally as a reduction kernel,
// in fact, we will use that technique in our kernel.

// Layernorm takes each element and (xij - meani) / (std_dev_i + EPS), EPS for numerical stability,
// then scales and shifts by the learned per-feature gamma and beta.

// We can fast track the std_dev calc by using E[(X = E[X])^2] = var = E[X^2] - E[X]^2 (simply expanding square and combining).
// So, we will simultaneously do 2 sum reductions, xij for the mean calculation and xij^2 for the variance calculation,
// enabling us to do this all in one pass.

// For simplicity, we are launching 1 block per row, each thread loading 8 elements at a time, as otherwise we would have to cast the results globally,
// and we assume N (M x N matrix) is divisible by 8.

// d_model = N = 768, so we're going to launch 96 threads per block, each doing the full work of 8 half elements.

// The residual add is fused into the front. In a pre-norm block the stream is
//     h = x + Attn(LN(x))        then        y = h + FFN(LN(h))
// so the add that closes one sub-layer is immediately followed by the norm that opens
// the next. Doing them in one kernel saves a whole read+write of the residual stream,
// and it is why this kernel has two outputs: `out` is the normalised tensor the next
// GEMM consumes, and `resid_out` is the un-normalised sum, which is what the *next*
// residual add needs. Only the very first LayerNorm of block 0 has nothing to add to,
// which is what the null `residual` case covers.

#include "kernels.cuh"

// Shared with the test harness so the reference cannot drift from the kernel.
using ln_cfg::EPS;


__device__ __forceinline__ void warp_sum_red(float& reg_sum, float& sq_sum) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        reg_sum += __shfl_down_sync(FULL_MASK, reg_sum, offset);
        sq_sum += __shfl_down_sync(FULL_MASK, sq_sum, offset);
    }

}



// Launch with M blocks, 96 tpb.
//   inp       - this sub-layer's output (or the embeddings, for the first call)
//   residual  - the incoming residual stream, or nullptr when there is nothing to add
//   resid_out - inp + residual, written for the next residual add; nullptr to skip
//   out       - normalise(inp + residual) * gamma + beta
__global__ void layerNorm2D(const half* __restrict__ inp, const half* __restrict__ residual,
                            half* __restrict__ resid_out, half* __restrict__ out,
                            const half* __restrict__ gamma, const half* __restrict__ beta,
                            const int M, const int N) {
    __shared__ float smem_reg[3]; __shared__ float smem_sq[3]; // 96 / 32.
    int tid = threadIdx.x; int warp = tid / 32; int lane = tid % 32;

    const int vecs_per_row = N / 8;   // 96, so one float4 per thread covers the row
    const size_t off = (size_t)blockIdx.x * vecs_per_row + tid;

    float4 val = reinterpret_cast<const float4*>(inp)[off];
    const half2* h2 = reinterpret_cast<const half2*>(&val);

    float v[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i * 2 + 0] = __low2float(h2[i]);
        v[i * 2 + 1] = __high2float(h2[i]);
    }

    // The branch is uniform across the whole grid, so it costs a predicate and nothing else.
    if (residual != nullptr) {
        float4 rval = reinterpret_cast<const float4*>(residual)[off];
        const half2* r2 = reinterpret_cast<const half2*>(&rval);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i * 2 + 0] += __low2float(r2[i]);
            v[i * 2 + 1] += __high2float(r2[i]);
        }
    }

    // The sum leaves before the statistics do: downstream needs it un-normalised, and it
    // is already sitting in registers.
    if (resid_out != nullptr) {
        uint4 rs;
        rs.x = pack_half2(v[0], v[1]);
        rs.y = pack_half2(v[2], v[3]);
        rs.z = pack_half2(v[4], v[5]);
        rs.w = pack_half2(v[6], v[7]);
        reinterpret_cast<uint4*>(resid_out)[off] = rs;
    }

    float reg_sum = 0.0f; float sq_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        reg_sum += v[i];
        sq_sum += v[i] * v[i];
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

    // gamma and beta are indexed by column only, so every block reads the same 1.5 KB and
    // L2 serves all but the first.
    float4 gval = reinterpret_cast<const float4*>(gamma)[tid];
    float4 bval = reinterpret_cast<const float4*>(beta)[tid];
    const half2* g2 = reinterpret_cast<const half2*>(&gval);
    const half2* b2 = reinterpret_cast<const half2*>(&bval);

    float g[8]; float b[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        g[i * 2 + 0] = __low2float(g2[i]);  g[i * 2 + 1] = __high2float(g2[i]);
        b[i * 2 + 0] = __low2float(b2[i]);  b[i * 2 + 1] = __high2float(b2[i]);
    }

    uint4 res;
    res.x = pack_half2((v[0] - mean) * rstd * g[0] + b[0], (v[1] - mean) * rstd * g[1] + b[1]);
    res.y = pack_half2((v[2] - mean) * rstd * g[2] + b[2], (v[3] - mean) * rstd * g[3] + b[3]);
    res.z = pack_half2((v[4] - mean) * rstd * g[4] + b[4], (v[5] - mean) * rstd * g[5] + b[5]);
    res.w = pack_half2((v[6] - mean) * rstd * g[6] + b[6], (v[7] - mean) * rstd * g[7] + b[7]);

    reinterpret_cast<uint4*>(out)[off] = res;
}
