// Standalone Softmax kernel

// This will not work for alternative sizes, curated specifically for our inference purposes.

#include "common.cuh"
#include <cuda_profiler_api.h>
#include <cmath>
#include <cfloat>

// Anonymous namespace so these names stay local to this translation unit; WARPS in
// particular means something different in attention.cu.
namespace {
constexpr int BLOCK = 1024;         // threads per block, one block per row
constexpr int WARPS = BLOCK / 32;   // 32, which is exactly the lane count of one warp
constexpr int VPT   = 7;            // float4s per thread; 1024 * 7 * 8 = 57344 > 50304
}

// Butterfly rather than shfl_down: every lane ends up holding the result, so the block
// reduction below needs no broadcast step.
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


#ifndef GPT2_NO_MAIN

static void reference_row(const half* inp, float* probs, int row, int N, int V) {
    const half* r = inp + (size_t)row * N;

    float m = -FLT_MAX;
    for (int c = 0; c < V; ++c) m = fmaxf(m, __half2float(r[c]));

    float s = 0.0f;
    for (int c = 0; c < V; ++c) {
        probs[c] = expf(__half2float(r[c]) - m);
        s += probs[c];
    }
    for (int c = 0; c < V; ++c) probs[c] /= s;
}


int main() {
    const int V = 50257;   // GPT-2 BPE vocabulary
    const int N = 50304;   // padded to a multiple of 128 for the GEMM tiling
    const int M = 512;     // rows of logits

    if (N / 8 > BLOCK * VPT) { fprintf(stderr, "row too wide for BLOCK * VPT\n"); return 1; }

    half* h_inp; CUDA_CHECK(cudaMallocHost(&h_inp, sizeof(half) * (size_t)M * N));
    half* h_out; CUDA_CHECK(cudaMallocHost(&h_out, sizeof(half) * (size_t)M * N));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            h_inp[(size_t)m * N + n] = static_cast<half>(((m + 1 + n * 2) % 17) / 19.0f);
        }
    }

    half* d_inp; CUDA_CHECK(cudaMalloc(&d_inp, sizeof(half) * (size_t)M * N));
    half* d_out; CUDA_CHECK(cudaMalloc(&d_out, sizeof(half) * (size_t)M * N));

    CUDA_CHECK(cudaMemcpy(d_inp, h_inp, sizeof(half) * (size_t)M * N, cudaMemcpyHostToDevice));

    dim3 grid(M);
    dim3 block(BLOCK);

    for (int i = 0; i < 50; ++i) {
        softmax<<<grid, block>>>(d_inp, d_out, N, V); // warming up
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStart());
    softmax<<<grid, block>>>(d_inp, d_out, N, V);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(half) * (size_t)M * N, cudaMemcpyDeviceToHost));

    float* ref = (float*)malloc(sizeof(float) * V);

    float max_rel = 0.0f;
    double max_sum_err = 0.0;
    const int probe_rows[] = {0, 1, 17, 255, 511};

    for (int p = 0; p < (int)(sizeof(probe_rows) / sizeof(int)); ++p) {
        const int row = probe_rows[p];
        reference_row(h_inp, ref, row, N, V);

        double got_sum = 0.0;
        for (int c = 0; c < V; ++c) {
            const float got = __half2float(h_out[(size_t)row * N + c]);
            got_sum += got;
            const float rel = fabsf(got - ref[c]) / fmaxf(ref[c], 1e-9f);
            max_rel = fmaxf(max_rel, rel);
        }
        max_sum_err = fmax(max_sum_err, fabs(got_sum - 1.0));

        // Padding columns must be exactly zero, never garbage from the GEMM.
        for (int c = V; c < N; ++c) {
            if (__half2float(h_out[(size_t)row * N + c]) != 0.0f) {
                printf("padding column %d in row %d is nonzero\n", c, row);
            }
        }
    }

    printf("max relative error vs reference: %.5f  (%s)\n",
           max_rel, max_rel < 5e-2f ? "PASS" : "FAIL");
    printf("worst |sum(probs) - 1|:          %.5f  (%s)\n",
           max_sum_err, max_sum_err < 5e-2 ? "PASS" : "FAIL");


    free(ref);

    CUDA_CHECK(cudaFreeHost(h_inp));
    CUDA_CHECK(cudaFreeHost(h_out));

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}

#endif // GPT2_NO_MAIN
