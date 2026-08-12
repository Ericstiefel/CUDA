// Standalone correctness + benchmark harness for gemm.cu.
#include "../kernels.cuh"
#include <cuda_profiler_api.h>
#include <cmath>
using namespace gemm_cfg;


int main() {
    // Grab at least one of the dims of the matmuls we're going to use the matmul for.
    // Must be exact multiples of the block tiles (no boundary masking in the kernel):
    //   M % BM(128) == 0,  N % BN(128) == 0,  K % BK(32) == 0
    int M = 4096;
    int N = 4096;
    int K = 1024;

    half *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, sizeof(half) * M * K));
    CUDA_CHECK(cudaMallocHost(&h_B, sizeof(half) * K * N));
    CUDA_CHECK(cudaMallocHost(&h_C, sizeof(half) * M * N));

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(half) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(half) * K * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(half) * M * N));


    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < K; ++n) {
            h_A[m * K + n] = static_cast<half>(((m + 1 + n * 2) % 17) / 19.0f);
        }
    }

    for (int m = 0; m < K; ++m) {
        for (int n = 0; n < N; ++n) {
            h_B[m * N + n] = static_cast<half>(((m + 1 + n * 2) % 17) / 21.0f);
        }
    }


    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(half) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(half) * K * N, cudaMemcpyHostToDevice));

    dim3 tpb(256);                    // 8 warps (2 x 4 warp grid) -> fixed by the kernel
    dim3 bpg(N / BN, M / BM);         // x -> N tiles, y -> M tiles

    for (int i = 0; i < 50; ++i) {
        gemm<<<bpg, tpb>>>(d_A, d_B, d_C, M, K, N); // warming up  (grid, block)
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStart());


    gemm<<<bpg, tpb>>>(d_A, d_B, d_C, M, K, N);   // grid, block
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaProfilerStop());


    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(half) * M * N, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Spot-check both entry points against a CPU reference. fp16 operands with fp32
    // accumulation over K=1024 lands around 1e-3 relative.
    const int probe_rows[] = {0, 1, 127, 128, 2047, 4095};
    const int probe_cols[] = {0, 1, 63, 128, 1023, 4095};

    const int d_head = 128;                // matches attn::BN
    const char* names[] = {"gemm", "gemm_gelu", "gemm_qkv_rope"};

    for (int variant = 0; variant < 3; ++variant) {
        if (variant == 1) gemm_gelu<<<bpg, tpb>>>(d_A, d_B, d_C, M, K, N);
        if (variant == 2) gemm_qkv_rope<<<bpg, tpb>>>(d_A, d_B, d_C, M, K, N, d_head);
        if (variant) {
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(half) * M * N, cudaMemcpyDeviceToHost));
        }

        float max_rel = 0.0f;
        for (int r = 0; r < (int)(sizeof(probe_rows) / sizeof(int)); ++r) {
            for (int c = 0; c < (int)(sizeof(probe_cols) / sizeof(int)); ++c) {
                const int row = probe_rows[r], col = probe_cols[c];

                // The RoPE epilogue mixes column pairs, so the reference needs both
                // halves of the pair. col is even for every probe that matters; for an
                // odd probe the partner is the column below.
                const int c_even = col & ~1;

                float acc0 = 0.0f, acc1 = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float a = __half2float(h_A[row * K + k]);
                    acc0 += a * __half2float(h_B[k * N + c_even]);
                    acc1 += a * __half2float(h_B[k * N + c_even + 1]);
                }

                float want;
                if (variant == 0) {
                    want = (col == c_even) ? acc0 : acc1;
                } else if (variant == 1) {
                    const float x = (col == c_even) ? acc0 : acc1;
                    const float kc = 0.7978845608028654f;
                    want = 0.5f * x * (1.0f + tanhf(kc * (x + 0.044715f * x * x * x)));
                } else {
                    const int d_model = N / 3;
                    if (c_even < 2 * d_model) {
                        const int pair = (c_even % d_head) / 2;
                        const float inv_freq = powf(10000.0f, -(float)(2 * pair) / (float)d_head);
                        const float th = (float)row * inv_freq;
                        want = (col == c_even) ? acc0 * cosf(th) - acc1 * sinf(th)
                                               : acc0 * sinf(th) + acc1 * cosf(th);
                    } else {
                        want = (col == c_even) ? acc0 : acc1;   // V is not rotated
                    }
                }

                const float got = __half2float(h_C[row * N + col]);
                max_rel = fmaxf(max_rel, fabsf(got - want) / fmaxf(fabsf(want), 1e-2f));
            }
        }
        printf("%-13s max relative error: %.5f  (%s)\n", names[variant],
               max_rel, max_rel < 2e-2f ? "PASS" : "FAIL");
    }



    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}