// Standalone correctness + benchmark harness for softmax.cu.
#include "../kernels.cuh"
#include <cuda_profiler_api.h>
#include <cmath>
#include <cfloat>
using namespace sm_cfg;


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