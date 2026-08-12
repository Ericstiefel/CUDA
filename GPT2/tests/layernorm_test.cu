// Standalone correctness harness for layernorm.cu.
#include "../kernels.cuh"
#include <cmath>
using ln_cfg::EPS;


// Reference for one row, used to spot-check the kernel.
static void reference_row(const half* inp, const half* residual, const half* gamma,
                          const half* beta, float* sum_out, float* norm_out,
                          int row, int N) {
    for (int c = 0; c < N; ++c) {
        float x = __half2float(inp[(size_t)row * N + c]);
        if (residual) x += __half2float(residual[(size_t)row * N + c]);
        sum_out[c] = x;
    }

    float mean = 0.0f;
    for (int c = 0; c < N; ++c) mean += sum_out[c];
    mean /= (float)N;

    float var = 0.0f;
    for (int c = 0; c < N; ++c) var += (sum_out[c] - mean) * (sum_out[c] - mean);
    var /= (float)N;

    const float rstd = 1.0f / sqrtf(var + EPS);
    for (int c = 0; c < N; ++c) {
        norm_out[c] = (sum_out[c] - mean) * rstd * __half2float(gamma[c])
                    + __half2float(beta[c]);
    }
}


int main() {
    const int N = 768;    // d_model
    const int M = 4096;   // tokens

    const size_t eles = (size_t)M * N;

    half *h_inp, *h_res, *h_resout, *h_out, *h_gamma, *h_beta;
    CUDA_CHECK(cudaMallocHost(&h_inp,    sizeof(half) * eles));
    CUDA_CHECK(cudaMallocHost(&h_res,    sizeof(half) * eles));
    CUDA_CHECK(cudaMallocHost(&h_resout, sizeof(half) * eles));
    CUDA_CHECK(cudaMallocHost(&h_out,    sizeof(half) * eles));
    CUDA_CHECK(cudaMallocHost(&h_gamma,  sizeof(half) * N));
    CUDA_CHECK(cudaMallocHost(&h_beta,   sizeof(half) * N));

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            h_inp[(size_t)m * N + n] = static_cast<half>(((m + 1 + n * 2) % 17) / 19.0f);
            h_res[(size_t)m * N + n] = static_cast<half>(((m * 3 + n * 5) % 13) / 23.0f);
        }
    }
    for (int n = 0; n < N; ++n) {
        h_gamma[n] = static_cast<half>(0.8f + (n % 7) / 10.0f);
        h_beta[n]  = static_cast<half>(((n % 5) - 2) / 20.0f);
    }

    half *d_inp, *d_res, *d_resout, *d_out, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_inp,    sizeof(half) * eles));
    CUDA_CHECK(cudaMalloc(&d_res,    sizeof(half) * eles));
    CUDA_CHECK(cudaMalloc(&d_resout, sizeof(half) * eles));
    CUDA_CHECK(cudaMalloc(&d_out,    sizeof(half) * eles));
    CUDA_CHECK(cudaMalloc(&d_gamma,  sizeof(half) * N));
    CUDA_CHECK(cudaMalloc(&d_beta,   sizeof(half) * N));

    CUDA_CHECK(cudaMemcpy(d_inp,   h_inp,   sizeof(half) * eles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res,   h_res,   sizeof(half) * eles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, sizeof(half) * N,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  h_beta,  sizeof(half) * N,    cudaMemcpyHostToDevice));

    dim3 grid(M);
    dim3 block(N / 8);

    float* ref_sum;  ref_sum  = (float*)malloc(sizeof(float) * N);
    float* ref_norm; ref_norm = (float*)malloc(sizeof(float) * N);
    const int probe_rows[] = {0, 1, 7, 255, 1024, 4095};

    // Both paths matter: every LayerNorm in the model fuses the add except the very
    // first one in block 0, which passes a null residual.
    for (int with_res = 0; with_res < 2; ++with_res) {
        const half* res_in = with_res ? d_res : nullptr;

        for (int i = 0; i < 50; ++i) {
            layerNorm2D<<<grid, block>>>(d_inp, res_in, d_resout, d_out, d_gamma, d_beta, M, N);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out,    d_out,    sizeof(half) * eles, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_resout, d_resout, sizeof(half) * eles, cudaMemcpyDeviceToHost));

        float max_rel_norm = 0.0f, max_rel_sum = 0.0f;
        for (int p = 0; p < (int)(sizeof(probe_rows) / sizeof(int)); ++p) {
            const int row = probe_rows[p];
            reference_row(h_inp, with_res ? h_res : nullptr, h_gamma, h_beta,
                          ref_sum, ref_norm, row, N);

            for (int c = 0; c < N; ++c) {
                const float gn = __half2float(h_out[(size_t)row * N + c]);
                const float gs = __half2float(h_resout[(size_t)row * N + c]);
                max_rel_norm = fmaxf(max_rel_norm,
                                     fabsf(gn - ref_norm[c]) / fmaxf(fabsf(ref_norm[c]), 1e-3f));
                max_rel_sum  = fmaxf(max_rel_sum,
                                     fabsf(gs - ref_sum[c]) / fmaxf(fabsf(ref_sum[c]), 1e-3f));
            }
        }

        printf("residual %s | norm rel err %.5f (%s) | resid_out rel err %.5f (%s)\n",
               with_res ? "fused " : "absent",
               max_rel_norm, max_rel_norm < 2e-2f ? "PASS" : "FAIL",
               max_rel_sum,  max_rel_sum  < 2e-2f ? "PASS" : "FAIL");
    }

    free(ref_sum); free(ref_norm);

    CUDA_CHECK(cudaFreeHost(h_inp));  CUDA_CHECK(cudaFreeHost(h_res));
    CUDA_CHECK(cudaFreeHost(h_resout)); CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFreeHost(h_gamma)); CUDA_CHECK(cudaFreeHost(h_beta));

    CUDA_CHECK(cudaFree(d_inp));  CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_resout)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma)); CUDA_CHECK(cudaFree(d_beta));

    return 0;
}