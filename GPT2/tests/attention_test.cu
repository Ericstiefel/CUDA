// Standalone correctness + benchmark harness for attention.cu.
#include "../kernels.cuh"
#include <cuda_profiler_api.h>
#include <cmath>
#include <cfloat>
using namespace attn;


// Reference attention for a single output row, used to spot-check the kernel.
static void reference_row(const half* Q, const half* K, const half* V, float* o_row,
                          int plane, int row, int seq, int d_head) {
    const size_t off = (size_t)plane * seq * d_head;
    const half* q = Q + off + (size_t)row * d_head;
    const half* k = K + off;
    const half* v = V + off;

    const float scale = 1.0f / sqrtf((float)d_head);
    float* s = (float*)malloc(sizeof(float) * seq);

    // Causal: row r attends to keys 0..r inclusive and nothing beyond.
    float m = -FLT_MAX;
    for (int j = 0; j <= row; ++j) {
        float dot = 0.0f;
        for (int c = 0; c < d_head; ++c) {
            dot += __half2float(q[c]) * __half2float(k[(size_t)j * d_head + c]);
        }
        s[j] = dot * scale;
        m = fmaxf(m, s[j]);
    }

    float l = 0.0f;
    for (int c = 0; c < d_head; ++c) o_row[c] = 0.0f;
    for (int j = 0; j <= row; ++j) {
        const float p = expf(s[j] - m);
        l += p;
        for (int c = 0; c < d_head; ++c) {
            o_row[c] += p * __half2float(v[(size_t)j * d_head + c]);
        }
    }
    for (int c = 0; c < d_head; ++c) o_row[c] /= l;

    free(s);
}


// Exercises the layout the model actually uses: Q/K/V interleaved as [T, 3*d_model]
// straight out of the projection GEMM, output written as [T, d_model] ready for the
// output projection. No split, no merge -- only the stride arguments change.
static void check_packed_layout() {
    const int heads   = 6;
    const int d_head  = BN;                  // 128
    const int seq     = 256;                 // % BM and % BK
    const int d_model = heads * d_head;      // 768
    const int qkv_cols = 3 * d_model;        // 2304

    const size_t qkv_eles = (size_t)seq * qkv_cols;
    const size_t out_eles = (size_t)seq * d_model;

    half *h_qkv, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_qkv, sizeof(half) * qkv_eles));
    CUDA_CHECK(cudaMallocHost(&h_out, sizeof(half) * out_eles));
    for (size_t i = 0; i < qkv_eles; ++i) {
        h_qkv[i] = static_cast<half>((float)((i * 11 + 3) % 17) / 23.0f);
    }

    half *d_qkv, *d_out;
    CUDA_CHECK(cudaMalloc(&d_qkv, sizeof(half) * qkv_eles));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(half) * out_eles));
    CUDA_CHECK(cudaMemcpy(d_qkv, h_qkv, sizeof(half) * qkv_eles, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaFuncSetAttribute(flash_attention,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)SMEM_BYTES));

    flash_attention<<<dim3(heads, seq / BM), dim3(THREADS), SMEM_BYTES>>>(
        d_qkv, d_qkv + d_model, d_qkv + 2 * d_model, d_out,
        seq, seq, d_head,
        qkv_cols, d_model,      // row strides in / out
        d_head, d_head);        // plane strides in / out
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(half) * out_eles, cudaMemcpyDeviceToHost));

    const float scale = 1.0f / sqrtf((float)d_head);
    float* s = (float*)malloc(sizeof(float) * seq);
    float max_rel = 0.0f;
    const int probe_rows[] = {0, 1, 63, 128, 255};

    for (int h = 0; h < heads; h += 2) {
        for (int p = 0; p < (int)(sizeof(probe_rows) / sizeof(int)); ++p) {
            const int r = probe_rows[p];
            const half* q = h_qkv + (size_t)r * qkv_cols + h * d_head;

            float m = -FLT_MAX;
            for (int j = 0; j <= r; ++j) {
                const half* k = h_qkv + (size_t)j * qkv_cols + d_model + h * d_head;
                float dot = 0.0f;
                for (int c = 0; c < d_head; ++c) dot += __half2float(q[c]) * __half2float(k[c]);
                s[j] = dot * scale;
                m = fmaxf(m, s[j]);
            }

            float l = 0.0f;
            float acc[128] = {0.0f};
            for (int j = 0; j <= r; ++j) {
                const half* v = h_qkv + (size_t)j * qkv_cols + 2 * d_model + h * d_head;
                const float pw = expf(s[j] - m);
                l += pw;
                for (int c = 0; c < d_head; ++c) acc[c] += pw * __half2float(v[c]);
            }

            for (int c = 0; c < d_head; ++c) {
                const float want = acc[c] / l;
                const float got = __half2float(h_out[(size_t)r * d_model + h * d_head + c]);
                max_rel = fmaxf(max_rel, fabsf(got - want) / fmaxf(fabsf(want), 1e-6f));
            }
        }
    }
    printf("packed [T, 3*d_model] layout max relative error: %.5f  (%s)\n",
           max_rel, max_rel < 2e-2f ? "PASS" : "FAIL");

    free(s);
    CUDA_CHECK(cudaFreeHost(h_qkv)); CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_qkv));     CUDA_CHECK(cudaFree(d_out));
}


int main() {
    // No boundary masking in the kernel, so the shapes must tile exactly:
    //   seq % BM == 0 (queries), seq % BK == 0 (keys), d_head == BN.
    // 14 * (2048/128) = 224 blocks = exactly 4 waves on 56 SMs, so no partial-wave
    // tail contaminates the measurement.
    const int planes = 14;     // batch * heads -> grid.x
    const int seq    = 2048;   // queries and keys/values
    const int d_head = BN;     // 128

    const size_t plane_eles = (size_t)seq * d_head;
    const size_t total_eles = (size_t)planes * plane_eles;
    const size_t bytes = sizeof(half) * total_eles;

    half *h_Q, *h_K, *h_V, *h_O;
    CUDA_CHECK(cudaMallocHost(&h_Q, bytes));
    CUDA_CHECK(cudaMallocHost(&h_K, bytes));
    CUDA_CHECK(cudaMallocHost(&h_V, bytes));
    CUDA_CHECK(cudaMallocHost(&h_O, bytes));

    half *d_Q, *d_Kk, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_Kk, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    for (size_t i = 0; i < total_eles; ++i) {
        h_Q[i] = static_cast<half>(((i * 7 + 1) % 17) / 19.0f);
        h_K[i] = static_cast<half>(((i * 5 + 3) % 17) / 21.0f);
        h_V[i] = static_cast<half>(((i * 3 + 2) % 13) / 23.0f);
    }

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Kk, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

    // 64 KB of dynamic smem is over the 48 KB default cap, so it must be opted into.
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)SMEM_BYTES));

    dim3 tpb(WARPS * 32);              // 8 warps, one 16-row Q slice each
    dim3 bpg(planes, seq / BM);        // x -> (batch, head) planes, y -> Q row tiles

    for (int i = 0; i < 50; ++i) {
        flash_attention<<<bpg, tpb, SMEM_BYTES>>>(d_Q, d_Kk, d_V, d_O, seq, seq, d_head,
                                              d_head, d_head, (int)plane_eles, (int)plane_eles); // warm up
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaProfilerStart());


    flash_attention<<<bpg, tpb, SMEM_BYTES>>>(d_Q, d_Kk, d_V, d_O, seq, seq, d_head,
                                              d_head, d_head, (int)plane_eles, (int)plane_eles);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaProfilerStop());


    CUDA_CHECK(cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // Spot-check a spread of rows against the CPU reference. fp16 operands with fp32
    // accumulation over 2048 keys puts the expected relative error around 1e-3.
    float* ref = (float*)malloc(sizeof(float) * d_head);
    float max_rel = 0.0f;
    const int probe_rows[] = {0, 1, 7, 8, 127, 128, 1023, 2047};
    for (int p = 0; p < planes; p += 5) {
        for (int r = 0; r < (int)(sizeof(probe_rows) / sizeof(int)); ++r) {
            const int row = probe_rows[r];
            reference_row(h_Q, h_K, h_V, ref, p, row, seq, d_head);
            for (int c = 0; c < d_head; ++c) {
                const float got = __half2float(h_O[(size_t)p * plane_eles + (size_t)row * d_head + c]);
                const float rel = fabsf(got - ref[c]) / fmaxf(fabsf(ref[c]), 1e-6f);
                max_rel = fmaxf(max_rel, rel);
            }
        }
    }
    printf("per-plane [plane][seq][d_head] layout max relative error: %.5f  (%s)\n",
           max_rel, max_rel < 2e-2f ? "PASS" : "FAIL");
    free(ref);

    check_packed_layout();

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_Kk));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    CUDA_CHECK(cudaFreeHost(h_Q));
    CUDA_CHECK(cudaFreeHost(h_K));
    CUDA_CHECK(cudaFreeHost(h_V));
    CUDA_CHECK(cudaFreeHost(h_O));

    return 0;
}