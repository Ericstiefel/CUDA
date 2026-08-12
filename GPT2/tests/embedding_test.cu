// Standalone correctness harness for embedding.cu.
#include "../kernels.cuh"
#include <cmath>


int main() {
    const int N = 768;      // d_model
    const int V = 50304;    // padded vocab
    const int T = 4096;     // tokens

    int* h_ids;   CUDA_CHECK(cudaMallocHost(&h_ids, sizeof(int) * T));
    half* h_wte;  CUDA_CHECK(cudaMallocHost(&h_wte, sizeof(half) * (size_t)V * N));
    half* h_out;  CUDA_CHECK(cudaMallocHost(&h_out, sizeof(half) * (size_t)T * N));

    for (int t = 0; t < T; ++t) h_ids[t] = (t * 7919) % V;
    for (size_t i = 0; i < (size_t)V * N; ++i) {
        h_wte[i] = static_cast<half>((float)(i % 23) / 29.0f);
    }

    int* d_ids;   CUDA_CHECK(cudaMalloc(&d_ids, sizeof(int) * T));
    half* d_wte;  CUDA_CHECK(cudaMalloc(&d_wte, sizeof(half) * (size_t)V * N));
    half* d_out;  CUDA_CHECK(cudaMalloc(&d_out, sizeof(half) * (size_t)T * N));

    CUDA_CHECK(cudaMemcpy(d_ids, h_ids, sizeof(int) * T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wte, h_wte, sizeof(half) * (size_t)V * N, cudaMemcpyHostToDevice));

    embedding_gather<<<T, ln_cfg::threads_for(N)>>>(d_ids, d_wte, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(half) * (size_t)T * N, cudaMemcpyDeviceToHost));

    // A gather is exact -- any mismatch at all is a bug, so compare bit for bit.
    long long bad = 0;
    for (int t = 0; t < T; ++t) {
        const half* want = h_wte + (size_t)h_ids[t] * N;
        const half* got  = h_out + (size_t)t * N;
        for (int c = 0; c < N; ++c) {
            if (__half2float(want[c]) != __half2float(got[c])) ++bad;
        }
    }
    printf("embedding_gather mismatched elements: %lld  (%s)\n", bad, bad == 0 ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFreeHost(h_ids)); CUDA_CHECK(cudaFreeHost(h_wte)); CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_ids)); CUDA_CHECK(cudaFree(d_wte)); CUDA_CHECK(cudaFree(d_out));

    return 0;
}