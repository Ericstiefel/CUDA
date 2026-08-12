// Main file for the project
//   nvcc -arch=sm_86 -o gpt2 gpt2.cu gemm.cu attention.cu softmax.cu layernorm.cu embedding.cu

#include "kernels.cuh"
#include <cmath>

// The dims of the config are different, mainly to match the limits of my A4500, this is more about it being a learning project
struct Cfg {
    static constexpr int n_layers = 12;
    static constexpr int d_model  = 768;
    static constexpr int n_heads  = 6;                    // 6 x 128, not GPT-2's 12 x 64:
    static constexpr int d_head   = attn::BN;             // the attention kernel fixes d_head at 128
    static constexpr int d_ff     = 4 * d_model;          // 3072
    static constexpr int vocab    = 50257;                // real BPE vocabulary
    static constexpr int vocab_pad = 50304;               // 393 * 128, so the GEMM tiles exactly
    static constexpr int T        = 2048;                 // tokens; must be a multiple of 128 

    static_assert(d_model == n_heads * d_head, "d_model must split into whole heads");
    static_assert(T % attn::BM == 0 && T % attn::BK == 0, "T must tile for attention");
    static_assert(d_model % gemm_cfg::BN == 0 && d_ff % gemm_cfg::BN == 0, "N must tile");
    static_assert(vocab_pad % gemm_cfg::BN == 0, "padded vocab must tile");
    static_assert(d_model % gemm_cfg::BK == 0 && d_ff % gemm_cfg::BK == 0, "K must tile");
    static_assert(vocab_pad / 8 <= sm_cfg::BLOCK * sm_cfg::VPT, "row too wide for softmax");
};


struct LayerWeights {
    half* ln1_g;  half* ln1_b;
    half* ln2_g;  half* ln2_b;
    half* w_qkv;                 // [d_model, 3 * d_model]
    half* w_o;                   // [d_model, d_model]
    half* w_ff1;                 // [d_model, d_ff]
    half* w_ff2;                 // [d_ff, d_model]
};

struct Model {
    half* wte;                   // [vocab_pad, d_model], gathered by token id
    half* w_lm;                  // [d_model, vocab_pad], the LM head
    half* lnf_g;  half* lnf_b;
    LayerWeights layers[Cfg::n_layers];
};

struct Buffers {
    int*  token_ids;             // [T]
    half* resid[2];              // [T, d_model], ping-ponged across residual adds
    half* ln_out;                // [T, d_model], normalised input to the next GEMM
    half* qkv;                   // [T, 3 * d_model], rotated in the projection epilogue
    half* attn_out;              // [T, d_model]
    half* proj_out;              // [T, d_model]
    half* ffn;                   // [T, d_ff]
    half* ffn_out;               // [T, d_model]
    half* logits;                // [T, vocab_pad]
    half* probs;                 // [T, vocab_pad]
};



static unsigned rng_state = 0x9E3779B9u;

static float next_weight() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((float)(rng_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static half* alloc_weights(size_t n, float scale) {
    half* h = (half*)malloc(sizeof(half) * n);
    for (size_t i = 0; i < n; ++i) h[i] = static_cast<half>(next_weight() * scale);
    half* d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(half) * n));
    CUDA_CHECK(cudaMemcpy(d, h, sizeof(half) * n, cudaMemcpyHostToDevice));
    free(h);
    return d;
}

static half* alloc_const(size_t n, float value) {
    half* h = (half*)malloc(sizeof(half) * n);
    for (size_t i = 0; i < n; ++i) h[i] = static_cast<half>(value);
    half* d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(half) * n));
    CUDA_CHECK(cudaMemcpy(d, h, sizeof(half) * n, cudaMemcpyHostToDevice));
    free(h);
    return d;
}

static half* alloc_buf(size_t n) {
    half* d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(half) * n));
    return d;
}

static size_t build_model(Model& m) {
    const float s_model = 1.0f / sqrtf((float)Cfg::d_model);
    const float s_ff    = 1.0f / sqrtf((float)Cfg::d_ff);

    size_t bytes = 0;
    auto acc = [&bytes](size_t n) { bytes += n * sizeof(half); };


    m.wte   = alloc_weights((size_t)Cfg::vocab_pad * Cfg::d_model, s_model);
    m.w_lm  = alloc_weights((size_t)Cfg::d_model * Cfg::vocab_pad, s_model);
    acc((size_t)Cfg::vocab_pad * Cfg::d_model * 2);

    m.lnf_g = alloc_const(Cfg::d_model, 1.0f);
    m.lnf_b = alloc_const(Cfg::d_model, 0.0f);
    acc(Cfg::d_model * 2);

    for (int i = 0; i < Cfg::n_layers; ++i) {
        LayerWeights& L = m.layers[i];
        L.ln1_g = alloc_const(Cfg::d_model, 1.0f);
        L.ln1_b = alloc_const(Cfg::d_model, 0.0f);
        L.ln2_g = alloc_const(Cfg::d_model, 1.0f);
        L.ln2_b = alloc_const(Cfg::d_model, 0.0f);
        L.w_qkv = alloc_weights((size_t)Cfg::d_model * 3 * Cfg::d_model, s_model);
        L.w_o   = alloc_weights((size_t)Cfg::d_model * Cfg::d_model,     s_model);
        L.w_ff1 = alloc_weights((size_t)Cfg::d_model * Cfg::d_ff,        s_model);
        L.w_ff2 = alloc_weights((size_t)Cfg::d_ff * Cfg::d_model,        s_ff);
        acc(Cfg::d_model * 4);
        acc((size_t)Cfg::d_model * 3 * Cfg::d_model);
        acc((size_t)Cfg::d_model * Cfg::d_model);
        acc((size_t)Cfg::d_model * Cfg::d_ff * 2);
    }
    return bytes;
}

static size_t build_buffers(Buffers& b) {
    const size_t stream_eles = (size_t)Cfg::T * Cfg::d_model;
    const size_t logit_eles  = (size_t)Cfg::T * Cfg::vocab_pad;

    CUDA_CHECK(cudaMalloc(&b.token_ids, sizeof(int) * Cfg::T));
    int* h_ids = (int*)malloc(sizeof(int) * Cfg::T);
    for (int t = 0; t < Cfg::T; ++t) h_ids[t] = (t * 7919 + 13) % Cfg::vocab;
    CUDA_CHECK(cudaMemcpy(b.token_ids, h_ids, sizeof(int) * Cfg::T, cudaMemcpyHostToDevice));
    free(h_ids);

    b.resid[0] = alloc_buf(stream_eles);
    b.resid[1] = alloc_buf(stream_eles);
    b.ln_out   = alloc_buf(stream_eles);
    b.qkv      = alloc_buf((size_t)Cfg::T * 3 * Cfg::d_model);
    b.attn_out = alloc_buf(stream_eles);
    b.proj_out = alloc_buf(stream_eles);
    b.ffn      = alloc_buf((size_t)Cfg::T * Cfg::d_ff);
    b.ffn_out  = alloc_buf(stream_eles);
    b.logits   = alloc_buf(logit_eles);
    b.probs    = alloc_buf(logit_eles);

    return (stream_eles * 6 + (size_t)Cfg::T * 3 * Cfg::d_model
            + (size_t)Cfg::T * Cfg::d_ff + logit_eles * 2) * sizeof(half)
           + sizeof(int) * Cfg::T;
}


// Forward pass
static dim3 gemm_grid(int M, int N) {
    return dim3(N / gemm_cfg::BN, M / gemm_cfg::BM);
}

static void forward(cudaStream_t s, const Model& m, Buffers& b) {
    constexpr int T   = Cfg::T;
    constexpr int D   = Cfg::d_model;
    const dim3 ln_block(ln_cfg::threads_for(D));

    embedding_gather<<<T, ln_block, 0, s>>>(b.token_ids, m.wte, b.resid[0], D);


    layerNorm2D<<<T, ln_block, 0, s>>>(b.resid[0], nullptr, nullptr, b.ln_out,
                                       m.layers[0].ln1_g, m.layers[0].ln1_b, T, D);

    int r = 0;   // which  buffer currently holds the stream

    for (int i = 0; i < Cfg::n_layers; ++i) {
        const LayerWeights& L = m.layers[i];

        // QKV projection, RoPE applied to Q and K inside.
        gemm_qkv_rope<<<gemm_grid(T, 3 * D), gemm_cfg::THREADS, 0, s>>>(
            b.ln_out, L.w_qkv, b.qkv, T, D, 3 * D, Cfg::d_head);

        flash_attention<<<dim3(Cfg::n_heads, T / attn::BM), attn::THREADS, attn::SMEM_BYTES, s>>>(
            b.qkv, b.qkv + D, b.qkv + 2 * D, b.attn_out,
            T, T, Cfg::d_head,
            3 * D, D,                       // row strides in / out
            Cfg::d_head, Cfg::d_head);      // plane strides in / out

        gemm<<<gemm_grid(T, D), gemm_cfg::THREADS, 0, s>>>(
            b.attn_out, L.w_o, b.proj_out, T, D, D);

        layerNorm2D<<<T, ln_block, 0, s>>>(b.proj_out, b.resid[r], b.resid[r ^ 1], b.ln_out,
                                           L.ln2_g, L.ln2_b, T, D);
        r ^= 1;

        gemm_gelu<<<gemm_grid(T, Cfg::d_ff), gemm_cfg::THREADS, 0, s>>>(
            b.ln_out, L.w_ff1, b.ffn, T, D, Cfg::d_ff);
        gemm<<<gemm_grid(T, D), gemm_cfg::THREADS, 0, s>>>(
            b.ffn, L.w_ff2, b.ffn_out, T, Cfg::d_ff, D);


        const bool last = (i + 1 == Cfg::n_layers);
        const half* g = last ? m.lnf_g : m.layers[i + 1].ln1_g;
        const half* bt = last ? m.lnf_b : m.layers[i + 1].ln1_b;
        layerNorm2D<<<T, ln_block, 0, s>>>(b.ffn_out, b.resid[r], b.resid[r ^ 1], b.ln_out,
                                           g, bt, T, D);
        r ^= 1;
    }

    gemm<<<gemm_grid(T, Cfg::vocab_pad), gemm_cfg::THREADS, 0, s>>>(
        b.ln_out, m.w_lm, b.logits, T, D, Cfg::vocab_pad);

    softmax<<<T, sm_cfg::BLOCK, 0, s>>>(b.logits, b.probs, Cfg::vocab_pad, Cfg::vocab);
}



static bool check_output(const Buffers& b) {
    const int probe_rows[] = {0, 1, 1023, 2047};
    half* row = (half*)malloc(sizeof(half) * Cfg::vocab_pad);

    bool ok = true;
    for (int p = 0; p < (int)(sizeof(probe_rows) / sizeof(int)); ++p) {
        const int t = probe_rows[p];
        CUDA_CHECK(cudaMemcpy(row, b.probs + (size_t)t * Cfg::vocab_pad,
                              sizeof(half) * Cfg::vocab_pad, cudaMemcpyDeviceToHost));

        double sum = 0.0;
        float peak = 0.0f;
        int argmax = -1;
        long long bad = 0, pad_nonzero = 0;

        for (int c = 0; c < Cfg::vocab; ++c) {
            const float v = __half2float(row[c]);
            if (!isfinite(v) || v < 0.0f) ++bad;
            sum += v;
            if (v > peak) { peak = v; argmax = c; }
        }
        for (int c = Cfg::vocab; c < Cfg::vocab_pad; ++c) {
            if (__half2float(row[c]) != 0.0f) ++pad_nonzero;
        }

        const bool row_ok = (bad == 0) && (pad_nonzero == 0) && (fabs(sum - 1.0) < 5e-2);
        ok = ok && row_ok;
        printf("  token %4d | sum %.5f | argmax %5d (p=%.3e) | bad %lld | pad!=0 %lld | %s\n",
               t, sum, argmax, peak, bad, pad_nonzero, row_ok ? "ok" : "FAIL");
    }
    free(row);
    return ok;
}


int main() {
    Model m;
    Buffers b;

    const size_t wbytes = build_model(m);
    const size_t abytes = build_buffers(b);
    printf("GPT-2  %d layers, d_model %d, %d heads x %d, d_ff %d, vocab %d (pad %d), T %d\n",
           Cfg::n_layers, Cfg::d_model, Cfg::n_heads, Cfg::d_head, Cfg::d_ff,
           Cfg::vocab, Cfg::vocab_pad, Cfg::T);
    printf("weights %.1f MB   activations %.1f MB\n\n",
           wbytes / 1048576.0, abytes / 1048576.0);

    // 64 KB of dynamic smem is over the 48 KB default cap for allocation.
    CUDA_CHECK(cudaFuncSetAttribute(flash_attention,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)attn::SMEM_BYTES));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    forward(stream, m, b);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    printf("output check (after 1 forward):\n");
    check_output(b);
    printf("\n");

    for (int i = 0; i < 3; ++i) forward(stream, m, b);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    const int iters = 10;
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) forward(stream, m, b);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float eager_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&eager_ms, t0, t1));
    eager_ms /= iters;

    printf("eager   %8.3f ms / forward\n", eager_ms);
    printf("output check (eager):\n");
    const bool eager_ok = check_output(b);

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    forward(stream, m, b);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    size_t num_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    for (int i = 0; i < 3; ++i) CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float graph_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&graph_ms, t0, t1));
    graph_ms /= iters;

    printf("\ngraph   %8.3f ms / forward   (%zu nodes, %.3f ms saved, %.1f%%)\n",
           graph_ms, num_nodes, eager_ms - graph_ms,
           100.0 * (eager_ms - graph_ms) / eager_ms);
    printf("output check (graph):\n");
    const bool graph_ok = check_output(b);

    printf("\n%s\n", (eager_ok && graph_ok) ? "PASS" : "FAIL");

    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
