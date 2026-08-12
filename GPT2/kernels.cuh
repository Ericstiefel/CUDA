#pragma once

// The launch contract for the model.
//
// Tile sizes live here rather than inside each .cu because a driver has to compute grid
// shapes and the attention kernel's dynamic shared-memory size, and it cannot see
// constants with internal linkage. Named namespaces also let BM/BK/WARPS keep the two
// different meanings they legitimately have -- attn::BK is 64, gemm_cfg::BK is 32,
// attn::WARPS is 8, sm_cfg::WARPS is 32 -- without either one shadowing the other.

#include "common.cuh"


namespace attn {
constexpr int BM = 128;   // Q rows per block tile
constexpr int BN = 128;   // head dim, also the smem row stride
constexpr int BK = 64;    // K/V rows (keys) per streamed tile

constexpr int WARPS   = 8;
constexpr int WARP_M  = BM / WARPS;      // 16 Q rows per warp -> one m16 mma tile
constexpr int THREADS = WARPS * 32;

constexpr int QK_K_STEPS = BN / 16;      // 8 reduction steps over head dim
constexpr int QK_N_TILES = BK / 8;       // n-tiles of S, 8 keys each
constexpr int PV_K_STEPS = BK / 16;      // reduction steps over keys
constexpr int PV_N_TILES = BN / 8;       // 16 n-tiles of O, 8 head-dim cols each

constexpr size_t Q_ELES = BM * BN;
constexpr size_t K_ELES = 2 * BK * BN;
constexpr size_t V_ELES = 2 * BK * BN;

// P never reaches shared memory, so there is no sP staging buffer here. Over the 48 KB
// default cap, so the driver must cudaFuncSetAttribute before the first launch.
constexpr size_t SMEM_BYTES = (Q_ELES + K_ELES + V_ELES) * sizeof(half);
}


namespace gemm_cfg {
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int THREADS = 256;   // 2 x 4 warp grid, fixed by the kernel
}


namespace sm_cfg {
constexpr int BLOCK = 1024;          // one block per row of logits
constexpr int WARPS = BLOCK / 32;    // 32, exactly the lane count of one warp
constexpr int VPT   = 7;             // float4s per thread
}


namespace ln_cfg {
// One block per row, one float4 (8 halves) per thread.
constexpr int threads_for(int N) { return N / 8; }
constexpr float EPS = 1e-6f;
}


// ---- entry points ------------------------------------------------------------------

// C = A * B, A is MxK and B is KxN, both half, accumulated in fp32 and narrowed on store.
// Requires M % 128 == 0, N % 128 == 0, K % 32 == 0.
__global__ void gemm(const half* __restrict__ A, const half* __restrict__ B,
                     half* __restrict__ C, const int M, const int K, const int N);

// Same GEMM with GELU folded into the epilogue. Only the FFN-up projection uses it.
__global__ void gemm_gelu(const half* __restrict__ A, const half* __restrict__ B,
                          half* __restrict__ C, const int M, const int K, const int N);

// The QKV projection with RoPE folded into the epilogue. C is [M, 3 * d_model] laid out
// [Q | K | V]; Q and K are rotated in the accumulator registers and V is left alone, so
// RoPE costs no extra pass over memory. N must be 3 * d_model, d_model a whole number
// of heads. Rotation uses the interleaved convention (element 2i paired with 2i+1).
__global__ void gemm_qkv_rope(const half* __restrict__ A, const half* __restrict__ B,
                              half* __restrict__ C, const int M, const int K, const int N,
                              const int d_head);

// Causal self-attention. Requires d_N == attn::BN, seq % attn::BM == 0, seq % attn::BK == 0.
// grid = (planes, seq / attn::BM), block = attn::THREADS, smem = attn::SMEM_BYTES.
//
// Row and plane strides are parameters so the same kernel serves two layouts. Reading a
// packed QKV buffer straight out of the projection GEMM is what lets the model skip the
// split and merge passes entirely:
//
//   packed  [T, 3*d_model] -> [T, d_model]
//     Q = qkv, K = qkv + d_model, V = qkv + 2*d_model
//     in_row = 3*d_model, out_row = d_model, in_plane = out_plane = d_head
//
//   per-plane [plane][seq][d_head]
//     in_row = out_row = d_head, in_plane = out_plane = seq * d_head
__global__ void flash_attention(const half* __restrict__ Q, const half* __restrict__ K,
                                const half* __restrict__ V, half* __restrict__ out,
                                int d_M, int d_K, int d_N,
                                int in_row_stride, int out_row_stride,
                                int in_plane_stride, int out_plane_stride);

// Row-wise softmax over the vocabulary axis. N is the padded width the GEMM wrote,
// V the real vocab; columns in between are forced to zero.
// grid = (rows), block = sm_cfg::BLOCK.
__global__ void softmax(const half* __restrict__ inp, half* __restrict__ out,
                        const int N, const int V);

// LayerNorm with the residual add fused into the front.
//   residual  == nullptr -> nothing to add (first LayerNorm of block 0 only)
//   resid_out == nullptr -> do not write the un-normalised sum
// grid = (M), block = ln_cfg::threads_for(N).
__global__ void layerNorm2D(const half* __restrict__ inp, const half* __restrict__ residual,
                            half* __restrict__ resid_out, half* __restrict__ out,
                            const half* __restrict__ gamma, const half* __restrict__ beta,
                            const int M, const int N);

// token_ids[T] -> out[T, N], gathering rows of the embedding table.
// grid = (T), block = ln_cfg::threads_for(N).
__global__ void embedding_gather(const int* __restrict__ token_ids,
                                 const half* __restrict__ wte,
                                 half* __restrict__ out, const int N);
