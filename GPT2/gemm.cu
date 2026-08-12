#include "kernels.cuh"
#include <cuda_profiler_api.h>
#include <cfloat>

using namespace gemm_cfg;   // BM, BN, BK -- see kernels.cuh for why these are namespaced

// Tune these, for max arithmetic intensity on matmul, M & N >> K (K Doesn't contribute).

// The body is shared and the activation is a compile-time flag, so gemm and gemm_gelu
// come out as two genuinely separate kernels with no duplicated source to drift apart.
// GELU is never its own launch: elementwise over [T, 3072] would be pure bandwidth for
// one multiply-add per element, and here the values are already in registers.

// Epilogue selector. Nothing here is ever launched on its own: GELU elementwise over
// [T, 3072] and RoPE elementwise over [T, 2304] would both be pure bandwidth for a
// handful of flops, and in the epilogue the values are already in registers.
namespace ep {
constexpr int NONE     = 0;
constexpr int GELU     = 1;
constexpr int QKV_ROPE = 2;
}

// 10000^(-2i/d_head) == exp2(-2i/d_head * log2(10000))
constexpr float LOG2_ROPE_BASE = 13.287712379549449f;


// One k-tile of the warp's 64x32 output, accumulated into rC. Lifted out of the main
// loop because the pipelined loop and the drain iteration both need it -- when this was
// copy-pasted, a fix to one copy silently missed the other.
__device__ __forceinline__ void mma_tile(const half* sA_stage, const half* sB_stage,
                                         const int lane, const int warp_row_offset,
                                         const int warp_col_offset, float rC[4][4][4]) {
    const int lane_group = lane / 8;
    const int row_in_group = lane % 8;

    for (int k_dim = 0; k_dim < 2; ++k_dim) {
        for (int m_dim = 0; m_dim < 4; ++m_dim) {
            // tensor core ldmatrix register pattern has it as 4 8x8 tiles
            int a_row = warp_row_offset + m_dim * 16 + row_in_group;
            a_row = (lane_group == 0 || lane_group == 2) ? a_row : a_row + 8;

            const int a_col = (lane_group == 0 || lane_group == 1) ? k_dim * 16 : k_dim * 16 + 8;

            uint32_t rA[4];
            ld_matrix_x4(rA, swizzled_ptr(sA_stage, a_row, a_col, BK));

            for (int n_dim = 0; n_dim < 4; ++n_dim) {
                int b_row = k_dim * 16 + row_in_group;
                b_row = (lane_group == 0 || lane_group == 2) ? b_row : b_row + 8;

                const int b_col = warp_col_offset + n_dim * 8;

                // B is row-major [k][n] in smem but the mma wants it column-major over
                // (k, n); the transposing ldmatrix does that for free, exactly as
                // attention.cu does for V.
                uint32_t rB[2];
                ld_matrix_x2_trans(rB, swizzled_ptr(sB_stage, b_row, b_col, BN));
                mma(rA, rB, rC[m_dim][n_dim]);
            }
        }
    }
}


// A MxK, B KxN.  d_head is only read by the QKV_ROPE epilogue.
template <int EP>
__device__ __forceinline__ void gemm_body(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, const int M, const int K, const int N, const int d_head) {
    __shared__ half sA[2][BM][BK];
    __shared__ half sB[2][BK][BN];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // A has width 32, so we're loading 8 eles at a time, 32 / 8 = 4.
    int load_A_row = blockIdx.y * BM + (tid / 4);
    int load_A_col = (tid % 4) * 8;
    
    // B tile has width 128, so we use 16 here
    int load_B_row = tid / 16;
    int load_B_col = blockIdx.x * BN + (tid % 16) * 8;

    float rC[4][4][4] = {0.0f};   // rA / rB now live inside mma_tile

    int write_stage = 0; int read_stage = 0;

    uint32_t smem_A_p1 = swizzled_ptr(&sA[0][0][0], tid / 4, load_A_col, BK);
    uint32_t smem_A_p2 = swizzled_ptr(&sA[0][0][0], tid / 4 + 64, load_A_col, BK);
    cp_async_128(smem_A_p1, &A[load_A_row * K + load_A_col]);
    cp_async_128(smem_A_p2, &A[(load_A_row + 64) * K + load_A_col]);

    uint32_t smem_B_p1 = swizzled_ptr(&sB[0][0][0], tid / 16, (tid % 16) * 8, BN);
    uint32_t smem_B_p2 = swizzled_ptr(&sB[0][0][0], tid / 16 + 16, (tid % 16) * 8, BN);
    cp_async_128(smem_B_p1, &B[load_B_row * N + load_B_col]);
    cp_async_128(smem_B_p2, &B[(load_B_row + 16) * N + load_B_col]);

    cp_async_commit_group(); cp_async_wait_group<0>(); __syncthreads();

    int warp_row = warp_id / 4; int warp_col = warp_id % 4; // where that warp is inside of the 2x4 warp grid.

    int warp_row_offset = warp_row * 64; int warp_col_offset = warp_col * 32;

    for (int tile_k = BK; tile_k < K; tile_k += BK) {
        // preload next set of tiles
        write_stage ^= 1;

        uint32_t smem_A_p1 = swizzled_ptr(&sA[write_stage][0][0], tid / 4, load_A_col, BK);
        uint32_t smem_A_p2 = swizzled_ptr(&sA[write_stage][0][0], tid / 4 + 64, load_A_col, BK);
        cp_async_128(smem_A_p1, &A[load_A_row * K + load_A_col + tile_k]);
        cp_async_128(smem_A_p2, &A[(load_A_row + 64) * K + load_A_col + tile_k]);

        uint32_t smem_B_p1 = swizzled_ptr(&sB[write_stage][0][0], tid / 16, (tid % 16) * 8, BN);
        uint32_t smem_B_p2 = swizzled_ptr(&sB[write_stage][0][0], tid / 16 + 16, (tid % 16) * 8, BN);
        cp_async_128(smem_B_p1, &B[load_B_row * N + load_B_col + tile_k * N]);
        cp_async_128(smem_B_p2, &B[(load_B_row + 16) * N + load_B_col + tile_k * N]);

        cp_async_commit_group();

        mma_tile(&sA[read_stage][0][0], &sB[read_stage][0][0],
                 lane, warp_row_offset, warp_col_offset, rC);

        read_stage ^= 1;
        // Must be <0>, not <1>. The group committed above is precisely the one holding
        // the tile the *next* iteration reads, so allowing one outstanding group lets
        // the compute run against shared memory the copy has not finished writing. It
        // races rather than fails: correct whenever the copy happens to land in time,
        // which is why a per-kernel cudaDeviceSynchronize hides it completely.
        // Overlap is unaffected -- the copy still runs underneath this tile's mmas.
        cp_async_wait_group<0>();
        __syncthreads();

    }

    // last tile
    mma_tile(&sA[read_stage][0][0], &sB[read_stage][0][0],
             lane, warp_row_offset, warp_col_offset, rC);

    // output is 16 x 8 per tile, each thread has 2 consecutive items in each of its 2 output rowws.
    int frag_row = lane / 4;
    int frag_col = (lane % 4) * 2;

    int warp_global_row = blockIdx.y * BM + warp_row_offset;
    int warp_global_col = blockIdx.x * BN + warp_col_offset;

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            int block_global_row = warp_global_row + (m * 16);
            int block_global_col = warp_global_col + (n * 8);

            int global_row = block_global_row + frag_row;
            int global_col = block_global_col + frag_col;

            // Accumulation stays fp32 the whole way down; the epilogue runs on the fp32
            // accumulator and the narrowing happens once, here, so the next kernel in
            // the chain can consume C as half directly.
            float c0 = rC[m][n][0], c1 = rC[m][n][1];
            float c2 = rC[m][n][2], c3 = rC[m][n][3];

            if (EP == ep::GELU) {
                c0 = gelu(c0); c1 = gelu(c1); c2 = gelu(c2); c3 = gelu(c3);
            } else if (EP == ep::QKV_ROPE) {
                // This C fragment is already in exactly the shape RoPE wants. Registers
                // 0,1 hold columns (col, col+1) of one row and 2,3 the same two columns
                // eight rows down, and col is always even -- so with the interleaved
                // convention each register pair *is* one rotation pair. No shuffles, no
                // shared memory, no separate pass over [T, 2304].
                //
                // Columns are laid out [Q | K | V], each d_model wide. Q and K rotate,
                // V does not. d_model is a whole number of heads, so the offset within a
                // head is just col % d_head for both.
                const int d_model = N / 3;
                if (global_col < 2 * d_model) {
                    const int pair = (global_col % d_head) >> 1;
                    const float inv_freq =
                        exp2f(-((float)(2 * pair) / (float)d_head) * LOG2_ROPE_BASE);

                    // The position is the token index, which is the global row.
                    float s, c;
                    sincosf((float)global_row * inv_freq, &s, &c);
                    const float r0 = c0 * c - c1 * s;
                    const float r1 = c0 * s + c1 * c;

                    sincosf((float)(global_row + 8) * inv_freq, &s, &c);
                    const float r2 = c2 * c - c3 * s;
                    const float r3 = c2 * s + c3 * c;

                    c0 = r0; c1 = r1; c2 = r2; c3 = r3;
                }
            }

            C[global_row * N + global_col] = static_cast<half>(c0);
            C[global_row * N + global_col + 1] = static_cast<half>(c1);
            C[(global_row + 8) * N + global_col] = static_cast<half>(c2);
            C[(global_row + 8) * N + global_col + 1] = static_cast<half>(c3);
        }
    }


}


__global__ void gemm(const half* __restrict__ A, const half* __restrict__ B,
                     half* __restrict__ C, const int M, const int K, const int N) {
    gemm_body<ep::NONE>(A, B, C, M, K, N, 0);
}

__global__ void gemm_gelu(const half* __restrict__ A, const half* __restrict__ B,
                          half* __restrict__ C, const int M, const int K, const int N) {
    gemm_body<ep::GELU>(A, B, C, M, K, N, 0);
}

// The QKV projection with RoPE folded into the store. N must be 3 * d_model.
__global__ void gemm_qkv_rope(const half* __restrict__ A, const half* __restrict__ B,
                              half* __restrict__ C, const int M, const int K, const int N,
                              const int d_head) {
    gemm_body<ep::QKV_ROPE>(A, B, C, M, K, N, d_head);
}
