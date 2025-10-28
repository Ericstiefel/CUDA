#include <mma.h>
#include <cuda_runtime.h>
#include <cmath>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul(const half* __restrict__ A,
                       const half* __restrict__ B,
                       half* __restrict__ C,
                       const int M,
                       const int N,
                       const int K)
{
    int global_row = blockIdx.y * blockDim.y + threadIdx.y;
    int global_col = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_m = global_row / WMMA_M;
    int warp_n = global_col / WMMA_N;

    // shared buffers (3-stage)
    __shared__ half A_tile[3][WMMA_M][WMMA_K];
    __shared__ half B_tile[3][WMMA_K][WMMA_N];

    wmma::fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag[3];
    wmma::fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag[3];

    wmma::fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fill_fragment(C_frag, 0.0f);

    int num_k_tiles = (K + WMMA_K - 1) / WMMA_K; 

    int threads_per_block = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    auto load_tile_to_shared = [&](int tile_idx, int buf_idx) {
        {
            const int A_elems = WMMA_M * WMMA_K;
            for (int i = tid; i < A_elems; i += threads_per_block) {
                int r = i / WMMA_K;
                int c = i % WMMA_K;
                int global_r = (warp_m * WMMA_M) + r;
                int global_c = tile_idx * WMMA_K + c;

                A_tile[buf_idx][r][c] = (global_r < M && global_c < N) ? A[global_r * N + global_c] : __float2half(0.0f);
            }
        }
        {
            const int B_elems = WMMA_K * WMMA_N;
            for (int i = tid; i < B_elems; i += threads_per_block) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int global_r = tile_idx * WMMA_K + r;              
                int global_c = (warp_n * WMMA_N) + c;         

                B_tile[buf_idx][r][c] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : __float2half(0.0f);
                
            }
        }
    };

    load_tile_to_shared(0, 0);
    if (num_k_tiles > 1) load_tile_to_shared(1, 1);
    __syncthreads();

    if (num_k_tiles > 0) {
        wmma::load_matrix_sync(A_frag[0], &A_tile[0][0][0], WMMA_K);
        wmma::load_matrix_sync(B_frag[0], &B_tile[0][0][0], WMMA_N);
    }
    if (num_k_tiles > 1) {
        wmma::load_matrix_sync(A_frag[1], &A_tile[1][0][0], WMMA_K);
        wmma::load_matrix_sync(B_frag[1], &B_tile[1][0][0], WMMA_N);
    }

    for (int i = 0; i < num_k_tiles; ++i) {
        int buf_i   = i % 3;
        int buf_i1  = (i + 1) % 3;
        int buf_i2  = (i + 2) % 3;

        if (i + 2 < num_k_tiles) {
            load_tile_to_shared(i + 2, buf_i2);
        }

        __syncthreads();

        if (i + 1 < num_k_tiles) {
            wmma::load_matrix_sync(A_frag[buf_i1], &A_tile[buf_i1][0][0], WMMA_K);
            wmma::load_matrix_sync(B_frag[buf_i1], &B_tile[buf_i1][0][0], WMMA_N);
        }

        wmma::mma_sync(C_frag, A_frag[buf_i], B_frag[buf_i], C_frag);
    }

    // write results
    int out_row = warp_m * WMMA_M;
    int out_col = warp_n * WMMA_N;

    float c_tmp[WMMA_M * WMMA_N];
    wmma::store_matrix_sync(c_tmp, C_frag, WMMA_N, wmma::mem_row_major);

    for (int r = 0; r < WMMA_M; ++r) {
        int global_r = out_row + r;
        if (global_r >= M) continue;
        for (int c = 0; c < WMMA_N; ++c) {
            int global_c = out_col + c;
            if (global_c >= N) continue;
            C[ global_r * N + global_c ] = __float2half( c_tmp[r * WMMA_N + c]);
        }
    }
}
