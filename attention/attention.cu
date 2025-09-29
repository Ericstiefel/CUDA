// attention_with_existing_kernels_fused_init.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <ctime>
#include <cmath>

using namespace nvcuda;

// ------------------ User-tunable model dims ------------------
#define SEQ_LEN 32      // number of tokens (M)
#define D_MODEL 128     // embedding dim (input feature size)
#define D_K 64          // query/key dim
#define D_V 64          // value dim

// WMMA tile params 
#define TILE_WIDTH 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CEIL_DIV(x,y) (((x)+(y)-1)/(y))

#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA Error in file %s at line %d: %s\n", __FILE__,    \
                __LINE__, cudaGetErrorString(err));                            \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)


__global__ void init_random_matrix(half* ptr, int rows, int cols, unsigned long seed) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        curandState localState;
        curand_init(seed, idx, 0, &localState);
        float rv = curand_uniform(&localState);
        ptr[idx] = __float2half(rv);
    }
}


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


    if (num_k_tiles > 0) load_tile_to_shared(0, 0);
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

__global__ void split_fused(const half* __restrict__ fused,
                            half* __restrict__ Q_out,
                            half* __restrict__ K_out,
                            half* __restrict__ V_out,
                            int seq_len, int Dk, int Dv)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Nf = 2 * Dk + Dv;
    if (row >= seq_len || col >= (Dk > Dv ? Dk : Dv)) return; 

    if (row < seq_len) {
        int base = row * Nf;
        if (col < Dk) {
            Q_out[row * Dk + col] = fused[base + col];
        }
        if (col < Dk) {
            K_out[col * seq_len + row] = fused[base + Dk + col]; // Transpose K here (most opportune time)
        }
        if (col < Dv) {
            V_out[row * Dv + col] = fused[base + 2 * Dk + col];
        }
    }
}

// ------------------ Host flow ------------------
int main() {

    // dims to match matmul's M/K/N naming:
    const int M = SEQ_LEN;
    const int K_dim = D_MODEL;          // shared dimension for X * W  (X: M x K_dim)
    const int Dk = D_K;
    const int Dv = D_V;
    const int NFUSED = 2 * Dk + Dv;     // columns of fused weight

    cudaStream_t s0;
    CUDA_CHECK(cudaStreamCreate(&s0));

    // allocate X: (M x K_dim)
    half* d_X;
    CUDA_CHECK(cudaMalloc((void**)&d_X, sizeof(half) * M * K_dim));

    half* d_fusedW;
    CUDA_CHECK(cudaMalloc((void**)&d_fusedW, sizeof(half) * K_dim * NFUSED));

    dim3 tpb(16,16);
    dim3 bpg_X( CEIL_DIV(K_dim, tpb.x), CEIL_DIV(M, tpb.y) );
    CUDA_CHECK(init_random_matrix<<<bpg_X, tpb, 0, s0>>>(d_X, M, K_dim, (unsigned long)time(NULL)));

    dim3 bpg_fused_init( CEIL_DIV(NFUSED, tpb.x), CEIL_DIV(K_dim, tpb.y) );
    CUDA_CHECK(init_random_matrix<<<bpg_fused_init, tpb, 0, s0>>>(d_fusedW, K_dim, NFUSED, (unsigned long)time(NULL)+42));
    CUDA_CHECK(cudaStreamSynchronize(s0));

    half* d_fused_out;
    CUDA_CHECK(cudaMalloc((void**)&d_fused_out, sizeof(half) * M * NFUSED));
    CUDA_CHECK(cudaMemset(d_fused_out, 0, sizeof(half) * M * NFUSED));
    dim3 mm_tpb(TILE_WIDTH, TILE_WIDTH);
    dim3 mm_bpg( CEIL_DIV(NFUSED, mm_tpb.x), CEIL_DIV(M, mm_tpb.y) );

    CUDA_CHECK(matmul<<<mm_bpg, mm_tpb, 0, s0>>>(d_X, d_fusedW, d_fused_out, M, NFUSED, K_dim));
    CUDA_CHECK(cudaStreamSynchronize(s0));


    half *d_Q, *d_K_T, *d_V;
    CUDA_CHECK(cudaMalloc((void**)&d_Q, sizeof(half) * M * Dk));
    CUDA_CHECK(cudaMalloc((void**)&d_K_T, sizeof(half) * M * Dk));
    CUDA_CHECK(cudaMalloc((void**)&d_V, sizeof(half) * M * Dv));

    dim3 split_tpb(16, 16);
    dim3 split_bpg( CEIL_DIV( (Dk> Dv? Dk : Dv), split_tpb.x ), CEIL_DIV(M, split_tpb.y) );
    CUDA_CHECK(split_fused<<<split_bpg, split_tpb, 0, s0>>>(d_fused_out, d_Q, d_K_T, d_V, M, Dk, Dv));
    CUDA_CHECK(cudaStreamSynchronize(s0));


    // --- cleanup ---
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_fusedW));
    CUDA_CHECK(cudaFree(d_fused_out));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K_T));
    CUDA_CHECK(cudaFree(d_V));

    CUDA_CHECK(cudaStreamDestroy(s0));

    return 0;
}
