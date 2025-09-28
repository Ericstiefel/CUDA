#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

/*
Steps:
- Init random weight matrices
- Matmul with input
- QK^T / sqrt(d_k)
- Softmax result with matmul V joint

*/

using namespace nvcuda;


#define TILE_WIDTH 16
#define N 128

#define d_k 64 // Dims of input

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define CEIL_DIV (((N) + (TILE_WIDTH) - 1) / TILE_WIDTH)
#define INP_CEIL_DIV (((d_k) + (TILE_WIDTH) - 1) / TILE_WIDTH)


#define CUDA_CHECK(call)\
do {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        fprintf(stderr, "CUDA Error in file %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);\
    }\
} while (0)

__global__ void init_random_matrix(half* ptr, unsigned long seed) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < N) {
        int idx = row * N + col;

        curandState_t localState;

        curand_init(seed, idx, 0, &localState);

        float random_val = __float2half(curand_uniform(&localState));
        ptr[idx] = random_val;
    }
}


__global__ void matmul_3stage(const half* __restrict__ A,
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

    __shared__ half A_tile[3][WMMA_M][WMMA_K];
    __shared__ half B_tile[3][WMMA_K][WMMA_N];

    wmma::fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> A_frag[3];
    wmma::fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> B_frag[3];

    wmma::fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fill_fragment(C_frag, 0.0f);

    int num_k_tiles = K / WMMA_K; // Need to ceil_div (currently assumes an even division)

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
                half v = __float2half(0.0f);
                if (global_r < M && global_c < K) v = A[ global_r * K + global_c ];
                A_tile[buf_idx][r][c] = v;
            }
        }
        {
            const int B_elems = WMMA_K * WMMA_N;
            for (int i = tid; i < B_elems; i += threads_per_block) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                int global_r = tile_idx * WMMA_K + r;
                int global_c = (warp_n * WMMA_N) + c;
                half v = __float2half(0.0f);
                if (global_r < K && global_c < N) v = B[ global_r * N + global_c ];
                B_tile[buf_idx][r][c] = v;
            }
        }
    };


    load_tile_to_shared(0, 0);
    if (num_k_tiles > 1) load_tile_to_shared(1, 1);

    __syncthreads();

    wmma::load_matrix_sync(A_frag[0], &A_tile[0][0][0], WMMA_K);
    wmma::load_matrix_sync(B_frag[0], &B_tile[0][0][0], WMMA_N);
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
            C[ global_r * N + global_c ] = __float2half( c_tmp[r * WMMA_N + c] );
        }
    }
}

__global__ void pack_weights_fused(const half* __restrict__ Wq,
                                   const half* __restrict__ Wk,
                                   const half* __restrict__ Wv,
                                   half* __restrict__ Wf,
                                   int K, int N, int N3) 
{
    int r = blockIdx.y * blockDim.y + threadIdx.y; 
    int c = blockIdx.x * blockDim.x + threadIdx.x; 

    if (r >= K || c >= N3) return;

    if (c < N) {
        // Wq
        Wf[r * N3 + c] = Wq[r * N + c];
    } else if (c < 2 * N) {
        // Wk
        int kc = c - N;
        Wf[r * N3 + c] = Wk[r * N + kc];
    } else {
        // Wv
        int vc = c - 2 * N;
        Wf[r * N3 + c] = Wv[r * N + vc];
    }
}


int main() {

    half* input;

    size_t inp_size = sizeof(half) * d_k * d_k;

    CUDA_CHECK(cudaMalloc((void**)&input, inp_size));

    cudaStream_t streamA;

    CUDA_CHECK(cudaStreamCreate(&streamA));

    dim3 inpThreadPerBlock(N, N);
    dim3 inpBlocksPerGrid(INP_CEIL_DIV, INP_CEIL_DIV);

    CUDA_CHECK(init_random_matrix<<<inpBlocksPerGrid, inpThreadPerBlock, 0, streamA>>>(input, time(NULL));

    size_t weight_size =  sizeof(half) * d_k * N;

    half *d_Q_W, *d_K_W, *d_V_W; // Assumed to be the same weight_size

    CUDA_CHECK(cudaMalloc((void**)&d_Q_W, weight_size));
    CUDA_CHECK(cudaMalloc((void**)&d_K_W, weight_size));
    CUDA_CHECK(cudaMalloc((void**)&d_V_W, weight_size));

    cudaStream_t streamB, streamC, streamD;

    CUDA_CHECK(cudaStreamCreate(&streamB));
    CUDA_CHECK(cudaStreamCreate(&streamC));
    CUDA_CHECK(cudaStreamCreate(&streamD));

    cudaEvent_t weight_init_doneB, weight_init_doneC, weight_init_doneD;

    CUDA_CHECK(cudaEventCreate(&weight_init_doneB));
    CUDA_CHECK(cudaEventCreate(&weight_init_doneC));
    CUDA_CHECK(cudaEventCreate(&weight_init_doneD));

    dim3 weightThreadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 weightBlocksPerGrid(CEIL_DIV, CEIL_DIV);
    
    CUDA_CHECK(cudaRecordEvent(weight_init_doneB, streamB));
    CUDA_CHECK(cudaRecordEvent(weight_init_doneC, streamC));
    CUDA_CHECK(cudaRecordEvent(weight_init_doneD, streamD));
    

    CUDA_CHECK(init_random_matrix<<<weightBlocksPerGrid, weightThreadsPerBlock, 0, streamB>>>(d_Q_W, time(NULL) + 1));
    CUDA_CHECK(init_random_matrix<<<weightBlocksPerGrid, weightThreadsPerBlock, 0, streamC>>>(d_K_W, time(NULL) + 2));
    CUDA_CHECK(init_random_matrix<<<weightBlocksPerGrid, weightThreadsPerBlock, 0, streamD>>>(d_V_W, time(NULL) + 3));

    CUDA_CHECK(cudaStreamWaitEvent(streamA, weight_init_doneB));
    CUDA_CHECK(cudaStreamWaitEvent(streamA, weight_init_doneC));
    CUDA_CHECK(cudaStreamWaitEvent(streamA, weight_init_doneD));

    half *d_Q, *d_K, *d_V;
    CUDA_CHECK(cudaMalloc((void**)&d_Q, weight_size));
    CUDA_CHECK(cudaMalloc((void**)&d_K, weight_size));
    CUDA_CHECK(cudaMalloc((void**)&d_V, weight_size));

    cudaGraph_t graph; 
    cudaGraphExec_t graph_executable;

    cudaEvent_t record_attention;

    half* fused_QKV;

    int N3 = N * 3;

    size_t fused_size = sizeof(half) * N3 * d_k;

    dim3 fuse_tpb(16, 16);
    dim3 fuse_bpg((N3 + fuse_tpb.x - 1) / fuse_tpb.x, (N3 + fuse_tpb.y - 1) / fuse_tpb.y);

    CUDA_CHECK(cudaMalloc((void**)&fused_QKV, fused_size));

    CUDA_CHECK(cudaEventCreate(&record_attention));

    CUDA_CHECK(cudaStreamRecord(streamA, record_attention));

    CUDA_CHECK(pack_weights_fused<<<fuse_bpg, fuse_tpb, 0, streamA>>>(d_Q_W, d_K_W, d_V_W, fused_QKV, N, N, N3));




    CUDA_CHECK(cudaEventDestroy(weight_init_doneB));
    CUDA_CHECK(cudaEventDestroy(weight_init_doneC));
    CUDA_CHECK(cudaEventDestroy(weight_init_doneD));

    CUDA_CHECK(cudaEventDestroy(matmulWB));
    CUDA_CHECK(cudaEventDestroy(matmulWC));

    
}