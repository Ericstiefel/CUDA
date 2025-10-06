// attention_with_existing_kernels_fused_init.cu
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <ctime>
#include <cmath>

// Reminder for later: Convert tile loads from DRAM to Asynch

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


__device__ float warpReduceMax(float val, unsigned mask = 0xffffffff) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    return val;
}

__device__ float warpReduceSum(float val, unsigned mask = 0xffffffff) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}



__global__ void attention(const half* __restrict__ d_Q,
                          const half* __restrict__ d_K_T,
                          const half* __restrict__ d_V,
                          half* __restrict__ d_out,
                          const int N,    // sequence length
                          const int d)    // head dimension
{
    // thread / block coords
    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int tid = local_row * blockDim.x + local_col;

    const int tile_row_idx = blockIdx.y; // which Q-row tile (along N)
    const int tile_col_idx = blockIdx.x; // which output-column tile (along d)

    const int num_k_tiles = (d + WMMA_K - 1) / WMMA_K;

    // Shared memory tiles & buffers

    __shared__ half Q_tile[2][WMMA_M][WMMA_K];        // M x K  (row-major)
    __shared__ half Kt_tile[2][WMMA_K][WMMA_N];      // K x N  (row-major)
    __shared__ half V_tile[2][WMMA_N][WMMA_K];       // N x K  (row-major)

    // keep float intermediates for accuracy
    __shared__ float qkt_shared[2][WMMA_M][WMMA_N];  // M x N (float)
    __shared__ float softmax_out[2][WMMA_M][WMMA_N]; // numerators as float

    __shared__ float row_max[WMMA_M];
    __shared__ float row_den[WMMA_M];
    // rescale factor to apply to previous numerator O_old when max changes:
    __shared__ float rescale_factors[WMMA_M];

    // -----------------------------
    // WMMA fragments (row-major layout)
    // For Q*K^T : A(MxK) * B(KxN) -> C(MxN)   -> use (M,N,K)
    // For softmax*V: A(MxN) * B(NxK) -> C(MxK) -> use (M,K,N)
    // -----------------------------
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> sm_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> final_c_frag;

    // -----------------------------
    // Load helpers (shared mem fills)
    // -----------------------------
    auto load_q_tile = [&](int k_tile_idx, int buf) {
        const int flat = WMMA_M * WMMA_K;
        for (int i = tid; i < flat; i += threads_per_block) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int global_r = tile_row_idx * WMMA_M + r;       // along N
            int global_c = k_tile_idx * WMMA_K + c;         // along d (head)
            bool in_bounds = (global_r < N && global_c < d);
            Q_tile[buf][r][c] = in_bounds ? d_Q[global_r * d + global_c] : __float2half(0.0f);
        }
    };

    auto load_kt_tile = [&](int k_tile_idx, int buf) {
        const int flat = WMMA_K * WMMA_N;
        for (int i = tid; i < flat; i += threads_per_block) {
            int r = i / WMMA_N; // K index (0..WMMA_K-1)
            int c = i % WMMA_N; // N index (0..WMMA_N-1)
            int global_r = k_tile_idx * WMMA_K + r;         // K (head) index
            int global_c = tile_row_idx * WMMA_N + c;       // sequence index (N)
            bool in_bounds = (global_r < d && global_c < N);
            // d_K_T is stored as K x N with stride N, so row-major access is: d_K_T[row * N + col]
            Kt_tile[buf][r][c] = in_bounds ? d_K_T[global_r * N + global_c] : __float2half(0.0f);
        }
    };

    auto load_v_tile = [&](int k_tile_idx, int buf) {
        const int flat = WMMA_N * WMMA_K;
        for (int i = tid; i < flat; i += threads_per_block) {
            int r = i / WMMA_K; // N index (0..WMMA_N-1)
            int c = i % WMMA_K; // K index (0..WMMA_K-1)
            int global_r = tile_row_idx * WMMA_N + r;       // sequence index (N)
            int global_c = k_tile_idx * WMMA_K + c;         // head index (d)
            bool in_bounds = (global_r < N && global_c < d);
            V_tile[buf][r][c] = in_bounds ? d_V[global_r * d + global_c] : __float2half(0.0f);
        }
    };


    wmma::fill_fragment(final_c_frag, 0.0f); 

    if (local_col == 0 && local_row < WMMA_M) {
        row_max[local_row] = -1e30f;
        row_den[local_row] = 0.0f;
        rescale_factors[local_row] = 1.0f; // neutral initial scale
    }
    __syncthreads();


    // PREP: Preloading into first buffer
    if (num_k_tiles > 0) {
        load_q_tile(0, 0);
        load_kt_tile(0, 0);
        load_v_tile(0, 0);
    }
    __syncthreads();

    for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        int compute_buf = k_tile_idx & 1;
        int next_k = k_tile_idx + 1;
        // start loading next tile into the other buffer (if it exists)
        if (next_k < num_k_tiles) {
            int load_buf = next_k & 1;
            load_q_tile(next_k, load_buf);
            load_kt_tile(next_k, load_buf);
            load_v_tile(next_k, load_buf);
        }
        __syncthreads(); // ensure compute_buf is ready

        // ---------------------
        // Q * K^T -> c_frag (M x N)
        // ---------------------
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, &Q_tile[compute_buf][0][0], WMMA_K);   // ld = WMMA_K (Q row length)
        wmma::load_matrix_sync(b_frag, &Kt_tile[compute_buf][0][0], WMMA_N);  // ld = WMMA_N (Kt row length)
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        const float scale = 1.0f / sqrtf((float)d);
        for (int i = 0; i < c_frag.num_elements; ++i) c_frag.x[i] *= scale;

        // store partial (M x N) into float qkt_shared
        wmma::store_matrix_sync(&qkt_shared[compute_buf][0][0], c_frag, WMMA_N, wmma::mem_row_major);
        __syncthreads();

        // ---------------------
        // Online Softmax (per-row, warp-parallel)
        // ---------------------
        {
            const int r = local_row; // row in WMMA_M
            const int c = local_col; // col in WMMA_N
            float my_val = (r < WMMA_M && c < WMMA_N) ? qkt_shared[compute_buf][r][c] : -INFINITY;

            // each row handled by one warp: warp_id = r, lane_id = c
            unsigned mask = __activemask();

            // compute row max across warp lanes
            float chunk_max = warpReduceMax(my_val, mask);

            // exponentiate relative to chunk_max
            float my_exp = (r < WMMA_M && c < WMMA_N) ? expf(my_val - chunk_max) : 0.0f;

            // compute row sum across warp lanes
            float chunk_sum = warpReduceSum(my_exp, mask);

            // We will compute a per-row rescale factor (= exp(prev_max - new_max))
            // lane 0 updates running stats and writes rescale_factors[row]
            float new_max_local = -INFINITY;
            if (c == 0 && r < WMMA_M) {
                float prev_max = row_max[r];
                float prev_den = row_den[r];

                if (prev_den == 0.0f) {
                    // first tile for this row
                    row_max[r] = chunk_max;
                    row_den[r] = chunk_sum;
                    // previous numerator is zero so scaling factor = 0 (O_old * 0 = 0)
                    rescale_factors[r] = 0.0f;
                } else {
                    float new_max = fmaxf(prev_max, chunk_max);
                    float scaled_prev = expf(prev_max - new_max) * prev_den;
                    float scaled_chunk = expf(chunk_max - new_max) * chunk_sum;
                    // scaling factor to apply to the previous numerator O_old:
                    rescale_factors[r] = expf(prev_max - new_max); // = scaled_prev / prev_den
                    row_max[r] = new_max;
                    row_den[r] = scaled_prev + scaled_chunk;
                }
                new_max_local = row_max[r];
            }

            // broadcast updated max to all lanes in warp from lane 0
            float new_max = __shfl_sync(mask, new_max_local, 0);

            // normalized numerator (w.r.t. new_max)
            float my_num = (r < WMMA_M && c < WMMA_N) ? expf(my_val - new_max) : 0.0f;
            if (r < WMMA_M && c < WMMA_N) {
                softmax_out[compute_buf][r][c] = my_num;
            }
        } // end online softmax

        // convert softmax_out (float) -> half buffer for WMMA
        __shared__ half softmax_half_buf[2][WMMA_M][WMMA_N];
        {
            const int flat = WMMA_M * WMMA_N;
            for (int i = tid; i < flat; i += threads_per_block) {
                int r = i / WMMA_N;
                int c = i % WMMA_N;
                float v = softmax_out[compute_buf][r][c];
                softmax_half_buf[compute_buf][r][c] = __float2half(v);
            }
        }
        __syncthreads();

        // ---------------------
        // Rescale the existing numerator accumulator (final_c_frag) if needed
        // then compute softmax_chunk * V_tile and accumulate into final_c_frag
        // ---------------------
        // ensure rescale_factors are visible
        __syncthreads();

        // Apply per-row rescale factor to the accumulator fragment.
        // final_c_frag is small; iterate over its elements and multiply by the row scale.
        // Mapping: element index i -> row = i / WMMA_K (matches mem_row_major packing used elsewhere)
        for (int i = 0; i < final_c_frag.num_elements; ++i) {
            int rr = i / WMMA_K; // row within 0..WMMA_M-1
            float scale_factor = rescale_factors[rr];
            // if scale_factor==1.0f this is a cheap multiply; you may guard it if you prefer
            final_c_frag.x[i] *= scale_factor;
        }

        // Now add the new chunk: softmax_chunk (M x N) * V_tile (N x K) -> accumulate into final_c_frag (M x K)
        wmma::load_matrix_sync(sm_frag, &softmax_half_buf[compute_buf][0][0], WMMA_N); // ld = WMMA_N
        wmma::load_matrix_sync(v_frag,  &V_tile[compute_buf][0][0],           WMMA_K); // ld = WMMA_K
        wmma::mma_sync(final_c_frag, sm_frag, v_frag, final_c_frag);
        __syncthreads();
    } // end k_tile loop

    // ---------------------
    // Write back final_c_frag (M x K) -> d_out (with normalization by row_den)
    // ---------------------
    {
        // temporary storage in registers/shared: store fragment to local float array
        float tmpC[WMMA_M * WMMA_K];
        wmma::store_matrix_sync(tmpC, final_c_frag, WMMA_K, wmma::mem_row_major); // ld = WMMA_K

        const int total_out_elems = WMMA_M * WMMA_K;
        for (int i = tid; i < total_out_elems; i += threads_per_block) {
            int rr = i / WMMA_K; // local row in tile
            int cc = i % WMMA_K; // local col in tile
            int global_r = tile_row_idx * WMMA_M + rr;
            int global_c = tile_col_idx * WMMA_K + cc;
            if (global_r < N && global_c < d) {
                // divide by running denominator row_den[rr] to finalize softmax normalization
                float denom = row_den[rr];
                float val = tmpC[ rr * WMMA_K + cc ];
                float scaled = denom > 0.0f ? (val / denom) : 0.0f;
                d_out[ global_r * d + global_c ] = __float2half(scaled);
            }
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
