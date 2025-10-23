/*
Multi-Head Flash Attention Optimized

Effectively use Flash Attention, but for each head, expand in the Z dimension.
*/

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


__device__ __forceinline__ float warpReduceMax(float val, unsigned mask = 0xffffffffu) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val, unsigned mask = 0xffffffffu) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// __forceinline__ releases the overhead of a function call, keeping optimization
__device__ __forceinline__ void load_q_tile(
    half* __restrict__ d_Q,
    half* __restrict__ Q_tile,
    int tile_idx,
    int buf,
    int N,
    int d)
{
    const int elems_per_load = 8;  // 16 bytes / half = 8
    const int bytes_per_cp = 16;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    int tile_size = WMMA_M * WMMA_K;
    half* smem_tile_dest = Q_tile + buf * tile_size;

    for (int i = tid * elems_per_load; i < tile_size; i += threads_per_block * elems_per_load) {
        int r = i / WMMA_K;
        int c = i % WMMA_K;

        int global_row = blockIdx.y * WMMA_M + r;
        int global_col = tile_idx * WMMA_K + c;

        const half* gmem_ptr = d_Q + global_row * d + global_col;
        half* smem_ptr = smem_tile_dest + r * WMMA_K + c;

        if (global_row < N && global_col < d) {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "r"(smem_ptr), "l"(gmem_ptr), "n"(bytes_per_cp)
            );
        }
    }
}


__device__ __forceinline__ void load_kt_tile(
    half* __restrict__ d_K_t,
    half* __restrict__ Kt_tile,
    int tile_idx,
    int buf,
    int N,
    int d)
{
    const int bytes_per_cp = 16;
    const int elems_per_load = 8;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    int tile_size = WMMA_K * WMMA_N;
    half* smem_tile_dest = Kt_tile + buf * tile_size;

    for (int i = tid * elems_per_load; i < tile_size; i += threads_per_block * elems_per_load) {
        int r = i / WMMA_N;
        int c = i % WMMA_N;

        int global_row = tile_idx * WMMA_K + r;   // traverse d dimension
        int global_col = blockIdx.x * WMMA_N + c; // traverse N dimension

        const half* gmem_ptr = d_K_t + global_row * N + global_col;
        half* smem_ptr = smem_tile_dest + r * WMMA_N + c;

        if (global_row < d && global_col < N) {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "r"(smem_ptr), "l"(gmem_ptr), "n"(bytes_per_cp)
            );
        }
    }
}


__device__ __forceinline__ void load_v_tile(
    half* __restrict__ d_V,
    half* __restrict__ V_tile,
    int tile_idx,
    int buf,
    int N,
    int d)
{
    const int bytes_per_cp = 16;
    const int elems_per_load = 8;

    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    int tile_size = WMMA_K * WMMA_N;
    half* smem_dst = V_tile + buf * tile_size;

    for (int i = tid * elems_per_load; i < tile_size; i += threads_per_block * elems_per_load) {
        int r = i / WMMA_K;
        int c = i % WMMA_K;

        int global_row = tile_idx * WMMA_M + r;
        int global_col = blockIdx.x * WMMA_K + c;

        const half* gmem_ptr = d_V + global_row * d + global_col;
        half* smem_ptr = smem_dst + r * WMMA_K + c;

        if (global_row < N && global_col < d) {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "r"(smem_ptr), "l"(gmem_ptr), "n"(bytes_per_cp)
            );
        }
    }
}

// Each of the Q, K^T, V pointers now point to all the weight matrices of that group (d_Qs points to _ Q weight matrices). Z dim will indicate which one.
__global__ void masked_attention(const half* __restrict__ d_Qs,
                                const half* __restrict__ d_K_Ts,
                                const half* __restrict__ d_Vs,
                                half* __restrict__ d_out,
                                const int N,
                                const int d) 
{
    int weight_group = blockIdx.z;

    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;


    int start_point = N * d * weight_group;
    int end_point = N * d * (weight_group + 1); // Not inclusive

    const int threads_per_block = blockDim.x * blockDim.y;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int num_k_tiles = (d + WMMA_K - 1) / WMMA_K;

    __shared__ half Q_tile[2][WMMA_M][WMMA_K];
    __shared__ half Kt_tile[2][WMMA_K][WMMA_N];
    __shared__ half V_tile[2][WMMA_N][WMMA_K];

    __shared__ float qkt_shared[2][WMMA_M][WMMA_N];
    __shared__ float softmax_out[2][WMMA_M][WMMA_N];

    __shared__ half softmax_half_buf[2][WMMA_M][WMMA_N];

    __shared__ float row_max[WMMA_M];
    __shared__ float row_den[WMMA_M];

    __shared__ float rescale_factors[WMMA_M];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag; 
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> sm_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, float> final_c_frag;

    const half* d_Q = d_Qs + start_point;
    const half* d_K_T = d_K_Ts + start_point;
    const half* d_V = d_Vs + start_point;

    if (num_k_tiles > 0) {
        load_q_tile(d_Q, *Q_tile, 0, 0, N, d);
        load_kt_tile(d_K_T, *Kt_tile, 0, 0, N, d);
        load_v_tile(d_V, *V_tile, 0, 0, N, d);
    }
    asm volatile("cp.async.commit_group;");

    wmma::fill_fragment(final_c_frag, 0.0f);

    if (local_col == 0 && local_row < WMMA_M) {
        row_max[local_row] = -1e30f;
        row_den[local_row] = 0.0f;
        rescale_factors[local_row] = 1.0f;
    }
    __syncthreads();


    // Bit masking for ignoring lanes with incomplete data (the dims of threads and data don't align perfectly)

    unsigned full_mask = __activemask();
    unsigned active_mask;

    if (WMMA_N == 32) active_mask = full_mask;
    else active_mask = ( (WMMA_N >= 32) ? 0xffffffffu : ((1u << WMMA_N) - 1u) );

    for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx) {
        int compute_buf = k_tile_idx & 1;
        int next_k = k_tile_idx + 1;

        if (next_k < num_k_tiles) {
            int load_buf = next_k & 1;
            load_q_tile(d_Q, *Q_tile, next_k, load_buf, N, d);
            load_kt_tile(d_K_T, *Kt_tile, next_k, load_buf, N, d);
            load_v_tile(d_V, *V_tile, next_k, load_buf, N, d);
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");

        // Matmul Q @ K^t

        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, &Q_tile[compute_buf][0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &Kt_tile[compute_buf][0][0], WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        const float scale = (1.0f / sqrtf((float)d * gridDim.z) );
        for (int i = 0; i < c_frag.num_elements; ++i) { c_frag.x[i] *= scale;}

        wmma::store_matrix_sync(&qkt_shared[compute_buf][0][0], c_frag, WMMA_N, wmma::row_major);
        __syncthreads();


        // Online Softmax

        {
            const int r = local_row;
            const int lane = local_col;


            // compute global indices for this tile element
            int global_q = blockIdx.y * WMMA_M + r;        // query position (sequence index)
            int global_k = blockIdx.x * WMMA_N + lane;     // key position (sequence index)

            bool valid_q = (global_q < N);
            bool valid_k = (global_k < N);

            // causal mask condition: allow only key positions <= query position (<= for include self-attn)
            bool allowed = valid_q && valid_k && (global_k <= global_q);


            float my_val = (r < WMMA_M && lane < WMMA_N && allowed) 
                            ? qkt_shared[compute_buf][r][lane] 
                            : -1e9f; // mask out future / invalid positions


            unsigned mask = active_mask;

            float chunk_max = warpReduceMax(my_val, mask);
            float my_exp = (r < WMMA_M && lane < WMMA_N && allowed)
                 ? expf(my_val - chunk_max)
                 : 0.0f;
            float chunk_sum = warpReduceSum(my_exp, mask);


            float new_max_local = -1e9f;
            if ((lane & 0x1f) == 0 && r < WMMA_M) {
                float prev_max = row_max[r];
                float prev_den = row_den[r];

                if (prev_den == 0.0f) {
                    row_max[r] = chunk_max;
                    row_den[r] = chunk_sum;
                    rescale_factors[r] = 0.0f;
                }
                else {
                    float new_max = fmaxf(prev_max, chunk_max);
                    float scaled_prev = expf(prev_max - new_max) * prev_den;
                    float scaled_chunk = expf(chunk_max - new_max) * chunk_sum;
                    rescale_factors[r] = expf(prev_max - new_max);
                    row_max[r] = new_max;
                    row_den[r] = scaled_prev + scaled_chunk;
                }
                new_max_local = row_max[r];
            }

            float new_max = __shfl_sync(mask, new_max_local, 0); // Write local max to each thread in its warp

            float my_num = (r < WMMA_M && lane < WMMA_N) ? expf(my_val - new_max) : 0.0f;

            if (r < WMMA_M && lane < WMMA_N) {
                softmax_out[compute_buf][r][lane] = my_num;
            }
        }
        // End of online softmax
        // Converting to right sizing, scaling, and writing output steps now

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

        for (int i = 0; i < final_c_frag.num_elements; ++i) {
            int rr = i / WMMA_K;
            float scale_factor = rescale_factors[rr];
            final_c_frag.x[i] *= scale_factor;
        }


        wmma::load_matrix_sync(sm_frag, &softmax_half_buf[compute_buf][0][0], WMMA_N);
        wmma::load_matrix_sync(v_frag, &V_tile[compute_buf][0][0], WMMA_K);
        wmma::mma_sync(final_c_frag, sm_frag, v_frag, final_c_frag);
        // Everything is sync since last syncthreads, no need to call it again
    }

    // Write to global memory

    {
        float tmpC[WMMA_M * WMMA_K];
        wmma::store_matrix_sync(tmpC, final_c_frag, WMMA_K, wmma::row_major);

        const int total_out_elems = WMMA_M * WMMA_K;
        for (int i = tid; i < total_out_elems; i += threads_per_block) {
            int rr = i / WMMA_K;
            int cc = i % WMMA_K;
            int global_r = blockIdx.y * WMMA_M + rr;
            int global_c = blockIdx.x * WMMA_K + cc;

            if (global_r < N && global_c < d) {
                float denom = row_den[rr];
                float val = tmpC[rr * WMMA_K + cc];
                float scaled = (denom > 0.0f) ? (val / denom) : 0.0f;
                d_out[(global_r * d + global_c) + start_point] = __float2half(scaled);
                // d_out will be of dims num_heads * N x d
            }
        }

    }




    
    
}