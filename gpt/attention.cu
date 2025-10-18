/*
Multi-Head Flash Attention Optimized

Effectively use Flash Attention, but for each head, expand in the Z dimension.
*/

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


__device__ float warpReduceMax(float val, unsigned mask = 0xffffffffu) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val, unsigned mask = 0xffffffffu) {
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

        if (global_row < d && global_col < N) {
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], %2;\n" ::
                "r"(smem_ptr), "l"(gmem_ptr), "n"(bytes_per_cp)
            );
        }
    }
}
