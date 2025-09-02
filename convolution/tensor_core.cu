#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace nvcuda;


template <int KERNEL_WIDTH>
__constant__ half d_kernel[KERNEL_WIDTH][KERNEL_WIDTH];

template <int TILE_WIDTH, int KERNEL_WIDTH>
__global__ void tc(
    const __restrict__ half* input,
    __restrict__ half* output,
    const int M, 
    const int N
) {
    constexpr int TILE_DIM = TILE_WIDTH + KERNEL_WIDTH - 1;


    __shared__ half tiles_in[2][TILE_DIM][TILE_DIM];
    __shared__ half tiles_out[TILE_WIDTH][TILE_WIDTH];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;


    constexpr int radius = KERNEL_WIDTH / 2;

    const int start_inp_row = by * TILE_WIDTH - radius;

    // TILE_WIDTH must be a multiple of 16.
    constexpr int WARP_TILE_X = 16;
    constexpr int WARP_TILE_Y = 16;
    constexpr int WARPS_PER_BLOCK_X = TILE_WIDTH / WARP_TILE_X;
    constexpr int WARPS_PER_BLOCK_Y = TILE_WIDTH / WARP_TILE_Y;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    const int inp_col_preload = bx * TILE_WIDTH - radius;
    const int thread_load_row = start_inp_row + ty;
    const int thread_load_col = inp_col_preload + tx;

    for (int i = 0; i < TILE_DIM; i += blockDim.y) {
        for (int j = 0; j < TILE_DIM; j += blockDim.x) {
            int inp_r = start_inp_row + ty + i;
            int inp_c = inp_col_preload + tx + j;

            half value = (inp_r >= 0 && inp_r < M && inp_c >= 0 && inp_c < N) ? input[inp_r * N + inp_c] : __float2half(0.0f);
            tiles_in[0][ty + i][tx + j] = value;
        }
    }

    __syncthreads();

    int compute_idx = 0;
    const int num_cols_tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;


    for (int tile = 0; tile < num_cols_tiles; ++tile) {

        const int next_tile_col = tile + 1;
        const int load_idx = compute_idx ^ 1;

        // Asynchronously load the next input tile if it exists.
        if (next_tile_col < num_cols_tiles) {
            const int inp_c_nxt = next_tile_col * TILE_WIDTH - radius;
            for (int i = 0; i < TILE_DIM; i += blockDim.y) {
                for (int j = 0; j < TILE_DIM; j += blockDim.x) {
                    int inp_r = start_inp_row + ty + i;
                    int inp_c = inp_c_nxt + tx + j;
                    half value = (inp_r >= 0 && inp_r < M && inp_c >= 0 && inp_c < N) ? input[inp_r * N + inp_c] : __float2half(0.0f);
                    tiles_in[load_idx][ty + i][tx + j] = value;
                }
            }
        }


        __syncthreads();

        for (int i_frag = 0; i_frag < TILE_WIDTH; i_frag += 16) {
            for (int j_frag = 0; j_frag < TILE_WIDTH; j_frag += 16) {

                for (int k_frag = 0; k_frag < KERNEL_WIDTH; ++k_frag) {

                    wmma::load_matrix_sync(a_frag, &tiles_in[compute_idx][i_frag][j_frag + k_frag], TILE_DIM);
                    wmma::load_matrix_sync(b_frag, &d_kernel[0][k_frag]);

                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                wmma::store_matrix_sync(&tiles_out[i_frag][j_frag], c_frag, TILE_WIDTH, wmma::row_major);
            }
        }

        __syncthreads();

        const int global_output_row_start = by * TILE_WIDTH;
        const int global_output_col_start = bx * TILE_WIDTH;
        const int global_output_row = global_output_row_start + ty;
        const int global_output_col = global_output_col_start + tx;

        if (global_output_row < M && global_output_col < N) {
            output[global_output_row * N + global_output_col] = tiles_out[ty][tx];
        }


        __syncthreads();
        
    }
}
