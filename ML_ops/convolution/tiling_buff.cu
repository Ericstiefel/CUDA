/*
This file will use double buffering in an attempt to hide memory latency during tile loading.

The strategy is as follows:
Each thread block is assigned a ROW of output tiles.
The block iterates through the tiles in its assigned row.

Each block will compute (N + TILE_WIDTH - 1) / TILE_WIDTH tiles.

For each tile:
    (1) Asynchronously load the NEXT input tile into one shared memory buffer (e.g., Buffer B).
    (2) Simultaneously, compute the convolution using the CURRENT input tile, which is already in the other shared memory buffer (e.g., Buffer A).
    (3) Store the result for the current tile to global memory.
    (4) Synchronize all threads in the block.
    (5) Swap the roles of Buffer A and B and move to the next tile.
*/

// Declare Constant Memory for the convolution filter (Loaded in the host launcher)
template <int KERNEL_WIDTH>
__constant__ half d_kernel[KERNEL_WIDTH][KERNEL_WIDTH];


template <int TILE_WIDTH, int KERNEL_WIDTH>
__global__ void tiling_buff(
    half* __restrict__ output, 
    const half* __restrict__ input,
    const int M, // Input Height
    const int N  // Input Width
) {

    constexpr int TILE_DIM = TILE_WIDTH + KERNEL_WIDTH - 1;

    __shared__ half tiles[2][TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // This block is responsible for processing one row of output tiles.
    // blockIdx.y selects the row. The kernel will loop over the columns.
    const int block_row = blockIdx.y;
    
    constexpr int r = KERNEL_WIDTH / 2;

    // Calculate the base row for all input tiles this block will access.
    // This is constant for a given block.
    const int inp_r_base = block_row * TILE_WIDTH - r;

    const int num_tiles_x = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    // Preparation: Loading in first tile
    int inp_c_base_prologue = 0 * TILE_WIDTH - r; // For tile_x = 0
    for (int i = ty; i < TILE_DIM; i += TILE_WIDTH) {
        for (int j = tx; j < TILE_DIM; j += TILE_WIDTH) {
            int inp_r = inp_r_base + i;
            int inp_c = inp_c_base_prologue + j;

            // Check boundaries and pad with 0 if out of bounds
            if (inp_r >= 0 && inp_r < M && inp_c >= 0 && inp_c < N) {
                tiles[0][i][j] = input[inp_r * N + inp_c];
            } else {
                tiles[0][i][j] = __float2half(0.0f);
            }
        }
    }
    
    __syncthreads();

    int compute_buffer_idx = 0;
    

    for (int tile_x = 0; tile_x < num_tiles_x; ++tile_x) {

        const int next_tile_x = tile_x + 1;
        const int load_buffer_idx = compute_buffer_idx ^ 1; 

        if (next_tile_x < num_tiles_x) {
            const int inp_c_base_next = next_tile_x * TILE_WIDTH - r;
            
            for (int i = ty; i < TILE_DIM; i += TILE_WIDTH) {
                for (int j = tx; j < TILE_DIM; j += TILE_WIDTH) {
                    int inp_r = inp_r_base + i;
                    int inp_c = inp_c_base_next + j;
                    
                    if (inp_r >= 0 && inp_r < M && inp_c >= 0 && inp_c < N) {
                        tiles[load_buffer_idx][i][j] = input[inp_r * N + inp_c];
                    } else {
                        tiles[load_buffer_idx][i][j] = __float2half(0.0f);
                    }
                }
            }
        }

        half sum = __float2half(0.0f);
        
        for (int i = 0; i < KERNEL_WIDTH; ++i) {
            for (int j = 0; j < KERNEL_WIDTH; ++j) {
                sum += tiles[compute_buffer_idx][ty + i][tx + j] * d_kernel[i][j];
            }
        }

        const int out_r = block_row * TILE_WIDTH + ty;
        const int out_c = tile_x * TILE_WIDTH + tx;
    
        if (out_r < M && out_c < N) {
            output[out_r * N + out_c] = sum;
        }
        __syncthreads();

        compute_buffer_idx ^= 1; 
    }
}