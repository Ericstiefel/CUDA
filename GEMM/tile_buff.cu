#include <cuda_runtime.h>

/*
The strategy here is to load one tile from global memory into one buffer, while the second is computing. Because there is no dependency, these operations can execute simultaneously.
Must preload a tile into memory for intialization
*/


template <int TILE_WIDTH_T> 
__global__ void tiled_double_buffer(const float* A, const float* B, float* C, const int M, const int K, const int N) {
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int global_row = blockIdx.y * TILE_WIDTH_T + local_row;
    int global_col = blockIdx.x * TILE_WIDTH_T + local_col;


    __shared__ float A_tile[2][TILE_WIDTH_T][TILE_WIDTH_T];
    __shared__ float B_tile[2][TILE_WIDTH_T][TILE_WIDTH_T];

    float sum = 0.0f;
    int num_tiles = (K + TILE_WIDTH_T - 1) / TILE_WIDTH_T;
    
    int read_buf = 0;
    int write_buf = 1;

    int first_tile_a_col = 0 * TILE_WIDTH_T + local_col;
    int first_tile_b_row = 0 * TILE_WIDTH_T + local_row;
    
    if (global_row < M && first_tile_a_col < K) {
        A_tile[read_buf][local_row][local_col] = A[global_row * K + first_tile_a_col];
    } else {
        A_tile[read_buf][local_row][local_col] = 0.0f;
    }
    if (first_tile_b_row < K && global_col < N) {
        B_tile[read_buf][local_row][local_col] = B[first_tile_b_row * N + global_col];
    } else {
        B_tile[read_buf][local_row][local_col] = 0.0f;
    }
    __syncthreads();



    for (int tile = 1; tile < num_tiles; ++tile) {
        

        int a_col = tile * TILE_WIDTH_T + local_col;
        int b_row = tile * TILE_WIDTH_T + local_row;

        if (global_row < M && a_col < K) {
            A_tile[write_buf][local_row][local_col] = A[global_row * K + a_col];
        } else {
            A_tile[write_buf][local_row][local_col] = 0.0f;
        }
        if (b_row < K && global_col < N) {
            B_tile[write_buf][local_row][local_col] = B[b_row * N + global_col];
        } else {
            B_tile[write_buf][local_row][local_col] = 0.0f;
        }
        
        for (int i = 0; i < TILE_WIDTH_T; ++i) {
            sum += A_tile[read_buf][local_row][i] * B_tile[read_buf][i][local_col];
        }
        
        int temp = read_buf;
        read_buf = write_buf;
        write_buf = temp;

        __syncthreads();
    }

    for (int i = 0; i < TILE_WIDTH_T; ++i) {
        sum += A_tile[read_buf][local_row][i] * B_tile[read_buf][i][local_col];
    }
    
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = sum;
    }
}

template <int TILE_WIDTH>
void tiled_buff_launcher(const float* d_A, const float* d_B, float* d_C, const int M, const int K, const int N) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);

    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_double_buffer<TILE_WIDTH><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
}

// This line tells the compiler to create the code for the TILE_WIDTH = 32 version.
template void tiled_buff_launcher<32>(const float*, const float*, float*, int, int, int);