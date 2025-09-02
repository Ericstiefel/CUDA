#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --- Constant memory for kernel ---
template <int KERNEL_WIDTH>
__constant__ half d_kernel[KERNEL_WIDTH * KERNEL_WIDTH];

// --- Kernel ---
template <int TILE_WIDTH, int KERNEL_WIDTH>
__global__ void convolutionTiling(const half* __restrict__ input,
                                  half* __restrict__ output,
                                  const int M, const int N) {
    // Global output coordinates
    int out_r = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int out_c = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Shared memory tile
    __shared__ half tile[TILE_WIDTH + KERNEL_WIDTH - 1][TILE_WIDTH + KERNEL_WIDTH - 1];

    int tile_r = threadIdx.y;
    int tile_c = threadIdx.x;

    int in_r = out_r - KERNEL_WIDTH / 2;
    int in_c = out_c - KERNEL_WIDTH / 2;

    if (in_r >= 0 && in_r < M && in_c >= 0 && in_c < N) {
        tile[tile_r][tile_c] = input[in_r * N + in_c];
    } else {
        tile[tile_r][tile_c] = __float2half(0.0f);
    }

    __syncthreads();

    if (out_r < M && out_c < N && tile_r < TILE_WIDTH && tile_c < TILE_WIDTH) {
        float sum = 0.0f;
        for (int i = 0; i < KERNEL_WIDTH; ++i) {
            for (int j = 0; j < KERNEL_WIDTH; ++j) {
                sum += __half2float(tile[tile_r + i][tile_c + j]) *
                       __half2float(d_kernel[i * KERNEL_WIDTH + j]);
            }
        }
        output[out_r * N + out_c] = __float2half(sum);
    }
}

// --- Launcher ---
template <int TILE_WIDTH, int KERNEL_WIDTH>
void tiling_launcher(const half* h_input,
                     const half* h_kernel,
                     half* d_output,
                     const int M, const int N) {
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel<KERNEL_WIDTH>, h_kernel,
                       KERNEL_WIDTH * KERNEL_WIDTH * sizeof(half));

    // Block and grid
    dim3 threadsPerBlock(TILE_WIDTH + KERNEL_WIDTH - 1,
                         TILE_WIDTH + KERNEL_WIDTH - 1);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                       (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch
    convolutionTiling<TILE_WIDTH, KERNEL_WIDTH>
        <<<blocksPerGrid, threadsPerBlock>>>(h_input, d_output, M, N);
}
