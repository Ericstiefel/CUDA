#include <cuda_runtime.h>

/*
2D Convolution Kernel, naive

input \in\mathbb{M x N}

Each thread will compute one element of the output matrix

No padding

2D kernel \in\mathbb{h x k}


To remain within bounds, row +- kernel_radius && col +- kernel_radius < M, N

*/

#include <cuda_runtime.h>

__global__ void naive_corrected(const half* input, const half* kernel, half* output, const int M, const int N, const int kernel_radius) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int kernel_width = kernel_radius * 2 + 1;

    if (row >= kernel_radius && row < (M - kernel_radius) &&
        col >= kernel_radius && col < (N - kernel_radius)) {
        
        float sum = 0.0f;

        for (int ker_row = -kernel_radius; ker_row <= kernel_radius; ++ker_row) {
            for (int ker_col = -kernel_radius; ker_col <= kernel_radius; ++ker_col) {
                
                int input_row = row + ker_row;
                int input_col = col + ker_col;

                // Map kernel indices from [-radius, radius] to [0, 2*radius]
                int kernel_idx_row = ker_row + kernel_radius;
                int kernel_idx_col = ker_col + kernel_radius;

                sum += __half2float(input[input_row * N + input_col]) *
                       __half2float(kernel[kernel_idx_row * kernel_width + kernel_idx_col]);
            }
        }
        
        output[row * N + col] = __float2half(sum);
    }
}

// M rows, N cols
dim3 block(16, 16);
dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

conv2d_naive<<<grid, block>>>(d_input, d_kernel, d_output, M, N, kernel_radius);
cudaDeviceSynchronize();
