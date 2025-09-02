#include <cuda_runtime.h>


__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (i >= output_size) return;

    float val = 0.0f;
    for (int m_i = 0; m_i < kernel_size; ++m_i) {
        val += input[i + m_i] * kernel[m_i];
    }
    output[i] = val;
}
