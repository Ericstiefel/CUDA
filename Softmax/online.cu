#include <cuda_runtime.h>
#include <cmath>

/*
Taking advantage of e^{ab} = e^a * e^b, we can process normalization and summation in a single step
*/


__global__ void online_ker(const float* __restrict__ input, float* __restrict__ output, const int M, const int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        int row_i = blockIdx.x * blockDim.x;

        float curr_max = input[row_i];
        float sum = 1.0f; // First element will always be 1 (e^{curr_max - input[row_i]}, where curr_max = input[row_i])

        for (int k = 1; k < N; ++k) {
            float curr_num = input[row_i + k];

            if (curr_num > curr_max) { 
                sum = sum * exp(curr_max - curr_num);
                curr_max = curr_num;
            }

            sum = sum + exp(curr_num - curr_max);
        }

        for (int k = 0; k < N; ++k) {
            output[row_i + k] /= sum;
        }
    }
}

void online_launcher(float* input, float* output, int M, int N) {
    dim3 threadsPerBlock(1024); // 32 * 32
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    online_ker<<<blocksPerGrid, threadsPerBlock>>>(input, output, M, N);
}