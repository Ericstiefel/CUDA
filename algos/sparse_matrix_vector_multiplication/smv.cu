#include <cuda_runtime.h>
#include <stdio.h>

__global__ void get_elems_per_row(const int* __restrict__ A, int* __restrict__ row_sums, const int row_len) {
    int row = threadIdx.x;
    int start = row_len * threadIdx.x;
    int end = row_len * (threadIdx.x + 1);

    int val = 0;
    for (int i = start; i < end; i += 1) {
        val += (A[i] != 0); 
    }
    row_sums[row] = val;
}

// Launch with 1 block, N threads
// Kogge Stone is an inclusive scan (belloch is exclusive), so we shift the numbers and add 0 to the front to transition
__global__ void prefix_sum(const int* __restrict__ A, int* __restrict__ out, const int N) {
    extern __shared__ int vals[]; 
    int id = threadIdx.x;

    // Load data
    if (id < N) {
        vals[id] = A[id];
    }

    int tin = 0;
    int tout = N;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads(); 
        
        if (id >= stride) {
            vals[tout + id] = vals[tin + id] + vals[tin + id - stride];
        }
        else {
            vals[tout + id] = vals[tin + id];
        }
        
        // Swap buffers
        int temp = tin;
        tin = tout;
        tout = temp;
    }
    
    __syncthreads();

    if (id < N) {
        if (id == 0) {
            out[0] = 0;
        }
        out[id + 1] = vals[tin + id];
    }
}

// Launch with 1 block, N threads
__global__ void get_csr(
    const int* __restrict__ A, 
    int* __restrict__ vals, 
    int* __restrict__ cols, 
    const int* __restrict__ row_ptr, 
    const int N,
    const int row_len
    ) {
        int id = threadIdx.x;
        if (id >= N) return;

        int row_start = row_len * id;
        int row_end = row_len * (id + 1);

        int current_write_pos = row_ptr[id]; 

        for (int i = row_start; i < row_end; i += 1) {
            int val = A[i];

            if (val != 0) { 
                vals[current_write_pos] = val;
                cols[current_write_pos] = i % row_len; 
                current_write_pos++; 
            }
        }
}

__global__ void smv(
    const int* __restrict__ vec, 
    const int* __restrict__ vals,
    const int* __restrict__ cols,
    const int* __restrict__ row_ptr,
    int* __restrict__ out,
    const int N) {
        
        int row = threadIdx.x;
        if (row >= N) return;

        int num = 0;

        // Iterate from row start to row end
        for (int i = row_ptr[row]; i < row_ptr[row + 1]; i += 1) {
            num += vec[cols[i]] * vals[i];
        }

        out[row] = num;
}