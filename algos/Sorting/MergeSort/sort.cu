#include <cuda_runtime.h>

__global__ void bitonic_local_sort(const int* __restrict__ inp, int* __restrict__ block_sorts, const int N) {
    extern __shared__ int local_vals[]; // sizeof(int) * tpb.x
    int lid = threadIdx.x;
    int gid = lid + blockDim.x * blockIdx.x;

    if (gid < N) {
        local_vals[lid] = inp[gid];
    }
    else {
        local_vals[lid] = 2147483647;
    }

    // One reason we assume N = 2^k, k integer, is because otherwise we'd be pairing some numbers that did not input
    // to shared memory in the last block, and run into some errors. Another is the pure efficiency of the loop.
    __syncthreads();

    // here we're assuming tpb.x is also 2^p, p integer.
    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int stride = k / 2; stride > 0; stride /= 2) {
            bool ascending = (lid & k) == 0;

            int partner = stride ^ lid;

            int your_val = local_vals[lid];
            int partner_val = local_vals[partner];

            int max_val = (your_val > partner_val) ? your_val : partner_val;
            int min_val = (your_val <= partner_val) ? your_val : partner_val;

            if (lid < partner) {
                if (ascending) {
                    local_vals[lid] = min_val;
                    local_vals[partner] = max_val;
                }
                else {
                    local_vals[lid] = max_val;
                    local_vals[partner] = min_val;
                }
            }

            __syncthreads();

        }
        
    }

    if (gid < N) {
        block_sorts[gid] = local_vals[lid];
    }
}

__device__ int binary_search_lower(const int* __restrict__ vals, const int num, const int size) {
    int l = 0; int r = size;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (vals[mid] < num) { l = mid + 1; }
        else { r = mid; }
    }
    return l;
}

__device__ int binary_search_upper(const int* __restrict__ vals, const int num, const int size) {
    int l = 0; int r = size;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (vals[mid] <= num) { l = mid + 1; }
        else { r = mid; }
    }
    return l;
}

/*
  Global Merge Sort:
  Merges two adjacent sorted chunks (width) into a sorted chunk of size (2*width).
  Reads directly from Global Memory, allowing infinite chunk sizes.
*/
__global__ void global_merge(const int* __restrict__ inp, int* __restrict__ out, int N, int width) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= N) return;


    int chunk_idx = idx / width;
    

    int neighbor_chunk_idx = chunk_idx ^ 1;

    int my_chunk_start = chunk_idx * width;
    int neighbor_chunk_start = neighbor_chunk_idx * width;
    
    if (neighbor_chunk_start >= N) {
        out[idx] = inp[idx];
        return;
    }

    int my_valid_size = min(width, N - my_chunk_start);
    int neighbor_valid_size = min(width, N - neighbor_chunk_start);

    int my_val = inp[idx];
    
    int my_offset = idx - my_chunk_start;
    
    int rank_in_neighbor;
    

    if (chunk_idx < neighbor_chunk_idx) {
        rank_in_neighbor = binary_search_lower(
            &inp[neighbor_chunk_start], 
            my_val, 
            neighbor_valid_size
        );
    } 

    else {
        rank_in_neighbor = binary_search_upper(
            &inp[neighbor_chunk_start], 
            my_val, 
            neighbor_valid_size
        );
    }

    int merged_pair_start = min(my_chunk_start, neighbor_chunk_start);
    
    int destination_idx = merged_pair_start + my_offset + rank_in_neighbor;

    out[destination_idx] = my_val;
}


void run_merge_sort(int* d_data, int* d_buffer, int N) {
    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    size_t sharedMemSize = threadsPerBlock * sizeof(int);

    int* d_in = d_buffer;
    int* d_out = d_data;


    bitonic_block_sort<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_data, d_in, N);
    

    for (int width = threadsPerBlock; width < N; width *= 2) {
        
        int global_blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        
        global_merge<<<global_blocks, threadsPerBlock>>>(d_in, d_out, N, width);
        
        int* temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    // If the final result ended up in d_buffer (which is d_in now), copy it back to d_data
    if (d_in != d_data) {
        cudaMemcpy(d_data, d_in, N * sizeof(int), cudaMemcpyDeviceToDevice);
    }
}

