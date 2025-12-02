// Histogramming is a topic which is simple, and surprisingly useful. 

// How it works:
// We must use atomics to collapse race conditions
// But, if we would do global atomics, we would kill productivity, because no different blocks can add to
// the same number simultaneously.
// So, we gain a per block result histogram, and simply atomic add from there to create a global.


// For simplicity, we're assuming the min val is 0 (this leads to obvious changes when this is not the case)
__global__ void histogram(const int* __restrict__ inp, int* __restrict__ out, const int num_eles, const int max_val) {
    extern __shared__ int local_hist[]; // sizeof(int) * max_val

    int lid = threadIdx.x;
    int gid = lid + blockDim.x * blockIdx.x;

    for (int i = lid; i < max_val; i += blockDim.x) {
        local_hist[i] = 0;
    }

    __syncthreads();

    for (int i = gid; i < num_eles; i += blockDim.x * gridDim.x) {
        atomicAdd(&local_hist[inp[i]], 1);
    }

    __syncthreads();

    for (int i = lid; i < max_val; i += blockDim.x) {
        atomicAdd(&out[i], local_hist[i]);
    }

}