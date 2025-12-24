#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define KW 4
#define M 256
#define N 256

#define BM 32
#define BN 32

#define PAD 8
#define SMEM_STRIDE (PAD + BN)
#define SMEM_STRIDE_VEC (SMEM_STRIDE / 8)

// Further optimization is using tensor cores.

__constant__ half kernel[KW * KW]; 



__device__ __forceinline__ void async_128_bit(void* smem, void* gmem) {
    asm volatile (
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(__cvta_generic_to_shared(smem)), "l"(gmem)
    );
}

__device__ __forceinline__ void commit_group() {
    asm volatile (
        "cp.async.commit_group;\n" ::
    );
}

template <int N>
__device__ __forceinline__ void wait_group() {
    asm volatile (
        "cp.async.wait_group %0;\n" :: "n"(N)
    );
}

__device__ __forceinline__ void load_tile(
    half* __restrict__ smem,        
    const half* __restrict__ gmem, 
) {
    float4* smem4 = reinterpret_cast<float4*>(smem);
    const float4 gmem4 = reinterpret_cast<const float4*>(gmem);
    
    int data_width_vec = BN / 8; 

    int total_chunks = BM * data_width_vec;

    for (int i = threadIdx.x; i < total_chunks; i += blockDim.x) {

        int r = i / data_width_vec;
        int c = i % data_width_vec;

        const float4* from = gmem4 + (r * N / 8) + c;
        float4* to = smem4 + (r * SMEM_STRIDE_VEC + c);

        async_128_bit(to, from);
    }
}

__global__ void conv2d(const half* __restrict__ inp, half* __restrict__ out) {
    __shared__ alignas(16) half smem[BM * SMEM_STRIDE];

    int block_offset = (blockIdx.y * BM) * N + (blockIdx.x * BN);
    const half* tile_ptr = inp + block_offset;

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    load_tile(smem, tile_ptr);
    commit_group();
    wait_group<0>();
    __syncthreads();

    if (threadIdx.y < (BM - KW + 1) && threadIdx.x < (BN - KW + 1)) {
        float sum = 0.0f;

        #pragma unroll
        for (int fy = 0; fy < KW; ++fy) {
            #pragma unroll
            for (int fx = 0; fx < KW; ++fx) {
                int smem_idx = (threadIdx.y + fy) * SMEM_STRIDE + (threadIdx.x + fx);

                half val = smem[smem_idx];
                half w = kernel[fy * KW + fx];

                sum += __half2float(val) * __half2float(w);
            }
        }
    }

    int out_row = blockIdx.y * BM + threadIdx.y;
    int out_col = blockIdx.x * BN + threadIdx.x;

    out[out_row * N + out_col] = sum;
}