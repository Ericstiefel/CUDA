// This file is for a hyperoptimized Matmul that is the best I can do (from scratch obviously)
// From the Tensor Core optimizations, we're keeping everything async, but with a caveat.
// We now require M, N, K to be multiples of 8 (if not just pad with 0s and remove dims from matrix later).
// Here, we use CUDA's "special" float4 object, simply created to move textures really fast with 4 32 bit floats
// packed into a data type.

// We're going to continue using float16s for tensor core usage and pack 8 of these numbers into a float4.
// Now, we never deliberately "pack" it into a float4, but we just cast the memory locations into float4s.

// Since we now have 2 float16s where 1 float32 usually is, we can either use async(we do here) which 
// does this automatically, or if you're manually loading, use __low2half(float4.x,y,z,w) or __high2half() to 
// grab the lower or higher number packed in.



/*
All Simple Math Involved:

Conversion to Float4:
sizeof(half) = 2 bytes, 16 in a float4 ( 4 * 4 ).
Each 16 byte async transfer (GMEM to Shared) contains 8 half elements.

Global To Shared Loads Per Thread:
Tile dims: 128 x 32 = 4096 half values. K is smaller because the inner dim being smaller 
is more continuous of a process (less of the largest iteration)
= 512 float4s.
256 tpb, 2 float4s loaded per thread.

Warps Per Block:
256 / 32 = 8 warps per block.

128 x 128 output. 8 Warps.
MMA Design:
1 x 8: 128 x 16 = 128 loaded rows/cols & 16 loaded rows/cols = 144 rows/cols loaded
2 x 4: 64 x 32 = 96 rows/cols loaded
More square the pattern is, more efficient in this context.

acc[4][2] for this reason (16 x 16 output)


k_step = BK / 16 = 2 (main loop) 


*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 128
#define BN 128 
#define BK 32
#define WARPS 8 

// Bank Conflict Avoidance 
#define PAD 8            // Padding to skew banks (8 halves = 16 bytes)
#define (BK + PAD) // New Stride = 40 (divisible by 8 for float4 alignment!)

// Helper for Vector Arithmetic
// 40 halves / 8 per float4 = 5 float4s
#define BK_PAD_VEC (BK_PAD / 8) 

__device__ __forceinline__ void cp_async_128(void* smem_ptr, const void* gmem_ptr) {
    asm volatile (
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(__cvta_generic_to_shared(smem_ptr)), "l"(gmem_ptr)
    );
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile ("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile ("cp.async.wait_group %0;\n " :: "n"(N));
}


__device__ __forceinline__ void load_A_tile(const half* __restrict__ A_ptr, 
                                            half* __restrict__ smemA_ptr, 
                                            const int M, const int K, const int k_start_offset) 
{
    const float4* A_4ptr = reinterpret_cast<const float4*>(A_ptr);
    float4* smemA4_ptr = reinterpret_cast<float4*>(smemA_ptr);

    int block_row_start = blockIdx.y * 128;
    int current_k_col = k_start_offset;
    int k_vec_stride = K / 8;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int vec_id = threadIdx.x + (i * 256);
        
        int row_offset = vec_id / 4; 
        int col_offset = vec_id % 4;

        int final_row = block_row_start + row_offset;
        int final_col_vec = (current_k_col / 8) + col_offset;
        const float4* src = A_4ptr + (final_row * k_vec_stride) + final_col_vec;
        
        int smem_idx = (row_offset * BK_PAD_VEC) + col_offset;
        float4* dst = smemA4_ptr + smem_idx;

        cp_async_128(dst, src);
    }
}

__device__ __forceinline__ void load_B_tile(const half* __restrict__ B_ptr, 
                                            half* __restrict__ smemB_ptr, 
                                            const int K, const int N, const int k_start_offset) 
{
    const float4* B_4ptr = reinterpret_cast<const float4*>(B_ptr);
    float4* smemB4_ptr = reinterpret_cast<float4*>(smemB_ptr);

    int block_col_start = blockIdx.x * 128;
    int current_k_row = k_start_offset;
    int k_vec_stride = K / 8;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int vec_id = threadIdx.x + (i * 256);
        
        int col_offset = vec_id / 4;   
        int row_offset = vec_id % 4;     

        // Global Load
        int final_col = block_col_start + col_offset;
        int final_row_vec = (current_k_row / 8) + row_offset;
        const float4* src = B_4ptr + (final_col * k_vec_stride) + final_row_vec;

        int smem_idx = (col_offset * BK_PAD_VEC) + row_offset;
        float4* dst = smemB4_ptr + smem_idx;

        cp_async_128(dst, src);
    }
}

__global__ void gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C,
                     const int M, const int N, const int K) 
{
    // ALLOCATION SIZE UPDATE: Use Padded Stride
    // Size = 2 buffers * (Dim1 * Padded_Dim2)
    // A: BM * BK_PAD
    // B: BN * BK_PAD
    extern __shared__ alignas(16) half smem[]; 
    
    // Offset Logic must use PADDED sizes
    int A_size = BM * BK_PAD;
    int B_size = BN * BK_PAD;

    half* smem_A[2] = { smem, smem + A_size };
    half* smem_B[2] = { smem + (2 * A_size), smem + (2 * A_size) + B_size };

    int tid = threadIdx.x;
    int warp_num = tid / 32;
    int warp_row = warp_num / 4; // 4 here is the grid width of acc, not the same 4 used in the loaders.
    int warp_col = warp_num % 4;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) { 
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    load_A_tile(A, smem_A[0], M, K, 0);
    load_B_tile(B, smem_B[0], K, N, 0);

    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    int curr_buf = 0;
    int next_buf = 1;

    for (int k = 0; k < K - BK; k += BK) {
        
        load_A_tile(A, smem_A[next_buf], M, K, k + BK);
        load_B_tile(B, smem_B[next_buf], K, N, k + BK);
        cp_async_commit();

        #pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            
            // Load A Fragments
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = (warp_row * 64) + (i * 16);
                int col = k_step * 16;
                // USE PADDED STRIDE for pointer math
                const half* addr = smem_A[curr_buf] + (row * BK_PAD) + col;
                // TELL WMMA THE STRIDE IS PADDED
                wmma::load_matrix_sync(frag_a[i], addr, BK_PAD);
            }

            // Load B Fragments
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int col = (warp_col * 32) + (j * 16);
                int row = k_step * 16;
                // B is Col Major. Stride is BK_PAD.
                const half* addr = smem_B[curr_buf] + (col * BK_PAD) + row;
                wmma::load_matrix_sync(frag_b[j], addr, BK_PAD);
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], frag_a[i], frag_b[j], acc[i][j]);
                }
            }
        }
        cp_async_wait<0>();
        __syncthreads();

        curr_buf ^= 1;
        next_buf ^= 1;
    }

    // Process Final Tile
    #pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int row = (warp_row * 64) + (i * 16);
            const half* addr = smem_A[curr_buf] + (row * BK_PAD) + (k_step * 16);
            wmma::load_matrix_sync(frag_a[i], addr, BK_PAD);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int col = (warp_col * 32) + (j * 16);
            const half* addr = smem_B[curr_buf] + (col * BK_PAD) + (k_step * 16);
            wmma::load_matrix_sync(frag_b[j], addr, BK_PAD);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(acc[i][j], frag_a[i], frag_b[j], acc[i][j]);
            }
        }
    }

    int g_row = (blockIdx.y * 128) + (warp_row * 64);
    int g_col = (blockIdx.x * 128) + (warp_col * 32);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float* ptr = C + (g_row + i * 16) * N + (g_col + j * 16);
            wmma::store_matrix_sync(ptr, acc[i][j], N, wmma::mem_row_major);
        }
    }
}