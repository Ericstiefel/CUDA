#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>


#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 32  

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARPS_M 2
#define WARPS_N 4
#define WARP_SIZE 32

#define WARP_TILE_M 64
#define WARP_TILE_N 32

// -------------------------------------------------------------------------
// PTX HELPERS
// -------------------------------------------------------------------------

__device__ __forceinline__ unsigned int cvta_to_shared(const void* ptr) {
    return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_ca(void* smem, const void* gmem) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(cvta_to_shared(smem)), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group(const int n) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ float4* get_swizzled_ptr(float4* base, int row, int col_idx, int stride_in_float4) {
    int logical_idx = (row * stride_in_float4) + col_idx;
    int swizzled_idx = logical_idx ^ row; // XOR Row into Bank bits
    return &base[swizzled_idx];
}

__device__ __forceinline__ void ldmatrix_x4(unsigned int* regs, void* smem_ptr) {
    unsigned int addr = cvta_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2(unsigned int* regs, void* smem_ptr) {
    unsigned int addr = cvta_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(addr)
    );
}

__device__ __forceinline__ void mma_m16n8k16(float* acc, unsigned int* reg_a, unsigned int* reg_b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(reg_a[0]), "r"(reg_a[1]), "r"(reg_a[2]), "r"(reg_a[3]),
          "r"(reg_b[0]), "r"(reg_b[1])
    );
}

__device__ __forceinline__ void load_A_tile(
    half* smem, const half* gmem, 
    int m_offset, int k_offset, int M, int K) 
{

    int tid = threadIdx.x;
    int row = tid / 2;     
    int col = tid % 2;    
    
    // Bounds check
    if (m_offset + row < M && k_offset < K) {
        const float4* src = reinterpret_cast<const float4*>(&gmem[(m_offset + row) * K + k_offset + (col * 8)]);
        float4* dst = get_swizzled_ptr(reinterpret_cast<float4*>(smem), row, col, 2);
        cp_async_ca(dst, src);
    }
}

__device__ __forceinline__ void load_B_tile(
    half* smem, const half* gmem, 
    int k_offset, int n_offset, int K, int N) 
{
    
    int tid = threadIdx.x;
    int col_n = tid / 2; 
    int row_k = tid % 2; 

    if (n_offset + col_n < N && k_offset < K) {
        const float4* src = reinterpret_cast<const float4*>(&gmem[(n_offset + col_n) * K + k_offset + (row_k * 8)]);
        
        float4* dst = get_swizzled_ptr(reinterpret_cast<float4*>(smem), col_n, row_k, 2);
        cp_async_ca(dst, src);
    }
}


__global__ void optimized_gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    
    extern __shared__ alignas(128) half smem[];
    half* smem_A[2];
    half* smem_B[2];

    int stage_A_size = 128 * 16;
    int stage_B_size = 128 * 16;
    
    smem_A[0] = smem;
    smem_A[1] = smem + stage_A_size;
    smem_B[0] = smem + (2 * stage_A_size);
    smem_B[1] = smem + (2 * stage_A_size) + stage_B_size;
    
    float accum[4][4][4] = {0.0f}; 
    
    unsigned int frag_A[4][4]; 
    unsigned int frag_B[4][2]; 

    int tid = threadIdx.x;
    int warp_id = tid / 32;

    // Map Block to Global
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    // Warps are 2x4.
    int warp_row = (warp_id / 4); 
    int warp_col = (warp_id % 4); 
    
    int warp_m_offset = warp_row * 64;
    int warp_n_offset = warp_col * 32;

    int load_stage = 0;
    int comp_stage = 0;
    
    load_A_tile(smem_A[load_stage], A, block_row, 0, M, K);
    load_B_tile(smem_B[load_stage], B, 0, block_col, K, N);
    
    cp_async_commit();
    load_stage ^= 1;


    for (int k = 0; k < K; k += 16) {
        
        if (k + 16 < K) {
            load_A_tile(smem_A[load_stage], A, block_row, k + 16, M, K);
            load_B_tile(smem_B[load_stage], B, k + 16, block_col, K, N);
        }
        cp_async_commit();

        cp_async_wait_group(1);
        __syncthreads();

        #pragma unroll
        for(int m=0; m<4; m++) {
            float4* ptr = get_swizzled_ptr(reinterpret_cast<float4*>(smem_A[comp_stage]), warp_m_offset/8 + m*2, 0, 2); 
            ldmatrix_x4(frag_A[m], ptr);
        }
        #pragma unroll
        for(int n=0; n<4; n++) {
            float4* ptr = get_swizzled_ptr(reinterpret_cast<float4*>(smem_B[comp_stage]), warp_n_offset/8 + n*2, 0, 2); 
            ldmatrix_x2(frag_B[n], ptr);
        }

        #pragma unroll
        for(int m=0; m<4; m++) {
            #pragma unroll
            for(int n=0; n<4; n++) {
                mma_m16n8k16(accum[m][n], frag_A[m], frag_B[n]);
            }
        }

        __syncthreads(); 
        load_stage ^= 1;
        comp_stage ^= 1;
    }
    
    cp_async_wait_group(0);
    __syncthreads();

    
    
    int lane = tid % 32;
    int lane_row = lane / 4;
    int lane_col = (lane % 4) * 2;

    #pragma unroll
    for(int m=0; m<4; m++) {
        #pragma unroll
        for(int n=0; n<4; n++) {
            int global_row = block_row + warp_m_offset + (m * 16);
            int global_col = block_col + warp_n_offset + (n * 8);
            
            // Fragment contents
            // Reg 0: (row, col)
            // Reg 1: (row, col+1)
            // Reg 2: (row+8, col)
            // Reg 3: (row+8, col+1)
            
            if (global_row + lane_row < M && global_col + lane_col < N)
                C[(global_row + lane_row) * N + global_col + lane_col] = accum[m][n][0];
            
            if (global_row + lane_row < M && global_col + lane_col + 1 < N)
                C[(global_row + lane_row) * N + global_col + lane_col + 1] = accum[m][n][1];
                
            if (global_row + lane_row + 8 < M && global_col + lane_col < N)
                C[(global_row + lane_row + 8) * N + global_col + lane_col] = accum[m][n][2];
                
            if (global_row + lane_row + 8 < M && global_col + lane_col + 1 < N)
                C[(global_row + lane_row + 8) * N + global_col + lane_col + 1] = accum[m][n][3];
        }
    }
}