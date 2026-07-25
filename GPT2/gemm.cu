#include <cuda_runtime.h>
#include <cuda_fp16.h>


__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t swizzle_address(const half* base_smem, int row, int col, int stride) {
    int vals_per_vec = 16 / sizeof(half); // 8 2 byte half values fit inside 1 transfer 16 byte load
    int vecs_per_row = stride / vals_per_vec;

    int chunk_idx = col / vals_per_vec;
    int offset = col % vals_per_vec;

    int swizzled_col = chunk_idx ^ ((row % 8) % vecs_per_row);
    int flat_idx = row * stride + swizzled_col * vals_per_vec + offset;
    return smem_ptr_to_uint(base_smem + flat_idx);
}


__device__ __forceinline__ void cp_async_128(uint32_t* smem_addr, const void* gmem_ptr) { // gmem -> smem async 
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n" 
        :: "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile (
        "cp.async.commit_group;\n"
    );
}

template <int N> // must be available at runtime
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n"
    :: "n"(N) );
}

// b16 for 16 bits per element
__device__ __forceinline__ void ld_matrix_x4(uint32_t regs[4], uint32_t smem_addr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3} [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(smem_addr) 
    );
}

__device__ __forceinline__ void ld_matrix_x2(uint32_t regs[2], uint32_t smem_addr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1} [%2];\n"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void mma(const uint32_t rA[4], const uint32_t rB[2], uint32_t rC[4]) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"

        :"+f"(rC[0]), "+f"(rC[1]), "+f"(rC[2]), "+f"(rC[3])
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3])
        "r"(rB[0]), "r"(rB[1])
    );
}

#define BM 128
#define BN 128 
#define BK 32

// Tune these, for max arithmetic intensity on matmul, M & N >> K (K Doesn't contribute).

// A MxK, B KxN.
__global__ void gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, const int M, const int K, const int N) {
    __shared__ half sA[2][BM][BK];
    __shared__ half sB[2][BK][BN];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    // A has width 32, so we're loading 8 eles at a time, 32 / 8 = 4.
    int load_A_row = blockIdx.y * BM + (tid / 4);
    int load_A_col = (tid % 4) * 8;



}