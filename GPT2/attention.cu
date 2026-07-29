// This file will contain the attention mechanism (Self-Attention).
// After profiling, this kernel is not close to SOTA as it's extremely heavy on both register count
// And smem, despite only double buffering (I've seen triple done in adjacent configurations). 
// Perhaps if I shrink the Block dims, the register and shared memory used should scale with the 
// sizes (perhaps in half so we can load a block in 1 async load rather than 2).


#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cmath>


__device__ __forceinline__ uint32_t cvta_generic_to_shared(void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t swizzled_ptr(const void* smem_ptr, int row, int col, int stride) {
    int eles_per_vec = 16 / sizeof(half);
    int vecs_per_row = stride / eles_per_vec;

    int chunk_idx = col / eles_per_vec;
    int offset = col % eles_per_vec;

    int swizzled_col = chunk_idx ^ ((row % 8) % vecs_per_row);
    int flat_idx = (row * stride) + (col * eles_per_vec) + offset;
    return cvta_generic_to_shared(smem_ptr + flat_idx);
}

__device__ __forceinline__ void cp_async_128(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        "r"(smem_ptr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile ("cp.async.commit_group;\n");
}

template <int N> // availability at runtime
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ld_matrix_x4(const uint32_t smem_ptr, uint32_t* rA) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(rA[0]), "=r"(rA[1]), "=r"(rA[2]), "=r"(rA[3]),
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2(const uint32_t smem_ptr, uint32_t* rB) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rB[0]), "=r"(rB[1]),
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2_trans(const uint32_t smem_ptr, uint32_t* rB) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rB[0]), "=r"(rB[1]),
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void mma(uint32_t* rA, uint32_t* rB, uint32_t* rC) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(rC[0]), "+f"(rC[1]), "+f"(rC[2]), "+f"(rC[3]),
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]), 
        "r"(rB[0]), "r"(rB[1])
    );
}

// Stores 16 bits (2 bytes) as deliberate store instruction with smem ptr
__device__ __forceinline__ void st_shared_b16(uint32_t smem_ptr, half val) {
    unsigned int bits = __half_as_ushort(val);
    asm volatile (
        "st.shared.b16 [%0], %1;\n"
        :: "r"(smem_ptr), "h"(bits)
    );
}

template <int WIDTH>
__device__ __forceinline__ void softmax_online_reduction(float& l_max, float& l_sum) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int i = WIDTH / 2; i > 0; offset >>= 1) {
        float neighbor_max = __shfl_down_sync(mask, l_max, i, WIDTH);
        float neighbor_sum = __shfl_down_sync(mask, l_sum, i, WIDTH);

        float joint_max = fmaxf(l_max, neighbor_max);
        l_sum = l_sum * expf(l_max - joint_max) + neighbor_sum * (neighbor_max - joint_max);
        l_max = joint_max
    }
}