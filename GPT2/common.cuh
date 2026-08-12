#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>


#define FULL_MASK 0xffffffffu

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s \n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)



__device__ __forceinline__ uint32_t cvta_generic_to_shared(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(ptr)));
}


__device__ __forceinline__ uint32_t swizzled_ptr(const void* smem_ptr, int row, int col, int stride) {
    int eles_per_vec = 16 / (int)sizeof(half);
    int vecs_per_row = stride / eles_per_vec;

    int chunk_idx = col / eles_per_vec;
    int offset = col % eles_per_vec;

    int swizzled_chunk = chunk_idx ^ ((row % 8) % vecs_per_row);
    int flat_idx = (row * stride) + (swizzled_chunk * eles_per_vec) + offset;
    return cvta_generic_to_shared(static_cast<const half*>(smem_ptr) + flat_idx);
}



__device__ __forceinline__ void cp_async_128(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile ("cp.async.commit_group;\n");
}

template <int N> 
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n" :: "n"(N));
}



__device__ __forceinline__ void ld_matrix_x4(uint32_t rA[4], uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(rA[0]), "=r"(rA[1]), "=r"(rA[2]), "=r"(rA[3])
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2(uint32_t rB[2], uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rB[0]), "=r"(rB[1])
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void ld_matrix_x2_trans(uint32_t rB[2], uint32_t smem_ptr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(rB[0]), "=r"(rB[1])
        : "r"(smem_ptr)
    );
}

__device__ __forceinline__ void mma(const uint32_t rA[4], const uint32_t rB[2], float rC[4]) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        " {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(rC[0]), "+f"(rC[1]), "+f"(rC[2]), "+f"(rC[3])
        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]),
          "r"(rB[0]), "r"(rB[1])
    );
}


__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}


__device__ __forceinline__ float gelu(float x) {
    const float k = 0.7978845608028654f;   // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
