#pragma once

// Device-side primitives shared by every kernel in this model.
//
// Before this file existed each .cu carried its own copy of the PTX wrappers, and the
// copies had drifted: gemm.cu called the address helpers smem_ptr_to_uint/swizzle_addr
// while attention.cu called the identical functions cvta_generic_to_shared/swizzled_ptr.
// Worse, the tile macros collided outright -- BK was 32 in gemm.cu and 64 in
// attention.cu, WARPS was 8 in attention.cu and 32 in softmax.cu. Those stayed harmless
// only because nothing ever included two of them at once, which stops being true the
// moment a driver does. Tile sizes now live in an anonymous namespace inside the kernel
// that owns them; everything genuinely shared lives here.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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


// ---- shared memory addressing ------------------------------------------------------

__device__ __forceinline__ uint32_t cvta_generic_to_shared(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(const_cast<void*>(ptr)));
}

// XOR swizzle over 16-byte chunks: consecutive rows have their chunks rotated so that a
// column-wise access hits distinct banks instead of all landing in one.
__device__ __forceinline__ uint32_t swizzled_ptr(const void* smem_ptr, int row, int col, int stride) {
    int eles_per_vec = 16 / (int)sizeof(half);
    int vecs_per_row = stride / eles_per_vec;

    int chunk_idx = col / eles_per_vec;
    int offset = col % eles_per_vec;

    int swizzled_chunk = chunk_idx ^ ((row % 8) % vecs_per_row);
    int flat_idx = (row * stride) + (swizzled_chunk * eles_per_vec) + offset;
    return cvta_generic_to_shared(static_cast<const half*>(smem_ptr) + flat_idx);
}


// ---- async global -> shared copy ---------------------------------------------------

__device__ __forceinline__ void cp_async_128(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile ("cp.async.commit_group;\n");
}

template <int N> // must be an immediate, hence the template rather than a parameter
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile ("cp.async.wait_group %0;\n" :: "n"(N));
}


// ---- tensor core operand loads -----------------------------------------------------
//
// volatile is kept on every ldmatrix: without it the compiler may CSE a load across
// __syncthreads, and a double-buffered address repeats every two iterations while the
// data underneath it does not.

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


// ---- packing -----------------------------------------------------------------------

// Two floats -> one packed half2 register. F2FP.PACK_AB, a single instruction on sm_80+,
// where a convert-then-shift-then-or pack costs three.
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}
