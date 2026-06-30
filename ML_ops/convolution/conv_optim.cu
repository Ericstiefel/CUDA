#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Constant Cache in Peak Performance when the warp requests the same value from the constant cache (ensure they're on the same step).
// We're designing this without tensor cores more for the purpose of understanding than peak performance.
// For the same reason (I know there are multiple approaches to border handling), we're going to assume the kernel fits perfectly in the input dims.


/*
Architecture:
Kernel: 3x3 in constant memory
Thread: Will compute 1 output element
Block: Say, 14x14 for example (done below)
Batch: each block owns a fixed (blockIdx.x, blockIdx.y) output-tile position; tile_idx sweeps
the batch dimension, double-buffered so the next image's tile load overlaps this image's compute.
The grid is sized (host-side) to cover the full MxN output in one shot.

Pad Shared Memory (Swizzling is extremely unnecessary here, especially because reading the data is 3 continuous items (6 bytes), not a multiple of 16 bytes)

Input tile: (14 + 3 - 1) x (14 + 3 - 1) = (16 x 16) per block.
Input tile: (math to make the tiles divisible by 16 bytes or 8 elenents for async loading smoothness)
We're adding 2 to both dims, 6 + 8k for some k \in\mathbb{N}. Let's go with the arbitrary 14.

*/

#define KM 3
#define KN 3
#define BM 16
#define BN 16

__constant__ half c_ker[KM][KN];

__device__ __forceinline__ uint32_t gts(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_32(uint32_t smem, const void* gmem) {
    asm volatile (
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile (
        "cp.async.commit_group;\n"
    );
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile (
        "cp.async.wait_group %0;\n"
        ::"n"(N)
    );
}

// BM*BN (256) elements, 2 halfs (4 bytes) per thread -> needs 128 threads (4 warps)
__device__ __forceinline__ void load_tile(const half* gmem_ptr, half* smem_ptr, const int tile_idx, const int M, const int N) {
    int lid = threadIdx.y * blockDim.x + threadIdx.x; // first 128 threads of each block
    if (lid < 128) {
        int row = lid / 8;        
        int col = (lid % 8) * 2;  

        const half* gmem_tile_begin = gmem_ptr + (size_t)tile_idx * M * N + (blockIdx.y * 14) * N + blockIdx.x * 14; // 14 output dim per block (offsetting by sizes of the output tiles).
        const half* load_from = gmem_tile_begin + row * N + col;
        uint32_t smem_to = gts(smem_ptr + row * BN + col);
        cp_async_32(smem_to, load_from);
        cp_async_commit_group();
    }
}

__global__ void conv(const half* __restrict__ inp, half* __restrict__ out, const int M, const int N, const int B) {
    __shared__ half smem[2][BM * BN]; // Padding not necessary
    int tx = threadIdx.x; int ty = threadIdx.y;
    int gr = blockDim.y * blockIdx.y + ty;
    int gc = blockDim.x * blockIdx.x + tx;

    // No padding, output is smaller than input by K-1 in each dim.
    int M_out = M - KM + 1;
    int N_out = N - KN + 1;

    int read = 0; int write = 0;

    load_tile(inp, &smem[write][0], write, M, N);
    cp_async_wait_group<0>(); __syncthreads();

    for (int tile_idx = 0; tile_idx < B; ++tile_idx) {
        int next_tile = tile_idx + 1;
        bool has_next = next_tile < B;

        if (has_next) {
            write ^= 1;
            load_tile(inp, &smem[write][0], next_tile, write, M, N);
        }

        float result = 0.0f;
        #pragma unroll
        for (int m = 0; m < KM; ++m) {
            #pragma unroll
            for (int n = 0; n < KN; ++n) {
                // Per-thread sliding window: this thread's output pixel (ty, tx) reads input
                // starting at (ty, tx) and spanning the KMxKN kernel footprint.
                half val = smem[read][(ty + m) * BN + (tx + n)];
                half ker_val = c_ker[m][n];
                result += __half2float(val) * __half2float(ker_val);
            }
        }

        out[(size_t)tile_idx * M_out * N_out + gr * N_out + gc] = __float2half(result);

        if (has_next) {
            read ^= 1; cp_async_wait_group<1>(); __syncthreads();
        }
    }
}

// For checking purposes
__global__ void conv_naive(const half* __restrict__ inp, half* __restrict__ out, const int M, const int N, const int M_out, const int N_out) {
    int gr = blockIdx.y * blockDim.y + threadIdx.y;
    int gc = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    if (gr >= M_out || gc >= N_out) return;

    const half* img = inp + (size_t)b * M * N;
    float result = 0.0f;
    #pragma unroll
    for (int m = 0; m < KM; ++m) {
        #pragma unroll
        for (int n = 0; n < KN; ++n) {
            result += __half2float(img[(gr + m) * N + (gc + n)]) * __half2float(c_ker[m][n]);
        }
    }
    out[(size_t)b * M_out * N_out + gr * N_out + gc] = __float2half(result);
}

__global__ void check_identical(const half* __restrict__ A, const half* __restrict__ B, int n, int* correct) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int val = fabsf(__half2float(A[idx]) - __half2float(B[idx])) < 1e-3f ? 1 : 0;
    atomicAdd(correct, val);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

int main() {

    const int M = 128, N = 128, B = 8;
    const int M_out = M - KM + 1, N_out = N - KN + 1;
    srand(42);

    std::vector<half> hInp((size_t)B * M * N);
    for (size_t i = 0; i < hInp.size(); ++i) hInp[i] = __float2half(rand() / (float)RAND_MAX);

    half hKer[KM][KN];
    for (int m = 0; m < KM; ++m)
        for (int n = 0; n < KN; ++n)
            hKer[m][n] = __float2half(rand() / (float)RAND_MAX);

    half *dInp, *dOut, *dOutRef;
    int *dCorrect;
    CUDA_CHECK(cudaMalloc(&dInp, hInp.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOut, (size_t)B * M_out * N_out * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dOutRef, (size_t)B * M_out * N_out * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dCorrect, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dInp, hInp.data(), hInp.size() * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ker, hKer, sizeof(hKer)));
    CUDA_CHECK(cudaMemset(dCorrect, 0, sizeof(int)));

    dim3 block(14, 14);
    dim3 grid(N_out / 14, M_out / 14);
    conv<<<grid, block>>>(dInp, dOut, M, N, B);
    CUDA_CHECK(cudaGetLastError());

    dim3 refBlock(16, 16);
    dim3 refGrid((N_out + 15) / 16, (M_out + 15) / 16, B);
    conv_naive<<<refGrid, refBlock>>>(dInp, dOutRef, M, N, M_out, N_out);
    CUDA_CHECK(cudaGetLastError());

    int total = B * M_out * N_out;
    check_identical<<<(total + 255) / 256, 256>>>(dOut, dOutRef, total, dCorrect);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int correct = 0;
    CUDA_CHECK(cudaMemcpy(&correct, dCorrect, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<half> hOut(total), hOutRef(total);
    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, total * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hOutRef.data(), dOutRef, total * sizeof(half), cudaMemcpyDeviceToHost));

    float maxAbsErr = 0.0f, sumAbsErr = 0.0f;
    for (int i = 0; i < total; ++i) {
        float e = fabsf(__half2float(hOut[i]) - __half2float(hOutRef[i]));
        maxAbsErr = fmaxf(maxAbsErr, e);
        sumAbsErr += e;
    }

    printf("M=%d N=%d B=%d M_out=%d N_out=%d\n", M, N, B, M_out, N_out);
    printf("check_identical (tol 1e-3): %d / %d matched (%.2f%%)\n", correct, total, 100.0f * correct / total);
    printf("max abs error: %f, mean abs error: %f\n", maxAbsErr, sumAbsErr / total);

    return 0;
}
