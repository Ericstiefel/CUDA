#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

// Kernel to compare result against using a simple GPU GEMM reference implementation
__global__ void basic_gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, const int M, const int K, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = sum;
}

__device__ __forceinline__ uint32_t smem_ptr_to_uint(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t swizzle_addr(const half* base_ptr, int row, int col, int stride) {
    int vals_per_vec = 16 / sizeof(half); // 8 elements per 16 bytes
    int chunks_per_row = stride / vals_per_vec;
    int chunk_idx = col / vals_per_vec;
    int offset = col % vals_per_vec;

    // Mask to chunks_per_row's bit width: with fewer than 8 chunks/row (e.g. BK=32 -> 4
    // chunks), XORing with the full row%8 overflows past the valid chunk range and aliases
    // into the next row's slot. Masking keeps row%8 a no-op when chunks_per_row >= 8.
    int swizzled_col = chunk_idx ^ ((row % 8) % chunks_per_row);
    int flat_idx = (row * stride) + (swizzled_col * vals_per_vec) + offset;

    return smem_ptr_to_uint(base_ptr + flat_idx);
}

__device__ __forceinline__ void cp_async_128(uint32_t smem_addr, const void* global_ptr) {
    asm volatile (
        "cp.async.cg.shared.global [%0], [%1], 16;\n" 
        :: "r"(smem_addr), "l"(global_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile ("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void wait_group() {
    asm volatile ("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ld_matrix_x4(uint32_t regs[4], uint32_t smem_addr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) // Fixed output constraints
        : "r"(smem_addr)
    );
}

// B is stored row-major in shared memory, but mma's B operand needs a col-major
// fragment. .trans makes ldmatrix do that transpose during the load.
__device__ __forceinline__ void ld_matrix_x2(uint32_t regs[2], uint32_t smem_addr) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(regs[0]), "=r"(regs[1])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void mma(const uint32_t a[4], const uint32_t b[2], float c[4]) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3])
          "r"(b[0]), "r"(b[1])
    );
}


#define BM 128
#define BN 128 
#define BK 32

__global__ void gemm(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, const int M, const int K, const int N) {
    __shared__ half sA[2][BM][BK];
    __shared__ half sB[2][BK][BN];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int lane_id = tx % 32;
    int warp_id = tx / 32;

    // Base courier load coordinates for Pass 1
    int load_a_row = by * BM + (tx / 4);
    int load_a_col = (tx % 4) * 8;
    int load_b_row = tx / 16;
    int load_b_col = bx * BN + (tx % 16) * 8;

    // Fixed register naming consistency
    uint32_t rA[4]; 
    uint32_t rB[2]; 
    float rC[4][4][4] = {0.0f}; // 64 output values of C per thread (128 x 128 / (8 * 32))

    int write_stage = 0; 
    int read_stage = 0;

    
    uint32_t smem_wA_p1 = swizzle_addr(&sA[write_stage][0][0], tx / 4, (tx % 4) * 8, BK);
    uint32_t smem_wA_p2 = swizzle_addr(&sA[write_stage][0][0], tx / 4 + 64, (tx % 4) * 8, BK);
    cp_async_128(smem_wA_p1, &A[load_a_row * K + load_a_col]);
    cp_async_128(smem_wA_p2, &A[(load_a_row + 64) * K + load_a_col]);

    uint32_t smem_wB_p1 = swizzle_addr(&sB[write_stage][0][0], tx / 16, (tx % 16) * 8, BN);
    uint32_t smem_wB_p2 = swizzle_addr(&sB[write_stage][0][0], tx / 16 + 16, (tx % 16) * 8, BN);

    cp_async_128(smem_wB_p1, &B[load_b_row * N + load_b_col]);
    cp_async_128(smem_wB_p2, &B[(load_b_row + 16) * N + load_b_col]);
    cp_async_commit(); 
    wait_group<0>(); // Fixed function name call
    __syncthreads();

    int warp_row = warp_id / 4;
    int warp_col = warp_id % 4;

    // Each warp will compute a 64 x 32 block of C (128 x 128 / 2 x 4 warp grid)

    int warp_row_offset = warp_row * 64;
    int warp_col_offset = warp_col * 32;


    for (int tile_k = BK; tile_k < K; tile_k += BK) {
        write_stage ^= 1;

        // Fetch Next Global Data Tiles into background write stage (2 Passes)
        smem_wA_p1 = swizzle_addr(&sA[write_stage][0][0], tx / 4, (tx % 4) * 8, BK);
        smem_wA_p2 = swizzle_addr(&sA[write_stage][0][0], tx / 4 + 64, (tx % 4) * 8, BK);
        cp_async_128(smem_wA_p1, &A[load_a_row * K + load_a_col + tile_k]);
        cp_async_128(smem_wA_p2, &A[(load_a_row + 64) * K + load_a_col + tile_k]);

        smem_wB_p1 = swizzle_addr(&sB[write_stage][0][0], tx / 16, (tx % 16) * 8, BN);
        smem_wB_p2 = swizzle_addr(&sB[write_stage][0][0], tx / 16 + 16, (tx % 16) * 8, BN);

        cp_async_128(smem_wB_p1, &B[load_b_row * N + load_b_col + tile_k * N]);
        cp_async_128(smem_wB_p2, &B[(load_b_row + 16) * N + load_b_col + tile_k * N]);

        cp_async_commit();
        for (int k_dim = 0; k_dim < 2; ++k_dim) {

            for (int m_dim = 0; m_dim < 4; ++m_dim) {

                int lane_group   = lane_id / 8;
                int row_in_group = lane_id % 8;

                int a_row;
                if (lane_group == 0 || lane_group == 2)
                    a_row = warp_row_offset + m_dim * 16 + row_in_group;
                else
                    a_row = warp_row_offset + m_dim * 16 + 8 + row_in_group;

                int a_col;
                if (lane_group == 0 || lane_group == 1)
                    a_col = k_dim * 16;
                else
                    a_col = k_dim * 16 + 8;

                uint32_t smem_read_A = swizzle_addr(&sA[read_stage][0][0], a_row, a_col, BK);

                ld_matrix_x4(rA, smem_read_A);

                for (int n_dim = 0; n_dim < 4; ++n_dim) {

                    int b_row;

                    if (lane_group == 0 || lane_group == 2)
                        b_row = k_dim * 16 + row_in_group;
                    else
                        b_row = k_dim * 16 + 8 + row_in_group;

                    uint32_t smem_read_B = swizzle_addr(&sB[read_stage][0][0], b_row, warp_col_offset + n_dim * 8, BN);

                    ld_matrix_x2(rB, smem_read_B);

                    mma(rA, rB, rC[m_dim][n_dim]);
                }
            }
        }
        read_stage ^= 1;
        wait_group<1>(); 
        __syncthreads(); 
    }

    // Final computation:
    for (int k_dim = 0; k_dim < 2; ++k_dim) {
            for (int m_dim = 0; m_dim < 4; ++m_dim) {
                // Load A tile & loop over loading respective B tiles
                int a_row = warp_row_offset + (m_dim * 16) + (lane_id % 16);
                int a_col = (k_dim * 16) + (lane_id / 16) * 8;
                uint32_t smem_read_A = swizzle_addr(&sA[read_stage][0][0], a_row, a_col, BK);
                ld_matrix_x4(rA, smem_read_A);

                for (int n_dim = 0; n_dim < 4; ++n_dim) {

                    int b_row = (k_dim * 16) + lane_id % 16;
                    uint32_t smem_read_B = swizzle_addr(&sB[read_stage][0][0], b_row, warp_col_offset + n_dim * 8, BN);
                    ld_matrix_x2(rB, smem_read_B);
                    mma(rA, rB, rC[m_dim][n_dim]);

                }
            }
        }

    // mma.m16n8k16 accumulator fragment layout (per PTX ISA): row = (lane>>2) + 8*(l/2),
    // col = 2*(lane%4) + (l%2), for l = 0..3 matching rC[m][n][0..3].
    int frag_row = lane_id / 4;
    int frag_col = (lane_id % 4) * 2;

    // Added warp row/column spatial offsets so warps write to unique global memory slots
    int warp_global_row = by * BM + warp_row_offset; 
    int warp_global_col = bx * BN + warp_col_offset; 

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {

            int block_global_row = warp_global_row + (m * 16);
            int block_global_col = warp_global_col + (n * 8);

            int global_row = block_global_row + frag_row;
            int global_col = block_global_col + frag_col;

            // Map the 4 MMA outputs per thread into the 16x8 region as:
            // (row*8 + col), (row*8 + col + 1), ((row+8)*8 + col), ((row+8)*8 + col + 1)
            C[global_row * N + global_col] = rC[m][n][0];
            C[global_row * N + (global_col + 1)] = rC[m][n][1];
            C[(global_row + 8) * N + global_col] = rC[m][n][2];
            C[(global_row + 8) * N + (global_col + 1)] = rC[m][n][3];
            
        }
    }
}

__global__ void check_identical(const float* __restrict__ C1, const float* __restrict__ C2, int M, int N, int* correct) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int idx = row * N + col;
    int val = fabsf(C1[idx] - C2[idx]) < 1e-3f ? 1 : 0;
    atomicAdd(correct, val);
}

#define CEIL_DIV(larger, smaller) ((larger + smaller - 1) / smaller)

int main() {
    int M = 1024, N = 1024, K = 512;
    half *A, *B;
    float *C, *C2;

    cudaMalloc(&A, sizeof(half) * M * K);
    cudaMalloc(&B, sizeof(half) * K * N);
    cudaMalloc(&C, sizeof(float) * M * N);
    cudaMalloc(&C2, sizeof(float) * M * N);

    half* hA = (half*)malloc(sizeof(half) * M * K);
    half* hB = (half*)malloc(sizeof(half) * K * N);
    for (int i = 0; i < M * K; ++i) hA[i] = __float2half((float)(i % 7 + 1));
    for (int i = 0; i < K * N; ++i) hB[i] = __float2half((float)(i % 13 + 1));
    cudaMemcpy(A, hA, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB, sizeof(half) * K * N, cudaMemcpyHostToDevice);

    dim3 blockDimRef(16, 16);
    dim3 gridDimRef(CEIL_DIV(N, blockDimRef.x), CEIL_DIV(M, blockDimRef.y));
    basic_gemm<<<gridDimRef, blockDimRef>>>(A, B, C2, M, K, N);

    dim3 blockDimGemm(256);
    dim3 gridDimGemm(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm<<<gridDimGemm, blockDimGemm>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();

    int* d_correct;
    cudaMalloc(&d_correct, sizeof(int));
    cudaMemset(d_correct, 0, sizeof(int));
    dim3 blockDimChk(16, 16);
    dim3 gridDimChk(CEIL_DIV(N, blockDimChk.x), CEIL_DIV(M, blockDimChk.y));
    check_identical<<<gridDimChk, blockDimChk>>>(C, C2, M, N, d_correct);
    cudaDeviceSynchronize();

    int h_correct = 0;
    cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
    int total = M * N;
    std::cout << "Matches: " << h_correct << " / " << total << std::endl;

    free(hA);
    free(hB);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C2);
    cudaFree(d_correct);
    return 0;
}
