#ifndef GEMM_H
#define GEMM_H

#include <cuda_fp16.h>


void cpu_gemm(const float* A, const float* B, float* C, int M, int K, int N);



void naive_launcher(const float* d_A, const float* d_B, float* d_C, int M, int K, int N);



template <int TILE_WIDTH>
void tile_cm_launcher(const float* d_A, const float* d_B, float* d_C, int M, int K, int N);



template <int TILE_WIDTH>
void tiled_buff_launcher(const float* d_A, const float* d_B, float* d_C, int M, int K, int N);


void basic_tc_launcher(const half* d_A, const half* d_B, float* d_C, int M, int K, int N);

void tc_launcher(const half* d_A, const half* d_B, float* d_C, int M, int K, int N);


#endif // GEMM_H
