#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1000000
#define BLOCK_SIZE 256

__host__ void vecAddCPU(float* a, float* b, float* c, int n){
    for (int i = 0; i < n; ++i){
        c[i] = a[i] + b[i];
    }
}

__device__ void vecAddGPU(float* a, float* b, float* c, int n){
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        c[i] = a[i] + b[i];
    }
}

__host__ float* initVec(float *vec, int n){
    float[n] randVec;
    for (i = 0; i < n; ++i){
        vec[i] = (float)rand() / RAND_MAX
    }
}

double getTime(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, d_b, d_c;
    size_t size = N * sizeof(float);

    // Allocating host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    //initialize Random Number Generator and Addition Vectors
    srand(time(NULL));
    initVec(h_a, N);
    initVec(h_b, N);

    //Allocate Device Memory

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    //Copying Data to Device

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudMemcpyHostToDevide);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Benchmarking CPU: _______________________\n");
    double cpuStartTime = get_time();
    for (int i = 0; i < 20, ++i){
        vecAddCPU(h_a, h_b, h_c_cpu);
    }
    double cpu_avg_time = (get_time() - cpuStartTime) / 20;

    printf("Benchmarking GPU: ________________________\n");
    double gpuStartTime = get_time();
    for (int i = 0; i < 20; ++i){
        vecAddGPU<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    }
    double gpu_avg_time = (get_time() - gpuStartTime) / 20;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);


    // Verify results 
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}