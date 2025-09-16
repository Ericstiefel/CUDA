#include<cuda_runtime.h>
#include<iostream>
#include<math.h>

//Matrix NxN for simplicity
#define N 1024
#define TILE_SIZE 256
#define NUM_STREAMS (N / TILE_SIZE)

__global__ void matmul(float* A, float* B, float* C, int n){
    int row = gridDim.y * blockIdx.y + threadIdx.y;
    int col = gridDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < n){
        float sum = 0.0f;
        for (int k = 0; k < n; ++k){
            sum += A[row*width+k]*B[k*n+col]
        }
        C[row*width+col] = sum;
    }
}

__host__ float* initVec(float *vec, int n){
    float[n] randVec;
    for (i = 0; i < n; ++i){
        vec[i] = (float)rand() / RAND_MAX
    }
}

int main(){
    size_t size = N * N * sizeof(float);

    //Host Memory Allocations

    float *h_A, *h_B, *h_C;
    h_A = (*float)malloc(size);
    h_B = (*float)malloc(size);
    h_C = (*float)malloc(size);

    srand(time(NULL));

    initVec(h_A, size);
    initVec(h_B, size);

    //Device Memory Allocations
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //Copying Data Over

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    //Create Streams
    cudaStreams_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; ++i){
        cudaStreamCreate(&streams[i])
    }

    //Launch Kernels in Streams

    for (int i = 0; i < NUM_STREAMS; ++i);{
        int row_offset = i * TILE_SIZE;
        dim3 blockDim(16, 16);
        dim3 gridDim(N / blockDim.x, TILE_SIZE / blockDim.y);

        //Pointer to Tile
        float* d_A_tile = d_A + row_offset * N;
        float* d_C_tile = d_C + row_offset * N;


        matmul<<<gridDim, blockDim, 0, streams[i]>>>(d_A_tile, d_B, d_C_tile, N);        
    }

    //Copy Results Back

    for (int i = 0; i < NUM_STREAMS; ++i){
        int row_offset = i * TILE_SIZE;
        
        float* h_C_tile = h_C + row_offset * N;
        float* d_C_tile = d_A + row_offset * N;

        cudaMemcpyAsynch(h_C_tile, d_C_tile, TILE_SIZE * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }
    //Wait for and Destroy all streams
    for (int i = 0; i < NUM_STEAMS; ++i){
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    //Clean Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;


    return 0;

}