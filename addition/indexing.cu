#include<cuda_runtime.h>

__device__ int index1D(){
    return threadIdx.x + gridDim.x * blockIdx.x;
}
__device__ int index2D(int width){
    x = threadIdx.x + gridDim.x * gridIdx.x;
    y = threadIdx.y + gridDim.y * gridIdx.y;
    return y * width + x;
}

__device__ int index3D(int height, int width){
    x = threadIdx.x + gridDim.x * blockIdx.x;
    y = threadIdx.y + gridDim.y * blockIdx.y;
    z = threadIdx.z + gridDim.z * blockIdx.z;

    return z * (height * width) + y * width + x;
}

__global__ void 1Dadd(float* inp_a, float* inp_b, float* out, int n){
    i = index1D();

    if (i < n){
        out[i] = inp_a[i] + inp_b[i];
    }
}