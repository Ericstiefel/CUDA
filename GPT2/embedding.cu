
#include "kernels.cuh"


__global__ void embedding_gather(const int* __restrict__ token_ids,
                                 const half* __restrict__ wte,
                                 half* __restrict__ out, const int N) {
    const int vecs_per_row = N / 8;
    const int tok = token_ids[blockIdx.x];

    const float4* src = reinterpret_cast<const float4*>(wte) + (size_t)tok * vecs_per_row;
    float4* dst = reinterpret_cast<float4*>(out) + (size_t)blockIdx.x * vecs_per_row;

    dst[threadIdx.x] = src[threadIdx.x];
}
