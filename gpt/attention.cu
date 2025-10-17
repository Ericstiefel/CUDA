/*
Multi-Head Flash Attention Optimized

Effectively use Flash Attention, but for each head, expand in the Z dimension.
*/

__device__ float warpReduceMax(float val, unsigned mask = 0xffffffffu) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val, unsigned mask = 0xffffffffu) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}