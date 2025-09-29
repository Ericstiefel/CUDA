#include <cuda_runtime.h>

// Simple File with example general error check for synchronous API Launches

#define CUDA_CHECK_ERROR(call)  __cuda_check_errors((call), __FILE__, __LINE__)

inline void __cuda_check_errors(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s. \n", file, line, (int)err, cudaGetStringError(err));
        exit(EXIT_FAILURE);
    }
}