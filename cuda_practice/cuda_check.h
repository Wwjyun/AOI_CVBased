#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                            \
        if (error__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error__)          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (false)

inline void check_kernel(const char* name) {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << name << " completed successfully.\n";
}
