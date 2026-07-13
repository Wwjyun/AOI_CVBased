#include "cuda_check.h"

#include <iomanip>
#include <iostream>

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::cout << "CUDA device count: " << count << "\n";
    if (count == 0) {
        std::cerr << "No CUDA GPU found.\n";
        return 1;
    }

    for (int device = 0; device < count; ++device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::cout << "\nDevice " << device << ": " << prop.name << "\n"
                  << "  Compute capability: " << prop.major << "." << prop.minor << "\n"
                  << "  VRAM: " << std::fixed << std::setprecision(2)
                  << prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << " GB\n"
                  << "  SM count: " << prop.multiProcessorCount << "\n"
                  << "  Max threads/block: " << prop.maxThreadsPerBlock << "\n"
                  << "  Warp size: " << prop.warpSize << "\n";
    }
    return 0;
}
