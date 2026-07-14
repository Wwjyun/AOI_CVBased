#include <iostream>
#include <vector>
#include "visionflow_cuda.h"

int main() {
    std::cout << "VisionFlow CUDA ABI: " << vf_gpu_abi_version() << "\n";
    if (vf_gpu_abi_version() != VF_CUDA_ABI_VERSION) {
        std::cerr << "ABI mismatch\n";
        return 2;
    }

    int count = vf_gpu_device_count();
    std::cout << "CUDA device count: " << count << "\n";
    if (count <= 0) {
        std::cerr << "No CUDA device\n";
        return 3;
    }

    char name[256]{};
    int result = vf_gpu_device_name(name, static_cast<int>(sizeof(name)));
    if (result != VF_CUDA_OK) {
        char message[256]{};
        vf_gpu_error_message(result, message, static_cast<int>(sizeof(message)));
        std::cerr << "Device query failed: " << message << "\n";
        return 4;
    }

    int capability = vf_gpu_compute_capability();
    std::cout << "Device: " << name << "\n";
    std::cout << "Compute capability: " << capability / 10 << "." << capability % 10 << "\n";

    const int width = 8;
    const int height = 8;
    std::vector<uint8_t> bgr(width * height * 3, 128);
    std::vector<uint8_t> gray(width * height, 0);
    result = vf_bgr_to_gray_u8(
        bgr.data(), width, height, width * 3, 3,
        gray.data(), width, 1);
    if (result != VF_CUDA_OK) {
        char message[256]{};
        vf_gpu_error_message(result, message, static_cast<int>(sizeof(message)));
        std::cerr << "Grayscale smoke failed: " << message << "\n";
        return 5;
    }

    std::cout << "C ABI and grayscale smoke passed\n";
    return 0;
}
