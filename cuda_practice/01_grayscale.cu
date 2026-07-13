#include "cuda_check.h"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void bgr_to_gray(const unsigned char* bgr, unsigned char* gray, int pixel_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;
    int base = index * 3;
    float value = 0.114f * bgr[base] + 0.587f * bgr[base + 1] + 0.299f * bgr[base + 2];
    gray[index] = static_cast<unsigned char>(value + 0.5f);
}

int main() {
    constexpr int width = 4;
    constexpr int height = 2;
    std::vector<unsigned char> input = {
        0, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 0,
        255, 0, 0, 20, 100, 200, 50, 50, 50, 10, 20, 30};
    std::vector<unsigned char> output(width * height);
    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size()));
    CUDA_CHECK(cudaMalloc(&d_output, output.size()));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    bgr_to_gray<<<blocks, threads>>>(d_input, d_output, width * height);
    check_kernel("bgr_to_gray");
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost));

    for (int i = 0; i < width * height; ++i) std::cout << static_cast<int>(output[i]) << ' ';
    std::cout << "\n";
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
