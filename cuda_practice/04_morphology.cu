#include "cuda_check.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

__global__ void morphology_3x3(const unsigned char* input, unsigned char* output,
                               int width, int height, bool dilate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    unsigned char result = dilate ? 0 : 255;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            unsigned char value = input[py * width + px];
            result = dilate ? max(result, value) : min(result, value);
        }
    }
    output[y * width + x] = result;
}

int main(int argc, char** argv) {
    constexpr int width = 9, height = 7;
    bool dilate = argc < 2 || std::string(argv[1]) != "erode";
    std::vector<unsigned char> input(width * height, 0), output(width * height);
    input[3 * width + 4] = 255;
    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size()));
    CUDA_CHECK(cudaMalloc(&d_output, output.size()));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice));
    dim3 threads(16, 16), blocks((width + 15) / 16, (height + 15) / 16);
    morphology_3x3<<<blocks, threads>>>(d_input, d_output, width, height, dilate);
    check_kernel(dilate ? "dilate_3x3" : "erode_3x3");
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) std::cout << (output[y * width + x] ? "##" : "..");
        std::cout << '\n';
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
