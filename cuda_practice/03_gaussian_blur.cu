#include "cuda_check.h"

#include <iostream>
#include <vector>

__device__ int clamp_int(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

__global__ void gaussian_blur_3x3(const unsigned char* input, unsigned char* output,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int weights[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    int sum = 0;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int px = clamp_int(x + kx, 0, width - 1);
            int py = clamp_int(y + ky, 0, height - 1);
            sum += input[py * width + px] * weights[ky + 1][kx + 1];
        }
    }
    output[y * width + x] = static_cast<unsigned char>((sum + 8) / 16);
}

int main() {
    constexpr int width = 8, height = 8;
    std::vector<unsigned char> input(width * height, 0), output(width * height);
    input[(height / 2) * width + width / 2] = 255;
    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size()));
    CUDA_CHECK(cudaMalloc(&d_output, output.size()));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice));
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    gaussian_blur_3x3<<<blocks, threads>>>(d_input, d_output, width, height);
    check_kernel("gaussian_blur_3x3");
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) std::cout << static_cast<int>(output[y * width + x]) << '\t';
        std::cout << '\n';
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
