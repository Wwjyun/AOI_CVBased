#include "cuda_check.h"

#include <iostream>
#include <vector>

__global__ void threshold_u8(const unsigned char* input, unsigned char* output,
                             int count, unsigned char threshold, bool invert) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    bool white = input[index] > threshold;
    if (invert) white = !white;
    output[index] = white ? 255 : 0;
}

int main() {
    std::vector<unsigned char> input = {0, 20, 80, 127, 128, 129, 180, 255};
    std::vector<unsigned char> output(input.size());
    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size()));
    CUDA_CHECK(cudaMalloc(&d_output, output.size()));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice));

    int threads = 256;
    threshold_u8<<<1, threads>>>(d_input, d_output, static_cast<int>(input.size()), 128, false);
    check_kernel("threshold_u8");
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost));

    for (auto value : output) std::cout << static_cast<int>(value) << ' ';
    std::cout << "\nExpected: 0 0 0 0 0 255 255 255\n";
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
