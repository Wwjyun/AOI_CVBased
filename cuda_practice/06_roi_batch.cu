#include "cuda_check.h"

#include <iostream>
#include <vector>

__global__ void batch_threshold(const unsigned char* rois, unsigned char* output,
                                int roi_pixels, int roi_count, unsigned char threshold) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int roi = blockIdx.y;
    if (roi >= roi_count || pixel >= roi_pixels) return;
    int index = roi * roi_pixels + pixel;
    output[index] = rois[index] > threshold ? 255 : 0;
}

int main() {
    constexpr int roi_w = 8, roi_h = 8, roi_count = 4;
    constexpr int roi_pixels = roi_w * roi_h;
    std::vector<unsigned char> input(roi_count * roi_pixels), output(input.size());
    for (int roi = 0; roi < roi_count; ++roi)
        for (int pixel = 0; pixel < roi_pixels; ++pixel)
            input[roi * roi_pixels + pixel] = static_cast<unsigned char>(roi * 60 + pixel % 16);

    unsigned char *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input.size()));
    CUDA_CHECK(cudaMalloc(&d_output, output.size()));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size(), cudaMemcpyHostToDevice));
    int threads = 256;
    dim3 blocks((roi_pixels + threads - 1) / threads, roi_count);
    batch_threshold<<<blocks, threads>>>(d_input, d_output, roi_pixels, roi_count, 128);
    check_kernel("batch_threshold");
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost));
    for (int roi = 0; roi < roi_count; ++roi) {
        int white = 0;
        for (int pixel = 0; pixel < roi_pixels; ++pixel)
            white += output[roi * roi_pixels + pixel] == 255;
        std::cout << "ROI " << roi << ": white pixels=" << white << '/' << roi_pixels << '\n';
    }
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
