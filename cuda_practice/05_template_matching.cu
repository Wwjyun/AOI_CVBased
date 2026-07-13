#include "cuda_check.h"

#include <cfloat>
#include <iostream>
#include <vector>

__global__ void template_ssd(const unsigned char* image, const unsigned char* templ,
                             float* scores, int image_w, int image_h, int templ_w, int templ_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int result_w = image_w - templ_w + 1;
    int result_h = image_h - templ_h + 1;
    if (x >= result_w || y >= result_h) return;
    float sum = 0.0f;
    for (int ty = 0; ty < templ_h; ++ty) {
        for (int tx = 0; tx < templ_w; ++tx) {
            float difference = static_cast<float>(image[(y + ty) * image_w + x + tx]) -
                               static_cast<float>(templ[ty * templ_w + tx]);
            sum += difference * difference;
        }
    }
    scores[y * result_w + x] = sum / (templ_w * templ_h);
}

int main() {
    constexpr int image_w = 16, image_h = 12, templ_w = 3, templ_h = 3;
    std::vector<unsigned char> image(image_w * image_h, 10);
    std::vector<unsigned char> templ = {0, 50, 0, 50, 255, 50, 0, 50, 0};
    constexpr int expected_x = 7, expected_y = 5;
    for (int y = 0; y < templ_h; ++y)
        for (int x = 0; x < templ_w; ++x)
            image[(expected_y + y) * image_w + expected_x + x] = templ[y * templ_w + x];

    int result_w = image_w - templ_w + 1, result_h = image_h - templ_h + 1;
    std::vector<float> scores(result_w * result_h);
    unsigned char *d_image = nullptr, *d_templ = nullptr;
    float* d_scores = nullptr;
    CUDA_CHECK(cudaMalloc(&d_image, image.size()));
    CUDA_CHECK(cudaMalloc(&d_templ, templ.size()));
    CUDA_CHECK(cudaMalloc(&d_scores, scores.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_image, image.data(), image.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_templ, templ.data(), templ.size(), cudaMemcpyHostToDevice));
    dim3 threads(16, 16), blocks((result_w + 15) / 16, (result_h + 15) / 16);
    template_ssd<<<blocks, threads>>>(d_image, d_templ, d_scores, image_w, image_h, templ_w, templ_h);
    check_kernel("template_ssd");
    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float best = FLT_MAX;
    int best_index = -1;
    for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
        if (scores[i] < best) { best = scores[i]; best_index = i; }
    }
    std::cout << "Best match: x=" << best_index % result_w << ", y=" << best_index / result_w
              << ", mean SSD=" << best << "\nExpected: x=7, y=5, mean SSD=0\n";
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_templ));
    CUDA_CHECK(cudaFree(d_scores));
    return (best_index % result_w == expected_x && best_index / result_w == expected_y) ? 0 : 2;
}
