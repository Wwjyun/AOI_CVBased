#define VISIONFLOW_CUDA_EXPORTS
#include "visionflow_cuda.h"
#include "visionflow_cuda_internal.cuh"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <vector>

namespace {
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int SCAN_THREADS = 256;
constexpr int TRANSPOSE_TILE = 32;
constexpr int TRANSPOSE_ROWS = 8;
constexpr int MAX_GAUSSIAN_KERNEL = 127;

__constant__ float gaussian_weights[MAX_GAUSSIAN_KERNEL];

int cuda_result(cudaError_t error) { return visionflow_cuda::runtime_error(error); }

int alloc_copy(const uint8_t* host, int width, int height, int stride, int channels, uint8_t** device) {
    return visionflow_cuda::allocate_and_upload(host, width, height, stride, channels, device);
}

int copy_back_free(uint8_t* host, int stride, int width, int height, int channels, uint8_t* device) {
    return visionflow_cuda::download_and_free(host, stride, width, height, channels, device);
}

__device__ int reflect101(int value, int length) {
    if (length <= 1) return 0;
    while (value < 0 || value >= length) {
        value = value < 0 ? -value : 2 * length - value - 2;
    }
    return value;
}

__global__ void bgr_gray_kernel(const uint8_t* src, uint8_t* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int index = (y * width + x) * 3;
    dst[y * width + x] = static_cast<uint8_t>((29 * src[index] + 150 * src[index + 1] + 77 * src[index + 2] + 128) >> 8);
}

__global__ void bgr_rgb_kernel(const uint8_t* src, uint8_t* dst, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = (y * width + x) * 3;
    dst[i] = src[i + 2]; dst[i + 1] = src[i + 1]; dst[i + 2] = src[i];
}

__global__ void crop_kernel(const uint8_t* src, uint8_t* dst, int src_width, int x0, int y0, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) dst[(y * width + x) * channels + c] = src[((y + y0) * src_width + x + x0) * channels + c];
}

__global__ void resize_gray_kernel(const uint8_t* src, uint8_t* dst, int sw, int sh, int dw, int dh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    if (dw <= sw && dh <= sh) {
        float scale_x = static_cast<float>(sw) / dw;
        float scale_y = static_cast<float>(sh) / dh;
        float source_x0 = x * scale_x;
        float source_x1 = (x + 1) * scale_x;
        float source_y0 = y * scale_y;
        float source_y1 = (y + 1) * scale_y;
        int start_x = static_cast<int>(floorf(source_x0));
        int end_x = static_cast<int>(ceilf(source_x1));
        int start_y = static_cast<int>(floorf(source_y0));
        int end_y = static_cast<int>(ceilf(source_y1));
        float sum = 0.0f;
        for (int source_y = start_y; source_y < end_y; ++source_y) {
            float weight_y = fmaxf(0.0f, fminf(source_y1, source_y + 1.0f) - fmaxf(source_y0, static_cast<float>(source_y)));
            int clamped_y = max(0, min(sh - 1, source_y));
            for (int source_x = start_x; source_x < end_x; ++source_x) {
                float weight_x = fmaxf(0.0f, fminf(source_x1, source_x + 1.0f) - fmaxf(source_x0, static_cast<float>(source_x)));
                int clamped_x = max(0, min(sw - 1, source_x));
                sum += src[clamped_y * sw + clamped_x] * weight_x * weight_y;
            }
        }
        dst[y * dw + x] = static_cast<uint8_t>(sum / (scale_x * scale_y) + 0.5f);
        return;
    }
    float sx = (x + 0.5f) * sw / dw - 0.5f, sy = (y + 0.5f) * sh / dh - 0.5f;
    int raw_x0 = static_cast<int>(floorf(sx));
    int raw_y0 = static_cast<int>(floorf(sy));
    int x0 = max(0, min(sw - 1, raw_x0));
    int y0 = max(0, min(sh - 1, raw_y0));
    int x1 = max(0, min(sw - 1, raw_x0 + 1));
    int y1 = max(0, min(sh - 1, raw_y0 + 1));
    float ax = sx - floorf(sx), ay = sy - floorf(sy);
    float value = (1 - ay) * ((1 - ax) * src[y0 * sw + x0] + ax * src[y0 * sw + x1]) + ay * ((1 - ax) * src[y1 * sw + x0] + ax * src[y1 * sw + x1]);
    dst[y * dw + x] = static_cast<uint8_t>(value + 0.5f);
}

__global__ void gaussian_horizontal_kernel(
    const uint8_t* src,
    float* intermediate,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int kx = -radius; kx <= radius; ++kx) {
            int sx = reflect101(x + kx, width);
            sum += src[(y * width + sx) * channels + c] * gaussian_weights[kx + radius];
        }
        intermediate[(y * width + x) * channels + c] = sum;
    }
}

__global__ void gaussian_vertical_kernel(
    const float* intermediate,
    uint8_t* dst,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky) {
            int sy = reflect101(y + ky, height);
            sum += intermediate[(sy * width + x) * channels + c] * gaussian_weights[ky + radius];
        }
        dst[(y * width + x) * channels + c] =
            static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, sum + 0.5f)));
    }
}

__global__ void threshold_kernel(const uint8_t* src, uint8_t* dst, int count, int threshold, int max_value, int invert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    bool high = src[i] > threshold;
    dst[i] = static_cast<uint8_t>((invert ? !high : high) ? max_value : 0);
}

__global__ void replicate_border_kernel(
    const uint8_t* src,
    uint8_t* padded,
    int width,
    int height,
    int padded_width,
    int padded_height,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= padded_width || y >= padded_height) return;
    int source_x = max(0, min(width - 1, x - radius));
    int source_y = max(0, min(height - 1, y - radius));
    padded[y * padded_width + x] = src[source_y * width + source_x];
}

__global__ void row_prefix_u8_kernel(
    const uint8_t* src,
    unsigned long long* prefix,
    int width,
    int height) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= height) return;
    __shared__ unsigned long long scan[SCAN_THREADS];
    __shared__ unsigned long long carry;
    __shared__ unsigned long long chunk_carry;
    if (lane == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < width; base += SCAN_THREADS) {
        int column = base + lane;
        scan[lane] = column < width ? static_cast<unsigned long long>(src[row * width + column]) : 0ULL;
        __syncthreads();
        for (int offset = 1; offset < SCAN_THREADS; offset <<= 1) {
            unsigned long long add = lane >= offset ? scan[lane - offset] : 0ULL;
            __syncthreads();
            scan[lane] += add;
            __syncthreads();
        }
        if (lane == 0) chunk_carry = carry;
        __syncthreads();
        if (column < width) prefix[row * width + column] = scan[lane] + chunk_carry;
        __syncthreads();
        int valid = min(SCAN_THREADS, width - base);
        if (lane == 0) carry = chunk_carry + scan[valid - 1];
        __syncthreads();
    }
}

__global__ void transpose_u64_kernel(
    const unsigned long long* src,
    unsigned long long* dst,
    int width,
    int height) {
    __shared__ unsigned long long tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];
    int x = blockIdx.x * TRANSPOSE_TILE + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE + threadIdx.y;
    for (int offset = 0; offset < TRANSPOSE_TILE; offset += TRANSPOSE_ROWS) {
        if (x < width && y + offset < height) {
            tile[threadIdx.y + offset][threadIdx.x] = src[(y + offset) * width + x];
        }
    }
    __syncthreads();
    x = blockIdx.y * TRANSPOSE_TILE + threadIdx.x;
    y = blockIdx.x * TRANSPOSE_TILE + threadIdx.y;
    for (int offset = 0; offset < TRANSPOSE_TILE; offset += TRANSPOSE_ROWS) {
        if (x < height && y + offset < width) {
            dst[(y + offset) * height + x] = tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ void row_prefix_u64_inplace_kernel(
    unsigned long long* values,
    int width,
    int height) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    if (row >= height) return;
    __shared__ unsigned long long scan[SCAN_THREADS];
    __shared__ unsigned long long carry;
    __shared__ unsigned long long chunk_carry;
    if (lane == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < width; base += SCAN_THREADS) {
        int column = base + lane;
        scan[lane] = column < width ? values[row * width + column] : 0ULL;
        __syncthreads();
        for (int offset = 1; offset < SCAN_THREADS; offset <<= 1) {
            unsigned long long add = lane >= offset ? scan[lane - offset] : 0ULL;
            __syncthreads();
            scan[lane] += add;
            __syncthreads();
        }
        if (lane == 0) chunk_carry = carry;
        __syncthreads();
        if (column < width) values[row * width + column] = scan[lane] + chunk_carry;
        __syncthreads();
        int valid = min(SCAN_THREADS, width - base);
        if (lane == 0) carry = chunk_carry + scan[valid - 1];
        __syncthreads();
    }
}

__device__ unsigned long long integral_value_transposed(
    const unsigned long long* integral_transposed,
    int padded_height,
    int x,
    int y) {
    if (x < 0 || y < 0) return 0ULL;
    return integral_transposed[x * padded_height + y];
}

__global__ void adaptive_integral_kernel(
    const uint8_t* src,
    const unsigned long long* integral_transposed,
    uint8_t* dst,
    int width,
    int height,
    int padded_height,
    int block_size,
    float c,
    int max_value,
    int invert) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int x0 = x;
    int y0 = y;
    int x1 = x + block_size - 1;
    int y1 = y + block_size - 1;
    unsigned long long bottom_right =
        integral_value_transposed(integral_transposed, padded_height, x1, y1);
    unsigned long long above =
        integral_value_transposed(integral_transposed, padded_height, x1, y0 - 1);
    unsigned long long left =
        integral_value_transposed(integral_transposed, padded_height, x0 - 1, y1);
    unsigned long long above_left =
        integral_value_transposed(integral_transposed, padded_height, x0 - 1, y0 - 1);
    unsigned long long sum = (bottom_right + above_left) - (above + left);
    unsigned long long area = static_cast<unsigned long long>(block_size) * block_size;
    int mean = static_cast<int>((sum + area / 2ULL) / area);
    bool selected = invert
        ? static_cast<int>(src[y * width + x]) <= mean - static_cast<int>(floorf(c))
        : static_cast<int>(src[y * width + x]) > mean - static_cast<int>(ceilf(c));
    dst[y * width + x] = static_cast<uint8_t>(selected ? max_value : 0);
}

__global__ void morph_kernel(const uint8_t* src, uint8_t* dst, int width, int height, int channels, int radius, int dilate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) {
        int value = dilate ? 0 : 255;
        for (int ky = -radius; ky <= radius; ++ky) for (int kx = -radius; kx <= radius; ++kx) {
            int sx = x + kx, sy = y + ky;
            int sample = (sx < 0 || sx >= width || sy < 0 || sy >= height) ? (dilate ? 0 : 255) : src[(sy * width + sx) * channels + c];
            value = dilate ? max(value, sample) : min(value, sample);
        }
        dst[(y * width + x) * channels + c] = static_cast<uint8_t>(value);
    }
}

dim3 grid2d(int width, int height) { return dim3((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y); }
}

VF_CUDA_API int vf_gpu_abi_version() { return VF_CUDA_ABI_VERSION; }

VF_CUDA_API int vf_gpu_device_count() { int count = 0; return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0; }

VF_CUDA_API int vf_gpu_compute_capability() {
    cudaDeviceProp prop{};
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess ? prop.major * 10 + prop.minor : 0;
}

VF_CUDA_API int vf_gpu_device_name(char* output, int capacity) {
    if (!output || capacity <= 0) return 1;
    cudaDeviceProp prop{}; cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) return cuda_result(error);
    strncpy_s(output, capacity, prop.name, _TRUNCATE); return 0;
}

VF_CUDA_API int vf_gpu_error_message(int error_code, char* output, int capacity) {
    if (!output || capacity <= 0) return VF_CUDA_INVALID_ARGUMENT;
    const char* message = "Unknown VisionFlow CUDA error";
    switch (error_code) {
        case VF_CUDA_OK: message = "Success"; break;
        case VF_CUDA_INVALID_ARGUMENT: message = "Invalid argument"; break;
        case VF_CUDA_ALLOCATION_FAILED: message = "Device allocation failed"; break;
        case VF_CUDA_COPY_FAILED: message = "Host/device copy failed"; break;
        case VF_CUDA_KERNEL_FAILED: message = "CUDA kernel failed"; break;
        case VF_CUDA_DEVICE_UNAVAILABLE: message = "CUDA device unavailable"; break;
        case VF_CUDA_ABI_MISMATCH: message = "CUDA DLL ABI mismatch"; break;
        case VF_CUDA_INTERNAL_ERROR: message = "Internal CUDA DLL error"; break;
        default:
            if (error_code >= VF_CUDA_RUNTIME_ERROR_BASE) {
                message = cudaGetErrorString(static_cast<cudaError_t>(error_code - VF_CUDA_RUNTIME_ERROR_BASE));
            }
            break;
    }
    strncpy_s(output, capacity, message, _TRUNCATE);
    return VF_CUDA_OK;
}

VF_CUDA_API int vf_bgr_to_gray_u8(const uint8_t* src, int w, int h, int stride, int sc, uint8_t* dst, int dstride, int dc) {
    if (sc != 3 || dc != 1) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    bgr_gray_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_bgr_to_rgb_u8(const uint8_t* src, int w, int h, int stride, int sc, uint8_t* dst, int dstride, int dc) {
    if (sc != 3 || dc != 3) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h * 3);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    bgr_rgb_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 3, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_crop_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int x,int y,int cw,int ch) {
    if (sc != dc || (sc != 1 && sc != 3) || x < 0 || y < 0 || cw <= 0 || ch <= 0 || x + cw > w || y + ch > h) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(cw) * ch * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    crop_kernel<<<grid2d(cw, ch), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, x, y, cw, ch, sc);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, cw, ch, sc, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_resize_gray_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int dw,int dh) {
    if (sc != 1 || dc != 1 || dw <= 0 || dh <= 0) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(dw) * dh);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    resize_gray_kernel<<<grid2d(dw, dh), dim3(BLOCK_X, BLOCK_Y)>>>(ds, dd, w, h, dw, dh);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, dw, dh, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_gaussian_blur_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int kernel) {
    if (sc != dc || (sc != 1 && sc != 3) || kernel < 3 || kernel % 2 == 0 || kernel > MAX_GAUSSIAN_KERNEL) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *ds = nullptr, *dd = nullptr;
    float* intermediate = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    cudaError_t error = cudaMalloc(&intermediate, static_cast<size_t>(w) * h * sc * sizeof(float));
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return cuda_result(error);
    }

    double sigma = 0.3 * ((kernel - 1) * 0.5 - 1) + 0.8;
    std::vector<float> weights(kernel);
    float total = 0.0f;
    int radius = kernel / 2;
    for (int i = -radius; i <= radius; ++i) {
        weights[i + radius] = expf(-(i * i) / static_cast<float>(2.0 * sigma * sigma));
        total += weights[i + radius];
    }
    for (float& value : weights) value /= total;

    error = cudaMemcpyToSymbol(gaussian_weights, weights.data(), static_cast<size_t>(kernel) * sizeof(float));
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(intermediate);
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return cuda_result(error);
    }
    gaussian_horizontal_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(ds, intermediate, w, h, sc, radius);
    gaussian_vertical_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(intermediate, dd, w, h, sc, radius);
    result = visionflow_cuda::kernel_result();
    visionflow_cuda::free_device(intermediate);
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, sc, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_threshold_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int threshold,int max_value,int invert) {
    if (sc != 1 || dc != 1 || threshold < 0 || threshold > 255 || max_value < 0 || max_value > 255) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    int result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    int count = w * h;
    threshold_kernel<<<(count + 255) / 256, 256>>>(ds, dd, count, threshold, max_value, invert);
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_adaptive_mean_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int block,float c,int max_value,int invert) {
    if (w <= 0 || h <= 0 || sc != 1 || dc != 1 || block < 3 || block % 2 == 0 ||
        max_value < 0 || max_value > 255 || !std::isfinite(c)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    int radius = block / 2;
    if (radius > (INT_MAX - w) / 2 || radius > (INT_MAX - h) / 2) return VF_CUDA_INVALID_ARGUMENT;
    int padded_width = w + radius * 2;
    int padded_height = h + radius * 2;
    if (static_cast<size_t>(padded_width) > SIZE_MAX / static_cast<size_t>(padded_height)) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    size_t padded_count = static_cast<size_t>(padded_width) * static_cast<size_t>(padded_height);
    if (padded_count > SIZE_MAX / sizeof(unsigned long long)) return VF_CUDA_INVALID_ARGUMENT;
    uint8_t *ds = nullptr, *dd = nullptr;
    uint8_t* padded = nullptr;
    unsigned long long* row_prefix = nullptr;
    unsigned long long* integral_transposed = nullptr;
    int result = alloc_copy(src, w, h, stride, 1, &ds);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&dd, static_cast<size_t>(w) * h);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(ds); return result; }
    result = visionflow_cuda::allocate_bytes(&padded, padded_count);
    if (result != VF_CUDA_OK) {
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return result;
    }
    cudaError_t error = cudaMalloc(&row_prefix, padded_count * sizeof(unsigned long long));
    if (error == cudaSuccess) error = cudaMalloc(&integral_transposed, padded_count * sizeof(unsigned long long));
    if (error != cudaSuccess) {
        visionflow_cuda::free_device(integral_transposed);
        visionflow_cuda::free_device(row_prefix);
        visionflow_cuda::free_device(padded);
        visionflow_cuda::free_device(dd);
        visionflow_cuda::free_device(ds);
        return cuda_result(error);
    }
    replicate_border_kernel<<<grid2d(padded_width, padded_height), dim3(BLOCK_X, BLOCK_Y)>>>(
        ds, padded, w, h, padded_width, padded_height, radius);
    row_prefix_u8_kernel<<<padded_height, SCAN_THREADS>>>(padded, row_prefix, padded_width, padded_height);
    dim3 transpose_block(TRANSPOSE_TILE, TRANSPOSE_ROWS);
    dim3 transpose_grid(
        (padded_width + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        (padded_height + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
    transpose_u64_kernel<<<transpose_grid, transpose_block>>>(
        row_prefix, integral_transposed, padded_width, padded_height);
    row_prefix_u64_inplace_kernel<<<padded_width, SCAN_THREADS>>>(
        integral_transposed, padded_height, padded_width);
    adaptive_integral_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(
        ds, integral_transposed, dd, w, h, padded_height, block, c, max_value, invert);
    result = visionflow_cuda::kernel_result();
    visionflow_cuda::free_device(integral_transposed);
    visionflow_cuda::free_device(row_prefix);
    visionflow_cuda::free_device(padded);
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, 1, dd);
    else visionflow_cuda::free_device(dd);
    visionflow_cuda::free_device(ds);
    return result;
}

VF_CUDA_API int vf_morphology_rect_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int operation,int kernel,int iterations) {
    if (sc != dc || (sc != 1 && sc != 3) || kernel < 3 || kernel % 2 == 0 || iterations < 1 || operation < VF_MORPH_OPEN || operation > VF_MORPH_ERODE) {
        return VF_CUDA_INVALID_ARGUMENT;
    }
    uint8_t *a = nullptr, *b = nullptr;
    int result = alloc_copy(src, w, h, stride, sc, &a);
    if (result != VF_CUDA_OK) return result;
    result = visionflow_cuda::allocate_bytes(&b, static_cast<size_t>(w) * h * sc);
    if (result != VF_CUDA_OK) { visionflow_cuda::free_device(a); return result; }
    auto pass = [&](int dilate) {
        morph_kernel<<<grid2d(w, h), dim3(BLOCK_X, BLOCK_Y)>>>(a, b, w, h, sc, kernel / 2, dilate);
        std::swap(a, b);
    };
    if (operation == VF_MORPH_OPEN) {
        for (int i = 0; i < iterations; ++i) pass(0);
        for (int i = 0; i < iterations; ++i) pass(1);
    } else if (operation == VF_MORPH_CLOSE) {
        for (int i = 0; i < iterations; ++i) pass(1);
        for (int i = 0; i < iterations; ++i) pass(0);
    } else {
        for (int i = 0; i < iterations; ++i) pass(operation == VF_MORPH_DILATE);
    }
    result = visionflow_cuda::kernel_result();
    if (result == VF_CUDA_OK) result = copy_back_free(dst, dstride, w, h, sc, a);
    else visionflow_cuda::free_device(a);
    visionflow_cuda::free_device(b);
    return result;
}
