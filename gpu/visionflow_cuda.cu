#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#define VF_EXPORT extern "C" __declspec(dllexport)

namespace {
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

int cuda_result(cudaError_t error) { return error == cudaSuccess ? 0 : 1000 + static_cast<int>(error); }

int alloc_copy(const uint8_t* host, int width, int height, int stride, int channels, uint8_t** device) {
    const size_t row_bytes = static_cast<size_t>(width) * channels;
    cudaError_t error = cudaMalloc(device, row_bytes * height);
    if (error != cudaSuccess) return cuda_result(error);
    error = cudaMemcpy2D(*device, row_bytes, host, stride, row_bytes, height, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { cudaFree(*device); *device = nullptr; }
    return cuda_result(error);
}

int copy_back_free(uint8_t* host, int stride, int width, int height, int channels, uint8_t* device) {
    const size_t row_bytes = static_cast<size_t>(width) * channels;
    cudaError_t error = cudaMemcpy2D(host, stride, device, row_bytes, row_bytes, height, cudaMemcpyDeviceToHost);
    cudaFree(device);
    return cuda_result(error);
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
    float sx = (x + 0.5f) * sw / dw - 0.5f, sy = (y + 0.5f) * sh / dh - 0.5f;
    int x0 = max(0, min(sw - 1, static_cast<int>(floorf(sx))));
    int y0 = max(0, min(sh - 1, static_cast<int>(floorf(sy))));
    int x1 = min(sw - 1, x0 + 1), y1 = min(sh - 1, y0 + 1);
    float ax = sx - floorf(sx), ay = sy - floorf(sy);
    float value = (1 - ay) * ((1 - ax) * src[y0 * sw + x0] + ax * src[y0 * sw + x1]) + ay * ((1 - ax) * src[y1 * sw + x0] + ax * src[y1 * sw + x1]);
    dst[y * dw + x] = static_cast<uint8_t>(value + 0.5f);
}

__global__ void gaussian_kernel(const uint8_t* src, uint8_t* dst, int width, int height, int channels, const float* weights, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int ky = -radius; ky <= radius; ++ky) for (int kx = -radius; kx <= radius; ++kx) {
            int sx = reflect101(x + kx, width), sy = reflect101(y + ky, height);
            sum += src[(sy * width + sx) * channels + c] * weights[ky + radius] * weights[kx + radius];
        }
        dst[(y * width + x) * channels + c] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, sum + 0.5f)));
    }
}

__global__ void threshold_kernel(const uint8_t* src, uint8_t* dst, int count, int threshold, int max_value, int invert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    bool high = src[i] > threshold;
    dst[i] = static_cast<uint8_t>((invert ? !high : high) ? max_value : 0);
}

__global__ void adaptive_kernel(const uint8_t* src, uint8_t* dst, int width, int height, int radius, float c, int max_value, int invert) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    unsigned int sum = 0;
    for (int ky = -radius; ky <= radius; ++ky) for (int kx = -radius; kx <= radius; ++kx) {
        int sx = max(0, min(width - 1, x + kx)), sy = max(0, min(height - 1, y + ky));
        sum += src[sy * width + sx];
    }
    float mean = static_cast<float>(sum) / ((radius * 2 + 1) * (radius * 2 + 1));
    bool high = src[y * width + x] > mean - c;
    dst[y * width + x] = static_cast<uint8_t>((invert ? !high : high) ? max_value : 0);
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

VF_EXPORT int vf_gpu_device_count() { int count = 0; return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0; }

VF_EXPORT int vf_gpu_compute_capability() {
    cudaDeviceProp prop{};
    return cudaGetDeviceProperties(&prop, 0) == cudaSuccess ? prop.major * 10 + prop.minor : 0;
}

VF_EXPORT int vf_gpu_device_name(char* output, int capacity) {
    if (!output || capacity <= 0) return 1;
    cudaDeviceProp prop{}; cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess) return cuda_result(error);
    strncpy_s(output, capacity, prop.name, _TRUNCATE); return 0;
}

VF_EXPORT int vf_bgr_to_gray_u8(const uint8_t* src, int w, int h, int stride, int sc, uint8_t* dst, int dstride, int dc) {
    if (!src || !dst || sc != 3 || dc != 1 || w <= 0 || h <= 0) return 1;
    uint8_t *ds = nullptr, *dd = nullptr; int r = alloc_copy(src, w, h, stride, 3, &ds); if (r) return r;
    if (cudaMalloc(&dd, static_cast<size_t>(w) * h) != cudaSuccess) { cudaFree(ds); return 2; }
    bgr_gray_kernel<<<grid2d(w,h), dim3(BLOCK_X,BLOCK_Y)>>>(ds,dd,w,h); r = cuda_result(cudaGetLastError());
    if (!r) r = copy_back_free(dst,dstride,w,h,1,dd); else cudaFree(dd); cudaFree(ds); return r;
}

VF_EXPORT int vf_bgr_to_rgb_u8(const uint8_t* src, int w, int h, int stride, int sc, uint8_t* dst, int dstride, int dc) {
    if (!src || !dst || sc != 3 || dc != 3) return 1; uint8_t *ds=nullptr,*dd=nullptr; int r=alloc_copy(src,w,h,stride,3,&ds); if(r)return r;
    cudaMalloc(&dd,static_cast<size_t>(w)*h*3); bgr_rgb_kernel<<<grid2d(w,h),dim3(BLOCK_X,BLOCK_Y)>>>(ds,dd,w,h); r=cuda_result(cudaGetLastError());
    if(!r)r=copy_back_free(dst,dstride,w,h,3,dd);else cudaFree(dd);cudaFree(ds);return r;
}

VF_EXPORT int vf_crop_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int x,int y,int cw,int ch) {
    if(!src||!dst||sc!=dc||x<0||y<0||cw<=0||ch<=0||x+cw>w||y+ch>h)return 1;uint8_t *ds=nullptr,*dd=nullptr;int r=alloc_copy(src,w,h,stride,sc,&ds);if(r)return r;
    cudaMalloc(&dd,static_cast<size_t>(cw)*ch*sc);crop_kernel<<<grid2d(cw,ch),dim3(BLOCK_X,BLOCK_Y)>>>(ds,dd,w,x,y,cw,ch,sc);r=cuda_result(cudaGetLastError());
    if(!r)r=copy_back_free(dst,dstride,cw,ch,sc,dd);else cudaFree(dd);cudaFree(ds);return r;
}

VF_EXPORT int vf_resize_gray_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int dw,int dh) {
    if(sc!=1||dc!=1||dw<=0||dh<=0)return 1;uint8_t *ds=nullptr,*dd=nullptr;int r=alloc_copy(src,w,h,stride,1,&ds);if(r)return r;cudaMalloc(&dd,static_cast<size_t>(dw)*dh);
    resize_gray_kernel<<<grid2d(dw,dh),dim3(BLOCK_X,BLOCK_Y)>>>(ds,dd,w,h,dw,dh);r=cuda_result(cudaGetLastError());if(!r)r=copy_back_free(dst,dstride,dw,dh,1,dd);else cudaFree(dd);cudaFree(ds);return r;
}

VF_EXPORT int vf_gaussian_blur_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int kernel) {
    if(sc!=dc||(sc!=1&&sc!=3)||kernel<3||kernel%2==0)return 1;uint8_t *ds=nullptr,*dd=nullptr;int r=alloc_copy(src,w,h,stride,sc,&ds);if(r)return r;cudaMalloc(&dd,static_cast<size_t>(w)*h*sc);
    double sigma=0.3*((kernel-1)*0.5-1)+0.8;std::vector<float> weights(kernel);float total=0;int radius=kernel/2;for(int i=-radius;i<=radius;++i){weights[i+radius]=expf(-(i*i)/(2.0f*sigma*sigma));total+=weights[i+radius];}for(float& v:weights)v/=total;
    float* dweights=nullptr;cudaMalloc(&dweights,kernel*sizeof(float));cudaMemcpy(dweights,weights.data(),kernel*sizeof(float),cudaMemcpyHostToDevice);
    gaussian_kernel<<<grid2d(w,h),dim3(BLOCK_X,BLOCK_Y)>>>(ds,dd,w,h,sc,dweights,radius);r=cuda_result(cudaGetLastError());cudaFree(dweights);if(!r)r=copy_back_free(dst,dstride,w,h,sc,dd);else cudaFree(dd);cudaFree(ds);return r;
}

VF_EXPORT int vf_threshold_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int threshold,int max_value,int invert) {
    if(sc!=1||dc!=1)return 1;uint8_t *ds=nullptr,*dd=nullptr;int r=alloc_copy(src,w,h,stride,1,&ds);if(r)return r;cudaMalloc(&dd,static_cast<size_t>(w)*h);
    int count=w*h;threshold_kernel<<<(count+255)/256,256>>>(ds,dd,count,threshold,max_value,invert);r=cuda_result(cudaGetLastError());if(!r)r=copy_back_free(dst,dstride,w,h,1,dd);else cudaFree(dd);cudaFree(ds);return r;
}

VF_EXPORT int vf_adaptive_mean_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int block,float c,int max_value,int invert) {
    if(sc!=1||dc!=1||block<3||block%2==0)return 1;uint8_t *ds=nullptr,*dd=nullptr;int r=alloc_copy(src,w,h,stride,1,&ds);if(r)return r;cudaMalloc(&dd,static_cast<size_t>(w)*h);
    adaptive_kernel<<<grid2d(w,h),dim3(BLOCK_X,BLOCK_Y)>>>(ds,dd,w,h,block/2,c,max_value,invert);r=cuda_result(cudaGetLastError());if(!r)r=copy_back_free(dst,dstride,w,h,1,dd);else cudaFree(dd);cudaFree(ds);return r;
}

VF_EXPORT int vf_morphology_rect_u8(const uint8_t* src,int w,int h,int stride,int sc,uint8_t* dst,int dstride,int dc,int operation,int kernel,int iterations) {
    if(sc!=dc||(sc!=1&&sc!=3)||kernel<3||kernel%2==0||iterations<1||operation<0||operation>3)return 1;uint8_t *a=nullptr,*b=nullptr;int r=alloc_copy(src,w,h,stride,sc,&a);if(r)return r;cudaMalloc(&b,static_cast<size_t>(w)*h*sc);
    auto pass=[&](int dilate){morph_kernel<<<grid2d(w,h),dim3(BLOCK_X,BLOCK_Y)>>>(a,b,w,h,sc,kernel/2,dilate);std::swap(a,b);};
    if(operation==0){for(int i=0;i<iterations;++i)pass(0);for(int i=0;i<iterations;++i)pass(1);}else if(operation==1){for(int i=0;i<iterations;++i)pass(1);for(int i=0;i<iterations;++i)pass(0);}else for(int i=0;i<iterations;++i)pass(operation==2);
    r=cuda_result(cudaGetLastError());if(!r)r=copy_back_free(dst,dstride,w,h,sc,a);else cudaFree(a);cudaFree(b);return r;
}
