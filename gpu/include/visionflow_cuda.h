#ifndef VISIONFLOW_CUDA_H
#define VISIONFLOW_CUDA_H

#include <stdint.h>
#include "visionflow_cuda_errors.h"

#define VF_CUDA_ABI_VERSION 1
#define VF_CUDA_PLAN_VERSION 1
#define VF_PLAN_INPUT_NODE (-1)

/*
 * ABI rules:
 * - All image pointers are host pointers to uint8 interleaved data.
 * - Strides are byte counts, not pixel counts.
 * - The caller owns every input/output buffer and must allocate the output.
 * - Calls are synchronous: output is ready when the function returns.
 * - A return value of VF_CUDA_OK means success; other values are declared in
 *   visionflow_cuda_errors.h and can be described by vf_gpu_error_message().
 * - The Python bridge serializes calls sharing one GpuRuntime. Native callers
 *   should also serialize calls unless they provide their own higher-level
 *   synchronization.
 * - Context APIs are additive ABI v1 extensions. Callers may probe their
 *   exports and keep using the stateless primitive APIs with an older DLL.
 * - A context owns reusable device buffers and must be destroyed by the same
 *   module with vf_context_destroy(). It is not safe for concurrent calls.
 */

#if defined(_WIN32)
#  if defined(VISIONFLOW_CUDA_EXPORTS)
#    define VF_CUDA_API __declspec(dllexport)
#  else
#    define VF_CUDA_API __declspec(dllimport)
#  endif
#else
#  define VF_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum VisionFlowMorphologyOperation {
    VF_MORPH_OPEN = 0,
    VF_MORPH_CLOSE = 1,
    VF_MORPH_DILATE = 2,
    VF_MORPH_ERODE = 3
};

enum VisionFlowPlanOperatorKind {
    VF_PLAN_GRAY = 1,
    VF_PLAN_GAUSSIAN = 2,
    VF_PLAN_THRESHOLD = 3,
    VF_PLAN_ADAPTIVE_MEAN = 4,
    VF_PLAN_MORPHOLOGY = 5
};

typedef struct VfPlanOperatorV1 {
    uint32_t struct_size;
    int32_t kind;
    int32_t input_node;
    int32_t output_node;
    int32_t int_params[4];
    float float_params[2];
} VfPlanOperatorV1;

typedef struct VfPlanDescV1 {
    uint32_t struct_size;
    uint32_t version;
    int32_t input_channels;
    int32_t operator_count;
    const VfPlanOperatorV1* operators;
    int32_t output_node;
} VfPlanDescV1;

VF_CUDA_API int vf_gpu_abi_version(void);
VF_CUDA_API int vf_gpu_device_count(void);
VF_CUDA_API int vf_gpu_compute_capability(void);
VF_CUDA_API int vf_gpu_device_name(char* output, int capacity);
VF_CUDA_API int vf_gpu_error_message(int error_code, char* output, int capacity);

VF_CUDA_API int vf_context_create(void** context);
VF_CUDA_API int vf_context_destroy(void* context);
VF_CUDA_API int vf_context_stats(
    void* context, uint64_t* reserved_bytes, uint64_t* allocation_count);

/*
 * Optional generic plan ABI. The descriptor is backend-neutral and contains
 * no detector ID/name. vf_plan_create copies and validates the descriptor;
 * vf_plan_execute only transfers image data and launches the compiled plan.
 * A plan borrows its context and must be destroyed before that context.
 */
VF_CUDA_API int vf_plan_query(
    const VfPlanDescV1* desc, int width, int height,
    char* reason, int reason_capacity);
VF_CUDA_API int vf_plan_create(
    void* context, const VfPlanDescV1* desc, int width, int height, void** plan);
VF_CUDA_API int vf_plan_execute(
    void* plan,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);
VF_CUDA_API int vf_plan_destroy(void* plan);

VF_CUDA_API int vf_bgr_to_gray_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);

VF_CUDA_API int vf_bgr_to_rgb_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels);

VF_CUDA_API int vf_crop_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int crop_x, int crop_y, int crop_width, int crop_height);

VF_CUDA_API int vf_resize_gray_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int dst_width, int dst_height);

VF_CUDA_API int vf_gaussian_blur_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int kernel_size);

VF_CUDA_API int vf_threshold_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int threshold, int max_value, int invert);

VF_CUDA_API int vf_adaptive_mean_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int block_size, float c, int max_value, int invert);

VF_CUDA_API int vf_morphology_rect_u8(
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride, int dst_channels,
    int operation, int kernel_size, int iterations);

VF_CUDA_API int vf_preprocess_401_2_u8(
    void* context,
    const uint8_t* src, int width, int height, int src_stride, int src_channels,
    uint8_t* dst, int dst_stride,
    int gaussian_kernel_size,
    int adaptive_block_size, float adaptive_c,
    int max_value, int invert);

#ifdef __cplusplus
}
#endif

#endif
