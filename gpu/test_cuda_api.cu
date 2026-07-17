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

    void* context = nullptr;
    result = vf_context_create(&context);
    if (result != VF_CUDA_OK || context == nullptr) {
        std::cerr << "Persistent context creation failed\n";
        return 6;
    }
    std::vector<uint8_t> fused_binary(width * height, 0);
    result = vf_preprocess_401_2_u8(
        context,
        bgr.data(), width, height, width * 3, 3,
        fused_binary.data(), width,
        3, 3, -2.0f, 255, 1);
    uint64_t reserved_bytes = 0;
    uint64_t allocation_count = 0;
    int stats_result = vf_context_stats(context, &reserved_bytes, &allocation_count);
    if (result != VF_CUDA_OK || stats_result != VF_CUDA_OK ||
        reserved_bytes == 0 || allocation_count == 0) {
        char message[256]{};
        int failed = result != VF_CUDA_OK ? result : stats_result;
        vf_gpu_error_message(failed, message, static_cast<int>(sizeof(message)));
        std::cerr << "Fused 401-2 smoke failed: " << message << "\n";
        return 7;
    }

    VfPlanOperatorV1 operators[3]{};
    operators[0].struct_size = sizeof(VfPlanOperatorV1);
    operators[0].kind = VF_PLAN_GRAY;
    operators[0].input_node = VF_PLAN_INPUT_NODE;
    operators[0].output_node = 0;
    operators[1].struct_size = sizeof(VfPlanOperatorV1);
    operators[1].kind = VF_PLAN_GAUSSIAN;
    operators[1].input_node = 0;
    operators[1].output_node = 1;
    operators[1].int_params[0] = 3;
    operators[2].struct_size = sizeof(VfPlanOperatorV1);
    operators[2].kind = VF_PLAN_ADAPTIVE_MEAN;
    operators[2].input_node = 1;
    operators[2].output_node = 2;
    operators[2].int_params[0] = 3;
    operators[2].int_params[1] = 255;
    operators[2].int_params[2] = 1;
    operators[2].float_params[0] = -2.0f;
    VfPlanDescV1 descriptor{};
    descriptor.struct_size = sizeof(VfPlanDescV1);
    descriptor.version = VF_CUDA_PLAN_VERSION;
    descriptor.input_channels = 3;
    descriptor.operator_count = 3;
    descriptor.operators = operators;
    descriptor.output_node = 2;
    char plan_reason[256]{};
    result = vf_plan_query(&descriptor, width, height, plan_reason, sizeof(plan_reason));
    void* plan = nullptr;
    if (result == VF_CUDA_OK) result = vf_plan_create(context, &descriptor, width, height, &plan);
    std::vector<uint8_t> plan_binary(width * height, 0);
    if (result == VF_CUDA_OK) {
        result = vf_plan_execute(
            plan, bgr.data(), width, height, width * 3, 3,
            plan_binary.data(), width, 1);
    }
    uint64_t plan_allocation_count = 0;
    if (result == VF_CUDA_OK) {
        result = vf_context_stats(context, &reserved_bytes, &plan_allocation_count);
    }
    if (result == VF_CUDA_OK) {
        result = vf_plan_execute(
            plan, bgr.data(), width, height, width * 3, 3,
            plan_binary.data(), width, 1);
    }
    uint64_t repeated_allocation_count = 0;
    if (result == VF_CUDA_OK) {
        result = vf_context_stats(context, &reserved_bytes, &repeated_allocation_count);
    }
    int plan_destroy_result = vf_plan_destroy(plan);
    int destroy_result = vf_context_destroy(context);
    if (result != VF_CUDA_OK || plan_destroy_result != VF_CUDA_OK ||
        destroy_result != VF_CUDA_OK || plan_allocation_count != repeated_allocation_count) {
        char message[256]{};
        int failed = result != VF_CUDA_OK ? result :
            plan_destroy_result != VF_CUDA_OK ? plan_destroy_result : destroy_result;
        vf_gpu_error_message(failed, message, static_cast<int>(sizeof(message)));
        std::cerr << "Generic native plan smoke failed: " << message
                  << " reason=" << plan_reason << "\n";
        return 8;
    }

    std::cout << "C ABI, grayscale, fused 401-2 and generic plan smoke passed\n";
    return 0;
}
