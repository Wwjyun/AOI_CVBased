from __future__ import annotations

import unittest
from pathlib import Path

from gpu.preflight_cuda_build import (
    OPTIONAL_GENERIC_PLAN_EXPORTS,
    REQUIRED_ABI_V1_EXPORTS,
    inspect_contract,
)


class CudaSourceContractTests(unittest.TestCase):
    def test_header_source_runtime_smoke_and_build_manifest_are_synchronized(self):
        result = inspect_contract()

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["abi_version"], 1)
        self.assertTrue(REQUIRED_ABI_V1_EXPORTS.issubset(result["exports"]))
        self.assertEqual(set(result["optional_generic_plan_exports"]), OPTIONAL_GENERIC_PLAN_EXPORTS)
        self.assertEqual(result["dll_sources"], ["gpu/visionflow_cuda.cu"])
        self.assertEqual(result["smoke_sources"], ["gpu/test_cuda_api.cu"])

    def test_generic_plan_execute_has_one_upload_one_download_and_no_allocation(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        execute = source.split("VF_CUDA_API int vf_plan_execute(", 1)[1].split(
            "VF_CUDA_API int vf_plan_destroy(", 1
        )[0]
        header = (root / "gpu" / "include" / "visionflow_cuda.h").read_text(encoding="utf-8")
        descriptor = header.split("typedef struct VfPlanOperatorV1", 1)[1].split(
            "VF_CUDA_API int vf_gpu_abi_version", 1
        )[0]

        self.assertEqual(execute.count("cudaMemcpy2DAsync("), 2)
        self.assertEqual(execute.count("cudaStreamSynchronize"), 0)
        self.assertEqual(execute.count("stream_result"), 1)
        self.assertNotIn("cudaMalloc", execute)
        self.assertNotIn("reserve_plan_buffers", execute)
        self.assertIn("context->stream", execute)
        self.assertIn("context->u8[4]", execute)
        self.assertNotIn("detector", descriptor.lower())

    def test_persistent_context_owns_stream_and_fused_path_uses_it(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        context = source.split("struct PersistentContext", 1)[1].split("struct NativePlan", 1)[0]
        fused = source.split("VF_CUDA_API int vf_preprocess_401_2_u8(", 1)[1].split(
            "VF_CUDA_API int vf_morphology_rect_u8(", 1
        )[0]

        self.assertIn("cudaStreamCreateWithFlags", context)
        self.assertIn("cudaStreamDestroy", context)
        self.assertIn("persistent->stream", fused)
        self.assertEqual(fused.count("cudaMemcpy2DAsync("), 2)


if __name__ == "__main__":
    unittest.main()
