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

        self.assertEqual(execute.count("cudaMemcpy2D("), 2)
        self.assertNotIn("cudaMalloc", execute)
        self.assertNotIn("reserve_plan_buffers", execute)
        self.assertNotIn("detector", descriptor.lower())


if __name__ == "__main__":
    unittest.main()
