from __future__ import annotations

import unittest

from gpu.preflight_cuda_build import REQUIRED_ABI_V1_EXPORTS, inspect_contract


class CudaSourceContractTests(unittest.TestCase):
    def test_header_source_runtime_smoke_and_build_manifest_are_synchronized(self):
        result = inspect_contract()

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["abi_version"], 1)
        self.assertTrue(REQUIRED_ABI_V1_EXPORTS.issubset(result["exports"]))
        self.assertEqual(result["dll_sources"], ["gpu/visionflow_cuda.cu"])
        self.assertEqual(result["smoke_sources"], ["gpu/test_cuda_api.cu"])


if __name__ == "__main__":
    unittest.main()
