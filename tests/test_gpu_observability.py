from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.gpu_runtime import GpuRuntime
from core.performance import PipelineProfiler
from core.pipeline import AOIPipeline
from gpu.validate_cuda_dll import compare


class _SuccessfulDll:
    @staticmethod
    def vf_bgr_to_gray_u8(*_args):
        return 0


class PipelineProfilerTests(unittest.TestCase):
    def test_snapshot_separates_pipeline_detector_and_report_stages(self):
        profiler = PipelineProfiler()
        with profiler.measure("tiling"):
            pass
        with profiler.measure("detector:401"):
            pass
        with profiler.measure("report:json"):
            pass

        snapshot = profiler.snapshot()

        self.assertEqual(snapshot["schema_version"], 1)
        self.assertIn("tiling", snapshot["stages_sec"])
        self.assertIn("401", snapshot["detectors_sec"])
        self.assertIn("json", snapshot["reporting_sec"])
        self.assertGreaterEqual(snapshot["end_to_end_sec"], 0.0)


class GpuRuntimeMetricsTests(unittest.TestCase):
    def test_disabled_runtime_has_zero_cuda_calls(self):
        runtime = GpuRuntime(enabled=False)

        metrics = runtime.performance_stats()

        self.assertEqual(metrics["call_count"], 0)
        self.assertEqual(metrics["host_to_device_bytes"], 0)
        self.assertEqual(metrics["device_to_host_bytes"], 0)

    def test_synchronous_call_records_estimated_transfer_bytes(self):
        runtime = GpuRuntime(enabled=False)
        runtime._dll = _SuccessfulDll()
        runtime.device_count = 1
        image = np.zeros((2, 3, 3), dtype=np.uint8)

        runtime.bgr_to_gray(image)
        metrics = runtime.performance_stats()

        self.assertEqual(metrics["call_count"], 1)
        self.assertEqual(metrics["estimated_round_trips"], 1)
        self.assertEqual(metrics["host_to_device_bytes"], image.nbytes)
        self.assertEqual(metrics["device_to_host_bytes"], 6)
        self.assertEqual(metrics["functions"]["vf_bgr_to_gray_u8"]["calls"], 1)


class ComparisonToleranceTests(unittest.TestCase):
    def test_max_diff_is_applied_per_pixel(self):
        expected = np.zeros((2, 2), dtype=np.uint8)
        actual = np.ones((2, 2), dtype=np.uint8)

        result = compare("within_one", actual, expected, max_diff=1)

        self.assertEqual(result["out_of_tolerance_ratio"], 0.0)

    def test_excessive_pixel_ratio_fails(self):
        expected = np.zeros((2, 2), dtype=np.uint8)
        actual = np.array([[2, 0], [0, 0]], dtype=np.uint8)

        with self.assertRaises(AssertionError):
            compare("one_bad_pixel", actual, expected, max_diff=1, mismatch_ratio=0.0)

    def test_small_excessive_pixel_ratio_can_be_explicitly_allowed(self):
        expected = np.zeros((2, 2), dtype=np.uint8)
        actual = np.array([[2, 0], [0, 0]], dtype=np.uint8)

        result = compare("one_allowed_pixel", actual, expected, max_diff=1, mismatch_ratio=0.25)

        self.assertEqual(result["out_of_tolerance_ratio"], 0.25)


class CpuFallbackRegressionTests(unittest.TestCase):
    @staticmethod
    def _recipe() -> dict:
        return {
            "recipe_name": "GPU_OBSERVABILITY_TEST",
            "product_id": "TEST",
            "machine_id": "TEST",
            "version": "1.0.0",
            "gpu": {
                "tiling": False,
                "display": False,
                "dll_path": "missing.dll",
                "fallback_to_cpu": True,
            },
            "tile": {"mode": "grid", "width": 64, "height": 64, "overlap_x": 0, "overlap_y": 0},
            "decision": {"mode": "all_detectors_must_pass", "important_detectors": ["401-1"], "max_ng_count": 0},
            "detectors": {
                "401-1": {
                    "enabled": True,
                    "use_gpu": False,
                    "display_name": "fallback regression",
                    "params": {
                        "blur_size": 3,
                        "adaptive_block_size": 3,
                        "adaptive_c": -2.0,
                        "roi_inset_px": 0,
                        "contour_mode": "external",
                        "morph_operation": "none",
                        "process_scale": 1.0,
                        "min_area": 0,
                        "max_area": 0,
                        "min_circularity": 0,
                        "min_fill_ratio": 0,
                        "max_fill_ratio": 0,
                    },
                }
            },
            "output": {
                "save_overlay": False,
                "save_ng_tiles": False,
                "save_csv": False,
                "save_matrix_csv": False,
                "save_json": False,
            },
        }

    @staticmethod
    def _normalized(result: dict) -> dict:
        normalized = deepcopy(result)
        for key in ("duration_sec", "outputs", "execution"):
            normalized.pop(key, None)
        for tile_result in normalized["tiles"]:
            for detector_result in tile_result["detectors"]:
                detector_result.pop("execution", None)
        return normalized

    def test_missing_gpu_fallback_matches_cpu_only_result(self):
        image = np.random.default_rng(20260714).integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix="visionflow_cpu_fallback_") as temporary:
            root = Path(temporary)
            image_path = root / "input.png"
            encoded, payload = cv2.imencode(".png", image)
            self.assertTrue(encoded)
            image_path.write_bytes(payload.tobytes())

            cpu_recipe = self._recipe()
            fallback_recipe = deepcopy(cpu_recipe)
            fallback_recipe["gpu"]["tiling"] = True
            fallback_recipe["gpu"]["dll_path"] = str(root / "definitely_missing.dll")
            fallback_recipe["detectors"]["401-1"]["use_gpu"] = True
            cpu_path = root / "cpu.yaml"
            fallback_path = root / "fallback.yaml"
            cpu_path.write_text(yaml.safe_dump(cpu_recipe, sort_keys=False), encoding="utf-8")
            fallback_path.write_text(yaml.safe_dump(fallback_recipe, sort_keys=False), encoding="utf-8")

            cpu_result = AOIPipeline(cpu_path, root / "cpu_output").run(image_path)
            fallback_result = AOIPipeline(fallback_path, root / "fallback_output").run(image_path)

        self.assertEqual(self._normalized(cpu_result), self._normalized(fallback_result))
        self.assertEqual(cpu_result["execution"]["gpu"]["metrics"]["call_count"], 0)
        self.assertFalse(fallback_result["execution"]["gpu"]["tiling"]["active"])
        self.assertEqual(fallback_result["execution"]["gpu"]["metrics"]["call_count"], 0)
        self.assertIn("performance", cpu_result["execution"])
        self.assertIn("401-1", cpu_result["execution"]["performance"]["detectors_sec"])


if __name__ == "__main__":
    unittest.main()
