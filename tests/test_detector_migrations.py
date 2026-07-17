from __future__ import annotations

import unittest

import cv2
import numpy as np

from detectors.detector_401_1 import Detector401_1


class _AreaUnsupportedRuntime:
    available = True
    unavailable_reason = ""
    supports_fused_401_2 = False

    def __init__(self):
        self.gray_calls = 0

    def bgr_to_gray(self, image):
        self.gray_calls += 1
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class Detector4011PlanMigrationTests(unittest.TestCase):
    @staticmethod
    def _params() -> dict:
        return {
            "process_scale": 0.63,
            "blur_size": 4,
            "adaptive_block_size": 6,
            "adaptive_c": -1.5,
            "max_value": 255,
            "invert": True,
            "morph_operation": "close",
            "morph_kernel": 4,
            "morph_iterations": 2,
            "roi_inset_px": 3,
            "contour_mode": "external",
            "min_area": 0,
            "max_area": 0,
            "min_circularity": 0,
            "min_fill_ratio": 0,
            "max_fill_ratio": 0,
        }

    @staticmethod
    def _legacy_reference(image: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = min(max(float(params["process_scale"]), 0.05), 1.0)
        target = (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale)))
        work = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
        blur_size = 5
        work = cv2.GaussianBlur(work, (blur_size, blur_size), 0)
        binary = cv2.adaptiveThreshold(
            work,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            7,
            -1.5,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2), scale

    def test_shared_plan_matches_legacy_cpu_preprocessing(self):
        image = np.random.default_rng(4011).integers(0, 256, size=(93, 117, 3), dtype=np.uint8)
        detector = Detector401_1(params=self._params())

        actual, scale = detector._make_binary(image)
        expected, expected_scale = self._legacy_reference(image, self._params())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(scale, expected_scale)
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")
        self.assertEqual(detector.preprocess_plan_cache_size, 1)

        detector._make_binary(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector.params["adaptive_c"] = -2.5
        detector._make_binary(image)
        self.assertEqual(detector.preprocess_plan_cache_size, 2)

    def test_area_unsupported_cuda_restarts_full_detector_on_cpu(self):
        image = np.random.default_rng(4012).integers(0, 256, size=(96, 112, 3), dtype=np.uint8)
        params = self._params()
        cpu_result = Detector401_1(params=params).run(image)
        runtime = _AreaUnsupportedRuntime()

        fallback_result = Detector401_1(
            params=params,
            use_gpu=True,
            gpu_runtime=runtime,
        ).run(image)

        self.assertEqual(fallback_result["defects"], cpu_result["defects"])
        self.assertEqual(fallback_result["pass"], cpu_result["pass"])
        self.assertEqual(fallback_result["score"], cpu_result["score"])
        self.assertEqual(runtime.gray_calls, 1)
        execution = fallback_result["execution"]
        self.assertEqual(execution["backend"], "cpu")
        self.assertEqual(execution["preprocess_capability"]["route"], "fallback")
        self.assertIn("area", execution["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
