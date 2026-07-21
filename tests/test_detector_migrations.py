from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.preprocess_plan import CpuPreprocessDagExecutor, CpuPreprocessExecutor
from core.preprocess_cache import TilePreprocessCache
from detectors.detector_401 import Detector401
from detectors.detector_401_1 import Detector401_1
from detectors.detector_401_2 import Detector401_2
from detectors.detector_900 import Detector900
from detectors.base_detector import BaseDetector


class _AreaUnsupportedRuntime:
    available = True
    unavailable_reason = ""
    supports_fused_401_2 = False
    fallback_to_cpu = True

    def __init__(self):
        self.gray_calls = 0

    def bgr_to_gray(self, image):
        self.gray_calls += 1
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class _NativeAreaRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = True
    supports_fused_401_2 = False

    def __init__(self):
        self.calls = 0

    @staticmethod
    def native_plan_capability(_plan, _image):
        return True, "native area resize supported"

    def execute_plan(self, image, plan):
        self.calls += 1
        return CpuPreprocessExecutor().execute(image, plan)


class _Failing401Runtime:
    available = True
    unavailable_reason = ""
    supports_fused_401_2 = False

    @staticmethod
    def gaussian_blur(image, kernel_size):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def morphology(*_args):
        raise RuntimeError("injected 401 morphology failure")

    @staticmethod
    def bgr_to_gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def adaptive_threshold(image, block_size, c, max_value, invert):
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type, block_size, c
        )


class _AvailableRuntimeWithoutDag:
    available = True
    unavailable_reason = ""
    supports_fused_401_2 = False


class _NativeDagRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_dag_plan = True

    def __init__(self):
        self.calls = 0

    @staticmethod
    def native_dag_plan_capability(_plan, _image):
        return True, "supported fake native DAG plan"

    def execute_dag_plan(self, image, plan):
        self.calls += 1
        return CpuPreprocessDagExecutor().execute(image, plan)


class _MeanDetector(BaseDetector):
    detector_id = "mean"

    def detect(self, image):
        return [{"confidence": float(np.mean(image))}]


class DetectorBatchContractTests(unittest.TestCase):
    def test_default_batch_and_roi_contract_runs_in_input_order(self):
        detector = _MeanDetector()
        images = [
            np.full((6, 7), 10, dtype=np.uint8),
            np.full((6, 7), 20, dtype=np.uint8),
        ]

        full = detector.run_batch(images)
        roi = detector.run_batch(images, rois=[(1, 2, 3, 2), (0, 0, 4, 5)])

        self.assertEqual([item["score"] for item in full], [10.0, 20.0])
        self.assertEqual([item["score"] for item in roi], [10.0, 20.0])

    def test_default_batch_rejects_invalid_roi(self):
        detector = _MeanDetector()
        with self.assertRaisesRegex(ValueError, "exceeds image bounds"):
            detector.run_batch([np.zeros((5, 5), dtype=np.uint8)], rois=[(4, 4, 2, 2)])


class SharedTilePreprocessCacheTests(unittest.TestCase):
    def test_cpu_detectors_share_one_gray_conversion_for_same_tile(self):
        image = np.zeros((32, 40, 3), dtype=np.uint8)
        cache = TilePreprocessCache(image)
        detectors = (Detector401_1(), Detector401_2(), Detector900())

        outputs = []
        for detector in detectors:
            detector._active_preprocess_cache = cache
            outputs.append(detector.preprocess(image))

        self.assertIs(outputs[0], outputs[1])
        self.assertIs(outputs[1], outputs[2])
        self.assertEqual(outputs[0].shape, (32, 40))


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
        self.assertEqual(runtime.gray_calls, 0)
        execution = fallback_result["execution"]
        self.assertEqual(execution["backend"], "cpu")
        self.assertEqual(execution["preprocess_capability"]["route"], "fallback")
        self.assertIn("area", execution["fallback_reason"])

    def test_area_supported_native_plan_runs_401_1_in_one_gpu_call(self):
        image = np.random.default_rng(4014).integers(0, 256, size=(96, 112, 3), dtype=np.uint8)
        params = self._params()
        expected = Detector401_1(params=params).run(image)
        runtime = _NativeAreaRuntime()

        actual = Detector401_1(params=params, use_gpu=True, gpu_runtime=runtime).run(image)

        self.assertEqual(actual["defects"], expected["defects"])
        self.assertEqual(actual["pass"], expected["pass"])
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(actual["execution"]["backend"], "cuda_dll")
        self.assertEqual(actual["execution"]["preprocess_capability"]["route"], "native_plan")

    def test_area_unsupported_cuda_fails_before_any_gpu_call_when_fallback_is_disabled(self):
        image = np.random.default_rng(4013).integers(0, 256, size=(96, 112, 3), dtype=np.uint8)
        runtime = _AreaUnsupportedRuntime()
        runtime.fallback_to_cpu = False
        detector = Detector401_1(
            params=self._params(),
            use_gpu=True,
            gpu_runtime=runtime,
        )

        with self.assertRaisesRegex(RuntimeError, "Resize.*area"):
            detector.run(image)

        self.assertEqual(runtime.gray_calls, 0)


class Detector401PlanMigrationTests(unittest.TestCase):
    @staticmethod
    def _params() -> dict:
        return {
            "roi_inset_px": 4,
            "blur_size": 4,
            "morph_operation": "close",
            "morph_kernel": 4,
            "morph_iterations": 2,
            "adaptive_block_size": 6,
            "adaptive_c": 2.5,
            "binary_inv": True,
            "max_value": 255,
            "contour_mode": "external",
            "min_area": 0,
            "max_area": 0,
        }

    @staticmethod
    def _legacy_reference(image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=2)
        gray = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            7,
            2.5,
        )

    def test_shared_plan_preserves_bgr_morphology_order_and_cache(self):
        image = np.random.default_rng(401).integers(0, 256, size=(91, 113, 3), dtype=np.uint8)
        detector = Detector401(params=self._params())

        actual = detector._make_binary(image)

        np.testing.assert_array_equal(actual, self._legacy_reference(image))
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector._make_binary(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector.params["adaptive_c"] = 3.5
        detector._make_binary(image)
        self.assertEqual(detector.preprocess_plan_cache_size, 2)

    def test_gpu_primitive_failure_restarts_full_detector_on_cpu(self):
        image = np.random.default_rng(402).integers(0, 256, size=(96, 112, 3), dtype=np.uint8)
        params = self._params()
        cpu_result = Detector401(params=params).run(image)

        fallback_result = Detector401(
            params=params,
            use_gpu=True,
            gpu_runtime=_Failing401Runtime(),
        ).run(image)

        self.assertEqual(fallback_result["defects"], cpu_result["defects"])
        self.assertEqual(fallback_result["pass"], cpu_result["pass"])
        self.assertEqual(fallback_result["score"], cpu_result["score"])
        execution = fallback_result["execution"]
        self.assertEqual(execution["backend"], "cpu")
        self.assertEqual(execution["preprocess_capability"]["route"], "fallback")
        self.assertIn("injected 401 morphology failure", execution["fallback_reason"])


class Detector4012TileWhiteRatioTests(unittest.TestCase):
    def test_white_ratio_work_is_profiled(self):
        detector = Detector401_2(params={
            "blur_size": 3,
            "adaptive_block_size": 3,
            "adaptive_c": 0.0,
            "white_pixel_ratio_threshold": 0.0,
        })
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.rectangle(image, (8, 8), (23, 23), (255, 255, 255), thickness=-1)

        result = detector.run(image)

        stages = result["execution"]["performance"]["stages_sec"]
        self.assertIn("white_ratio_analysis", stages)
        self.assertGreaterEqual(stages["white_ratio_analysis"], 0.0)

    def test_whole_tile_ratio_returns_at_most_one_defect(self):
        binary = np.zeros((60, 76), dtype=np.uint8)
        binary[:, :38] = 255
        detector = Detector401_2(
            params={"roi_inset_px": 2, "white_pixel_ratio_threshold": 0.5}
        )

        with patch.object(detector, "_make_binary", return_value=binary):
            defects = detector.detect(np.zeros((64, 80), dtype=np.uint8))

        self.assertEqual(len(defects), 1)
        defect = defects[0]
        self.assertEqual(defect["bbox_local"], [2, 2, 76, 60])
        self.assertEqual(defect["area"], 4560.0)
        self.assertEqual(defect["metadata"]["shape"], "tile_roi")
        self.assertEqual(defect["metadata"]["white_pixel_count"], 2280)
        self.assertEqual(defect["metadata"]["total_pixel_count"], 4560)
        self.assertEqual(defect["metadata"]["white_pixel_ratio"], 0.5)
        self.assertEqual(defect["metadata"]["roi_offset_local"], [2, 2])
        self.assertEqual(defect["metadata"]["roi_size"], [76, 60])

    def test_ratio_below_threshold_passes(self):
        binary = np.zeros((20, 20), dtype=np.uint8)
        binary[:10, :10] = 255
        detector = Detector401_2(params={"white_pixel_ratio_threshold": 0.251})

        with patch.object(detector, "_make_binary", return_value=binary):
            defects = detector.detect(np.zeros((20, 20), dtype=np.uint8))

        self.assertEqual(defects, [])

    def test_legacy_0625_threshold_means_62_5_percent_and_is_inclusive(self):
        detector = Detector401_2(params={"white_pixel_ratio_threshold": 0.625})
        image = np.zeros((20, 20), dtype=np.uint8)
        at_threshold = np.zeros((20, 20), dtype=np.uint8)
        at_threshold.flat[:250] = 255
        below_threshold = at_threshold.copy()
        below_threshold.flat[249] = 0

        with patch.object(detector, "_make_binary", return_value=at_threshold):
            defects = detector.detect(image)
        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["metadata"]["white_pixel_ratio"], 0.625)
        self.assertEqual(defects[0]["metadata"]["white_pixel_ratio_threshold_percent"], 62.5)

        with patch.object(detector, "_make_binary", return_value=below_threshold):
            self.assertEqual(detector.detect(image), [])

    def test_many_disconnected_white_points_still_return_one_defect(self):
        binary = np.zeros((128, 128), dtype=np.uint8)
        binary[::2, ::2] = 255
        detector = Detector401_2(params={"white_pixel_ratio_threshold": 0.2})

        with patch.object(detector, "_make_binary", return_value=binary):
            defects = detector.detect(np.zeros((128, 128), dtype=np.uint8))

        self.assertEqual(len(defects), 1)
        self.assertEqual(defects[0]["metadata"]["white_pixel_count"], 4096)


class Detector900DagMigrationTests(unittest.TestCase):
    @staticmethod
    def _params() -> dict:
        return {
            "max_value": 255,
            "outer_threshold": 123,
            "outer_invert": True,
            "inner_adaptive_block_size": 6,
            "inner_adaptive_c": -1.25,
            "inner_invert": False,
            "roi_inset_px": 2,
            "outer_target_width": 40,
            "outer_width_tolerance": 40,
            "outer_target_height": 40,
            "outer_height_tolerance": 40,
            "inner_target_width": 30,
            "inner_width_tolerance": 30,
            "inner_target_height": 30,
            "inner_height_tolerance": 30,
        }

    def test_cpu_dag_shares_gray_and_matches_legacy_masks(self):
        image = np.random.default_rng(900).integers(0, 256, size=(71, 83, 3), dtype=np.uint8)
        detector = Detector900(params=self._params())

        masks = detector._make_masks(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        expected_outer = cv2.threshold(gray, 123, 255, cv2.THRESH_BINARY_INV)[1]
        expected_inner = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, -1.25
        )

        np.testing.assert_array_equal(masks["outer_mask"], expected_outer)
        np.testing.assert_array_equal(masks["inner_mask"], expected_inner)
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector._make_masks(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector.params["outer_threshold"] = 124
        detector._make_masks(image)
        self.assertEqual(detector.preprocess_plan_cache_size, 2)

    def test_missing_cuda_dag_restarts_full_detector_on_cpu(self):
        image = np.random.default_rng(901).integers(0, 256, size=(72, 84, 3), dtype=np.uint8)
        params = self._params()
        cpu_result = Detector900(params=params).run(image)

        fallback_result = Detector900(
            params=params,
            use_gpu=True,
            gpu_runtime=_AvailableRuntimeWithoutDag(),
        ).run(image)

        self.assertEqual(fallback_result["defects"], cpu_result["defects"])
        self.assertEqual(fallback_result["pass"], cpu_result["pass"])
        execution = fallback_result["execution"]
        self.assertEqual(execution["backend"], "cpu")
        self.assertEqual(execution["preprocess_capability"]["route"], "fallback")
        self.assertIn("CUDA DAG executor is not available", execution["fallback_reason"])

    def test_native_cuda_dag_routes_900_once_and_preserves_results(self):
        image = np.random.default_rng(902).integers(0, 256, size=(72, 84, 3), dtype=np.uint8)
        params = self._params()
        cpu_result = Detector900(params=params).run(image)
        runtime = _NativeDagRuntime()

        gpu_result = Detector900(params=params, use_gpu=True, gpu_runtime=runtime).run(image)

        self.assertEqual(runtime.calls, 1)
        self.assertEqual(gpu_result["defects"], cpu_result["defects"])
        self.assertEqual(gpu_result["pass"], cpu_result["pass"])
        self.assertEqual(
            gpu_result["execution"]["preprocess_capability"]["route"],
            "native_dag_plan",
        )


if __name__ == "__main__":
    unittest.main()
