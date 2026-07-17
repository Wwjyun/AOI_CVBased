from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.gpu_runtime import GpuRuntime, GpuRuntimeError
from core.performance import PipelineProfiler
from core.pipeline import AOIPipeline
from core.preprocess_plan import (
    AdaptiveMean,
    CpuPreprocessExecutor,
    CpuPreprocessDagExecutor,
    CudaPreprocessExecutor,
    Gaussian,
    Gray,
    InvalidPreprocessPlan,
    Morphology,
    PreprocessPlan,
    PreprocessDagNode,
    PreprocessDagPlan,
    PreprocessPlanCache,
    Resize,
    Threshold,
    UnsupportedPreprocessPlan,
)
from detectors.detector_401_2 import Detector401_2
from gpu.validate_cuda_dll import compare


class _SuccessfulDll:
    @staticmethod
    def vf_bgr_to_gray_u8(*_args):
        return 0


class _Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _FusedDll:
    def __init__(self):
        self.destroyed = []
        self.vf_context_create = _Function(self._create)
        self.vf_context_destroy = _Function(self._destroy)
        self.vf_context_stats = _Function(self._stats)
        self.vf_preprocess_401_2_u8 = _Function(lambda *_args: 0)

    @staticmethod
    def _create(output):
        output._obj.value = 1234
        return 0

    def _destroy(self, context):
        self.destroyed.append(context.value if hasattr(context, "value") else int(context))
        return 0

    @staticmethod
    def _stats(_context, reserved_bytes, allocation_count):
        reserved_bytes._obj.value = 4096
        allocation_count._obj.value = 7
        return 0


class _FusedRuntimeStub:
    available = True
    supports_fused_401_2 = True

    def __init__(self):
        self.calls = 0

    def preprocess_401_2(self, image, *_args):
        self.calls += 1
        return np.zeros(image.shape[:2], dtype=np.uint8)


class _FailingFusedRuntimeStub:
    available = True
    supports_fused_401_2 = True

    @staticmethod
    def preprocess_401_2(*_args):
        raise RuntimeError("injected fused failure")


class _PrimitiveRuntimeStub:
    supports_fused_401_2 = False

    def __init__(self):
        self.calls = []

    def bgr_to_gray(self, image):
        self.calls.append("gray")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, image, kernel_size):
        self.calls.append("gaussian")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def adaptive_threshold(self, image, block_size, c, max_value, invert):
        self.calls.append("adaptive_mean")
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type, block_size, c
        )


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

        self.assertFalse(runtime.supports_fused_401_2)
        runtime.bgr_to_gray(image)
        metrics = runtime.performance_stats()

        self.assertEqual(metrics["call_count"], 1)
        self.assertEqual(metrics["estimated_round_trips"], 1)
        self.assertEqual(metrics["host_to_device_bytes"], image.nbytes)
        self.assertEqual(metrics["device_to_host_bytes"], 6)
        self.assertEqual(metrics["functions"]["vf_bgr_to_gray_u8"]["calls"], 1)

    def test_optional_context_enables_fused_call_and_is_destroyed(self):
        runtime = GpuRuntime(enabled=False)
        dll = _FusedDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()

        self.assertTrue(runtime.supports_fused_401_2)
        output = runtime.preprocess_401_2(np.zeros((4, 5, 3), dtype=np.uint8), 3, 3, -2.0, 255)
        self.assertEqual(output.shape, (4, 5))
        metrics = runtime.performance_stats()
        self.assertEqual(metrics["functions"]["vf_preprocess_401_2_u8"]["calls"], 1)
        self.assertEqual(metrics["persistent_context"]["reserved_bytes"], 4096)
        self.assertEqual(metrics["persistent_context"]["allocation_count"], 7)

        runtime.close()
        self.assertFalse(runtime.supports_fused_401_2)
        self.assertEqual(dll.destroyed, [1234])


class DetectorFusedRoutingTests(unittest.TestCase):
    def test_detector_401_2_uses_fused_preprocessing_without_changing_cpu_contract(self):
        runtime = _FusedRuntimeStub()
        detector = Detector401_2(use_gpu=True, gpu_runtime=runtime)
        image = np.zeros((16, 20, 3), dtype=np.uint8)

        processed = detector.preprocess(image)
        binary = detector._make_binary(processed)

        self.assertEqual(processed.shape, image.shape)
        self.assertEqual(binary.shape, image.shape[:2])
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(detector.last_preprocess_capability["route"], "fused")

    def test_detector_401_2_fused_failure_restarts_entire_detector_on_cpu(self):
        detector = Detector401_2(
            use_gpu=True,
            gpu_runtime=_FailingFusedRuntimeStub(),
            params={"blur_size": 3, "adaptive_block_size": 3, "roi_inset_px": 0},
        )
        image = np.zeros((16, 20, 3), dtype=np.uint8)

        result = detector.run(image)

        self.assertFalse(result["execution"]["gpu_active"])
        self.assertEqual(result["execution"]["backend"], "cpu")
        self.assertIn("injected fused failure", result["execution"]["fallback_reason"])
        capability = result["execution"]["preprocess_capability"]
        self.assertEqual(capability["route"], "fallback")
        self.assertEqual(capability["selected_backend"], "cpu")
        self.assertIn("injected fused failure", capability["reason"])

    def test_detector_401_2_cpu_execution_reports_cpu_route(self):
        detector = Detector401_2(params={"blur_size": 3, "adaptive_block_size": 3})

        result = detector.run(np.zeros((16, 20, 3), dtype=np.uint8))

        capability = result["execution"]["preprocess_capability"]
        self.assertEqual(capability["route"], "cpu")
        self.assertEqual(capability["requested_backend"], "cpu")


class PreprocessPlanTests(unittest.TestCase):
    @staticmethod
    def _plan():
        return PreprocessPlan(
            name="shared_threshold_plan",
            operations=(Gray(), Gaussian(3), AdaptiveMean(3, -2.0, 255, True)),
        )

    def test_cpu_executor_matches_direct_opencv_pipeline(self):
        image = np.random.default_rng(17).integers(0, 256, size=(31, 37, 3), dtype=np.uint8)
        expected_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        expected_blur = cv2.GaussianBlur(expected_gray, (3, 3), 0)
        expected = cv2.adaptiveThreshold(
            expected_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, -2.0
        )

        actual = CpuPreprocessExecutor().execute(image, self._plan())

        np.testing.assert_array_equal(actual, expected)

    def test_cuda_executor_uses_reusable_primitives_when_fused_export_is_missing(self):
        image = np.random.default_rng(18).integers(0, 256, size=(21, 25, 3), dtype=np.uint8)
        runtime = _PrimitiveRuntimeStub()

        actual = CudaPreprocessExecutor(runtime).execute(image, self._plan())
        expected = CpuPreprocessExecutor().execute(image, self._plan())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(runtime.calls, ["gray", "gaussian", "adaptive_mean"])
        report = CudaPreprocessExecutor(runtime).capability_report(self._plan(), image).to_dict()
        self.assertEqual(report["route"], "primitive")
        self.assertEqual(report["selected_backend"], "cuda")

    def test_cuda_executor_rejects_non_equivalent_area_resize(self):
        plan = PreprocessPlan(operations=(Gray(), Resize(10, 10, "area")))

        with self.assertRaisesRegex(UnsupportedPreprocessPlan, "cannot preserve area"):
            CudaPreprocessExecutor(_PrimitiveRuntimeStub()).execute(
                np.zeros((20, 20, 3), dtype=np.uint8), plan
            )
        report = CudaPreprocessExecutor(_PrimitiveRuntimeStub()).capability_report(plan).to_dict()
        self.assertEqual(report["route"], "fallback")
        self.assertEqual(report["selected_backend"], "cpu")
        self.assertIn("Resize(area)", report["reason"])

    def test_plan_cache_reuses_shape_and_signature_with_lru_bound(self):
        cache = PreprocessPlanCache(max_entries=2)
        builds = []

        def build(name):
            builds.append(name)
            return PreprocessPlan((Gray(),), name=name)

        image = np.zeros((8, 9, 3), dtype=np.uint8)
        first = cache.get_or_create(image, ("gray", 1), lambda: build("first"))
        reused = cache.get_or_create(image, ("gray", 1), lambda: build("unused"))
        different_shape = cache.get_or_create(image[:7], ("gray", 1), lambda: build("shape"))
        different_dtype = cache.get_or_create(
            image.astype(np.float32), ("gray", 1), lambda: build("dtype")
        )
        different_params = cache.get_or_create(image, ("gray", 2), lambda: build("params"))

        self.assertIs(first, reused)
        self.assertIsNot(first, different_shape)
        self.assertIsNot(first, different_dtype)
        self.assertIsNot(first, different_params)
        self.assertEqual(builds, ["first", "shape", "dtype", "params"])
        self.assertEqual(cache.size, 2)

    def test_detector_401_2_caches_plan_by_shape_and_params(self):
        class CapturingExecutor:
            def __init__(self):
                self.plans = []

            @staticmethod
            def capability_report(plan):
                return CpuPreprocessExecutor.capability_report(plan)

            def execute(self, image, plan):
                self.plans.append(plan)
                return np.zeros(image.shape[:2], dtype=np.uint8)

        detector = Detector401_2(
            params={"blur_size": 4, "adaptive_block_size": 6, "adaptive_c": -2.0}
        )
        executor = CapturingExecutor()
        detector._cpu_preprocess_executor = executor
        image = np.zeros((64, 64), dtype=np.uint8)

        detector._make_binary(image)
        detector._make_binary(image.copy())
        detector._make_binary(np.zeros((65, 64), dtype=np.uint8))
        detector.params["adaptive_c"] = -3.0
        detector._make_binary(image)

        self.assertIs(executor.plans[0], executor.plans[1])
        self.assertIsNot(executor.plans[0], executor.plans[2])
        self.assertIsNot(executor.plans[0], executor.plans[3])
        self.assertEqual(executor.plans[0].operations[1], Gaussian(5))
        self.assertEqual(executor.plans[0].operations[2].block_size, 7)
        self.assertEqual(executor.plans[0].operations[2].c, -2.0)
        self.assertEqual(executor.plans[3].operations[2].c, -3.0)
        self.assertEqual(detector.preprocess_plan_cache_size, 3)

    def test_plan_signature_and_tensor_spec_are_deterministic(self):
        operations = (
            Gray(),
            Resize(10, 8, "area"),
            Gaussian(3),
            Threshold(120, 255, True),
            Morphology("open", 3, 1),
        )
        first = PreprocessPlan(operations, name="first")
        second = PreprocessPlan(operations, name="display_name_does_not_change_semantics")
        changed = PreprocessPlan(operations[:-2] + (Threshold(121, 255, True), operations[-1]))

        self.assertEqual(first.signature, second.signature)
        self.assertNotEqual(first.signature, changed.signature)
        spec = first.validate_input(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(spec.shape, (8, 10))
        self.assertEqual(spec.dtype, "uint8")
        self.assertEqual(spec.channels, 1)
        cpu_report = CpuPreprocessExecutor.capability_report(first).to_dict()
        self.assertEqual(cpu_report["route"], "cpu")
        self.assertEqual(cpu_report["plan_signature"], first.signature)

    def test_invalid_operator_parameters_are_rejected_at_plan_creation(self):
        invalid_operators = (
            Resize(0, 10),
            Gaussian(4),
            Threshold(-1),
            AdaptiveMean(2, 0.0),
            AdaptiveMean(3, float("inf")),
            Morphology("unknown", 3, 1),
            Morphology("open", 0, 1),
            Morphology("open", 3, -1),
        )
        for operator in invalid_operators:
            with self.subTest(operator=operator):
                with self.assertRaises(InvalidPreprocessPlan):
                    PreprocessPlan((operator,))

    def test_invalid_input_dtype_channel_shape_and_order_are_rejected(self):
        gray_plan = PreprocessPlan((Gray(),))
        invalid_inputs = (
            [1, 2, 3],
            np.zeros((4, 5), dtype=np.float32),
            np.zeros((4,), dtype=np.uint8),
            np.zeros((0, 5), dtype=np.uint8),
            np.zeros((4, 5, 1), dtype=np.uint8),
            np.zeros((4, 5, 4), dtype=np.uint8),
        )
        for image in invalid_inputs:
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(InvalidPreprocessPlan):
                    gray_plan.validate_input(image)

        with self.assertRaisesRegex(InvalidPreprocessPlan, "requires single-channel"):
            PreprocessPlan((Threshold(127),)).validate_input(
                np.zeros((4, 5, 3), dtype=np.uint8)
            )

    def test_executor_rejects_wrong_output_shape_or_dtype(self):
        class BadDtypeRuntime:
            supports_fused_401_2 = False

            @staticmethod
            def bgr_to_gray(image):
                return np.zeros(image.shape[:2], dtype=np.float32)

        class BadShapeRuntime:
            supports_fused_401_2 = False

            @staticmethod
            def bgr_to_gray(image):
                return np.zeros((image.shape[0], image.shape[1] + 1), dtype=np.uint8)

        with self.assertRaisesRegex(InvalidPreprocessPlan, "output dtype"):
            CudaPreprocessExecutor(BadDtypeRuntime()).execute(
                np.zeros((4, 5, 3), dtype=np.uint8), PreprocessPlan((Gray(),))
            )
        with self.assertRaisesRegex(InvalidPreprocessPlan, "output shape"):
            CudaPreprocessExecutor(BadShapeRuntime()).execute(
                np.zeros((4, 5, 3), dtype=np.uint8), PreprocessPlan((Gray(),))
            )

    def test_cpu_dag_validates_topology_and_returns_named_outputs(self):
        plan = PreprocessDagPlan(
            nodes=(
                PreprocessDagNode("gray", "root", Gray()),
                PreprocessDagNode("dark", "gray", Threshold(127, invert=True)),
                PreprocessDagNode("light", "gray", Threshold(127, invert=False)),
            ),
            outputs=("dark", "light"),
        )
        image = np.zeros((4, 5, 3), dtype=np.uint8)

        outputs = CpuPreprocessDagExecutor().execute(image, plan)

        self.assertEqual(tuple(outputs), ("dark", "light"))
        self.assertTrue(np.all(outputs["dark"] == 255))
        self.assertTrue(np.all(outputs["light"] == 0))
        with self.assertRaisesRegex(InvalidPreprocessPlan, "not available"):
            PreprocessDagPlan(
                nodes=(PreprocessDagNode("late", "missing", Gray()),),
                outputs=("late",),
            )


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

    def test_missing_gpu_without_cpu_fallback_fails_explicitly(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_strict_gpu_") as temporary:
            root = Path(temporary)
            recipe = self._recipe()
            recipe["gpu"]["tiling"] = True
            recipe["gpu"]["fallback_to_cpu"] = False
            recipe["gpu"]["dll_path"] = str(root / "definitely_missing.dll")
            recipe_path = root / "strict_gpu.yaml"
            recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(GpuRuntimeError, "CUDA DLL not found"):
                AOIPipeline(recipe_path, root / "output").run(root / "image_is_not_read.png")


if __name__ == "__main__":
    unittest.main()
