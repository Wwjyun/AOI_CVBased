from __future__ import annotations

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np
import yaml

from core.ai_runtime import (
    AiBackendUnavailable,
    AiInferenceError,
    AiModelError,
    AiModelSessionManager,
    LetterboxTransform,
    YoloXModelRegistry,
    compare_yolox_backend_results,
    decode_yolox_output,
    prepare_yolox_input,
)
from core.detector_manager import DetectorManager
from core.pipeline import AOIPipeline
from core.recipe_manager import RecipeError, RecipeManager


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "yolox"
EXAMPLE_RECIPE = ROOT / "recipes" / "examples" / "YOLOX_TINY_REFERENCE_AOI_01.yaml"


class _FakeNode:
    def __init__(self, name):
        self.name = name


class _FakeSessionOptions:
    def __init__(self):
        self.graph_optimization_level = None
        self.entries = {}

    def add_session_config_entry(self, key, value):
        self.entries[key] = value


class _FakeOrtSession:
    def __init__(self, owner, providers, options):
        self.owner = owner
        self.providers = list(providers)
        self.options = options
        self.run_calls = 0

    def get_providers(self):
        return list(self.providers)

    def get_inputs(self):
        return [_FakeNode("images")]

    def get_outputs(self):
        return [_FakeNode("output")]

    def run(self, _output_names, _inputs):
        self.run_calls += 1
        if (
            self.providers[0] == "CUDAExecutionProvider"
            and self.owner.fail_cuda_inference
            and self.run_calls > 1
        ):
            raise RuntimeError("CUDA out of memory")
        return [self.owner.output.copy()]


class _FakeOrt:
    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"

    SessionOptions = _FakeSessionOptions

    def __init__(
        self,
        output,
        providers,
        *,
        fail_cuda_inference=False,
        fail_cuda_initialization=False,
    ):
        self.output = np.asarray(output, dtype=np.float32)
        self.providers = list(providers)
        self.fail_cuda_inference = fail_cuda_inference
        self.fail_cuda_initialization = fail_cuda_initialization
        self.sessions = []
        self.inference_session_calls = []

    def get_available_providers(self):
        return list(self.providers)

    def InferenceSession(self, _path, sess_options, providers):
        self.inference_session_calls.append(providers[0])
        if (
            providers[0] == "CUDAExecutionProvider"
            and self.fail_cuda_initialization
        ):
            raise RuntimeError("CUDA initialization failed")
        session = _FakeOrtSession(self, providers, sess_options)
        self.sessions.append(session)
        return session


def write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Cannot encode test image")
    encoded.tofile(str(path))


class YoloXRegistryAndSessionTests(unittest.TestCase):
    def test_registry_verifies_fixture_and_session_is_loaded_once(self):
        registry = YoloXModelRegistry(MODEL_ROOT)
        manifest = registry.get("yolox_tiny_fixture")
        manager = AiModelSessionManager(registry)

        first = manager.session_for(manifest)
        second = manager.session_for(manifest, backend="onnxruntime_cpu")

        self.assertIs(first, second)
        self.assertEqual(manager.load_count, 1)
        self.assertEqual(manager.session_count, 1)
        self.assertEqual(first.backend, "onnxruntime_cpu")
        self.assertEqual(first.device, "CPU")
        manager.close()
        self.assertEqual(manager.session_count, 0)

    def test_registry_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_registry_") as temporary:
            root = Path(temporary)
            (root / "fixture.onnx").write_bytes(
                (MODEL_ROOT / "yolox_tiny_fixture.onnx").read_bytes()
            )
            registry = (MODEL_ROOT / "registry.yaml").read_text(encoding="utf-8")
            registry = registry.replace(
                "38d2c79bf140c829ffef9fcd264bb5fb630bdc280a7a1a5ec27911888ada8188",
                "0" * 64,
            ).replace("yolox_tiny_fixture.onnx", "fixture.onnx")
            (root / "registry.yaml").write_text(registry, encoding="utf-8")
            with self.assertRaisesRegex(AiModelError, "SHA-256 驗證失敗"):
                YoloXModelRegistry(root)

    def test_cuda_and_cpu_sessions_have_distinct_cache_keys_and_no_ep_fallback(self):
        registry = YoloXModelRegistry(MODEL_ROOT)
        manifest = registry.get("yolox_tiny_fixture")
        fake_ort = _FakeOrt(
            np.zeros((1, 21, 7), dtype=np.float32),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        manager = AiModelSessionManager(
            registry,
            gpu_mode="cuda",
            fallback_to_cpu=False,
            ort_module=fake_ort,
        )

        cpu = manager.session_for(manifest, backend="onnxruntime_cpu")
        cuda = manager.session_for(manifest, backend="onnxruntime_cuda")

        self.assertIs(
            cuda, manager.session_for(manifest, backend="onnxruntime_cuda")
        )
        self.assertIsNot(cpu, cuda)
        self.assertEqual(manager.load_count, 2)
        self.assertEqual(cpu.device, "CPU")
        self.assertEqual(cuda.device, "CUDA")
        self.assertEqual(
            fake_ort.sessions[1].options.entries,
            {"session.disable_cpu_ep_fallback": "1"},
        )

    def test_missing_cuda_provider_falls_back_only_when_policy_allows_it(self):
        registry = YoloXModelRegistry(MODEL_ROOT)
        manifest = registry.get("yolox_tiny_fixture")
        fake_ort = _FakeOrt(
            np.zeros((1, 21, 7), dtype=np.float32),
            ["CPUExecutionProvider"],
        )
        fallback_manager = AiModelSessionManager(
            registry,
            gpu_mode="auto",
            fallback_to_cpu=True,
            ort_module=fake_ort,
        )

        selection = fallback_manager.select_session(
            manifest, backend="auto", prefer_gpu=True
        )

        self.assertEqual(selection.actual_backend, "onnxruntime_cpu")
        self.assertIn("CUDAExecutionProvider 不可用", selection.fallback_reason)
        strict_manager = AiModelSessionManager(
            registry,
            gpu_mode="cuda",
            fallback_to_cpu=False,
            ort_module=fake_ort,
        )
        with self.assertRaisesRegex(
            AiBackendUnavailable, "CUDAExecutionProvider 不可用"
        ):
            strict_manager.select_session(
                manifest, backend="auto", prefer_gpu=True
            )

    def test_cuda_initialization_failure_is_quarantined_after_cpu_fallback(self):
        registry = YoloXModelRegistry(MODEL_ROOT)
        manifest = registry.get("yolox_tiny_fixture")
        fake_ort = _FakeOrt(
            np.zeros((1, 21, 7), dtype=np.float32),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            fail_cuda_initialization=True,
        )
        manager = AiModelSessionManager(
            registry,
            gpu_mode="auto",
            fallback_to_cpu=True,
            ort_module=fake_ort,
        )

        first = manager.select_session(
            manifest, backend="auto", prefer_gpu=True
        )
        second = manager.select_session(
            manifest, backend="auto", prefer_gpu=True
        )

        self.assertEqual(first.actual_backend, "onnxruntime_cpu")
        self.assertEqual(second.actual_backend, "onnxruntime_cpu")
        self.assertIn("CUDA initialization failed", second.fallback_reason)
        self.assertEqual(
            fake_ort.inference_session_calls,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )


class YoloXPreprocessAndPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.manifest = YoloXModelRegistry(MODEL_ROOT).get("yolox_tiny_fixture")

    def test_letterbox_preserves_bgr_and_records_scale_and_padding(self):
        image = np.zeros((16, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 7
        image[:, :, 1] = 11
        image[:, :, 2] = 19

        tensor, transform = prepare_yolox_input(image, self.manifest)

        self.assertEqual(tensor.shape, (1, 3, 32, 32))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [7.0, 11.0, 19.0])
        self.assertEqual(tensor[0, :, 20, 0].tolist(), [114.0, 114.0, 114.0])
        self.assertEqual(transform.scale_x, 1.0)
        self.assertEqual(transform.scale_y, 1.0)
        self.assertEqual((transform.pad_x, transform.pad_y), (0, 0))

    def test_real_fixture_decodes_nms_coordinates_scores_and_order(self):
        manager = AiModelSessionManager(YoloXModelRegistry(MODEL_ROOT))
        tensor, transform = prepare_yolox_input(
            np.zeros((32, 32, 3), dtype=np.uint8), self.manifest
        )
        output = manager.session_for(self.manifest).infer(tensor)

        defects = self._decode(output, transform)

        self.assertEqual(
            [(item["type"], item["bbox_local"], item["confidence"]) for item in defects],
            [
                ("scratch", [0, 0, 16, 16], 0.81),
                ("stain", [12, 12, 8, 8], 0.76),
            ],
        )
        self.assertEqual([item["area"] for item in defects], [256.0, 64.0])
        self.assertEqual(defects[0]["metadata"]["class_id"], 0)
        self.assertEqual(defects[0]["metadata"]["model_id"], "yolox_tiny_fixture")
        self.assertEqual(len(defects[0]["metadata"]["model_sha256"]), 64)

    def test_target_class_area_max_count_and_shape_validation(self):
        manager = AiModelSessionManager(YoloXModelRegistry(MODEL_ROOT))
        tensor, transform = prepare_yolox_input(
            np.zeros((32, 32, 3), dtype=np.uint8), self.manifest
        )
        output = manager.session_for(self.manifest).infer(tensor)

        stain_only = self._decode(output, transform, target_class_ids="1")
        area_filtered = self._decode(output, transform, min_box_area_px=100.0)
        limited = self._decode(output, transform, max_detections=1)

        self.assertEqual([item["type"] for item in stain_only], ["stain"])
        self.assertEqual([item["type"] for item in area_filtered], ["scratch"])
        self.assertEqual([item["type"] for item in limited], ["scratch"])
        with self.assertRaisesRegex(AiModelError, "output shape mismatch"):
            self._decode(np.zeros((1, 1, 7), np.float32), transform)

    def test_nms_is_class_aware_and_uses_strict_greater_than_threshold(self):
        output = np.zeros((1, 21, 7), dtype=np.float32)
        ln2 = np.float32(np.log(2.0))
        output[0, 0] = [1.0, 1.0, ln2, ln2, 0.9, 0.9, 0.1]
        output[0, 1] = [0.0, 1.0, ln2, ln2, 0.8, 0.9, 0.1]
        output[0, 2] = [-1.0, 1.0, ln2, ln2, 0.85, 0.1, 0.9]
        transform = LetterboxTransform(32, 32, 32, 32, 1.0, 1.0, 0, 0)

        class_aware = self._decode(output, transform, nms_iou_threshold=0.999)
        class_agnostic = self._decode(
            output,
            transform,
            nms_iou_threshold=0.999,
            class_agnostic_nms=True,
        )
        threshold_equal = self._decode(
            output,
            transform,
            nms_iou_threshold=1.0,
            class_agnostic_nms=True,
        )

        self.assertEqual([item["type"] for item in class_aware], ["scratch", "stain"])
        self.assertEqual([item["type"] for item in class_agnostic], ["scratch"])
        self.assertEqual(
            [item["type"] for item in threshold_equal],
            ["scratch", "stain", "scratch"],
        )

    def test_backend_comparison_enforces_raw_bbox_class_and_confidence_tolerances(self):
        reference_output = np.zeros((1, 21, 7), dtype=np.float32)
        candidate_output = reference_output.copy()
        candidate_output[0, 0, 0] = 5e-6
        reference_defects = [
            {
                "bbox_local": [10, 10, 20, 20],
                "confidence": 0.8,
                "metadata": {"class_id": 0},
            }
        ]
        candidate_defects = [
            {
                "bbox_local": [11, 10, 20, 20],
                "confidence": 0.80005,
                "metadata": {"class_id": 0},
            }
        ]

        accepted = compare_yolox_backend_results(
            reference_output,
            candidate_output,
            reference_defects,
            candidate_defects,
        )
        rejected = compare_yolox_backend_results(
            reference_output,
            candidate_output,
            reference_defects,
            [
                {
                    "bbox_local": [12, 10, 20, 20],
                    "confidence": 0.8002,
                    "metadata": {"class_id": 1},
                }
            ],
        )

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["detections"]["same_classes_and_order"])

    def _decode(self, output, transform, **overrides):
        params = {
            "confidence_threshold": 0.25,
            "nms_iou_threshold": 0.45,
            "target_class_ids": "",
            "max_detections": 300,
            "min_box_area_px": 0.0,
            "class_agnostic_nms": False,
        }
        params.update(overrides)
        return decode_yolox_output(output, self.manifest, transform, **params)


class DetectorYoloXIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            "model_id": "yolox_tiny_fixture",
            "confidence_threshold": 0.25,
            "nms_iou_threshold": 0.45,
            "target_class_ids": "",
            "max_detections": 300,
            "min_box_area_px": 0.0,
            "inference_backend": "onnxruntime_cpu",
            "precision": "fp32",
            "class_agnostic_nms": False,
        }

    def test_detector_manager_registration_result_contract_and_session_reuse(self):
        sessions = AiModelSessionManager(YoloXModelRegistry(MODEL_ROOT))
        manager = DetectorManager(ai_session_manager=sessions)
        detector = manager.create("yolox", params=self.params)
        image = np.zeros((32, 32, 3), dtype=np.uint8)

        first = detector.run(image)
        second = detector.run(image)

        self.assertIn("yolox", manager.definitions())
        self.assertFalse(first["pass"])
        self.assertEqual(first["score"], 0.81)
        self.assertEqual(len(first["defects"]), 2)
        self.assertEqual(first["execution"]["backend"], "onnxruntime_cpu")
        self.assertEqual(first["execution"]["ai"]["device"], "CPU")
        self.assertEqual(second["execution"]["ai"]["session_inference_count"], 2)
        self.assertEqual(sessions.load_count, 1)
        self.assertIn("dl_preprocess", first["execution"]["performance"]["stages_sec"])
        self.assertIn("postprocess", first["execution"]["performance"]["stages_sec"])

    def test_high_confidence_threshold_is_pass(self):
        params = {**self.params, "confidence_threshold": 0.99}
        result = DetectorManager().create("yolox", params=params).run(
            np.zeros((32, 32, 3), dtype=np.uint8)
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["defects"], [])

    def test_cuda_inference_failure_restarts_detector_on_cpu(self):
        registry = YoloXModelRegistry(MODEL_ROOT)
        manifest = registry.get("yolox_tiny_fixture")
        reference = AiModelSessionManager(registry)
        tensor, _ = prepare_yolox_input(
            np.zeros((32, 32, 3), dtype=np.uint8), manifest
        )
        output = reference.session_for(manifest).infer(tensor)
        fake_ort = _FakeOrt(
            output,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            fail_cuda_inference=True,
        )
        sessions = AiModelSessionManager(
            registry,
            gpu_mode="auto",
            fallback_to_cpu=True,
            ort_module=fake_ort,
        )
        detector = DetectorManager(ai_session_manager=sessions).create(
            "yolox", params={**self.params, "inference_backend": "auto"}, use_gpu=True
        )

        result = detector.run(np.zeros((32, 32, 3), dtype=np.uint8))
        second = detector.run(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertEqual(len(result["defects"]), 2)
        self.assertEqual(result["execution"]["ai"]["actual_backend"], "onnxruntime_cpu")
        self.assertIn("CUDA out of memory", result["execution"]["ai"]["fallback_reason"])
        self.assertIn("CUDA out of memory", second["execution"]["ai"]["fallback_reason"])
        self.assertFalse(result["execution"]["gpu_active"])
        self.assertEqual([session.run_calls for session in fake_ort.sessions], [2, 3])

    def test_strict_cuda_inference_failure_does_not_fallback(self):
        registry = YoloXModelRegistry(MODEL_ROOT)
        manifest = registry.get("yolox_tiny_fixture")
        output = np.zeros((1, 21, 7), dtype=np.float32)
        fake_ort = _FakeOrt(
            output,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            fail_cuda_inference=True,
        )
        sessions = AiModelSessionManager(
            registry,
            gpu_mode="cuda",
            fallback_to_cpu=False,
            ort_module=fake_ort,
        )
        detector = DetectorManager(ai_session_manager=sessions).create(
            "yolox", params={**self.params, "inference_backend": "auto"}, use_gpu=True
        )

        with self.assertRaisesRegex(AiInferenceError, "CUDA out of memory"):
            detector.run(np.zeros((32, 32, 3), dtype=np.uint8))

    def test_recipe_round_trip_rejects_unknown_model_and_pipeline_runs(self):
        recipe = RecipeManager().load(EXAMPLE_RECIPE)
        self.assertEqual(
            recipe["detectors"]["yolox"]["params"]["model_id"],
            "yolox_tiny_fixture",
        )
        invalid = deepcopy(recipe)
        invalid["detectors"]["yolox"]["params"]["model_id"] = "missing_model"
        with self.assertRaisesRegex(RecipeError, "找不到 YOLOX model_id"):
            RecipeManager().validate(invalid)

        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_pipeline_") as temporary:
            root = Path(temporary)
            image_path = root / "input.png"
            write_png(image_path, np.zeros((32, 32, 3), dtype=np.uint8))
            result = AOIPipeline(EXAMPLE_RECIPE, root / "output").run(image_path)

        self.assertEqual(result["final_result"], "NG")
        self.assertEqual(result["summary"]["defect_count"], 2)
        self.assertEqual(
            result["tiles"][0]["detectors"][0]["defects"][0]["bbox_global"],
            [0, 0, 16, 16],
        )
        self.assertEqual(
            result["tiles"][0]["detectors"][0]["execution"]["ai"]["actual_backend"],
            "onnxruntime_cpu",
        )

    def test_pipeline_auto_fallback_reports_ort_provider_without_loading_cuda_dll(self):
        recipe = RecipeManager().load(EXAMPLE_RECIPE)
        recipe["gpu"]["mode"] = "auto"
        recipe["gpu"]["fallback_to_cpu"] = True
        recipe["detectors"]["yolox"]["use_gpu"] = True
        recipe["detectors"]["yolox"]["params"]["inference_backend"] = "auto"
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_fallback_") as temporary:
            root = Path(temporary)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            image_path = root / "input.png"
            write_png(image_path, np.zeros((32, 32, 3), dtype=np.uint8))

            result = AOIPipeline(recipe_path, root / "output").run(image_path)

        detector_execution = result["tiles"][0]["detectors"][0]["execution"]
        gpu_status = result["execution"]["gpu"]["detectors"]["yolox"]
        self.assertEqual(detector_execution["ai"]["actual_backend"], "onnxruntime_cpu")
        self.assertIn(
            "CUDAExecutionProvider 不可用",
            detector_execution["ai"]["fallback_reason"],
        )
        self.assertTrue(gpu_status["requested"])
        self.assertFalse(gpu_status["active"])
        self.assertEqual(gpu_status["backend"], "onnxruntime_cpu")
        self.assertEqual(result["execution"]["gpu"]["metrics"]["call_count"], 0)

    def test_pipeline_fake_cuda_reports_active_backend_and_device(self):
        registry = YoloXModelRegistry(MODEL_ROOT)
        manifest = registry.get("yolox_tiny_fixture")
        tensor, _ = prepare_yolox_input(
            np.zeros((32, 32, 3), dtype=np.uint8), manifest
        )
        output = AiModelSessionManager(registry).session_for(manifest).infer(tensor)
        fake_ort = _FakeOrt(
            output,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        sessions = AiModelSessionManager(
            registry,
            gpu_mode="auto",
            fallback_to_cpu=True,
            ort_module=fake_ort,
        )
        recipe = RecipeManager().load(EXAMPLE_RECIPE)
        recipe["gpu"]["mode"] = "auto"
        recipe["detectors"]["yolox"]["use_gpu"] = True
        recipe["detectors"]["yolox"]["params"]["inference_backend"] = "auto"
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_cuda_") as temporary:
            root = Path(temporary)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            image_path = root / "input.png"
            write_png(image_path, np.zeros((32, 32, 3), dtype=np.uint8))
            pipeline = AOIPipeline(recipe_path, root / "output")
            pipeline.detector_manager = DetectorManager(ai_session_manager=sessions)

            result = pipeline.run(image_path)

        gpu_status = result["execution"]["gpu"]["detectors"]["yolox"]
        self.assertTrue(gpu_status["requested"])
        self.assertTrue(gpu_status["active"])
        self.assertEqual(gpu_status["backend"], "onnxruntime_cuda")
        self.assertEqual(gpu_status["device_name"], "CUDA")
        self.assertEqual(result["execution"]["gpu"]["metrics"]["call_count"], 0)

    def test_pipeline_strict_cuda_rejects_missing_ort_provider(self):
        recipe = RecipeManager().load(EXAMPLE_RECIPE)
        recipe["gpu"]["mode"] = "cuda"
        recipe["gpu"]["fallback_to_cpu"] = False
        recipe["detectors"]["yolox"]["use_gpu"] = True
        recipe["detectors"]["yolox"]["params"]["inference_backend"] = "auto"
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_strict_") as temporary:
            root = Path(temporary)
            recipe_path = root / "recipe.yaml"
            recipe_path.write_text(
                yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            image_path = root / "input.png"
            write_png(image_path, np.zeros((32, 32, 3), dtype=np.uint8))

            with self.assertRaisesRegex(
                AiBackendUnavailable, "CUDAExecutionProvider 不可用"
            ):
                AOIPipeline(recipe_path, root / "output").run(image_path)


if __name__ == "__main__":
    unittest.main()
