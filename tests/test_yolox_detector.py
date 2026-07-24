from __future__ import annotations

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from core.ai_runtime import (
    AiModelError,
    AiModelSessionManager,
    LetterboxTransform,
    YoloXModelRegistry,
    decode_yolox_output,
    prepare_yolox_input,
)
from core.detector_manager import DetectorManager
from core.pipeline import AOIPipeline
from core.recipe_manager import RecipeError, RecipeManager


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "yolox"
EXAMPLE_RECIPE = ROOT / "recipes" / "examples" / "YOLOX_TINY_REFERENCE_AOI_01.yaml"


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


if __name__ == "__main__":
    unittest.main()
