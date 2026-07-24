from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from core.ai_runtime import YoloXModelRegistry
from gpu.validate_yolox_acceptance import (
    AcceptanceManifestError,
    evaluate_predictions,
    load_acceptance_manifest,
    run_validation,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "yolox"
REFERENCE_RECIPE = ROOT / "recipes" / "examples" / "YOLOX_TINY_REFERENCE_AOI_01.yaml"


def _write_png(path: Path) -> None:
    success, encoded = cv2.imencode(
        ".png", np.zeros((32, 32, 3), dtype=np.uint8)
    )
    if not success:
        raise RuntimeError("Cannot encode fixture")
    encoded.tofile(str(path))


class YoloXAcceptanceTests(unittest.TestCase):
    def test_metrics_include_map_false_kill_miss_and_confusion(self):
        cases = [
            {
                "id": "positive",
                "ground_truth": [{"class_id": 0, "bbox": [0, 0, 10, 10]}],
                "predictions": [
                    {"class_id": 0, "bbox": [0, 0, 10, 10], "confidence": 0.9}
                ],
            },
            {
                "id": "negative",
                "ground_truth": [],
                "predictions": [
                    {"class_id": 1, "bbox": [5, 5, 4, 4], "confidence": 0.8}
                ],
            },
        ]

        metrics = evaluate_predictions(
            cases, class_names=("scratch", "stain"), iou_threshold=0.5
        )

        self.assertEqual(metrics["counts"], {
            "tp": 1, "fp": 1, "fn": 0, "ground_truth": 1, "predictions": 2
        })
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["map50"], 1.0)
        self.assertEqual(metrics["false_kill_rate"], 1.0)
        self.assertEqual(metrics["miss_rate"], 0.0)
        self.assertEqual(
            metrics["confusion_matrix"]["values"],
            [[1, 0, 0], [0, 0, 0], [0, 1, 0]],
        )

    def test_manifest_rejects_test_model_without_explicit_test_override(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_manifest_") as temporary:
            root = Path(temporary)
            positive = root / "positive.png"
            negative = root / "negative.png"
            _write_png(positive)
            _write_png(negative)
            manifest = root / "acceptance.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "model_id": "yolox_tiny_fixture",
                        "recipe": str(REFERENCE_RECIPE),
                        "thresholds": {
                            "min_precision": 0.5,
                            "min_recall": 1.0,
                            "min_map50": 0.5,
                            "max_false_kill_rate": 1.0,
                            "max_miss_rate": 0.0,
                        },
                        "cases": [
                            {
                                "id": "positive",
                                "image": str(positive),
                                "ground_truth": [
                                    {"class_id": 0, "bbox_xywh": [0, 0, 16, 16]},
                                    {"class_id": 1, "bbox_xywh": [12, 12, 8, 8]},
                                ],
                            },
                            {
                                "id": "negative",
                                "image": str(negative),
                                "ground_truth": [],
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AcceptanceManifestError, "test_only"):
                load_acceptance_manifest(
                    manifest, registry=YoloXModelRegistry(MODEL_ROOT)
                )

            report = run_validation(manifest, allow_test_model=True)

        self.assertTrue(report["passed"])
        self.assertTrue(report["model"]["test_only"])
        self.assertEqual(report["session"]["load_count"], 1)
        self.assertEqual(report["session"]["session_count"], 1)
        self.assertEqual(report["metrics"]["counts"], {
            "tp": 2, "fp": 2, "fn": 0, "ground_truth": 2, "predictions": 4
        })

    def test_manifest_requires_positive_and_negative_cases(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_manifest_") as temporary:
            root = Path(temporary)
            image = root / "positive.png"
            _write_png(image)
            manifest = root / "acceptance.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "model_id": "yolox_tiny_fixture",
                        "recipe": str(REFERENCE_RECIPE),
                        "thresholds": {
                            "min_precision": 0,
                            "min_recall": 0,
                            "min_map50": 0,
                            "max_false_kill_rate": 1,
                            "max_miss_rate": 1,
                        },
                        "cases": [
                            {
                                "id": "positive",
                                "image": str(image),
                                "ground_truth": [
                                    {"class_id": 0, "bbox_xywh": [0, 0, 4, 4]}
                                ],
                            }
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                AcceptanceManifestError, "positive and one negative"
            ):
                load_acceptance_manifest(
                    manifest,
                    registry=YoloXModelRegistry(MODEL_ROOT),
                    allow_test_model=True,
                )


if __name__ == "__main__":
    unittest.main()
