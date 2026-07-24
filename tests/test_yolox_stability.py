from __future__ import annotations

import unittest

from core.ai_runtime import AiModelError
from gpu.validate_yolox_stability import _parse_checkpoints, run_stability


class YoloXStabilityTests(unittest.TestCase):
    def test_checkpoint_parser_adds_final_iteration(self):
        self.assertEqual(_parse_checkpoints("2,5", 10), (2, 5, 10))
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _parse_checkpoints("11", 10)

    def test_cpu_stability_reuses_one_session_and_reports_memory(self):
        report = run_stability(
            model_id="yolox_tiny_fixture",
            backend="onnxruntime_cpu",
            image_path=None,
            warmup=2,
            iterations=20,
            checkpoints=(5, 10, 20),
            max_rss_growth_mb=64.0,
            allow_test_model=True,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["session"]["load_count"], 1)
        self.assertEqual(report["session"]["session_count"], 1)
        self.assertEqual(
            [item["iteration"] for item in report["checkpoints"]],
            [5, 10, 20],
        )
        self.assertEqual(
            report["session"]["sessions"][0]["inference_count"],
            22,
        )
        self.assertTrue(report["checks"]["deterministic_output"])
        self.assertTrue(report["checks"]["rss_growth_within_limit"])

    def test_test_only_model_requires_explicit_override(self):
        with self.assertRaisesRegex(AiModelError, "test_only"):
            run_stability(
                model_id="yolox_tiny_fixture",
                backend="onnxruntime_cpu",
                image_path=None,
                warmup=0,
                iterations=1,
                checkpoints=(1,),
                max_rss_growth_mb=64.0,
                allow_test_model=False,
            )


if __name__ == "__main__":
    unittest.main()
