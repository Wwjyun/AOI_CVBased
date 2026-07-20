from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from core.gpu_metrics import GpuPerformanceRecorder
from core.gpu_runtime import GpuRuntime
from core.preprocess_plan import AdaptiveMean, Gaussian, Gray, Morphology, PreprocessPlan
from core.tiler import Tiler
from gpu.profile_401_pipeline import _summary


class Detector401ProfilingTests(unittest.TestCase):
    def test_native_timings_accumulate_all_rois_and_peak_memory(self):
        recorder = GpuPerformanceRecorder()
        recorder.record_native(
            {"context_create_ms": 3.0, "morphology_ms": 7.0},
            kernel_launch_count=28,
            reserved_bytes=100,
        )
        recorder.record_native(
            {"context_create_ms": 3.0, "morphology_ms": 11.0},
            kernel_launch_count=28,
            reserved_bytes=160,
        )
        snapshot = recorder.snapshot()
        self.assertEqual(snapshot["native_cumulative_ms"]["context_create_ms"], 3.0)
        self.assertEqual(snapshot["native_cumulative_ms"]["morphology_ms"], 18.0)
        self.assertEqual(snapshot["kernel_launch_count"], 56)
        self.assertEqual(snapshot["peak_vram_bytes"], 160)

    def test_401_plan_launch_count_matches_current_native_implementation(self):
        plan = PreprocessPlan(
            (
                Gaussian(15),
                Morphology("open", 5, 10),
                Gray(),
                AdaptiveMean(29, 5, 255, True),
            ),
            name="401_profile_test",
        )
        self.assertEqual(GpuRuntime._plan_kernel_launch_count(plan, 3), 28)

    def test_anchor_grid_profile_preserves_coordinates(self):
        rng = np.random.default_rng(7)
        template = rng.integers(0, 256, (12, 10, 3), dtype=np.uint8)
        image = np.zeros((100, 140, 3), dtype=np.uint8)
        image[20:32, 30:40] = template
        with tempfile.TemporaryDirectory() as temp_name:
            template_path = Path(temp_name) / "anchor.png"
            encoded, payload = cv2.imencode(".png", template)
            self.assertTrue(encoded)
            payload.tofile(template_path)
            tiler = Tiler.from_config({
                "mode": "grid", "width": 16, "height": 14,
                "overlap_x": 0, "overlap_y": 0,
                "template_path": str(template_path),
                "search_x": 0, "search_y": 0, "search_w": 80, "search_h": 70,
                "offset_x": 5, "offset_y": 6, "rows": 2, "cols": 3,
                "roi_w": 16, "roi_h": 14, "gap_x": 2, "gap_y": 3,
                "match_threshold": 0.9,
            })
            tiles = list(tiler.iter_tiles(image))
        self.assertEqual(
            [(tile.x, tile.y, tile.width, tile.height) for tile in tiles],
            [
                (35, 26, 16, 14), (53, 26, 16, 14), (71, 26, 16, 14),
                (35, 43, 16, 14), (53, 43, 16, 14), (71, 43, 16, 14),
            ],
        )
        self.assertGreaterEqual(tiler.last_profile_ms["template_match_ms"], 0.0)
        self.assertGreaterEqual(tiler.last_profile_ms["roi_generation_ms"], 0.0)

    def test_profile_summary_uses_nearest_rank_p95(self):
        rows = []
        for value in range(1, 11):
            row = {field: None for field in (
                "template_match_ms", "roi_generation_ms", "context_initialization_ms",
                "buffer_allocation_ms", "h2d_ms", "roi_gather_ms", "gaussian_ms",
                "morphology_erode_ms", "morphology_dilate_ms", "morphology_total_ms",
                "grayscale_ms", "adaptive_mean_ms", "d2h_ms", "cuda_synchronize_ms",
                "cpu_find_contours_ms", "detector_postprocess_ms", "total_gpu_pipeline_ms",
                "total_detector_ms", "roi_count", "kernel_launch_count", "peak_vram_bytes",
            )}
            row["total_detector_ms"] = value
            rows.append(row)
        summary = _summary(rows)
        self.assertEqual(summary["total_detector_ms"]["median"], 5.5)
        self.assertEqual(summary["total_detector_ms"]["p95"], 10.0)
        self.assertIsNone(summary["morphology_erode_ms"])


if __name__ == "__main__":
    unittest.main()
