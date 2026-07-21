from __future__ import annotations

import cv2
import numpy as np

from core.preprocess_plan import AdaptiveMean, Gaussian, Gray, PreprocessPlan
from core.parameter_schema import specs_from_defaults
from detectors.base_detector import BaseDetector


class Detector401_2(BaseDetector):
    detector_id = "401-2"
    detector_name = "adaptive_white_ratio_detector"
    display_name = "401-2 adaptive white ratio detector"
    default_params = {
        "max_value": 255,
        "blur_size": 25,
        "adaptive_block_size": 35,
        "adaptive_c": -2.0,
        "roi_inset_px": 0,
        "white_pixel_ratio_threshold": 0.625,
    }
    PARAM_SPEC = specs_from_defaults(default_params, {
        "max_value": {"minimum": 1, "maximum": 255, "engineer_visible": False},
        "blur_size": {"minimum": 3, "odd": True, "engineer_visible": False},
        "adaptive_block_size": {"minimum": 3, "odd": True, "engineer_visible": False},
        "adaptive_c": {"engineer_visible": False},
        "roi_inset_px": {"minimum": 0},
        "white_pixel_ratio_threshold": {"minimum": 0.0, "maximum": 1.0, "engineer_visible": False},
    })

    def preprocess(self, image):
        return image if self.gpu_active else self.shared_gray(image)

    def detect(self, image) -> list[dict]:
        roi, offset_x, offset_y = self._roi_image(image)
        with self.measure_detection_stage("preprocess"):
            binary = self._make_binary(roi, offset_x, offset_y)
        with self.measure_detection_stage("white_ratio_analysis"):
            total_pixel_count = int(binary.size)
            if total_pixel_count <= 0:
                return []
            white_pixel_count = int(cv2.countNonZero(binary))
            white_pixel_ratio = white_pixel_count / float(total_pixel_count)

        ratio_threshold = float(self.params.get("white_pixel_ratio_threshold", 0.625))
        if white_pixel_ratio < ratio_threshold:
            return []

        height, width = binary.shape[:2]
        return [
            {
                "type": "401_2_white_pixel_ratio_ng",
                "bbox_local": [int(offset_x), int(offset_y), int(width), int(height)],
                "area": float(total_pixel_count),
                "confidence": float(np.round(min(1.0, white_pixel_ratio), 4)),
                "metadata": {
                    "shape": "tile_roi",
                    "white_pixel_count": white_pixel_count,
                    "total_pixel_count": total_pixel_count,
                    "white_pixel_ratio": float(np.round(white_pixel_ratio, 6)),
                    "white_pixel_ratio_percent": float(np.round(white_pixel_ratio * 100.0, 3)),
                    "white_pixel_ratio_threshold": ratio_threshold,
                    "white_pixel_ratio_threshold_percent": float(np.round(ratio_threshold * 100.0, 3)),
                    "threshold_method": "adaptive_mean_inv",
                    "roi_inset_px": int(self.params.get("roi_inset_px", 0)),
                    "roi_offset_local": [int(offset_x), int(offset_y)],
                    "roi_size": [int(width), int(height)],
                    "blur_size": int(self.params.get("blur_size", 25)),
                    "adaptive_block_size": int(self.params.get("adaptive_block_size", 35)),
                    "adaptive_c": float(self.params.get("adaptive_c", -2.0)),
                },
            }
        ]

    def _roi_image(self, gray):
        inset = max(0, int(self.params.get("roi_inset_px", 0)))
        if inset <= 0:
            return gray, 0, 0

        height, width = gray.shape[:2]
        if width <= inset * 2 or height <= inset * 2:
            return gray, 0, 0

        return gray[inset : height - inset, inset : width - inset], inset, inset

    def _make_binary(self, gray, offset_x: int = 0, offset_y: int = 0):
        blur_size = self._odd_at_least(int(self.params.get("blur_size", 25)), 3)
        block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 35)), 3)
        adaptive_c = float(self.params.get("adaptive_c", -2.0))
        max_value = int(self.params.get("max_value", 255))
        signature = (
            "gray_gaussian_adaptive_mean",
            blur_size,
            block_size,
            adaptive_c,
            max_value,
            True,
        )
        plan = self.cached_preprocess_plan(
            gray,
            signature,
            lambda: PreprocessPlan(
                name="gray_gaussian_adaptive_mean",
                operations=(
                    Gray(),
                    Gaussian(blur_size),
                    AdaptiveMean(
                        block_size=block_size,
                        c=adaptive_c,
                        max_value=max_value,
                        invert=True,
                    ),
                ),
            ),
        )
        return self.execute_preprocess_plan(gray, plan, (offset_x, offset_y))

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
