from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector305(BaseDetector):
    detector_id = "305"
    detector_name = "brightness_uniformity_detector"
    display_name = "Brightness / uniformity detector"
    default_params = {
        "mean_min": 20,
        "mean_max": 235,
        "std_max": 60,
        "percentile_low": 1,
        "percentile_high": 99,
        "cell_size": 128,
    }

    def preprocess(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def detect(self, image) -> list[dict]:
        cell_size = max(16, int(self.params.get("cell_size", 128)))
        defects = []
        for y in range(0, image.shape[0], cell_size):
            for x in range(0, image.shape[1], cell_size):
                cell = image[y : min(y + cell_size, image.shape[0]), x : min(x + cell_size, image.shape[1])]
                defect = self._inspect_cell(cell, x, y)
                if defect:
                    defects.append(defect)

        defects.sort(key=lambda item: item["confidence"], reverse=True)
        return defects

    def _inspect_cell(self, cell, x: int, y: int) -> dict | None:
        mean = float(np.mean(cell))
        std = float(np.std(cell))
        p_low = float(np.percentile(cell, float(self.params.get("percentile_low", 1))))
        p_high = float(np.percentile(cell, float(self.params.get("percentile_high", 99))))

        mean_min = float(self.params.get("mean_min", 20))
        mean_max = float(self.params.get("mean_max", 235))
        std_max = float(self.params.get("std_max", 60))

        reasons = []
        severity = 0.0
        if mean < mean_min:
            reasons.append("mean_below_min")
            severity = max(severity, (mean_min - mean) / max(mean_min, 1.0))
        if mean > mean_max:
            reasons.append("mean_above_max")
            severity = max(severity, (mean - mean_max) / max(255.0 - mean_max, 1.0))
        if std > std_max:
            reasons.append("std_above_max")
            severity = max(severity, (std - std_max) / max(std_max, 1.0))

        if not reasons:
            return None

        height, width = cell.shape[:2]
        return {
            "type": "brightness_uniformity",
            "bbox_local": [int(x), int(y), int(width), int(height)],
            "area": float(width * height),
            "confidence": float(np.round(min(1.0, severity), 4)),
            "metadata": {
                "mean": float(np.round(mean, 4)),
                "std": float(np.round(std, 4)),
                "percentile_low": float(np.round(p_low, 4)),
                "percentile_high": float(np.round(p_high, 4)),
                "reasons": reasons,
            },
        }
