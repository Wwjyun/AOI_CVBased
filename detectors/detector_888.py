from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector888(BaseDetector):
    detector_id = "888"
    detector_name = "texture_detector"
    display_name = "Texture / blur detector"
    default_params = {
        "laplacian_var_min": 20,
        "local_std_min": 3,
        "local_std_max": 80,
        "block_size": 128,
    }

    def preprocess(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def detect(self, image) -> list[dict]:
        block_size = max(16, int(self.params.get("block_size", 128)))
        defects = []
        for y in range(0, image.shape[0], block_size):
            for x in range(0, image.shape[1], block_size):
                block = image[y : min(y + block_size, image.shape[0]), x : min(x + block_size, image.shape[1])]
                defect = self._inspect_block(block, x, y)
                if defect:
                    defects.append(defect)

        defects.sort(key=lambda item: item["confidence"], reverse=True)
        return defects

    def _inspect_block(self, block, x: int, y: int) -> dict | None:
        laplacian_var = float(cv2.Laplacian(block, cv2.CV_64F).var())
        local_std = float(np.std(block))
        laplacian_var_min = float(self.params.get("laplacian_var_min", 20))
        local_std_min = float(self.params.get("local_std_min", 3))
        local_std_max = float(self.params.get("local_std_max", 80))

        reasons = []
        severity = 0.0
        if laplacian_var < laplacian_var_min:
            reasons.append("laplacian_var_below_min")
            severity = max(severity, (laplacian_var_min - laplacian_var) / max(laplacian_var_min, 1.0))
        if local_std < local_std_min:
            reasons.append("local_std_below_min")
            severity = max(severity, (local_std_min - local_std) / max(local_std_min, 1.0))
        if local_std > local_std_max:
            reasons.append("local_std_above_max")
            severity = max(severity, (local_std - local_std_max) / max(local_std_max, 1.0))

        if not reasons:
            return None

        height, width = block.shape[:2]
        return {
            "type": "texture_anomaly",
            "bbox_local": [int(x), int(y), int(width), int(height)],
            "area": float(width * height),
            "confidence": float(np.round(min(1.0, severity), 4)),
            "metadata": {
                "laplacian_var": float(np.round(laplacian_var, 4)),
                "local_std": float(np.round(local_std, 4)),
                "reasons": reasons,
            },
        }
