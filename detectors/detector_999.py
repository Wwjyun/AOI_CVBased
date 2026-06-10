from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector999(BaseDetector):
    detector_id = "999"
    detector_name = "blob_detector"
    display_name = "Dark / bright blob detector"
    default_params = {
        "threshold": 45,
        "min_area": 20,
        "max_area": 5000,
        "blur_size": 3,
        "invert": False,
        "clahe_enabled": True,
    }

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        if self.params.get("clahe_enabled", True):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        blur_size = int(self.params.get("blur_size", 3))
        if blur_size > 1:
            if blur_size % 2 == 0:
                blur_size += 1
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        return gray

    def detect(self, image) -> list[dict]:
        threshold = int(self.params.get("threshold", 45))
        threshold_type = cv2.THRESH_BINARY_INV if self.params.get("invert", False) else cv2.THRESH_BINARY
        _, binary = cv2.threshold(image, threshold, 255, threshold_type)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(self.params.get("min_area", 20))
        max_area = float(self.params.get("max_area", 5000))
        image_area = max(float(image.shape[0] * image.shape[1]), 1.0)

        defects = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            confidence = min(1.0, area / image_area * 20.0)
            defects.append(
                {
                    "type": "blob",
                    "bbox_local": [int(x), int(y), int(width), int(height)],
                    "area": area,
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "threshold": threshold,
                        "invert": bool(self.params.get("invert", False)),
                    },
                }
            )

        defects.sort(key=lambda item: item["area"], reverse=True)
        return defects
