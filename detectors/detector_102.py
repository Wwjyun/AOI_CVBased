from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector102(BaseDetector):
    detector_id = "102"
    detector_name = "scratch_detector"
    display_name = "Scratch / thin line detector"
    default_params = {
        "canny_low": 30,
        "canny_high": 120,
        "min_length": 50,
        "max_width": 8,
        "morphology_kernel": 3,
        "blur_size": 3,
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
        edges = cv2.Canny(
            image,
            int(self.params.get("canny_low", 30)),
            int(self.params.get("canny_high", 120)),
        )

        kernel_size = max(1, int(self.params.get("morphology_kernel", 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_length = float(self.params.get("min_length", 50))
        max_width = float(self.params.get("max_width", 8))
        defects = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            length = float(max(width, height))
            thickness = float(min(width, height))
            if length < min_length or thickness > max_width:
                continue

            area = float(cv2.contourArea(contour))
            confidence = min(1.0, length / max(min_length, 1.0))
            defects.append(
                {
                    "type": "scratch",
                    "bbox_local": [int(x), int(y), int(width), int(height)],
                    "area": area,
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "length": length,
                        "thickness": thickness,
                    },
                }
            )

        defects.sort(key=lambda item: item["metadata"]["length"], reverse=True)
        return defects
