from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector000(BaseDetector):
    detector_id = "000"
    detector_name = "binary_contour_area_guard"
    display_name = "Binary contour area guard"
    default_params = {
        "threshold_method": "adaptive_gaussian",
        "threshold": 128,
        "max_value": 255,
        "invert": True,
        "adaptive_block_size": 31,
        "adaptive_c": 5.0,
        "blur_size": 3,
        "morph_open_kernel": 0,
        "morph_open_iterations": 1,
        "morph_close_kernel": 0,
        "morph_close_iterations": 1,
        "min_area": 0,
        "max_area": 0,
        "min_width": 0,
        "max_width": 0,
        "min_height": 0,
        "max_height": 0,
    }

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        blur_size = int(self.params.get("blur_size", 3))
        if blur_size > 1:
            blur_size = self._odd_at_least(blur_size, 3)
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        return gray

    def detect(self, image) -> list[dict]:
        binary = self._make_binary(image)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = max(float(image.shape[0] * image.shape[1]), 1.0)
        defects = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            area_status = self._area_status(area)
            size_status = self._size_status(width, height)
            if not area_status and not size_status:
                continue

            confidence = min(1.0, area / image_area)
            defects.append(
                {
                    "type": "binary_contour_area",
                    "bbox_local": [int(x), int(y), int(width), int(height)],
                    "area": area,
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "area_status": area_status,
                        "size_status": size_status,
                        "threshold_method": str(self.params.get("threshold_method", "adaptive_gaussian")),
                        "invert": bool(self.params.get("invert", True)),
                    },
                }
            )

        defects.sort(key=lambda item: item["area"], reverse=True)
        return defects

    def _make_binary(self, gray):
        method = str(self.params.get("threshold_method", "adaptive_gaussian")).lower()
        max_value = int(self.params.get("max_value", 255))
        threshold_type = cv2.THRESH_BINARY_INV if self.params.get("invert", True) else cv2.THRESH_BINARY

        if method == "global":
            threshold = int(self.params.get("threshold", 128))
            _, binary = cv2.threshold(gray, threshold, max_value, threshold_type)
        elif method == "otsu":
            _, binary = cv2.threshold(gray, 0, max_value, threshold_type | cv2.THRESH_OTSU)
        elif method in {"adaptive_mean", "adaptive_gaussian"}:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 31)), 3)
            adaptive_c = float(self.params.get("adaptive_c", 5.0))
            binary = cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, adaptive_c)
        else:
            raise ValueError(f"Unsupported threshold method: {method}")

        binary = self._morph(binary, cv2.MORPH_OPEN, "morph_open_kernel", "morph_open_iterations")
        binary = self._morph(binary, cv2.MORPH_CLOSE, "morph_close_kernel", "morph_close_iterations")
        return binary

    def _area_status(self, area: float) -> str:
        min_area = float(self.params.get("min_area", 0))
        max_area = float(self.params.get("max_area", 0))
        if min_area and area < min_area:
            return "below_min_area"
        if max_area and area > max_area:
            return "above_max_area"
        return ""

    def _size_status(self, width: int, height: int) -> str:
        min_width = float(self.params.get("min_width", 0))
        max_width = float(self.params.get("max_width", 0))
        min_height = float(self.params.get("min_height", 0))
        max_height = float(self.params.get("max_height", 0))
        if min_width and width < min_width:
            return "below_min_width"
        if max_width and width > max_width:
            return "above_max_width"
        if min_height and height < min_height:
            return "below_min_height"
        if max_height and height > max_height:
            return "above_max_height"
        return ""

    def _morph(self, binary, operation: int, kernel_key: str, iterations_key: str):
        kernel_size = int(self.params.get(kernel_key, 0))
        iterations = int(self.params.get(iterations_key, 1))
        if kernel_size <= 1 or iterations <= 0:
            return binary
        kernel_size = self._odd_at_least(kernel_size, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(binary, operation, kernel, iterations=iterations)

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
