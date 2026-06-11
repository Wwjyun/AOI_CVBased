from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector001(BaseDetector):
    detector_id = "001"
    detector_name = "circle_threshold_ng_detector"
    display_name = "Circle threshold NG detector"
    default_params = {
        "threshold_method": "adaptive_mean",
        "threshold": 128,
        "max_value": 255,
        "invert": False,
        "adaptive_block_size": 31,
        "adaptive_c": -10.0,
        "canny_low": 128,
        "canny_high": 200,
        "blur_size": 20,
        "contour_mode": "external",
        "morph_operation": "none",
        "morph_kernel": 3,
        "morph_iterations": 1,
        "process_scale": 1.0,
        "min_area": 0,
        "max_area": 0,
        "min_radius": 0,
        "max_radius": 0,
        "min_circularity": 0.70,
        "min_fill_ratio": 0.55,
    }

    def preprocess(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def detect(self, image) -> list[dict]:
        binary, process_scale = self._make_binary(image)
        contour_mode = self._contour_mode()
        contours, _ = cv2.findContours(binary, contour_mode, cv2.CHAIN_APPROX_SIMPLE)
        image_area = max(float(image.shape[0] * image.shape[1]), 1.0)
        defects = []

        for contour in contours:
            area_scaled = float(cv2.contourArea(contour))
            perimeter_scaled = float(cv2.arcLength(contour, True))
            if area_scaled <= 0.0 or perimeter_scaled <= 0.0:
                continue

            (cx_scaled, cy_scaled), radius_scaled = cv2.minEnclosingCircle(contour)
            if radius_scaled <= 0.0:
                continue

            circle_area_scaled = float(np.pi * radius_scaled * radius_scaled)
            circularity = float(4.0 * np.pi * area_scaled / (perimeter_scaled * perimeter_scaled))
            fill_ratio = float(area_scaled / circle_area_scaled) if circle_area_scaled > 0.0 else 0.0

            inv_scale = 1.0 / process_scale
            area = area_scaled * inv_scale * inv_scale
            radius = float(radius_scaled * inv_scale)
            cx = float(cx_scaled * inv_scale)
            cy = float(cy_scaled * inv_scale)

            if not self._passes_filters(area, radius, circularity, fill_ratio):
                continue

            x = int(round((cx_scaled - radius_scaled) * inv_scale))
            y = int(round((cy_scaled - radius_scaled) * inv_scale))
            diameter = int(round(radius * 2.0))
            x = max(0, x)
            y = max(0, y)
            width = max(1, diameter)
            height = max(1, diameter)
            confidence = min(1.0, area / image_area * 20.0)

            defects.append(
                {
                    "type": "circle_detected_ng",
                    "bbox_local": [x, y, width, height],
                    "area": float(np.round(area, 3)),
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "shape": "circle",
                        "center_local": [float(np.round(cx, 3)), float(np.round(cy, 3))],
                        "radius": float(np.round(radius, 3)),
                        "diameter": float(np.round(radius * 2.0, 3)),
                        "circularity": float(np.round(circularity, 4)),
                        "fill_ratio": float(np.round(fill_ratio, 4)),
                        "threshold_method": str(self.params.get("threshold_method", "adaptive_mean")),
                        "invert": bool(self.params.get("invert", False)),
                    },
                }
            )

        defects.sort(key=lambda item: item["area"], reverse=True)
        return defects

    def _make_binary(self, gray):
        process_scale = float(self.params.get("process_scale", 1.0))
        process_scale = min(max(process_scale, 0.05), 1.0)
        work = gray
        if process_scale < 0.999:
            height, width = gray.shape[:2]
            work = cv2.resize(
                gray,
                (max(1, int(width * process_scale)), max(1, int(height * process_scale))),
                interpolation=cv2.INTER_AREA,
            )

        blur_size = int(self.params.get("blur_size", 20))
        if blur_size >= 3:
            blur_size = self._odd_at_least(blur_size, 3)
            work = cv2.GaussianBlur(work, (blur_size, blur_size), 0)

        method = str(self.params.get("threshold_method", "adaptive_mean")).lower()
        max_value = int(self.params.get("max_value", 255))
        threshold_type = cv2.THRESH_BINARY_INV if self.params.get("invert", False) else cv2.THRESH_BINARY

        if method == "global":
            _, binary = cv2.threshold(work, int(self.params.get("threshold", 128)), max_value, threshold_type)
        elif method == "otsu":
            _, binary = cv2.threshold(work, 0, max_value, threshold_type | cv2.THRESH_OTSU)
        elif method in {"adaptive_mean", "adaptive_gaussian"}:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 31)), 3)
            adaptive_c = float(self.params.get("adaptive_c", -10.0))
            binary = cv2.adaptiveThreshold(work, max_value, adaptive_method, threshold_type, block_size, adaptive_c)
        elif method == "canny":
            binary = cv2.Canny(work, int(self.params.get("canny_low", 128)), int(self.params.get("canny_high", 200)))
            if self.params.get("invert", False):
                binary = cv2.bitwise_not(binary)
        else:
            raise ValueError(f"Unsupported threshold method: {method}")

        return self._morph(binary), process_scale

    def _morph(self, binary):
        operation = str(self.params.get("morph_operation", "none")).lower()
        iterations = int(self.params.get("morph_iterations", 1))
        kernel_size = int(self.params.get("morph_kernel", 3))
        if operation in {"none", ""} or iterations <= 0 or kernel_size <= 1:
            return binary

        kernel_size = self._odd_at_least(kernel_size, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        operations = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "dilate": cv2.MORPH_DILATE,
            "erode": cv2.MORPH_ERODE,
        }
        cv_operation = operations.get(operation)
        if cv_operation is None:
            raise ValueError(f"Unsupported morphology operation: {operation}")
        if cv_operation in {cv2.MORPH_DILATE, cv2.MORPH_ERODE}:
            return cv2.dilate(binary, kernel, iterations=iterations) if cv_operation == cv2.MORPH_DILATE else cv2.erode(binary, kernel, iterations=iterations)
        return cv2.morphologyEx(binary, cv_operation, kernel, iterations=iterations)

    def _passes_filters(self, area: float, radius: float, circularity: float, fill_ratio: float) -> bool:
        min_area = float(self.params.get("min_area", 0))
        max_area = float(self.params.get("max_area", 0))
        min_radius = float(self.params.get("min_radius", 0))
        max_radius = float(self.params.get("max_radius", 0))
        min_circularity = float(self.params.get("min_circularity", 0.70))
        min_fill_ratio = float(self.params.get("min_fill_ratio", 0.55))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        if min_radius and radius < min_radius:
            return False
        if max_radius and radius > max_radius:
            return False
        return circularity >= min_circularity and fill_ratio >= min_fill_ratio

    def _contour_mode(self) -> int:
        mode = str(self.params.get("contour_mode", "external")).lower()
        if mode == "all":
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
