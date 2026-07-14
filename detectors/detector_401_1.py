from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector401_1(BaseDetector):
    detector_id = "401-1"
    detector_name = "adaptive_circle_contour_detector"
    display_name = "401-1 adaptive circle contour detector"
    default_params = {
        "threshold_method": "adaptive_mean",
        "max_value": 255,
        "invert": False,
        "blur_size": 45,
        "adaptive_block_size": 33,
        "adaptive_c": -2.0,
        "roi_inset_px": 100,
        "contour_mode": "list",
        "morph_operation": "none",
        "morph_kernel": 3,
        "morph_iterations": 1,
        "process_scale": 1.0,
        "min_area": 100,
        "max_area": 1000,
        "min_circularity": 0.70,
        "min_fill_ratio": 0.55,
        "max_fill_ratio": 1.20,
    }

    def preprocess(self, image):
        if self.gpu_active and image.ndim == 3:
            return self.gpu_runtime.bgr_to_gray(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def detect(self, image) -> list[dict]:
        roi, offset_x, offset_y = self._roi_image(image)
        binary, process_scale = self._make_binary(roi)
        contours, _ = cv2.findContours(binary, self._contour_mode(), cv2.CHAIN_APPROX_SIMPLE)
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

            if not self._passes_filters(area, circularity, fill_ratio):
                continue

            radius = float(radius_scaled * inv_scale)
            cx = float(cx_scaled * inv_scale + offset_x)
            cy = float(cy_scaled * inv_scale + offset_y)
            x = max(0, int(round((cx_scaled - radius_scaled) * inv_scale + offset_x)))
            y = max(0, int(round((cy_scaled - radius_scaled) * inv_scale + offset_y)))
            diameter = max(1, int(round(radius * 2.0)))
            confidence = min(1.0, area / image_area * 20.0)

            defects.append(
                {
                    "type": "401_1_circle_detected_ng",
                    "bbox_local": [x, y, diameter, diameter],
                    "area": float(np.round(area, 3)),
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "shape": "circle",
                        "center_local": [float(np.round(cx, 3)), float(np.round(cy, 3))],
                        "radius": float(np.round(radius, 3)),
                        "diameter": float(np.round(radius * 2.0, 3)),
                        "circularity": float(np.round(circularity, 4)),
                        "fill_ratio": float(np.round(fill_ratio, 4)),
                        "threshold_method": "adaptive_mean",
                        "roi_inset_px": int(self.params.get("roi_inset_px", 100)),
                        "roi_offset_local": [int(offset_x), int(offset_y)],
                        "blur_size": int(self.params.get("blur_size", 45)),
                        "adaptive_block_size": int(self.params.get("adaptive_block_size", 33)),
                        "adaptive_c": float(self.params.get("adaptive_c", -2.0)),
                        "invert": bool(self.params.get("invert", False)),
                    },
                }
            )

        defects.sort(key=lambda item: item["area"], reverse=True)
        return defects

    def _roi_image(self, gray):
        inset = max(0, int(self.params.get("roi_inset_px", 100)))
        if inset <= 0:
            return gray, 0, 0

        height, width = gray.shape[:2]
        if width <= inset * 2 or height <= inset * 2:
            return gray, 0, 0

        return gray[inset : height - inset, inset : width - inset], inset, inset

    def _make_binary(self, gray):
        process_scale = min(max(float(self.params.get("process_scale", 1.0)), 0.05), 1.0)
        work = gray
        if process_scale < 0.999:
            height, width = gray.shape[:2]
            target_width = max(1, int(width * process_scale))
            target_height = max(1, int(height * process_scale))
            if self.gpu_active:
                work = self.gpu_runtime.resize_gray(gray, target_width, target_height)
            else:
                work = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

        blur_size = self._odd_at_least(int(self.params.get("blur_size", 45)), 3)
        work = self.gpu_runtime.gaussian_blur(work, blur_size) if self.gpu_active else cv2.GaussianBlur(work, (blur_size, blur_size), 0)

        threshold_type = cv2.THRESH_BINARY_INV if self.params.get("invert", False) else cv2.THRESH_BINARY
        block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 33)), 3)
        adaptive_c = float(self.params.get("adaptive_c", -2.0))
        if self.gpu_active:
            binary = self.gpu_runtime.adaptive_threshold(
                work, block_size, adaptive_c, int(self.params.get("max_value", 255)), bool(self.params.get("invert", False))
            )
        else:
            binary = cv2.adaptiveThreshold(
                work,
                int(self.params.get("max_value", 255)),
                cv2.ADAPTIVE_THRESH_MEAN_C,
                threshold_type,
                block_size,
                adaptive_c,
            )
        return self._morph(binary), process_scale

    def _morph(self, binary):
        operation = str(self.params.get("morph_operation", "none")).lower()
        iterations = int(self.params.get("morph_iterations", 1))
        kernel_size = int(self.params.get("morph_kernel", 3))
        if operation in {"none", ""} or iterations <= 0 or kernel_size <= 1:
            return binary

        kernel_size = self._odd_at_least(kernel_size, 3)
        if self.gpu_active:
            return self.gpu_runtime.morphology(binary, operation, kernel_size, iterations)
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
        if cv_operation == cv2.MORPH_DILATE:
            return cv2.dilate(binary, kernel, iterations=iterations)
        if cv_operation == cv2.MORPH_ERODE:
            return cv2.erode(binary, kernel, iterations=iterations)
        return cv2.morphologyEx(binary, cv_operation, kernel, iterations=iterations)

    def _passes_filters(self, area: float, circularity: float, fill_ratio: float) -> bool:
        min_area = float(self.params.get("min_area", 100))
        max_area = float(self.params.get("max_area", 1000))
        min_circularity = float(self.params.get("min_circularity", 0.70))
        min_fill_ratio = float(self.params.get("min_fill_ratio", 0.55))
        max_fill_ratio = float(self.params.get("max_fill_ratio", 1.20))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        if circularity < min_circularity:
            return False
        if fill_ratio < min_fill_ratio:
            return False
        return not max_fill_ratio or fill_ratio <= max_fill_ratio

    def _contour_mode(self) -> int:
        mode = str(self.params.get("contour_mode", "external")).lower()
        if mode in {"all", "list"}:
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
