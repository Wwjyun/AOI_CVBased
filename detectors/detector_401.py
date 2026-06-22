from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector401(BaseDetector):
    detector_id = "401"
    detector_name = "negative_rotated_rect_detector"
    display_name = "401 negative rotated rectangle detector"
    default_params = {
        "blur_size": 15,
        "morph_operation": "open",
        "morph_kernel": 5,
        "morph_iterations": 10,
        "adaptive_block_size": 29,
        "adaptive_c": 5,
        "max_value": 255,
        "contour_mode": "list",
        "min_area": 25,
        "max_area": 10000,
    }

    def preprocess(self, image):
        return image.copy()

    def detect(self, image) -> list[dict]:
        binary = self._make_binary(image)
        contours, _ = cv2.findContours(binary, self._contour_mode(), cv2.CHAIN_APPROX_SIMPLE)
        image_area = max(float(image.shape[0] * image.shape[1]), 1.0)
        defects = []

        for contour in contours:
            if len(contour) < 3:
                continue

            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            rect_area = float(width * height)
            if not self._passes_area_filter(rect_area):
                continue

            box = cv2.boxPoints(rect)
            box = np.round(box).astype(int)
            x, y, w, h = cv2.boundingRect(box.reshape(-1, 1, 2))
            confidence = min(1.0, rect_area / image_area * 20.0)

            defects.append(
                {
                    "type": "401_negative_rect_detected_ng",
                    "bbox_local": [int(x), int(y), int(w), int(h)],
                    "area": float(np.round(rect_area, 3)),
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "shape": "rotated_rectangle",
                        "center_local": [float(np.round(center_x, 3)), float(np.round(center_y, 3))],
                        "size": [float(np.round(width, 3)), float(np.round(height, 3))],
                        "angle": float(np.round(angle, 3)),
                        "box_points_local": box.astype(int).tolist(),
                        "blur_size": int(self.params.get("blur_size", 15)),
                        "morph_operation": str(self.params.get("morph_operation", "open")),
                        "morph_kernel": int(self.params.get("morph_kernel", 5)),
                        "morph_iterations": int(self.params.get("morph_iterations", 10)),
                        "adaptive_block_size": int(self.params.get("adaptive_block_size", 29)),
                        "adaptive_c": float(self.params.get("adaptive_c", 5)),
                        "threshold_type": "adaptive_mean_inv",
                        "contour_mode": str(self.params.get("contour_mode", "list")),
                    },
                }
            )

        defects.sort(key=lambda item: item["area"], reverse=True)
        return defects

    def _make_binary(self, image):
        blur_size = self._odd_at_least(int(self.params.get("blur_size", 15)), 3)
        blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        morphed = self._morph(blurred)
        gray = cv2.cvtColor(morphed, cv2.COLOR_BGR2GRAY) if morphed.ndim == 3 else morphed
        block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 29)), 3)
        return cv2.adaptiveThreshold(
            gray,
            int(self.params.get("max_value", 255)),
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            float(self.params.get("adaptive_c", 5)),
        )

    def _morph(self, image):
        operation = str(self.params.get("morph_operation", "open")).lower()
        iterations = int(self.params.get("morph_iterations", 10))
        kernel_size = int(self.params.get("morph_kernel", 5))
        if operation in {"none", ""} or iterations <= 0 or kernel_size <= 1:
            return image

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
        if cv_operation == cv2.MORPH_DILATE:
            return cv2.dilate(image, kernel, iterations=iterations)
        if cv_operation == cv2.MORPH_ERODE:
            return cv2.erode(image, kernel, iterations=iterations)
        return cv2.morphologyEx(image, cv_operation, kernel, iterations=iterations)

    def _passes_area_filter(self, area: float) -> bool:
        min_area = float(self.params.get("min_area", 25))
        max_area = float(self.params.get("max_area", 10000))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        return True

    def _contour_mode(self) -> int:
        mode = str(self.params.get("contour_mode", "list")).lower()
        if mode in {"all", "list"}:
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
