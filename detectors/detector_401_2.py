from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector401_2(BaseDetector):
    detector_id = "401-2"
    detector_name = "adaptive_min_gray_detector"
    display_name = "401-2 adaptive min gray detector"
    default_params = {
        "max_value": 255,
        "blur_size": 25,
        "adaptive_block_size": 35,
        "adaptive_c": -2.0,
        "roi_inset_px": 0,
        "contour_mode": "list",
        "min_area": 0,
        "max_area": 0,
        "min_gray_threshold": 16,
    }

    def preprocess(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def detect(self, image) -> list[dict]:
        roi, offset_x, offset_y = self._roi_image(image)
        binary = self._make_binary(roi)
        contours, _ = cv2.findContours(binary, self._contour_mode(), cv2.CHAIN_APPROX_SIMPLE)
        defects = []
        gray_threshold = int(self.params.get("min_gray_threshold", 16))

        for contour in contours:
            if len(contour) < 3:
                continue

            area = float(cv2.contourArea(contour))
            if area <= 0.0 or not self._passes_area_filter(area):
                continue

            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            contour_pixels = roi[mask > 0]
            if contour_pixels.size == 0:
                continue

            min_gray = int(np.min(contour_pixels))
            if min_gray >= gray_threshold:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x += offset_x
            y += offset_y
            confidence = min(1.0, max(0.0, (gray_threshold - min_gray) / max(float(gray_threshold), 1.0)))

            defects.append(
                {
                    "type": "401_2_min_gray_ng",
                    "bbox_local": [int(x), int(y), int(w), int(h)],
                    "area": float(np.round(area, 3)),
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "shape": "contour",
                        "min_gray": min_gray,
                        "min_gray_threshold": gray_threshold,
                        "threshold_method": "adaptive_mean_inv",
                        "roi_inset_px": int(self.params.get("roi_inset_px", 0)),
                        "roi_offset_local": [int(offset_x), int(offset_y)],
                        "blur_size": int(self.params.get("blur_size", 25)),
                        "adaptive_block_size": int(self.params.get("adaptive_block_size", 35)),
                        "adaptive_c": float(self.params.get("adaptive_c", -2.0)),
                        "contour_mode": str(self.params.get("contour_mode", "list")),
                        "min_area": float(self.params.get("min_area", 0)),
                        "max_area": float(self.params.get("max_area", 0)),
                    },
                }
            )

        defects.sort(key=lambda item: item["metadata"]["min_gray"])
        return defects

    def _roi_image(self, gray):
        inset = max(0, int(self.params.get("roi_inset_px", 0)))
        if inset <= 0:
            return gray, 0, 0

        height, width = gray.shape[:2]
        if width <= inset * 2 or height <= inset * 2:
            return gray, 0, 0

        return gray[inset : height - inset, inset : width - inset], inset, inset

    def _make_binary(self, gray):
        blur_size = self._odd_at_least(int(self.params.get("blur_size", 25)), 3)
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 35)), 3)
        return cv2.adaptiveThreshold(
            blurred,
            int(self.params.get("max_value", 255)),
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            float(self.params.get("adaptive_c", -2.0)),
        )

    def _contour_mode(self) -> int:
        mode = str(self.params.get("contour_mode", "list")).lower()
        if mode in {"all", "list"}:
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    def _passes_area_filter(self, area: float) -> bool:
        min_area = float(self.params.get("min_area", 0))
        max_area = float(self.params.get("max_area", 0))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        return True

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
