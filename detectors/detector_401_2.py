from __future__ import annotations

import cv2
import numpy as np

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
        "contour_mode": "list",
        "min_area": 0,
        "max_area": 0,
        "white_pixel_ratio_threshold": 0.625,
    }

    def preprocess(self, image):
        if self.gpu_active and self.gpu_runtime.supports_fused_401_2:
            return image.copy()
        if self.gpu_active and image.ndim == 3:
            return self.gpu_runtime.bgr_to_gray(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def detect(self, image) -> list[dict]:
        roi, offset_x, offset_y = self._roi_image(image)
        binary = self._make_binary(roi)
        contours, _ = cv2.findContours(binary, self._contour_mode(), cv2.CHAIN_APPROX_SIMPLE)
        defects = []
        ratio_threshold = float(self.params.get("white_pixel_ratio_threshold", 0.625))

        for contour in contours:
            if len(contour) < 3:
                continue

            area = float(cv2.contourArea(contour))
            if area <= 0.0 or not self._passes_area_filter(area):
                continue

            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            contour_pixel_count = int(np.count_nonzero(mask))
            if contour_pixel_count <= 0:
                continue

            white_pixel_count = int(np.count_nonzero((binary == 255) & (mask > 0)))
            white_pixel_ratio = white_pixel_count / float(contour_pixel_count)
            if white_pixel_ratio < ratio_threshold:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x += offset_x
            y += offset_y
            confidence = min(1.0, white_pixel_ratio)

            defects.append(
                {
                    "type": "401_2_white_pixel_ratio_ng",
                    "bbox_local": [int(x), int(y), int(w), int(h)],
                    "area": float(np.round(area, 3)),
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "shape": "contour",
                        "white_pixel_count": white_pixel_count,
                        "contour_pixel_count": contour_pixel_count,
                        "white_pixel_ratio": float(np.round(white_pixel_ratio, 6)),
                        "white_pixel_ratio_percent": float(np.round(white_pixel_ratio * 100.0, 3)),
                        "white_pixel_ratio_threshold": ratio_threshold,
                        "white_pixel_ratio_threshold_percent": float(np.round(ratio_threshold * 100.0, 3)),
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

        defects.sort(key=lambda item: item["metadata"]["white_pixel_ratio"], reverse=True)
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
        block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 35)), 3)
        if self.gpu_active and self.gpu_runtime.supports_fused_401_2:
            return self.gpu_runtime.preprocess_401_2(
                gray,
                blur_size,
                block_size,
                float(self.params.get("adaptive_c", -2.0)),
                int(self.params.get("max_value", 255)),
                True,
            )
        if self.gpu_active:
            blurred = self.gpu_runtime.gaussian_blur(gray, blur_size)
            return self.gpu_runtime.adaptive_threshold(
                blurred,
                block_size,
                float(self.params.get("adaptive_c", -2.0)),
                int(self.params.get("max_value", 255)),
                True,
            )
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
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
