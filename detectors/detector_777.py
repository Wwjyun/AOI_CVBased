from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector777(BaseDetector):
    detector_id = "777"
    detector_name = "pattern_match_detector"
    display_name = "Pattern match detector"
    default_params = {
        "template_path": "",
        "match_threshold": 0.8,
        "max_count": 999,
        "min_count": 1,
        "nms_threshold": 0.3,
    }

    def preprocess(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def detect(self, image) -> list[dict]:
        template_path = str(self.params.get("template_path", "")).strip()
        if not template_path:
            return []

        template = cv2.imread(str(Path(template_path)), cv2.IMREAD_GRAYSCALE)
        if template is None:
            return [
                {
                    "type": "pattern_template_missing",
                    "bbox_local": [0, 0, int(image.shape[1]), int(image.shape[0])],
                    "area": float(image.shape[0] * image.shape[1]),
                    "confidence": 1.0,
                    "metadata": {"template_path": template_path},
                }
            ]

        if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
            return []

        threshold = float(self.params.get("match_threshold", 0.8))
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= threshold)
        boxes = []
        scores = []
        for x, y in zip(xs, ys):
            boxes.append([int(x), int(y), int(template.shape[1]), int(template.shape[0])])
            scores.append(float(result[y, x]))

        keep = self._nms(boxes, scores, float(self.params.get("nms_threshold", 0.3)))
        max_count = int(self.params.get("max_count", 999))
        min_count = int(self.params.get("min_count", 1))
        matches = keep[:max_count]

        if len(matches) < min_count:
            return [
                {
                    "type": "pattern_missing",
                    "bbox_local": [0, 0, int(image.shape[1]), int(image.shape[0])],
                    "area": float(image.shape[0] * image.shape[1]),
                    "confidence": 1.0,
                    "metadata": {
                        "expected_min_count": min_count,
                        "actual_count": len(matches),
                        "template_path": template_path,
                    },
                }
            ]
        if len(keep) > max_count:
            return [
                {
                    "type": "pattern_excess_count",
                    "bbox_local": [0, 0, int(image.shape[1]), int(image.shape[0])],
                    "area": float(image.shape[0] * image.shape[1]),
                    "confidence": 1.0,
                    "metadata": {
                        "max_count": max_count,
                        "actual_count": len(keep),
                        "template_path": template_path,
                    },
                }
            ]
        return []

    @staticmethod
    def _nms(boxes: list[list[int]], scores: list[float], threshold: float) -> list[int]:
        if not boxes:
            return []

        order = sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True)
        keep = []
        while order:
            current = order.pop(0)
            keep.append(current)
            order = [
                index
                for index in order
                if Detector777._iou(boxes[current], boxes[index]) <= threshold
            ]
        return keep

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        intersection = iw * ih
        union = aw * ah + bw * bh - intersection
        return intersection / union if union else 0.0
