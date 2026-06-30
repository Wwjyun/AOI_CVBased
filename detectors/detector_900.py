from __future__ import annotations

import cv2
import numpy as np

from detectors.base_detector import BaseDetector


class Detector900(BaseDetector):
    detector_id = "900"
    detector_name = "dual_frame_spacing_detector"
    display_name = "900 dual frame spacing detector"
    default_params = {
        "max_value": 255,
        "outer_threshold": 160,
        "outer_invert": False,
        "outer_contour_mode": "list",
        "outer_area_metric": "component_pixels",
        "outer_min_area": 100000,
        "outer_max_area": 130000,
        "inner_adaptive_block_size": 11,
        "inner_adaptive_c": 0.0,
        "inner_invert": False,
        "inner_contour_mode": "list",
        "inner_area_metric": "component_pixels",
        "inner_min_area": 100000,
        "inner_max_area": 130000,
        "inner_target_width": 998,
        "inner_width_tolerance": 33,
        "inner_target_height": 1164,
        "inner_height_tolerance": 33,
        "max_edge_gap": 31,
        "roi_inset_px": 0,
    }

    def preprocess(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    def detect(self, image) -> list[dict]:
        roi, offset_x, offset_y = self._roi_image(image)
        outer_mask = self._make_outer_binary(roi)
        inner_mask = self._make_inner_binary(roi)

        outer_candidates = self._find_candidates(
            outer_mask,
            mode_param="outer_contour_mode",
            area_metric_param="outer_area_metric",
            min_area_param="outer_min_area",
            max_area_param="outer_max_area",
        )
        inner_candidates = self._find_candidates(
            inner_mask,
            mode_param="inner_contour_mode",
            area_metric_param="inner_area_metric",
            min_area_param="inner_min_area",
            max_area_param="inner_max_area",
        )
        sized_inner_candidates = [candidate for candidate in inner_candidates if self._passes_inner_size(candidate)]

        match = self._find_valid_pair(outer_candidates, sized_inner_candidates)
        if match is not None:
            return []

        failure_bbox = self._failure_bbox(outer_candidates, inner_candidates, image.shape[:2], offset_x, offset_y)
        reason = self._failure_reason(outer_candidates, inner_candidates, sized_inner_candidates)
        return [
            {
                "type": "900_frame_spacing_ng",
                "bbox_local": failure_bbox,
                "area": float(np.round(self._bbox_area(failure_bbox), 3)),
                "confidence": 1.0,
                "metadata": {
                    "reason": reason,
                    "outer_candidate_count": len(outer_candidates),
                    "inner_candidate_count": len(inner_candidates),
                    "inner_size_pass_count": len(sized_inner_candidates),
                    "outer_threshold": int(self.params.get("outer_threshold", 160)),
                    "outer_contour_mode": str(self.params.get("outer_contour_mode", "list")),
                    "outer_area_metric": str(self.params.get("outer_area_metric", "component_pixels")),
                    "outer_min_area": float(self.params.get("outer_min_area", 100000)),
                    "outer_max_area": float(self.params.get("outer_max_area", 130000)),
                    "inner_threshold_method": "adaptive_mean",
                    "inner_adaptive_block_size": int(self.params.get("inner_adaptive_block_size", 11)),
                    "inner_adaptive_c": float(self.params.get("inner_adaptive_c", 0.0)),
                    "inner_contour_mode": str(self.params.get("inner_contour_mode", "list")),
                    "inner_area_metric": str(self.params.get("inner_area_metric", "component_pixels")),
                    "inner_min_area": float(self.params.get("inner_min_area", 100000)),
                    "inner_max_area": float(self.params.get("inner_max_area", 130000)),
                    "inner_target_width": int(self.params.get("inner_target_width", 998)),
                    "inner_width_tolerance": int(self.params.get("inner_width_tolerance", 33)),
                    "inner_target_height": int(self.params.get("inner_target_height", 1164)),
                    "inner_height_tolerance": int(self.params.get("inner_height_tolerance", 33)),
                    "max_edge_gap": int(self.params.get("max_edge_gap", 31)),
                    "roi_inset_px": int(self.params.get("roi_inset_px", 0)),
                    "roi_offset_local": [int(offset_x), int(offset_y)],
                    "best_outer": self._offset_candidate(self._largest_candidate(outer_candidates), offset_x, offset_y),
                    "best_inner": self._offset_candidate(self._largest_candidate(inner_candidates), offset_x, offset_y),
                },
            }
        ]

    def _roi_image(self, gray):
        inset = max(0, int(self.params.get("roi_inset_px", 0)))
        if inset <= 0:
            return gray, 0, 0

        height, width = gray.shape[:2]
        if width <= inset * 2 or height <= inset * 2:
            return gray, 0, 0

        return gray[inset : height - inset, inset : width - inset], inset, inset

    def _make_outer_binary(self, gray):
        threshold_type = cv2.THRESH_BINARY_INV if self.params.get("outer_invert", False) else cv2.THRESH_BINARY
        _, binary = cv2.threshold(
            gray,
            int(self.params.get("outer_threshold", 160)),
            int(self.params.get("max_value", 255)),
            threshold_type,
        )
        return binary

    def _make_inner_binary(self, gray):
        block_size = self._odd_at_least(int(self.params.get("inner_adaptive_block_size", 11)), 3)
        threshold_type = cv2.THRESH_BINARY_INV if self.params.get("inner_invert", False) else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            gray,
            int(self.params.get("max_value", 255)),
            cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            block_size,
            float(self.params.get("inner_adaptive_c", 0.0)),
        )

    def _find_candidates(
        self,
        binary,
        mode_param: str,
        area_metric_param: str,
        min_area_param: str,
        max_area_param: str,
    ) -> list[dict]:
        if str(self.params.get(area_metric_param, "component_pixels")).lower() == "component_pixels":
            return self._find_component_candidates(binary, min_area_param, max_area_param)

        contours, _ = cv2.findContours(binary, self._contour_mode(mode_param), cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_area = float(self.params.get(min_area_param, 0))
        max_area = float(self.params.get(max_area_param, 0))
        for contour in contours:
            if len(contour) < 3:
                continue

            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue
            if min_area and area < min_area:
                continue
            if max_area and area > max_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            candidates.append(
                {
                    "bbox": [int(x), int(y), int(width), int(height)],
                    "area": area,
                    "contour_area": area,
                    "area_metric": "contour_area",
                }
            )

        candidates.sort(key=lambda item: item["area"], reverse=True)
        return candidates

    def _find_component_candidates(self, binary, min_area_param: str, max_area_param: str) -> list[dict]:
        component_count, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
        candidates = []
        min_area = float(self.params.get(min_area_param, 0))
        max_area = float(self.params.get(max_area_param, 0))
        for label in range(1, component_count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area <= 0.0:
                continue
            if min_area and area < min_area:
                continue
            if max_area and area > max_area:
                continue

            candidates.append(
                {
                    "bbox": [x, y, width, height],
                    "area": area,
                    "component_pixel_area": area,
                    "area_metric": "component_pixels",
                }
            )

        candidates.sort(key=lambda item: item["area"], reverse=True)
        return candidates

    def _passes_inner_size(self, candidate: dict) -> bool:
        _, _, width, height = candidate["bbox"]
        target_width = int(self.params.get("inner_target_width", 998))
        width_tolerance = int(self.params.get("inner_width_tolerance", 33))
        target_height = int(self.params.get("inner_target_height", 1164))
        height_tolerance = int(self.params.get("inner_height_tolerance", 33))
        return (
            abs(width - target_width) <= width_tolerance
            and abs(height - target_height) <= height_tolerance
        )

    def _find_valid_pair(self, outer_candidates: list[dict], inner_candidates: list[dict]) -> dict | None:
        for outer in outer_candidates:
            for inner in inner_candidates:
                edge_gaps = self._edge_gaps(outer["bbox"], inner["bbox"])
                if edge_gaps is None:
                    continue
                if max(edge_gaps.values()) <= int(self.params.get("max_edge_gap", 31)):
                    return {
                        "outer": outer,
                        "inner": inner,
                        "edge_gaps": edge_gaps,
                    }
        return None

    @staticmethod
    def _edge_gaps(outer_bbox: list[int], inner_bbox: list[int]) -> dict | None:
        outer_x, outer_y, outer_w, outer_h = outer_bbox
        inner_x, inner_y, inner_w, inner_h = inner_bbox
        outer_right = outer_x + outer_w
        outer_bottom = outer_y + outer_h
        inner_right = inner_x + inner_w
        inner_bottom = inner_y + inner_h
        if inner_x < outer_x or inner_y < outer_y or inner_right > outer_right or inner_bottom > outer_bottom:
            return None
        return {
            "left": int(inner_x - outer_x),
            "top": int(inner_y - outer_y),
            "right": int(outer_right - inner_right),
            "bottom": int(outer_bottom - inner_bottom),
        }

    def _failure_reason(
        self,
        outer_candidates: list[dict],
        inner_candidates: list[dict],
        sized_inner_candidates: list[dict],
    ) -> str:
        if not outer_candidates:
            return "no_outer_candidate"
        if not inner_candidates:
            return "no_inner_candidate"
        if not sized_inner_candidates:
            return "inner_size_out_of_tolerance"
        return "edge_gap_out_of_tolerance_or_inner_not_inside_outer"

    def _failure_bbox(
        self,
        outer_candidates: list[dict],
        inner_candidates: list[dict],
        image_shape: tuple[int, int],
        offset_x: int,
        offset_y: int,
    ) -> list[int]:
        candidate = self._largest_candidate(inner_candidates) or self._largest_candidate(outer_candidates)
        if candidate is not None:
            x, y, width, height = candidate["bbox"]
            return [int(x + offset_x), int(y + offset_y), int(width), int(height)]

        height, width = image_shape
        return [0, 0, int(width), int(height)]

    @staticmethod
    def _largest_candidate(candidates: list[dict]) -> dict | None:
        return candidates[0] if candidates else None

    @staticmethod
    def _offset_candidate(candidate: dict | None, offset_x: int, offset_y: int) -> dict | None:
        if candidate is None:
            return None
        x, y, width, height = candidate["bbox"]
        return {
            "bbox": [int(x + offset_x), int(y + offset_y), int(width), int(height)],
            "area": float(np.round(candidate["area"], 3)),
            "area_metric": candidate.get("area_metric", ""),
        }

    @staticmethod
    def _bbox_area(bbox: list[int]) -> float:
        return float(max(0, bbox[2]) * max(0, bbox[3]))

    def _contour_mode(self, param_name: str) -> int:
        mode = str(self.params.get(param_name, "list")).lower()
        if mode in {"all", "list"}:
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
