from __future__ import annotations

import csv
import json
import uuid
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from core.logging_system import LogMixin

if TYPE_CHECKING:
    from core.performance import PipelineProfiler


class Reporter(LogMixin):
    def __init__(self, output_dir: Path, output_config: dict, profiler: PipelineProfiler | None = None):
        self.output_dir = Path(output_dir)
        self.output_config = output_config or {}
        self.profiler = profiler
        self.overlay_dir = self.output_dir / "overlay"
        self.ng_tiles_dir = self.output_dir / "ng_tiles"
        self.csv_dir = self.output_dir / "csv"
        self.matrix_csv_dir = self.output_dir / "matrix_csv"
        self.json_dir = self.output_dir / "json"
        for directory in (self.overlay_dir, self.ng_tiles_dir, self.csv_dir, self.matrix_csv_dir, self.json_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def write(self, image, result: dict) -> dict[str, object]:
        stem = Path(result["image_name"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{stem}_{result['recipe_name']}_{timestamp}_{uuid.uuid4().hex[:8]}"
        outputs: dict[str, object] = {}
        self.logger.info("Writing report outputs: image=%s base=%s", result.get("image_name"), base_name)

        if self.output_config.get("save_overlay", True):
            with self._measure("overlay"):
                overlay_path = self.overlay_dir / f"{base_name}_overlay.png"
                self._write_png(overlay_path, self._make_overlay(image, result))
                outputs["overlay"] = str(overlay_path)

        if self.output_config.get("save_ng_tiles", True):
            with self._measure("ng_tiles"):
                sidecars = self._write_ng_tiles(result, base_name)
                outputs["ng_tiles_dir"] = str(self.ng_tiles_dir)
                outputs["ng_tile_sidecars"] = sidecars

        if self.output_config.get("save_csv", True):
            with self._measure("csv"):
                csv_path = self.csv_dir / f"{base_name}.csv"
                self._write_csv(csv_path, result)
                outputs["csv"] = str(csv_path)

        if self.output_config.get("save_matrix_csv", True):
            with self._measure("matrix_csv"):
                matrix_csv_path = self.matrix_csv_dir / f"{base_name}_matrix.csv"
                self._write_matrix_csv(matrix_csv_path, result)
                outputs["matrix_csv"] = str(matrix_csv_path)

        if self.output_config.get("save_json", True):
            if self.profiler is not None:
                result.setdefault("execution", {})["performance"] = self.profiler.snapshot()
            with self._measure("json"):
                json_path = self.json_dir / f"{base_name}.json"
                with json_path.open("w", encoding="utf-8") as handle:
                    json.dump(self._json_safe_result(result, outputs), handle, ensure_ascii=False, indent=2)
                outputs["json"] = str(json_path)

        self.logger.info("Report outputs written: outputs=%s", outputs)
        return outputs

    def _measure(self, name: str):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.measure(f"report:{name}")

    @staticmethod
    def _json_safe_result(result: dict, outputs: dict[str, object]) -> dict:
        cleaned = dict(result)
        cleaned["outputs"] = dict(outputs)
        cleaned["tiles"] = []
        for tile_result in result["tiles"]:
            cleaned_tile = dict(tile_result)
            cleaned_tile.pop("_tile_image", None)
            cleaned["tiles"].append(cleaned_tile)
        return cleaned

    @staticmethod
    def _make_overlay(image, result: dict):
        overlay = image.copy()
        if Reporter._has_status_tiles(result):
            Reporter._draw_tile_status_overlay(overlay, result)
            return overlay

        for tile_result in result["tiles"]:
            for detector_result in tile_result["detectors"]:
                for defect in detector_result.get("defects", []):
                    Reporter._draw_defect(overlay, defect)
                    x, y, _, _ = defect["bbox_global"]
                    label = f"{detector_result['detector_id']}:{defect['type']}"
                    cv2.putText(overlay, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return overlay

    @staticmethod
    def _has_status_tiles(result: dict) -> bool:
        return any(
            Reporter._is_status_tile(tile_result.get("tile", {}))
            for tile_result in result.get("tiles", [])
        )

    @staticmethod
    def _is_status_tile(tile: dict) -> bool:
        metadata = tile.get("metadata", {})
        return metadata.get("mode") in {"pattern_match", "grid"}

    @staticmethod
    def _draw_tile_status_overlay(overlay, result: dict) -> None:
        for tile_result in result.get("tiles", []):
            tile = tile_result.get("tile", {})
            metadata = tile.get("metadata", {})
            if not Reporter._is_status_tile(tile):
                continue

            bbox = Reporter._status_tile_bbox(tile)
            x, y, width, height = [int(round(value)) for value in bbox]
            is_ng = tile_result.get("result") == "NG"
            color = (0, 0, 255) if is_ng else (0, 180, 0)
            status = "NG" if is_ng else "OK"
            tile_id = str(tile.get("tile_id", ""))
            label = f"{tile_id} {status}".strip()

            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 4)
            label_y = y - 8 if y >= 18 else y + height + 22
            cv2.putText(overlay, label, (x, max(18, label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    @staticmethod
    def _status_tile_bbox(tile: dict) -> list:
        metadata = tile.get("metadata", {})
        if metadata.get("mode") == "pattern_match" and metadata.get("match_bbox"):
            return metadata["match_bbox"]
        return [tile.get("x", 0), tile.get("y", 0), tile.get("width", 0), tile.get("height", 0)]

    @staticmethod
    def _draw_defect(overlay, defect: dict) -> None:
        x, y, width, height = defect["bbox_global"]
        metadata = defect.get("metadata", {})
        if metadata.get("shape") == "circle" and metadata.get("center_global") and metadata.get("radius"):
            cx, cy = metadata["center_global"]
            radius = metadata["radius"]
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(radius)), (0, 0, 255), 4)
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
            return
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 255), 4)

    def _write_ng_tiles(self, result: dict, base_name: str) -> list[str]:
        sidecars: list[str] = []
        for tile_result in result["tiles"]:
            if tile_result.get("result") != "NG":
                continue
            tile = tile_result["tile"]
            tile_image = tile_result.get("_tile_image")
            if tile_image is None:
                continue
            path = self.ng_tiles_dir / f"{base_name}_{tile['tile_id']}.png"
            self._write_png(path, self._make_ng_tile_overlay(tile_image, tile_result))
            sidecar_path = path.with_suffix(".json")
            sidecar = self._ng_tile_sidecar(result, tile_result, path.name)
            with sidecar_path.open("w", encoding="utf-8") as handle:
                json.dump(sidecar, handle, ensure_ascii=False, indent=2)
            sidecars.append(str(sidecar_path))
        return sidecars

    @staticmethod
    def _ng_tile_sidecar(result: dict, tile_result: dict, image_file: str) -> dict:
        provenance = result.get("provenance", {})
        detector_params = provenance.get("detector_params", {})
        detectors = []
        for detector_result in tile_result.get("detectors", []):
            detector_id = str(detector_result.get("detector_id", ""))
            detectors.append({
                "detector_id": detector_id,
                "display_name": detector_result.get("display_name", ""),
                "params": detector_params.get(detector_id, {}),
                "pass": detector_result.get("pass"),
                "score": detector_result.get("score"),
                "defects": detector_result.get("defects", []),
            })
        return {
            "schema_version": 1,
            "dataset_role": "unreviewed_ng_candidate",
            "image_file": image_file,
            "source_image": result.get("image_name"),
            "recipe_name": result.get("recipe_name"),
            "recipe_version": result.get("recipe_version"),
            "provenance": provenance,
            "tile": tile_result.get("tile", {}),
            "detectors": detectors,
            "human_review": {
                "status": "pending",
                "label": None,
                "reviewer": None,
                "reviewed_at": None,
                "notes": "",
            },
        }

    @staticmethod
    def _make_ng_tile_overlay(tile_image, tile_result: dict):
        annotated = tile_image.copy()
        line_width = Reporter._ng_tile_line_width(annotated)
        for detector_result in tile_result.get("detectors", []):
            for defect in detector_result.get("defects", []):
                if detector_result.get("detector_id") == "900":
                    Reporter._draw_detector_900_ng_tile_debug(annotated, defect, line_width)
                    continue
                bbox = Reporter._clipped_local_bbox(defect.get("bbox_local"), annotated)
                if bbox is None:
                    continue
                x, y, width, height = bbox
                cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), line_width)
        return annotated

    @staticmethod
    def _draw_detector_900_ng_tile_debug(annotated, defect: dict, line_width: int) -> None:
        metadata = defect.get("metadata", {})
        Reporter._draw_900_candidate_group(
            annotated,
            metadata.get("debug_outer_candidates") or [metadata.get("best_outer")],
            color=(255, 255, 0),
            prefix="OUT",
            line_width=line_width,
            label_y_offset=0,
        )
        Reporter._draw_900_candidate_group(
            annotated,
            metadata.get("debug_outer_rejected_candidates") or [],
            color=(255, 0, 255),
            prefix="OUT_FAIL",
            line_width=line_width,
            label_y_offset=18,
        )
        Reporter._draw_900_candidate_group(
            annotated,
            metadata.get("debug_inner_candidates") or [metadata.get("best_inner")],
            color=(0, 255, 0),
            prefix="IN",
            line_width=line_width,
            label_y_offset=0,
        )
        Reporter._draw_900_candidate_group(
            annotated,
            metadata.get("debug_inner_rejected_candidates") or [],
            color=(0, 165, 255),
            prefix="IN_FAIL",
            line_width=line_width,
            label_y_offset=36,
        )

        bbox = Reporter._clipped_local_bbox(defect.get("bbox_local"), annotated)
        if bbox is not None:
            x, y, width, height = bbox
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), max(line_width + 1, 3))

        debug_pair = metadata.get("debug_pair") or {}
        Reporter._draw_900_edge_gaps(annotated, debug_pair, line_width)

        lines = Reporter._detector_900_debug_lines(defect)
        panel_x = max(10, annotated.shape[1] - 430)
        Reporter._draw_text_panel(annotated, lines, origin=(panel_x, 10))

    @staticmethod
    def _draw_900_candidate_group(
        annotated,
        candidates: object,
        color: tuple[int, int, int],
        prefix: str,
        line_width: int,
        label_y_offset: int = 0,
    ) -> None:
        if not isinstance(candidates, list):
            return
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            bbox = Reporter._clipped_local_bbox(candidate.get("bbox"), annotated)
            if bbox is None:
                continue
            x, y, width, height = bbox
            thickness = max(1, line_width - 1) if index > 1 else max(line_width, 2)
            cv2.rectangle(annotated, (x, y), (x + width, y + height), color, thickness)
            reject_reason = str(candidate.get("reject_reason", ""))
            reject_suffix = f" {reject_reason}" if reject_reason else ""
            label = f"{prefix}{index}{reject_suffix} {width}x{height}"
            Reporter._draw_label(annotated, label, x, y - 6 + label_y_offset, color)

    @staticmethod
    def _draw_900_edge_gaps(annotated, debug_pair: dict, line_width: int) -> None:
        outer = debug_pair.get("outer") if isinstance(debug_pair, dict) else None
        inner = debug_pair.get("inner") if isinstance(debug_pair, dict) else None
        edge_gaps = debug_pair.get("edge_gaps") if isinstance(debug_pair, dict) else None
        if not isinstance(outer, dict) or not isinstance(inner, dict) or not isinstance(edge_gaps, dict):
            return

        outer_bbox = Reporter._clipped_local_bbox(outer.get("bbox"), annotated)
        inner_bbox = Reporter._clipped_local_bbox(inner.get("bbox"), annotated)
        if outer_bbox is None or inner_bbox is None:
            return

        ox, oy, ow, oh = outer_bbox
        ix, iy, iw, ih = inner_bbox
        color = (0, 255, 255) if debug_pair.get("edge_gap_pass") else (0, 165, 255)
        thickness = max(1, line_width)
        segments = [
            ((ox, iy + ih // 2), (ix, iy + ih // 2), f"L{edge_gaps.get('left')}"),
            ((ix + iw, iy + ih // 2), (ox + ow, iy + ih // 2), f"R{edge_gaps.get('right')}"),
            ((ix + iw // 2, oy), (ix + iw // 2, iy), f"T{edge_gaps.get('top')}"),
            ((ix + iw // 2, iy + ih), (ix + iw // 2, oy + oh), f"B{edge_gaps.get('bottom')}"),
        ]
        for start, end, label in segments:
            cv2.line(annotated, start, end, color, thickness)
            label_x = int((start[0] + end[0]) / 2)
            label_y = int((start[1] + end[1]) / 2)
            Reporter._draw_label(annotated, label, label_x, label_y, color)

    @staticmethod
    def _detector_900_debug_lines(defect: dict) -> list[str]:
        metadata = defect.get("metadata", {})
        debug_pair = metadata.get("debug_pair") or {}
        edge_gaps = debug_pair.get("edge_gaps") if isinstance(debug_pair, dict) else None
        lines = [
            "Detector 900 NG debug",
            f"reason: {metadata.get('reason', '')}",
            (
                "outer pass/raw/fail: "
                f"{metadata.get('outer_candidate_count', 0)}/"
                f"{metadata.get('outer_raw_candidate_count', 0)}/"
                f"{metadata.get('outer_rejected_candidate_count', 0)}"
            ),
            (
                "inner pass/raw/fail: "
                f"{metadata.get('inner_candidate_count', 0)}/"
                f"{metadata.get('inner_raw_candidate_count', 0)}/"
                f"{metadata.get('inner_rejected_candidate_count', 0)}"
            ),
            (
                "target outer: "
                f"{metadata.get('outer_target_width')}+-{metadata.get('outer_width_tolerance')} x "
                f"{metadata.get('outer_target_height')}+-{metadata.get('outer_height_tolerance')}"
            ),
            (
                "target inner: "
                f"{metadata.get('inner_target_width')}+-{metadata.get('inner_width_tolerance')} x "
                f"{metadata.get('inner_target_height')}+-{metadata.get('inner_height_tolerance')}"
            ),
            f"max gap: {metadata.get('max_edge_gap')}",
        ]
        if isinstance(edge_gaps, dict):
            lines.append(
                "gaps L/T/R/B: "
                f"{edge_gaps.get('left')}/{edge_gaps.get('top')}/{edge_gaps.get('right')}/{edge_gaps.get('bottom')}"
            )
        return lines

    @staticmethod
    def _draw_text_panel(annotated, lines: list[str], origin: tuple[int, int]) -> None:
        if not lines:
            return
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.58
        thickness = 2
        line_height = 22
        max_width = 0
        for line in lines:
            size, _ = cv2.getTextSize(str(line), font, scale, thickness)
            max_width = max(max_width, size[0])
        panel_width = min(annotated.shape[1] - x - 1, max_width + 18)
        panel_height = min(annotated.shape[0] - y - 1, line_height * len(lines) + 12)
        cv2.rectangle(annotated, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(annotated, (x, y), (x + panel_width, y + panel_height), (255, 255, 255), 1)
        for index, line in enumerate(lines):
            text_y = y + 22 + index * line_height
            if text_y >= annotated.shape[0]:
                break
            cv2.putText(annotated, str(line), (x + 8, text_y), font, scale, (255, 255, 255), thickness)

    @staticmethod
    def _draw_label(annotated, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 2
        height, width = annotated.shape[:2]
        text = str(label)
        size, baseline = cv2.getTextSize(text, font, scale, thickness)
        text_x = max(0, min(width - size[0] - 4, int(x)))
        text_y = max(size[1] + 4, min(height - baseline - 2, int(y)))
        cv2.rectangle(
            annotated,
            (text_x - 2, text_y - size[1] - 4),
            (text_x + size[0] + 4, text_y + baseline + 3),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(annotated, text, (text_x, text_y), font, scale, color, thickness)

    @staticmethod
    def _fmt_num(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if abs(number) >= 1000:
            return f"{number:.0f}"
        return f"{number:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _ng_tile_line_width(image) -> int:
        return 2

    @staticmethod
    def _clipped_local_bbox(bbox: object, image) -> tuple[int, int, int, int] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None

        height, width = image.shape[:2]
        try:
            x, y, box_width, box_height = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            return None

        x1 = max(0, min(width - 1, x))
        y1 = max(0, min(height - 1, y))
        x2 = max(0, min(width - 1, x + max(1, box_width)))
        y2 = max(0, min(height - 1, y + max(1, box_height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _write_png(path: Path, image) -> None:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError(f"Cannot write empty PNG image: {path}")

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError(f"OpenCV failed to encode PNG image: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(encoded.tobytes())
        except OSError as exc:
            raise OSError(f"Failed to write PNG image to {path}: {exc}") from exc

    @staticmethod
    def _write_csv(path: Path, result: dict) -> None:
        fields = [
            "image_name",
            "recipe_name",
            "machine_id",
            "product_id",
            "final_result",
            "detector_id",
            "defect_type",
            "bbox_global",
            "bbox_local",
            "tile_id",
            "score",
            "area",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for tile_result in result["tiles"]:
                for detector_result in tile_result["detectors"]:
                    for defect in detector_result.get("defects", []):
                        writer.writerow(
                            {
                                "image_name": result["image_name"],
                                "recipe_name": result["recipe_name"],
                                "machine_id": result["machine_id"],
                                "product_id": result["product_id"],
                                "final_result": result["final_result"],
                                "detector_id": detector_result["detector_id"],
                                "defect_type": defect["type"],
                                "bbox_global": defect.get("bbox_global"),
                                "bbox_local": defect.get("bbox_local"),
                                "tile_id": defect.get("tile_id"),
                                "score": detector_result.get("score"),
                                "area": defect.get("area"),
                            }
                        )

    @staticmethod
    def _write_matrix_csv(path: Path, result: dict) -> None:
        tiles = result.get("tiles", [])
        check_mark = "\u2713"
        max_row = max(
            (Reporter._safe_int(tile_result.get("tile", {}).get("row", 0)) for tile_result in tiles),
            default=0,
        )
        max_col = max(
            (Reporter._safe_int(tile_result.get("tile", {}).get("col", 0)) for tile_result in tiles),
            default=0,
        )
        fields = ["id", *[f"c{col + 1}" for col in range(max_col + 1)]]
        image_stem = Path(str(result.get("image_name", ""))).stem

        matrix_rows: dict[int, dict[str, str]] = {
            row: {"id": f"{image_stem}-{max_row - row + 1}", **{field: "" for field in fields[1:]}}
            for row in range(max_row + 1)
        }
        for tile_result in tiles:
            tile = tile_result.get("tile", {})
            row = Reporter._safe_int(tile.get("row", 0))
            col = Reporter._safe_int(tile.get("col", 0))
            if row not in matrix_rows:
                matrix_rows[row] = {"id": f"{image_stem}-{max_row - row + 1}", **{field: "" for field in fields[1:]}}
            if tile_result.get("result") == "NG":
                column_name = f"c{col + 1}"
                if column_name in matrix_rows[row]:
                    matrix_rows[row][column_name] = check_mark

        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in sorted(matrix_rows):
                writer.writerow(matrix_rows[row])

    @staticmethod
    def _safe_int(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
