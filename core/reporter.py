from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from core.logging_system import LogMixin


class Reporter(LogMixin):
    def __init__(self, output_dir: Path, output_config: dict):
        self.output_dir = Path(output_dir)
        self.output_config = output_config or {}
        self.overlay_dir = self.output_dir / "overlay"
        self.ng_tiles_dir = self.output_dir / "ng_tiles"
        self.csv_dir = self.output_dir / "csv"
        self.matrix_csv_dir = self.output_dir / "matrix_csv"
        self.json_dir = self.output_dir / "json"
        for directory in (self.overlay_dir, self.ng_tiles_dir, self.csv_dir, self.matrix_csv_dir, self.json_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def write(self, image, result: dict) -> dict[str, str]:
        stem = Path(result["image_name"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{stem}_{result['recipe_name']}_{timestamp}_{uuid.uuid4().hex[:8]}"
        outputs: dict[str, str] = {}
        self.logger.info("Writing report outputs: image=%s base=%s", result.get("image_name"), base_name)

        if self.output_config.get("save_overlay", True):
            overlay_path = self.overlay_dir / f"{base_name}_overlay.png"
            self._write_png(overlay_path, self._make_overlay(image, result))
            outputs["overlay"] = str(overlay_path)

        if self.output_config.get("save_ng_tiles", True):
            self._write_ng_tiles(result, base_name)
            outputs["ng_tiles_dir"] = str(self.ng_tiles_dir)

        if self.output_config.get("save_csv", True):
            csv_path = self.csv_dir / f"{base_name}.csv"
            self._write_csv(csv_path, result)
            outputs["csv"] = str(csv_path)

        if self.output_config.get("save_matrix_csv", True):
            matrix_csv_path = self.matrix_csv_dir / f"{base_name}_matrix.csv"
            self._write_matrix_csv(matrix_csv_path, result)
            outputs["matrix_csv"] = str(matrix_csv_path)

        if self.output_config.get("save_json", True):
            json_path = self.json_dir / f"{base_name}.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(self._json_safe_result(result, outputs), handle, ensure_ascii=False, indent=2)
            outputs["json"] = str(json_path)

        self.logger.info("Report outputs written: outputs=%s", outputs)
        return outputs

    @staticmethod
    def _json_safe_result(result: dict, outputs: dict[str, str]) -> dict:
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

    def _write_ng_tiles(self, result: dict, base_name: str) -> None:
        for tile_result in result["tiles"]:
            if tile_result.get("result") != "NG":
                continue
            tile = tile_result["tile"]
            tile_image = tile_result.get("_tile_image")
            if tile_image is None:
                continue
            path = self.ng_tiles_dir / f"{base_name}_{tile['tile_id']}.png"
            self._write_png(path, tile_image)

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
