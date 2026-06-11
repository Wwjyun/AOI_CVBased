from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import cv2


class Reporter:
    def __init__(self, output_dir: Path, output_config: dict):
        self.output_dir = Path(output_dir)
        self.output_config = output_config or {}
        self.overlay_dir = self.output_dir / "overlay"
        self.ng_tiles_dir = self.output_dir / "ng_tiles"
        self.csv_dir = self.output_dir / "csv"
        self.json_dir = self.output_dir / "json"
        for directory in (self.overlay_dir, self.ng_tiles_dir, self.csv_dir, self.json_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def write(self, image, result: dict) -> dict[str, str]:
        stem = Path(result["image_name"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{stem}_{result['recipe_name']}_{timestamp}"
        outputs: dict[str, str] = {}

        if self.output_config.get("save_overlay", True):
            overlay_path = self.overlay_dir / f"{base_name}_overlay.png"
            cv2.imwrite(str(overlay_path), self._make_overlay(image, result))
            outputs["overlay"] = str(overlay_path)

        if self.output_config.get("save_ng_tiles", True):
            self._write_ng_tiles(result, base_name)
            outputs["ng_tiles_dir"] = str(self.ng_tiles_dir)

        if self.output_config.get("save_csv", True):
            csv_path = self.csv_dir / f"{base_name}.csv"
            self._write_csv(csv_path, result)
            outputs["csv"] = str(csv_path)

        if self.output_config.get("save_json", True):
            json_path = self.json_dir / f"{base_name}.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(self._json_safe_result(result, outputs), handle, ensure_ascii=False, indent=2)
            outputs["json"] = str(json_path)

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
        for tile_result in result["tiles"]:
            for detector_result in tile_result["detectors"]:
                for defect in detector_result.get("defects", []):
                    Reporter._draw_defect(overlay, defect)
                    x, y, _, _ = defect["bbox_global"]
                    label = f"{detector_result['detector_id']}:{defect['type']}"
                    cv2.putText(overlay, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return overlay

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
            cv2.imwrite(str(path), tile_image)

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
