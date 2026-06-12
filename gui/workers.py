from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

import cv2
import numpy as np

from core.image_loader import ImageLoader
from core.pipeline import AOIPipeline
from core.tiler import create_tiler


class ImagePreviewWorker(QObject):
    loaded = Signal(Path, object)
    failed = Signal(Path, str)
    progress = Signal(int, str)

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self.image_loader = ImageLoader()

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(0, "Loading image")
            image = self.image_loader.load_rgb(self.path)
            self.progress.emit(60, "Converting preview")
            height, width, channels = image.shape
            qimage = QImage(
                image.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            ).copy()
        except Exception as exc:
            self.failed.emit(self.path, str(exc))
            return

        self.progress.emit(100, "Preview ready")
        self.loaded.emit(self.path, qimage)


class InspectionWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, image_path: Path, recipe_path: Path, output_dir: Path, output_overrides: dict | None = None):
        super().__init__()
        self.image_path = Path(image_path)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides

    @Slot()
    def run(self) -> None:
        try:
            pipeline = AOIPipeline(
                recipe_path=self.recipe_path,
                output_dir=self.output_dir,
                progress_callback=self.progress.emit,
                output_overrides=self.output_overrides,
            )
            result = pipeline.run(self.image_path)
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(result)


class TilePreviewWorker(QObject):
    finished = Signal(object, int, dict)
    failed = Signal(str)
    progress = Signal(int, str)
    MAX_PREVIEW_SIDE = 2200

    def __init__(self, image_path: Path, tile_config: dict):
        super().__init__()
        self.image_path = Path(image_path)
        self.tile_config = dict(tile_config)
        self.image_loader = ImageLoader()

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(0, "Loading image for tile preview")
            image = self.image_loader.load_bgr(self.image_path)
            self.progress.emit(20, "Creating tiler")
            tiler = create_tiler(self.tile_config)
            tiles = list(tiler.iter_tiles(image))
            self.progress.emit(60, f"Drawing {len(tiles)} preview tiles")
            preview = self._draw_tiles(image, tiles)
            preview = self._resize_preview(preview)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            self.progress.emit(80, "Converting tile preview")
            height, width, channels = rgb.shape
            qimage = QImage(
                rgb.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            ).copy()
            shape_counts: dict[str, int] = {}
            best_score = None
            for tile in tiles:
                metadata = tile.metadata or {}
                mode = metadata.get("mode", "unknown")
                key = metadata.get("shape", mode)
                shape_counts[key] = shape_counts.get(key, 0) + 1
                if metadata.get("score") is not None:
                    score = float(metadata["score"])
                    best_score = score if best_score is None else max(best_score, score)
            shape_counts["best_score"] = best_score
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.progress.emit(100, "Tile preview ready")
        self.finished.emit(qimage, len(tiles), shape_counts)

    @staticmethod
    def _draw_tiles(image, tiles):
        preview = image.copy()
        colors = {
            "rectangle": (0, 180, 0),
            "circle": (255, 120, 0),
            "polygon": (180, 0, 180),
            "pattern_match": (0, 180, 255),
            "unknown": (0, 0, 255),
        }
        for tile in tiles:
            metadata = tile.metadata or {}
            shape = metadata.get("shape", metadata.get("mode", "unknown"))
            color = colors.get(shape, colors["unknown"])
            cv2.rectangle(preview, (tile.x, tile.y), (tile.x + tile.width, tile.y + tile.height), color, 4)
            score = metadata.get("score")
            label = f"{tile.tile_id}" if score is None else f"{tile.tile_id}:{score:.3f}"
            cv2.putText(preview, label, (tile.x, max(0, tile.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            match_bbox = metadata.get("match_bbox")
            if match_bbox:
                x, y, width, height = match_bbox
                cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 255), 3)

            vertices = metadata.get("vertices") or []
            if vertices:
                points = np.array(vertices, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(preview, [points], True, color, 3)
        return preview

    @classmethod
    def _resize_preview(cls, preview):
        height, width = preview.shape[:2]
        longest_side = max(width, height)
        if longest_side <= cls.MAX_PREVIEW_SIDE:
            return preview
        scale = cls.MAX_PREVIEW_SIDE / float(longest_side)
        target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(preview, target_size, interpolation=cv2.INTER_AREA)
