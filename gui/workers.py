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

    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)
        self.image_loader = ImageLoader()

    @Slot()
    def run(self) -> None:
        try:
            image = self.image_loader.load_rgb(self.path)
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

        self.loaded.emit(self.path, qimage)


class InspectionWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, image_path: Path, recipe_path: Path, output_dir: Path):
        super().__init__()
        self.image_path = Path(image_path)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)

    @Slot()
    def run(self) -> None:
        try:
            pipeline = AOIPipeline(
                recipe_path=self.recipe_path,
                output_dir=self.output_dir,
            )
            result = pipeline.run(self.image_path)
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(result)


class ContourTilePreviewWorker(QObject):
    finished = Signal(object, int, dict)
    failed = Signal(str)

    def __init__(self, image_path: Path, tile_config: dict):
        super().__init__()
        self.image_path = Path(image_path)
        self.tile_config = dict(tile_config)
        self.image_loader = ImageLoader()

    @Slot()
    def run(self) -> None:
        try:
            image = self.image_loader.load_bgr(self.image_path)
            tiler = create_tiler(self.tile_config)
            tiles = list(tiler.iter_tiles(image))
            preview = self._draw_tiles(image, tiles)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            qimage = QImage(
                rgb.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            ).copy()
            shape_counts: dict[str, int] = {}
            for tile in tiles:
                shape = (tile.metadata or {}).get("shape", "unknown")
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.finished.emit(qimage, len(tiles), shape_counts)

    @staticmethod
    def _draw_tiles(image, tiles):
        preview = image.copy()
        colors = {
            "rectangle": (0, 180, 0),
            "circle": (255, 120, 0),
            "polygon": (180, 0, 180),
            "unknown": (0, 0, 255),
        }
        for tile in tiles:
            metadata = tile.metadata or {}
            shape = metadata.get("shape", "unknown")
            color = colors.get(shape, colors["unknown"])
            cv2.rectangle(preview, (tile.x, tile.y), (tile.x + tile.width, tile.y + tile.height), color, 2)
            label = f"{tile.tile_id}"
            cv2.putText(preview, label, (tile.x, max(0, tile.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            vertices = metadata.get("vertices") or []
            if vertices:
                points = np.array(vertices, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(preview, [points], True, color, 1)
        return preview
