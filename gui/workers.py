from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

from core.image_loader import ImageLoader
from core.pipeline import AOIPipeline


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
