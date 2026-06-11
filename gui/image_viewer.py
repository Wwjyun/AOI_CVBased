from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from core.image_loader import ImageLoadError, ImageLoader


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_loader = ImageLoader()
        self.last_error = ""
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self._zoom = 0

    def load_image(self, path: Path) -> bool:
        try:
            pixmap = self._load_pixmap(path)
        except ImageLoadError as exc:
            self.last_error = str(exc)
            return False
        self.last_error = ""
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 0
        return True

    def _load_pixmap(self, path: Path) -> QPixmap:
        image = self.image_loader.load_rgb(path)
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.isNull():
            raise ImageLoadError(f"Qt failed to create preview pixmap: {path}")
        return pixmap

    def clear(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._zoom = 0

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap_item.pixmap().isNull():
            super().wheelEvent(event)
            return

        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom += 1 if factor > 1 else -1
        if self._zoom < -8:
            self._zoom = -8
            return
        if self._zoom > 20:
            self._zoom = 20
            return
        self.scale(factor, factor)
