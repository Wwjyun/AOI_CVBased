from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from core.batch_dashboard import ImageScatterModel
from gui.theme import COLORS


RESULT_COLORS = {
    "PASS": COLORS["pass"],
    "NG": COLORS["ng"],
    "ERROR": COLORS["warn"],
}


class ImageScatterChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = ImageScatterModel("", 0.0, 0.0, [])
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_model(self, model: ImageScatterModel) -> None:
        self._model = model
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["surface"]))

        if not self._model.points:
            painter.setPen(QColor(COLORS["text_3"]))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No tile points")
            return

        plot = QRectF(42, 18, max(1, self.width() - 64), max(1, self.height() - 52))
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawRect(plot)

        width = max(float(self._model.width), 1.0)
        height = max(float(self._model.height), 1.0)

        grid_pen = QPen(QColor(COLORS["surface_3"]), 1)
        painter.setPen(grid_pen)
        for index in range(1, 4):
            x = plot.left() + plot.width() * index / 4
            y = plot.top() + plot.height() * index / 4
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        for point in self._model.points:
            px = plot.left() + (float(point.x) / width) * plot.width()
            py = plot.top() + (float(point.y) / height) * plot.height()
            radius = 4 + min(8, int(point.defect_count))
            color = RESULT_COLORS.get(point.status, COLORS["text_3"])
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(px, py), radius, radius)

        painter.setPen(QColor(COLORS["text_3"]))
        font = QFont("Consolas")
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(QRectF(plot.left(), plot.bottom() + 6, plot.width(), 18), Qt.AlignmentFlag.AlignCenter, "tile x")
        painter.save()
        painter.translate(10, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot.height() / 2, 0, plot.height(), 18), Qt.AlignmentFlag.AlignCenter, "tile y")
        painter.restore()

        legend_y = plot.top() + 6
        legend_x = plot.right() - 126
        for index, status in enumerate(("PASS", "NG", "ERROR")):
            color = RESULT_COLORS.get(status, COLORS["text_3"])
            x = legend_x + index * 44
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(x, legend_y + 6), 4, 4)
            painter.setPen(QColor(COLORS["text_2"]))
            painter.drawText(QRectF(x + 7, legend_y, 38, 14), Qt.AlignmentFlag.AlignVCenter, status)
