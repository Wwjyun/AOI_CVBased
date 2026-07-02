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
    def __init__(
        self,
        parent=None,
        x_label: str = "tile x",
        y_label: str = "tile y",
        empty_text: str = "No tile points",
        y_origin_bottom: bool = False,
        defect_radius_scale: int = 8,
    ):
        super().__init__(parent)
        self._model = ImageScatterModel("", 0.0, 0.0, [])
        self._x_label = x_label
        self._y_label = y_label
        self._empty_text = empty_text
        self._y_origin_bottom = y_origin_bottom
        self._defect_radius_scale = max(0, int(defect_radius_scale))
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
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_text)
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
            y_ratio = float(point.y) / height
            if self._y_origin_bottom:
                py = plot.bottom() - y_ratio * plot.height()
            else:
                py = plot.top() + y_ratio * plot.height()
            radius = 4 + min(self._defect_radius_scale, int(point.defect_count))
            color = RESULT_COLORS.get(point.status, COLORS["text_3"])
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QPointF(px, py), radius, radius)

        painter.setPen(QColor(COLORS["text_3"]))
        font = QFont("Consolas")
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(QRectF(plot.left(), plot.bottom() + 6, plot.width(), 18), Qt.AlignmentFlag.AlignCenter, self._x_label)
        painter.save()
        painter.translate(10, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot.height() / 2, 0, plot.height(), 18), Qt.AlignmentFlag.AlignCenter, self._y_label)
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
