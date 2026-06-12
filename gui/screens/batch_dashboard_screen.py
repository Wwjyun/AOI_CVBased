from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.batch_dashboard import BatchDashboardBuilder, BatchDashboardModel
from gui import icons
from gui.theme import COLORS
from gui.widgets.common import EmptyState
from gui.widgets.panel import Panel


RESULT_COLORS = {
    "PASS": COLORS["pass"],
    "NG": COLORS["ng"],
    "ERROR": COLORS["warn"],
}


class ResultDonutChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._distribution: list[tuple[str, int]] = []
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_distribution(self, distribution: list[tuple[str, int]]) -> None:
        self._distribution = distribution
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(18, 18, min(self.width(), self.height()) - 36, min(self.width(), self.height()) - 36)
        rect.moveCenter(self.rect().center())
        total = sum(value for _name, value in self._distribution)

        pen = QPen(QColor(COLORS["surface_3"]))
        pen.setWidth(22)
        painter.setPen(pen)
        painter.drawArc(rect, 0, 360 * 16)

        if total:
            start_angle = 90 * 16
            for name, value in self._distribution:
                if value <= 0:
                    continue
                span = int(-360 * 16 * (value / total))
                pen = QPen(QColor(RESULT_COLORS.get(name, COLORS["text_3"])))
                pen.setWidth(22)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawArc(rect, start_angle, span)
                start_angle += span

        painter.setPen(QColor(COLORS["text"]))
        font = QFont("Microsoft JhengHei UI")
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(total))

        font.setPointSize(10)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(COLORS["text_3"]))
        label_rect = QRectF(rect.left(), rect.center().y() + 18, rect.width(), 24)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "images")


class DefectBarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []
        self.setMinimumHeight(220)

    def set_rows(self, rows: list[dict]) -> None:
        self._rows = rows
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["surface"]))
        if not self._rows:
            painter.setPen(QColor(COLORS["text_3"]))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No defect data")
            return

        left = 18
        right = 18
        top = 18
        bar_h = 18
        gap = 10
        label_w = 112
        max_defects = max(int(row.get("defect_count", 0) or 0) for row in self._rows) or 1
        width = max(1, self.width() - left - right - label_w - 42)

        font = QFont("Consolas")
        font.setPointSize(9)
        painter.setFont(font)
        for index, row in enumerate(self._rows[:8]):
            y = top + index * (bar_h + gap)
            name = str(row.get("image_name", "-"))
            defects = int(row.get("defect_count", 0) or 0)
            value_w = int(width * defects / max_defects)

            painter.setPen(QColor(COLORS["text_2"]))
            painter.drawText(QRectF(left, y - 1, label_w, bar_h + 2), Qt.AlignmentFlag.AlignVCenter, name[:18])

            track = QRectF(left + label_w, y, width, bar_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["surface_3"]))
            painter.drawRoundedRect(track, 4, 4)
            painter.setBrush(QColor(COLORS["ng"] if defects else COLORS["pass"]))
            painter.drawRoundedRect(QRectF(track.left(), track.top(), max(2, value_w), bar_h), 4, 4)

            painter.setPen(QColor(COLORS["text"]))
            painter.drawText(
                QRectF(track.right() + 8, y - 1, 34, bar_h + 2),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                str(defects),
            )


class BatchDashboardScreen(QWidget):
    go_to_run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = BatchDashboardBuilder(None).build()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._empty = self._build_empty_state()
        self._content = self._build_content()
        layout.addWidget(self._empty, 1)
        layout.addWidget(self._content, 1)
        self._show_empty(True)

    def _build_empty_state(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        go_button = QPushButton("Go to Batch")
        go_button.setProperty("variant", "primary")
        go_button.setProperty("size", "sm")
        go_button.setIcon(icons.icon("play", size=14, color="#ffffff"))
        go_button.clicked.connect(self.go_to_run_requested.emit)
        layout.addWidget(
            EmptyState(
                "table",
                "No batch data yet",
                "Run a folder batch inspection first, then review dashboard statistics here.",
                action=go_button,
            )
        )
        return wrapper

    def _build_content(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.output_label = QLabel("")
        self.output_label.setProperty("mono", "true")
        self.output_label.setStyleSheet(f"color: {COLORS['text_3']};")
        layout.addWidget(self.output_label)

        metric_grid = QGridLayout()
        metric_grid.setSpacing(12)
        layout.addLayout(metric_grid)
        self.total_value, total_card = _metric_card("Total Images")
        self.pass_rate_value, pass_card = _metric_card("Pass Rate", COLORS["pass"])
        self.ng_rate_value, ng_card = _metric_card("NG Rate", COLORS["ng"])
        self.avg_defects_value, avg_card = _metric_card("Avg Defects")
        metric_grid.addWidget(total_card, 0, 0)
        metric_grid.addWidget(pass_card, 0, 1)
        metric_grid.addWidget(ng_card, 0, 2)
        metric_grid.addWidget(avg_card, 0, 3)

        chart_row = QHBoxLayout()
        chart_row.setSpacing(12)
        layout.addLayout(chart_row, 1)

        distribution_panel = Panel(title="Result Distribution")
        self.donut_chart = ResultDonutChart()
        distribution_panel.add_widget(self.donut_chart, 1)
        chart_row.addWidget(distribution_panel, 1)

        defect_panel = Panel(title="Top Defect Images")
        self.defect_chart = DefectBarChart()
        defect_panel.add_widget(self.defect_chart, 1)
        chart_row.addWidget(defect_panel, 2)

        table_panel = Panel(title="Batch Image Data", flush=True)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Image", "Result", "Defects", "NG Tiles", "Error"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_panel.add_widget(self.table, 1)
        layout.addWidget(table_panel, 1)

        return wrapper

    def set_batch_result(self, batch_result: dict | None) -> None:
        self._model = BatchDashboardBuilder(batch_result).build()
        self._show_empty(self._model.total == 0)
        if self._model.total == 0:
            return
        self._render_model(self._model)

    def _render_model(self, model: BatchDashboardModel) -> None:
        self.output_label.setText(model.output_dir)
        self.total_value.setText(str(model.total))
        self.pass_rate_value.setText(f"{model.pass_rate:.1f}%")
        self.ng_rate_value.setText(f"{model.ng_rate:.1f}%")
        self.avg_defects_value.setText(f"{model.avg_defects:.2f}")
        self.donut_chart.set_distribution(model.result_distribution)
        self.defect_chart.set_rows(model.top_defect_images)
        self._populate_table(model.rows)

    def _populate_table(self, rows: list[dict]) -> None:
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                str(row.get("image_name", "")),
                str(row.get("final_result", "")),
                str(row.get("defect_count", 0)),
                str(row.get("ng_count", 0)),
                str(row.get("error", "")),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (2, 3):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 0:
                    item.setToolTip(str(row.get("image_path", "")))
                self.table.setItem(row_index, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _show_empty(self, show: bool) -> None:
        self._empty.setVisible(show)
        self._content.setVisible(not show)


def _metric_card(title: str, color: str | None = None) -> tuple[QLabel, QFrame]:
    card = QFrame()
    card.setProperty("role", "panel")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(4)

    title_label = QLabel(title)
    title_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px; font-weight: 600;")
    layout.addWidget(title_label)

    value_label = QLabel("0")
    value_label.setProperty("mono", "true")
    value_label.setStyleSheet(f"font-size: 24px; font-weight: 800; color: {color or COLORS['text']};")
    layout.addWidget(value_label)
    return value_label, card
