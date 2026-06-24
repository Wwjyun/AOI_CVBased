from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QHeaderView,
    QSizePolicy,
    QSplitter,
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
from gui.widgets.scatter_chart import ImageScatterChart, RESULT_COLORS


def _format_duration(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "-"
    if seconds <= 0:
        return "-"
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.1f}s"


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
        self._selected_row: dict | None = None

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
        self.pass_rate_value, pass_card = _metric_card("Image Pass Rate", COLORS["pass"])
        self.tile_pass_rate_value, tile_pass_card = _metric_card("Tile Pass Rate", COLORS["pass"])
        self.tile_summary_value, tile_summary_card = _metric_card("Tiles PASS / NG")
        self.avg_defects_value, avg_card = _metric_card("Avg Defects")
        metric_grid.addWidget(total_card, 0, 0)
        metric_grid.addWidget(pass_card, 0, 1)
        metric_grid.addWidget(tile_pass_card, 0, 2)
        metric_grid.addWidget(avg_card, 0, 3)
        metric_grid.addWidget(tile_summary_card, 1, 0, 1, 4)

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

        data_splitter = QSplitter(Qt.Orientation.Horizontal)
        data_splitter.setChildrenCollapsible(False)

        table_panel = Panel(title="Batch Image Data", flush=True)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Image", "Result", "Tiles", "PASS Tiles", "NG Tiles", "Tile Pass", "Defects", "Time", "Error"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        table_panel.add_widget(self.table, 1)
        data_splitter.addWidget(table_panel)

        detail_panel = Panel(title="Selected Image Detail", flush=True)
        detail_body = QWidget()
        detail_layout = QVBoxLayout(detail_body)
        detail_layout.setContentsMargins(12, 12, 12, 12)
        detail_layout.setSpacing(10)

        self.detail_title = QLabel("Select an image row")
        self.detail_title.setWordWrap(True)
        self.detail_title.setStyleSheet("font-size: 15px; font-weight: 700;")
        detail_layout.addWidget(self.detail_title)

        self.detail_summary = QLabel("")
        self.detail_summary.setProperty("mono", "true")
        self.detail_summary.setWordWrap(True)
        self.detail_summary.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 11px;")
        detail_layout.addWidget(self.detail_summary)

        self.detail_outputs = QLabel("")
        self.detail_outputs.setProperty("mono", "true")
        self.detail_outputs.setWordWrap(True)
        self.detail_outputs.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 10px;")
        detail_layout.addWidget(self.detail_outputs)

        detail_splitter = QSplitter(Qt.Orientation.Horizontal)
        detail_splitter.setChildrenCollapsible(False)

        self.detail_table = QTableWidget(0, 7)
        self.detail_table.setHorizontalHeaderLabels(
            ["Tile", "Tile Result", "Detector", "Detector Result", "Score", "Defects", "Defect Detail"]
        )
        self.detail_table.verticalHeader().setVisible(False)
        self.detail_table.setShowGrid(False)
        self.detail_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        detail_splitter.addWidget(self.detail_table)

        scatter_panel = Panel(title="Selected Tile Scatter")
        self.scatter_chart = ImageScatterChart()
        scatter_panel.add_widget(self.scatter_chart, 1)
        detail_splitter.addWidget(scatter_panel)
        detail_splitter.setSizes([520, 320])
        detail_layout.addWidget(detail_splitter, 1)
        detail_panel.add_widget(detail_body, 1)
        data_splitter.addWidget(detail_panel)
        data_splitter.setSizes([780, 420])
        layout.addWidget(data_splitter, 1)

        return wrapper

    def set_batch_result(self, batch_result: dict | None) -> None:
        self._model = BatchDashboardBuilder(batch_result).build()
        self._selected_row = None
        self._show_empty(self._model.total == 0)
        if self._model.total == 0:
            return
        self._render_model(self._model)

    def _render_model(self, model: BatchDashboardModel) -> None:
        self.output_label.setText(f"{model.output_dir}\nTotal time: {_format_duration(model.duration_sec)}")
        self.total_value.setText(str(model.total))
        self.pass_rate_value.setText(f"{model.pass_rate:.1f}%")
        self.tile_pass_rate_value.setText(f"{model.tile_pass_rate:.1f}%")
        self.tile_summary_value.setText(f"{model.pass_tile_count} / {model.ng_tile_count}")
        self.avg_defects_value.setText(f"{model.avg_defects:.2f}")
        self.donut_chart.set_distribution(model.result_distribution)
        self.defect_chart.set_rows(model.top_defect_images)
        self._populate_table(model.rows)
        if model.rows:
            self.table.selectRow(0)
        else:
            self._render_detail(None)

    def _populate_table(self, rows: list[dict]) -> None:
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                str(row.get("image_name", "")),
                str(row.get("final_result", "")),
                str(row.get("tile_count", 0)),
                str(row.get("pass_tile_count", 0)),
                str(row.get("ng_count", 0)),
                f"{float(row.get('tile_pass_rate', 0) or 0):.1f}%",
                str(row.get("defect_count", 0)),
                _format_duration(row.get("duration_sec")),
                str(row.get("error", "")),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (2, 3, 4, 5, 6, 7):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if col == 0:
                    item.setToolTip(str(row.get("image_path", "")))
                self.table.setItem(row_index, col, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

    def _on_table_selection_changed(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self._render_detail(None)
            return
        row_index = selected[0].row()
        rows = self._model.rows
        self._render_detail(rows[row_index] if 0 <= row_index < len(rows) else None)

    def _render_detail(self, row: dict | None) -> None:
        self._selected_row = row
        if not row:
            self.detail_title.setText("Select an image row")
            self.detail_summary.setText("")
            self.detail_outputs.setText("")
            self.detail_table.setRowCount(0)
            self.scatter_chart.set_model(BatchDashboardBuilder.build_image_scatter(None))
            return

        self.detail_title.setText(str(row.get("image_name", "-")))
        self.detail_summary.setText(
            "Result {result} | Time {duration} | Tiles {tiles} | PASS {pass_tiles} | NG {ng_tiles} | "
            "Tile pass {tile_pass:.1f}% | Defects {defects}".format(
                result=row.get("final_result", "-"),
                duration=_format_duration(row.get("duration_sec")),
                tiles=int(row.get("tile_count", 0) or 0),
                pass_tiles=int(row.get("pass_tile_count", 0) or 0),
                ng_tiles=int(row.get("ng_count", 0) or 0),
                tile_pass=float(row.get("tile_pass_rate", 0) or 0),
                defects=int(row.get("defect_count", 0) or 0),
            )
        )
        outputs = row.get("outputs", {}) or {}
        output_lines = [f"{key}: {value}" for key, value in outputs.items()]
        if row.get("error"):
            output_lines.insert(0, f"error: {row.get('error')}")
        self.detail_outputs.setText("\n".join(output_lines))
        self._populate_detail_table(row)
        self.scatter_chart.set_model(BatchDashboardBuilder.build_image_scatter(row))

    def _populate_detail_table(self, row: dict) -> None:
        detail = row.get("detail", {}) or {}
        rows: list[list[str]] = []
        for tile_result in detail.get("tiles", []):
            tile = tile_result.get("tile", {}) or {}
            tile_id = str(tile.get("tile_id", "-"))
            tile_status = str(tile_result.get("result", "-"))
            detectors = tile_result.get("detectors", []) or []
            if not detectors:
                rows.append([tile_id, tile_status, "-", "-", "-", "0", "-"])
                continue
            for detector in detectors:
                defects = detector.get("defects", []) or []
                detector_status = "PASS" if detector.get("pass", True) else "NG"
                rows.append(
                    [
                        tile_id,
                        tile_status,
                        str(detector.get("detector_id", "-")),
                        detector_status,
                        _format_score(detector.get("score")),
                        str(len(defects)),
                        _defect_summary(defects),
                    ]
                )

        self.detail_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (4, 5):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.detail_table.setItem(row_index, col, item)
        self.detail_table.resizeColumnsToContents()
        self.detail_table.horizontalHeader().setStretchLastSection(True)

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


def _format_score(score) -> str:
    if score is None:
        return "-"
    try:
        return f"{float(score):.3f}"
    except (TypeError, ValueError):
        return str(score)


def _defect_summary(defects: list[dict]) -> str:
    if not defects:
        return "-"
    parts: list[str] = []
    for defect in defects[:3]:
        defect_type = str(defect.get("type", "defect"))
        bbox = defect.get("bbox_global") or defect.get("bbox_local")
        bbox_text = ""
        if bbox:
            bbox_text = f" bbox={bbox}"
        parts.append(f"{defect_type}{bbox_text}")
    if len(defects) > 3:
        parts.append(f"+{len(defects) - 3} more")
    return "; ".join(parts)
