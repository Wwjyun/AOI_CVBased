from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import icons
from gui.theme import COLORS
from gui.widgets.common import EmptyState, ProgressBar, result_badge
from gui.widgets.panel import Panel


def _format_duration(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "-"
    if seconds <= 0:
        return "-"
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.1f}s"


class MonitorControlPanel(Panel):
    choose_folder_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(title="Monitor Folder", parent=parent)
        self._folder: str | None = None

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setProperty("mono", "true")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px;")
        self.add_widget(self.folder_label)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self.choose_button = QPushButton("Folder")
        self.choose_button.setProperty("variant", "secondary")
        self.choose_button.setProperty("size", "sm")
        self.choose_button.setIcon(icons.icon("folder", size=14, color=COLORS["text_2"]))
        self.choose_button.clicked.connect(self.choose_folder_requested.emit)
        button_layout.addWidget(self.choose_button)

        self.start_button = QPushButton("Start")
        self.start_button.setProperty("variant", "primary")
        self.start_button.setProperty("size", "sm")
        self.start_button.setIcon(icons.icon("play", size=14, color="#ffffff"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("variant", "danger-ghost")
        self.stop_button.setProperty("size", "sm")
        self.stop_button.setIcon(icons.icon("x", size=14, color=COLORS["ng"]))
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch(1)
        self.add_widget(button_row)

        progress_row = QWidget()
        progress_layout = QHBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)
        self.progress_bar = ProgressBar()
        progress_layout.addWidget(self.progress_bar, 1)
        self.progress_pct_label = QLabel("0%")
        self.progress_pct_label.setProperty("mono", "true")
        self.progress_pct_label.setFixedWidth(38)
        self.progress_pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_layout.addWidget(self.progress_pct_label)
        self.add_widget(progress_row)

        self.message_label = QLabel("Waiting for folder and Recipe")
        self.message_label.setProperty("mono", "true")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 11px;")
        self.add_widget(self.message_label)

    def set_folder(self, folder: str | None) -> None:
        self._folder = folder
        self.folder_label.setText(folder or "No folder selected")
        self.folder_label.setStyleSheet(
            f"color: {COLORS['text_2'] if folder else COLORS['text_3']}; font-size: 12px;"
        )

    def set_ready(self, ready: bool, running: bool) -> None:
        self.choose_button.setEnabled(not running)
        self.start_button.setEnabled(ready and not running)
        self.stop_button.setEnabled(running)
        self.start_button.setText("Monitoring" if running else "Start")

    def set_progress(self, pct: int, message: str) -> None:
        pct = max(0, min(100, int(pct)))
        self.progress_bar.setValue(pct)
        self.progress_pct_label.setText(f"{pct}%")
        self.message_label.setText(message)


class MonitorStatsPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(title="Monitor Status", parent=parent)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._labels: dict[str, QLabel] = {}
        for key, label in (("processed", "Processed"), ("pass", "PASS"), ("ng", "NG"), ("error", "ERR")):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            value_label = QLabel("0")
            value_label.setProperty("mono", "true")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 20px; font-weight: 700;")
            name_label = QLabel(label)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 10px;")
            cell_layout.addWidget(value_label)
            cell_layout.addWidget(name_label)
            layout.addWidget(cell, 1)
            self._labels[key] = value_label
        self.add_widget(row)

    def set_counts(self, items: list[dict]) -> None:
        counts = {
            "processed": len(items),
            "pass": sum(1 for item in items if item.get("final_result") == "PASS"),
            "ng": sum(1 for item in items if item.get("final_result") == "NG"),
            "error": sum(1 for item in items if item.get("final_result") == "ERROR"),
        }
        for key, value in counts.items():
            self._labels[key].setText(str(value))


class MonitorTablePanel(Panel):
    def __init__(self, parent=None):
        super().__init__(title="Processed Images", flush=True, parent=parent)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Time", "Image", "Result", "Def", "NG", "Duration"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.add_widget(self.table, 1)

    def set_items(self, items: list[dict]) -> None:
        self.table.setRowCount(len(items))
        for row, item in enumerate(items):
            time_item = QTableWidgetItem(str(item.get("processed_at", "")))
            name_item = QTableWidgetItem(str(item.get("image_name", "")))
            name_item.setToolTip(str(item.get("image_path", "")))
            defects_item = QTableWidgetItem(str(item.get("defect_count", 0)))
            ng_item = QTableWidgetItem(str(item.get("ng_count", 0)))
            duration_item = QTableWidgetItem(_format_duration(item.get("duration_sec")))
            for table_item in (time_item, name_item, defects_item, ng_item, duration_item):
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, time_item)
            self.table.setItem(row, 1, name_item)
            self.table.setCellWidget(row, 2, result_badge(item.get("final_result")))
            self.table.setItem(row, 3, defects_item)
            self.table.setItem(row, 4, ng_item)
            self.table.setItem(row, 5, duration_item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)


class MonitorScreen(QWidget):
    choose_folder_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)

        self.control_panel = MonitorControlPanel()
        self.stats_panel = MonitorStatsPanel()
        top_layout.addWidget(self.control_panel, 2)
        top_layout.addWidget(self.stats_panel, 1)
        layout.addWidget(top_row)

        self.table_panel = MonitorTablePanel()
        layout.addWidget(self.table_panel, 1)

        self.empty_state = QFrame()
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.addWidget(EmptyState("eye", "No processed images yet", "Start monitoring, then add images under the selected folder."))
        layout.addWidget(self.empty_state)

        self.control_panel.choose_folder_requested.connect(self.choose_folder_requested.emit)
        self.control_panel.start_requested.connect(self.start_requested.emit)
        self.control_panel.stop_requested.connect(self.stop_requested.emit)
        self._refresh()

    def set_folder(self, folder: str | None) -> None:
        self.control_panel.set_folder(folder)

    def set_ready(self, ready: bool, running: bool) -> None:
        self.control_panel.set_ready(ready, running)

    def set_progress(self, pct: int, message: str) -> None:
        self.control_panel.set_progress(pct, message)

    def clear_items(self) -> None:
        self._items = []
        self._refresh()

    def add_item(self, item: dict) -> None:
        self._items.insert(0, item)
        self._items = self._items[:200]
        self._refresh()

    def items(self) -> list[dict]:
        return list(self._items)

    def _refresh(self) -> None:
        has_items = bool(self._items)
        self.stats_panel.set_counts(self._items)
        self.table_panel.set_items(self._items)
        self.table_panel.setVisible(has_items)
        self.empty_state.setVisible(not has_items)
