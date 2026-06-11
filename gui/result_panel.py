from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class ResultPanel(QWidget):
    HEADERS = ["Tile", "檢測器", "類型", "全域框", "面積", "分數"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary = QLabel("尚無結果")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 1)

    def show_result(self, result: dict) -> None:
        summary = result.get("summary", {})
        self.summary.setText(
            f"{result.get('final_result', '-')} | "
            f"Tile 數：{summary.get('tile_count', 0)} | "
            f"NG Tile：{summary.get('ng_count', 0)} | "
            f"缺陷數：{summary.get('defect_count', 0)}"
        )

        rows = []
        for tile_result in result.get("tiles", []):
            tile_id = tile_result.get("tile", {}).get("tile_id", "")
            for detector_result in tile_result.get("detectors", []):
                detector_id = detector_result.get("detector_id", "")
                score = detector_result.get("score", 0.0)
                for defect in detector_result.get("defects", []):
                    rows.append(
                        [
                            tile_id,
                            detector_id,
                            defect.get("type", ""),
                            str(defect.get("bbox_global", "")),
                            str(defect.get("area", "")),
                            f"{score:.4f}",
                        ]
                    )

        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                self.table.setItem(row_index, col_index, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()

    def clear(self) -> None:
        self.summary.setText("尚無結果")
        self.table.setRowCount(0)
