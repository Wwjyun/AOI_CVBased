from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


class RecipeDesignerPanel(QWidget):
    preview_requested = Signal(dict)
    recipe_saved = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path: Path | None = None

        self.recipe_name = QLineEdit("PRODUCT_A_PATTERN_MATCH_AOI_01")
        self.product_id = QLineEdit("PRODUCT_A")
        self.machine_id = QLineEdit("AOI_01")
        self.version = QLineEdit("0.1.0")

        self.template_path = QLineEdit("")
        self.template_button = QPushButton("選擇")
        self.template_button.clicked.connect(self._choose_template)
        self.match_threshold = self._double_spin(0.0, 1.0, 0.8, decimals=3, step=0.01)
        self.max_count = self._spin(1, 100000, 999)
        self.nms_threshold = self._double_spin(0.0, 1.0, 0.3, decimals=3, step=0.01)
        self.crop_padding = self._spin(0, 10000, 8)
        self.sort_row_tolerance = self._spin(1, 10000, 20)

        self.status_label = QLabel("尚未預覽")
        self.preview_button = QPushButton("預覽 Pattern Match 切圖")
        self.save_button = QPushButton("儲存 Recipe")
        self.preview_button.clicked.connect(self._emit_preview)
        self.save_button.clicked.connect(self._save_recipe)

        self._build_layout()

    def set_image_path(self, path: Path | None) -> None:
        self.image_path = Path(path) if path else None

    def set_preview_running(self, running: bool) -> None:
        self.preview_button.setEnabled(not running)
        self.save_button.setEnabled(not running)
        if running:
            self.status_label.setText("Pattern Match 預覽執行中...")

    def show_preview_result(self, tile_count: int, match_counts: dict) -> None:
        score_text = ""
        if match_counts.get("best_score") is not None:
            score_text = f"；最佳分數：{match_counts['best_score']:.4f}"
        self.status_label.setText(f"匹配 {tile_count} 張小圖{score_text}")

    def show_preview_error(self, message: str) -> None:
        self.status_label.setText(f"預覽失敗：{message}")

    def build_tile_config(self) -> dict:
        return {
            "mode": "pattern_match",
            "pattern_match": {
                "template_path": self.template_path.text().strip(),
                "match_threshold": self.match_threshold.value(),
                "max_count": self.max_count.value(),
                "nms_threshold": self.nms_threshold.value(),
                "crop_padding": self.crop_padding.value(),
                "sort_row_tolerance": self.sort_row_tolerance.value(),
            },
        }

    def build_recipe(self) -> dict:
        return {
            "recipe_name": self.recipe_name.text() or "PRODUCT_A_PATTERN_MATCH_AOI_01",
            "product_id": self.product_id.text() or "PRODUCT_A",
            "machine_id": self.machine_id.text() or "AOI_01",
            "version": self.version.text() or "0.1.0",
            "tile": self.build_tile_config(),
            "decision": {
                "mode": "all_detectors_must_pass",
                "important_detectors": ["999"],
                "max_ng_count": 0,
            },
            "detectors": {
                "999": {
                    "enabled": True,
                    "display_name": "Dark / bright blob detector",
                    "params": {
                        "threshold": 45,
                        "min_area": 20,
                        "max_area": 5000,
                        "blur_size": 3,
                        "invert": False,
                        "clahe_enabled": True,
                    },
                }
            },
            "output": {
                "save_overlay": True,
                "save_ng_tiles": True,
                "save_csv": True,
                "save_json": True,
            },
        }

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._recipe_group())
        layout.addWidget(self._pattern_match_group())

        button_row = QHBoxLayout()
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.save_button)
        layout.addLayout(button_row)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _recipe_group(self) -> QGroupBox:
        form = QFormLayout()
        form.addRow("Recipe 名稱", self.recipe_name)
        form.addRow("產品", self.product_id)
        form.addRow("機台", self.machine_id)
        form.addRow("版本", self.version)
        group = QGroupBox("Recipe 資訊")
        group.setLayout(form)
        return group

    def _pattern_match_group(self) -> QGroupBox:
        template_row = QHBoxLayout()
        template_row.addWidget(self.template_path, 1)
        template_row.addWidget(self.template_button)

        form = QFormLayout()
        form.addRow("Template", template_row)
        form.addRow("匹配門檻", self.match_threshold)
        form.addRow("最大匹配數", self.max_count)
        form.addRow("NMS 門檻", self.nms_threshold)
        form.addRow("裁切外擴", self.crop_padding)
        form.addRow("排序列容差", self.sort_row_tolerance)
        group = QGroupBox("Pattern Match 切圖")
        group.setLayout(form)
        return group

    def _emit_preview(self) -> None:
        self.preview_requested.emit(self.build_tile_config())

    def _save_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存 Recipe",
            f"recipes/{self.recipe_name.text() or 'PRODUCT_A_PATTERN_MATCH_AOI_01'}.yaml",
            "YAML 檔案 (*.yaml *.yml)",
        )
        if not path:
            return
        recipe_path = Path(path)
        with recipe_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.build_recipe(), handle, allow_unicode=True, sort_keys=False)
        self.recipe_saved.emit(recipe_path)
        self.status_label.setText(f"Recipe 已儲存：{recipe_path}")

    def _choose_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 Template",
            "",
            "圖片檔案 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.template_path.setText(path)

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        return spin

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 3,
        step: float = 1.0,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin
