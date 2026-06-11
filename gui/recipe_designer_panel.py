from __future__ import annotations

from pathlib import Path

import yaml
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class RecipeDesignerPanel(QWidget):
    preview_requested = Signal(dict)
    recipe_saved = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path: Path | None = None

        self.recipe_name = QLineEdit("PRODUCT_A_CONTOUR_TILE")
        self.product_id = QLineEdit("PRODUCT_A")
        self.machine_id = QLineEdit("AOI_01")
        self.version = QLineEdit("0.1.0")

        self.threshold_method = QComboBox()
        self.threshold_method.addItems(["global", "otsu", "adaptive_mean", "adaptive_gaussian"])
        self.threshold_method.setCurrentText("adaptive_gaussian")
        self.threshold = self._spin(0, 255, 128)
        self.max_value = self._spin(1, 255, 255)
        self.invert = QCheckBox("反向")
        self.invert.setChecked(True)
        self.adaptive_block_size = self._spin(3, 255, 31, step=2)
        self.adaptive_c = self._double_spin(-100.0, 100.0, 5.0)
        self.blur_size = self._spin(0, 99, 3)
        self.morph_open_kernel = self._spin(0, 99, 3)
        self.morph_open_iterations = self._spin(0, 20, 1)
        self.morph_close_kernel = self._spin(0, 99, 3)
        self.morph_close_iterations = self._spin(0, 20, 1)

        self.rectangle_enabled = QCheckBox("矩形")
        self.circle_enabled = QCheckBox("圓形")
        self.polygon_enabled = QCheckBox("多邊形")
        for checkbox in (self.rectangle_enabled, self.circle_enabled, self.polygon_enabled):
            checkbox.setChecked(True)

        self.min_area = self._double_spin(0.0, 1_000_000_000.0, 100.0)
        self.max_area = self._double_spin(0.0, 1_000_000_000.0, 200000.0)
        self.min_width = self._double_spin(0.0, 100000.0, 10.0)
        self.max_width = self._double_spin(0.0, 100000.0, 1000.0)
        self.min_height = self._double_spin(0.0, 100000.0, 10.0)
        self.max_height = self._double_spin(0.0, 100000.0, 1000.0)
        self.min_aspect_ratio = self._double_spin(0.0, 1000.0, 0.0)
        self.max_aspect_ratio = self._double_spin(0.0, 1000.0, 20.0)
        self.min_radius = self._double_spin(0.0, 100000.0, 5.0)
        self.max_radius = self._double_spin(0.0, 100000.0, 500.0)
        self.min_circularity = self._double_spin(0.0, 1.0, 0.75, decimals=3, step=0.01)
        self.polygon_min_vertices = self._spin(3, 99, 3)
        self.polygon_max_vertices = self._spin(3, 99, 12)
        self.approx_epsilon_ratio = self._double_spin(0.001, 1.0, 0.02, decimals=4, step=0.005)
        self.subpixel_enabled = QCheckBox("啟用亞像素角點")
        self.subpixel_enabled.setChecked(True)
        self.subpixel_window = self._spin(1, 99, 5)
        self.crop_padding = self._spin(0, 10000, 8)

        self.status_label = QLabel("尚未預覽")
        self.preview_button = QPushButton("預覽二值化切圖")
        self.save_button = QPushButton("儲存 Contour Recipe")
        self.preview_button.clicked.connect(self._emit_preview)
        self.save_button.clicked.connect(self._save_recipe)

        self._build_layout()

    def set_image_path(self, path: Path | None) -> None:
        self.image_path = Path(path) if path else None

    def set_preview_running(self, running: bool) -> None:
        self.preview_button.setEnabled(not running)
        self.save_button.setEnabled(not running)
        if running:
            self.status_label.setText("切圖預覽執行中...")

    def show_preview_result(self, tile_count: int, shape_counts: dict) -> None:
        parts = [f"{shape}: {count}" for shape, count in sorted(shape_counts.items())]
        suffix = "，".join(parts) if parts else "無形狀"
        self.status_label.setText(f"切出 {tile_count} 張小圖；{suffix}")

    def show_preview_error(self, message: str) -> None:
        self.status_label.setText(f"預覽失敗：{message}")

    def build_tile_config(self) -> dict:
        return {
            "mode": "contour",
            "threshold": {
                "method": self.threshold_method.currentText(),
                "threshold": self.threshold.value(),
                "max_value": self.max_value.value(),
                "invert": self.invert.isChecked(),
                "adaptive_block_size": self._odd_value(self.adaptive_block_size.value()),
                "adaptive_c": self.adaptive_c.value(),
                "blur_size": self.blur_size.value(),
                "morph_open_kernel": self.morph_open_kernel.value(),
                "morph_open_iterations": self.morph_open_iterations.value(),
                "morph_close_kernel": self.morph_close_kernel.value(),
                "morph_close_iterations": self.morph_close_iterations.value(),
            },
            "shapes": {
                "enabled_shapes": self._enabled_shapes(),
                "min_area": self.min_area.value(),
                "max_area": self.max_area.value(),
                "min_width": self.min_width.value(),
                "max_width": self.max_width.value(),
                "min_height": self.min_height.value(),
                "max_height": self.max_height.value(),
                "min_aspect_ratio": self.min_aspect_ratio.value(),
                "max_aspect_ratio": self.max_aspect_ratio.value(),
                "min_radius": self.min_radius.value(),
                "max_radius": self.max_radius.value(),
                "min_circularity": self.min_circularity.value(),
                "polygon_min_vertices": self.polygon_min_vertices.value(),
                "polygon_max_vertices": self.polygon_max_vertices.value(),
                "approx_epsilon_ratio": self.approx_epsilon_ratio.value(),
                "subpixel_enabled": self.subpixel_enabled.isChecked(),
                "subpixel_window": self.subpixel_window.value(),
                "crop_padding": self.crop_padding.value(),
            },
        }

    def build_recipe(self) -> dict:
        return {
            "recipe_name": self.recipe_name.text() or "PRODUCT_A_CONTOUR_TILE",
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
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self._recipe_group())
        content_layout.addWidget(self._threshold_group())
        content_layout.addWidget(self._shape_group())

        button_row = QHBoxLayout()
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.save_button)
        content_layout.addLayout(button_row)
        content_layout.addWidget(self.status_label)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

    def _recipe_group(self) -> QGroupBox:
        form = QFormLayout()
        form.addRow("配方名稱", self.recipe_name)
        form.addRow("產品", self.product_id)
        form.addRow("機台", self.machine_id)
        form.addRow("版本", self.version)
        group = QGroupBox("配方資訊")
        group.setLayout(form)
        return group

    def _threshold_group(self) -> QGroupBox:
        form = QFormLayout()
        form.addRow("二值化方法", self.threshold_method)
        form.addRow("固定閾值", self.threshold)
        form.addRow("最大值", self.max_value)
        form.addRow("", self.invert)
        form.addRow("自適應區塊", self.adaptive_block_size)
        form.addRow("自適應 C", self.adaptive_c)
        form.addRow("模糊尺寸", self.blur_size)
        form.addRow("Opening 核大小", self.morph_open_kernel)
        form.addRow("Opening 次數", self.morph_open_iterations)
        form.addRow("Closing 核大小", self.morph_close_kernel)
        form.addRow("Closing 次數", self.morph_close_iterations)
        group = QGroupBox("二值化切圖")
        group.setLayout(form)
        return group

    def _shape_group(self) -> QGroupBox:
        form = QFormLayout()
        shape_row = QHBoxLayout()
        shape_row.addWidget(self.rectangle_enabled)
        shape_row.addWidget(self.circle_enabled)
        shape_row.addWidget(self.polygon_enabled)
        form.addRow("形狀", shape_row)
        form.addRow("最小面積", self.min_area)
        form.addRow("最大面積", self.max_area)
        form.addRow("最小寬度", self.min_width)
        form.addRow("最大寬度", self.max_width)
        form.addRow("最小高度", self.min_height)
        form.addRow("最大高度", self.max_height)
        form.addRow("最小長寬比", self.min_aspect_ratio)
        form.addRow("最大長寬比", self.max_aspect_ratio)
        form.addRow("最小半徑", self.min_radius)
        form.addRow("最大半徑", self.max_radius)
        form.addRow("最小圓度", self.min_circularity)
        form.addRow("最少多邊形頂點", self.polygon_min_vertices)
        form.addRow("最多多邊形頂點", self.polygon_max_vertices)
        form.addRow("輪廓近似比例", self.approx_epsilon_ratio)
        form.addRow("", self.subpixel_enabled)
        form.addRow("亞像素視窗", self.subpixel_window)
        form.addRow("裁切外擴", self.crop_padding)
        group = QGroupBox("輪廓形狀篩選")
        group.setLayout(form)
        return group

    def _emit_preview(self) -> None:
        self.preview_requested.emit(self.build_tile_config())

    def _save_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存配方",
            f"recipes/{self.recipe_name.text() or 'PRODUCT_A_CONTOUR_TILE'}.yaml",
            "YAML 檔案 (*.yaml *.yml)",
        )
        if not path:
            return
        recipe_path = Path(path)
        with recipe_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.build_recipe(), handle, allow_unicode=True, sort_keys=False)
        self.recipe_saved.emit(recipe_path)
        self.status_label.setText(f"配方已儲存：{recipe_path}")

    def _enabled_shapes(self) -> list[str]:
        shapes = []
        if self.rectangle_enabled.isChecked():
            shapes.append("rectangle")
        if self.circle_enabled.isChecked():
            shapes.append("circle")
        if self.polygon_enabled.isChecked():
            shapes.append("polygon")
        return shapes

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

    @staticmethod
    def _odd_value(value: int) -> int:
        return value if value % 2 == 1 else value + 1
