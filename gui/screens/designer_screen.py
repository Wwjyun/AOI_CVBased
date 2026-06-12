from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.detector_manager import DetectorManager
from gui import icons
from gui.detector_labels import detector_zh_name
from gui.theme import COLORS, R_MD
from gui.widgets.common import Badge, NumStepper, Segmented, Toggle, make_param_widget, param_value
from gui.widgets.panel import Panel

# ============================================================
# AOI Console — Recipe 設計 screen
# ============================================================

TILE_MODES = [
    ("pattern_match", "Pattern Match"),
    ("grid", "Grid"),
    ("contour", "Contour"),
]

CONTOUR_DEFAULTS = {
    "threshold": {
        "method": "adaptive_gaussian",
        "threshold": 128,
        "max_value": 255,
        "invert": True,
        "adaptive_block_size": 31,
        "adaptive_c": 5,
        "blur_size": 3,
        "morph_open_kernel": 3,
        "morph_open_iterations": 1,
        "morph_close_kernel": 3,
        "morph_close_iterations": 1,
    },
    "shapes": {
        "enabled_shapes": ["rectangle", "circle", "polygon"],
        "min_area": 4000,
        "max_area": 200000,
        "min_width": 10,
        "max_width": 1000,
        "min_height": 10,
        "max_height": 1000,
        "min_aspect_ratio": 0,
        "max_aspect_ratio": 20,
        "min_radius": 5,
        "max_radius": 500,
        "min_circularity": 0.75,
        "polygon_min_vertices": 3,
        "polygon_max_vertices": 12,
        "approx_epsilon_ratio": 0.01,
        "subpixel_enabled": True,
        "subpixel_window": 5,
        "crop_padding": 8,
    },
}


def _form_grid() -> QFormLayout:
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


def _label(text: str, mono: bool = False) -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", "form-label")
    if mono:
        widget.setProperty("mono", "true")
    return widget


class TilePreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background: {COLORS['viewer_bg']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: {R_MD}px; color: rgba(255,255,255,0.4); font-size: 9pt;"
        )
        self.setText("尚未預覽")
        self._pixmap: QPixmap | None = None

    def set_image(self, image) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self.setText("")
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()


class DesignerScreen(QWidget):
    preview_requested = Signal(dict)
    recipe_saved = Signal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path: Path | None = None
        self.detector_definitions = DetectorManager().definitions()
        self._param_widgets: dict[str, dict[str, QWidget]] = {}
        self._enabled: dict[str, bool] = {detector_id: False for detector_id in self.detector_definitions}
        self._enabled["000"] = True
        self._row_widgets: dict[str, dict] = {}
        self._active_detector = "000"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        outer.addLayout(top_row, 1)

        # ---------------- left column ----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        left_layout.addWidget(self._build_recipe_info_panel())
        left_layout.addWidget(self._build_tiling_panel())
        left_layout.addWidget(self._build_preview_panel())
        left_layout.addStretch(1)

        left_scroll.setWidget(left)
        top_row.addWidget(left_scroll)

        # ---------------- right column ----------------
        top_row.addWidget(self._build_detector_panel(), 1)

        # ---------------- action bar ----------------
        outer.addWidget(self._build_action_bar())

    # ------------------------------------------------------------------
    # recipe info
    # ------------------------------------------------------------------
    def _build_recipe_info_panel(self) -> Panel:
        panel = Panel(title="Recipe 資訊")
        form = _form_grid()

        self.recipe_name_edit = QLineEdit("PRODUCT_A_PATTERN_MATCH_000_AOI_01")
        self.recipe_name_edit.setProperty("mono", "true")
        self.product_id_edit = QLineEdit("PRODUCT_A")
        self.product_id_edit.setProperty("mono", "true")
        self.machine_id_edit = QLineEdit("AOI_01")
        self.machine_id_edit.setProperty("mono", "true")
        self.version_edit = QLineEdit("0.1.0")
        self.version_edit.setProperty("mono", "true")

        form.addRow(_label("Recipe 名稱"), self.recipe_name_edit)
        form.addRow(_label("產品 Product"), self.product_id_edit)
        form.addRow(_label("機台 Machine"), self.machine_id_edit)
        form.addRow(_label("版本 Version"), self.version_edit)

        panel.add_layout(form)
        return panel

    # ------------------------------------------------------------------
    # tiling
    # ------------------------------------------------------------------
    def _build_tiling_panel(self) -> Panel:
        self.tile_mode = Segmented(TILE_MODES, value="pattern_match")
        self.tile_mode.currentChanged.connect(self._on_tile_mode_changed)
        panel = Panel(title="切圖 Tiling", actions=self.tile_mode)

        self.tile_stack = QStackedWidget()
        self.tile_stack.addWidget(self._build_pattern_match_form())
        self.tile_stack.addWidget(self._build_grid_form())
        self.tile_stack.addWidget(self._build_contour_form())
        panel.add_widget(self.tile_stack)
        return panel

    def _on_tile_mode_changed(self, value: str) -> None:
        index = {"pattern_match": 0, "grid": 1, "contour": 2}.get(value, 0)
        self.tile_stack.setCurrentIndex(index)

    def _build_pattern_match_form(self) -> QWidget:
        widget = QWidget()
        form = _form_grid()

        self.template_path_edit = QLineEdit("outputs_validation/pattern_template.png")
        self.template_path_edit.setProperty("mono", "true")
        template_button = QPushButton("選擇")
        template_button.setProperty("variant", "secondary")
        template_button.setProperty("size", "sm")
        template_button.setIcon(icons.icon("folder", size=13, color=COLORS["text_2"]))
        template_button.clicked.connect(self._choose_template)

        template_row = QHBoxLayout()
        template_row.setSpacing(6)
        template_row.addWidget(self.template_path_edit, 1)
        template_row.addWidget(template_button)
        form.addRow(_label("Template"), _wrap_layout(template_row))

        self.match_threshold = NumStepper(0.8, minimum=0, maximum=1, step=0.01, decimals=3)
        self.max_count = NumStepper(999, minimum=1, maximum=100000, step=1, decimals=0)
        self.nms_threshold = NumStepper(0.3, minimum=0, maximum=1, step=0.01, decimals=3)
        self.crop_padding = NumStepper(8, minimum=0, maximum=10000, step=1, decimals=0)
        self.sort_row_tolerance = NumStepper(20, minimum=1, maximum=10000, step=1, decimals=0)

        form.addRow(_label("匹配門檻"), self.match_threshold)
        form.addRow(_label("最大匹配數"), self.max_count)
        form.addRow(_label("NMS 門檻"), self.nms_threshold)
        form.addRow(_label("裁切外擴 px"), self.crop_padding)
        form.addRow(_label("排序列容差"), self.sort_row_tolerance)

        widget.setLayout(form)
        return widget

    def _build_grid_form(self) -> QWidget:
        widget = QWidget()
        form = _form_grid()

        self.grid_width = NumStepper(512, minimum=32, maximum=100000, step=1, decimals=0)
        self.grid_height = NumStepper(512, minimum=32, maximum=100000, step=1, decimals=0)
        self.grid_overlap_x = NumStepper(64, minimum=0, maximum=100000, step=1, decimals=0)
        self.grid_overlap_y = NumStepper(64, minimum=0, maximum=100000, step=1, decimals=0)

        form.addRow(_label("Tile 寬"), self.grid_width)
        form.addRow(_label("Tile 高"), self.grid_height)
        form.addRow(_label("重疊 X"), self.grid_overlap_x)
        form.addRow(_label("重疊 Y"), self.grid_overlap_y)

        widget.setLayout(form)
        return widget

    def _build_contour_form(self) -> QWidget:
        widget = QWidget()
        form = _form_grid()

        self.contour_min_area = NumStepper(4000, minimum=0, maximum=10_000_000, step=1, decimals=0)
        self.contour_approx_epsilon = NumStepper(0.01, minimum=0, maximum=1, step=0.005, decimals=3)

        form.addRow(_label("最小面積"), self.contour_min_area)
        form.addRow(_label("近似 ε"), self.contour_approx_epsilon)

        widget.setLayout(form)
        return widget

    def _choose_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 Template", "", "圖片檔案 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if path:
            self.template_path_edit.setText(path)

    # ------------------------------------------------------------------
    # tile preview
    # ------------------------------------------------------------------
    def _build_preview_panel(self) -> Panel:
        panel = Panel(title="切圖預覽")
        self.preview_label = TilePreviewLabel()
        panel.add_widget(self.preview_label)

        self.preview_status = QLabel("尚未預覽")
        self.preview_status.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 9pt;")
        self.preview_status.setWordWrap(True)
        panel.add_widget(self.preview_status)
        return panel

    def set_image_path(self, path: Path | None) -> None:
        self.image_path = Path(path) if path else None

    def set_preview_running(self, running: bool) -> None:
        self.preview_button.setEnabled(not running)
        self.save_button.setEnabled(not running)
        if running:
            self.preview_status.setText("切圖預覽執行中…")
            self.preview_status.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 9pt;")

    def show_preview_result(self, image, tile_count: int, shape_counts: dict) -> None:
        self.preview_label.set_image(image)
        score_text = ""
        best_score = shape_counts.get("best_score")
        if best_score is not None:
            score_text = f"；最佳分數：{best_score:.4f}"
        self.preview_status.setText(f"匹配 {tile_count} 張小圖{score_text}")
        self.preview_status.setStyleSheet(f"color: {COLORS['accent_text']}; font-size: 9pt;")

    def show_preview_error(self, message: str) -> None:
        self.preview_status.setText(f"預覽失敗：{message}")
        self.preview_status.setStyleSheet(f"color: {COLORS['ng']}; font-size: 9pt;")

    # ------------------------------------------------------------------
    # detector selection / params
    # ------------------------------------------------------------------
    def _build_detector_panel(self) -> Panel:
        panel = Panel(title="Detector 選用與參數", flush=True)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        list_scroll = QScrollArea()
        list_scroll.setWidgetResizable(True)
        list_scroll.setFixedWidth(280)
        list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        list_scroll.setStyleSheet(f"QScrollArea {{ border-right: 1px solid {COLORS['border']}; }}")

        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(0)

        for detector_id in sorted(self.detector_definitions):
            list_layout.addWidget(self._build_detector_row(detector_id))
        list_layout.addStretch(1)

        list_scroll.setWidget(list_widget)
        body_layout.addWidget(list_scroll)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.params_container = QWidget()
        params_outer = QVBoxLayout(self.params_container)
        params_outer.setContentsMargins(16, 16, 16, 16)
        params_outer.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        self.active_id_label = QLabel("000")
        self.active_id_label.setProperty("mono", "true")
        self.active_id_label.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.active_zh_label = QLabel("")
        self.active_zh_label.setStyleSheet(f"color: {COLORS['text_2']};")
        self.active_badge = Badge("啟用", kind="accent")
        header_row.addWidget(self.active_id_label)
        header_row.addWidget(self.active_zh_label)
        header_row.addWidget(self.active_badge)
        header_row.addStretch(1)
        params_outer.addLayout(header_row)

        self.param_form_container = QWidget()
        self.param_form_container.setMaximumWidth(420)
        self.param_form = _form_grid()
        self.param_form_container.setLayout(self.param_form)
        params_outer.addWidget(self.param_form_container)
        params_outer.addStretch(1)

        params_scroll.setWidget(self.params_container)
        body_layout.addWidget(params_scroll, 1)

        panel.add_widget(body, 1)
        self._select_detector("000")
        return panel

    def _build_detector_row(self, detector_id: str) -> QWidget:
        definition = self.detector_definitions[detector_id]

        row = QWidget()
        row.setProperty("role", "row-item")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setMinimumHeight(48)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(12, 8, 12, 8)
        row_layout.setSpacing(10)

        toggle = Toggle(checked=self._enabled.get(detector_id, False))
        toggle.toggled.connect(lambda checked, did=detector_id: self._on_detector_toggled(did, checked))
        row_layout.addWidget(toggle)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(7)
        id_label = QLabel(detector_id)
        id_label.setProperty("mono", "true")
        id_label.setStyleSheet("font-weight: 600;")
        zh_label = QLabel(detector_zh_name(detector_id))
        zh_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 12px;")
        title_row.addWidget(id_label)
        title_row.addWidget(zh_label, 1)
        text_col.addLayout(title_row)

        display_label = QLabel(definition["display_name"])
        display_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
        text_col.addWidget(display_label)

        row_layout.addLayout(text_col, 1)

        row.mousePressEvent = lambda _event, did=detector_id: self._select_detector(did)
        self._row_widgets[detector_id] = {"row": row, "toggle": toggle}
        return row

    def _on_detector_toggled(self, detector_id: str, checked: bool) -> None:
        self._enabled[detector_id] = checked
        if detector_id == self._active_detector:
            self.active_badge.setText("啟用" if checked else "停用")
            self.active_badge.set_kind("accent" if checked else "neutral")
        self._refresh_enabled_count()

    def _select_detector(self, detector_id: str) -> None:
        self._active_detector = detector_id
        for did, widgets in self._row_widgets.items():
            widgets["row"].setProperty("selected", "true" if did == detector_id else "false")
            widgets["row"].style().unpolish(widgets["row"])
            widgets["row"].style().polish(widgets["row"])

        definition = self.detector_definitions[detector_id]
        self.active_id_label.setText(detector_id)
        self.active_zh_label.setText(detector_zh_name(detector_id))
        enabled = self._enabled.get(detector_id, False)
        self.active_badge.setText("啟用" if enabled else "停用")
        self.active_badge.set_kind("accent" if enabled else "neutral")

        self._clear_param_form()
        widgets = self._param_widgets.setdefault(detector_id, {})
        for key, default_value in definition["default_params"].items():
            widget = widgets.get(key)
            if widget is None:
                widget = make_param_widget(default_value)
                widgets[key] = widget
            self.param_form.addRow(_label(key, mono=True), widget)

    def _clear_param_form(self) -> None:
        while self.param_form.rowCount():
            row = self.param_form.takeRow(0)
            for item in (row.labelItem, row.fieldItem):
                if item and item.widget():
                    item.widget().setParent(None)

    # ------------------------------------------------------------------
    # action bar
    # ------------------------------------------------------------------
    def _build_action_bar(self) -> Panel:
        panel = Panel()
        panel.body_layout.setContentsMargins(16, 10, 16, 10)
        row = QHBoxLayout()
        row.setSpacing(10)

        self.enabled_count_label = QLabel("")
        self.enabled_count_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px;")
        row.addWidget(self.enabled_count_label)
        row.addStretch(1)

        self.preview_button = QPushButton("預覽切圖")
        self.preview_button.setProperty("variant", "secondary")
        self.preview_button.setIcon(icons.icon("eye", size=15, color=COLORS["text_2"]))
        self.preview_button.clicked.connect(self._emit_preview)
        row.addWidget(self.preview_button)

        self.save_button = QPushButton("儲存 Recipe")
        self.save_button.setProperty("variant", "primary")
        self.save_button.setIcon(icons.icon("save", size=15, color="#ffffff"))
        self.save_button.clicked.connect(self._save_recipe)
        row.addWidget(self.save_button)

        panel.add_layout(row)
        self._refresh_enabled_count()
        return panel

    def _refresh_enabled_count(self) -> None:
        count = sum(1 for value in self._enabled.values() if value)
        self.enabled_count_label.setText(f"已啟用 {count} 個 detector")

    # ------------------------------------------------------------------
    # build / save
    # ------------------------------------------------------------------
    def build_tile_config(self) -> dict:
        mode = self.tile_mode.value()
        if mode == "grid":
            return {
                "mode": "grid",
                "width": int(self.grid_width.value()),
                "height": int(self.grid_height.value()),
                "overlap_x": int(self.grid_overlap_x.value()),
                "overlap_y": int(self.grid_overlap_y.value()),
            }
        if mode == "contour":
            config = deepcopy(CONTOUR_DEFAULTS)
            config["shapes"]["min_area"] = int(self.contour_min_area.value())
            config["shapes"]["approx_epsilon_ratio"] = float(self.contour_approx_epsilon.value())
            return {"mode": "contour", **config}
        return {
            "mode": "pattern_match",
            "pattern_match": {
                "template_path": self.template_path_edit.text().strip(),
                "match_threshold": float(self.match_threshold.value()),
                "max_count": int(self.max_count.value()),
                "nms_threshold": float(self.nms_threshold.value()),
                "crop_padding": int(self.crop_padding.value()),
                "sort_row_tolerance": int(self.sort_row_tolerance.value()),
            },
        }

    def _selected_detectors(self) -> dict:
        selected = {}
        for detector_id, enabled in self._enabled.items():
            if not enabled:
                continue
            definition = self.detector_definitions[detector_id]
            selected[detector_id] = {
                "enabled": True,
                "display_name": definition["display_name"],
                "params": self._params_for_detector(detector_id),
            }
        return selected

    def _params_for_detector(self, detector_id: str) -> dict:
        definition = self.detector_definitions[detector_id]
        widgets = self._param_widgets.get(detector_id, {})
        params = {}
        for key, default_value in definition["default_params"].items():
            widget = widgets.get(key)
            params[key] = default_value if widget is None else param_value(widget)
        return params

    def build_recipe(self) -> dict:
        detectors = self._selected_detectors()
        return {
            "recipe_name": self.recipe_name_edit.text() or "PRODUCT_A_PATTERN_MATCH_000_AOI_01",
            "product_id": self.product_id_edit.text() or "PRODUCT_A",
            "machine_id": self.machine_id_edit.text() or "AOI_01",
            "version": self.version_edit.text() or "0.1.0",
            "tile": self.build_tile_config(),
            "decision": {
                "mode": "all_detectors_must_pass",
                "important_detectors": list(detectors),
                "max_ng_count": 0,
            },
            "detectors": detectors,
            "output": {
                "save_overlay": True,
                "save_ng_tiles": True,
                "save_csv": True,
                "save_json": True,
            },
        }

    def _emit_preview(self) -> None:
        self.preview_requested.emit(self.build_tile_config())

    def _save_recipe(self) -> None:
        if not any(self._enabled.values()):
            self.preview_status.setText("請至少啟用一個 detector")
            self.preview_status.setStyleSheet(f"color: {COLORS['ng']}; font-size: 9pt;")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存 Recipe",
            f"recipes/{self.recipe_name_edit.text() or 'PRODUCT_A_PATTERN_MATCH_000_AOI_01'}.yaml",
            "YAML 檔案 (*.yaml *.yml)",
        )
        if not path:
            return
        recipe_path = Path(path)
        with recipe_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.build_recipe(), handle, allow_unicode=True, sort_keys=False)
        self.recipe_saved.emit(recipe_path)
        self.preview_status.setText(f"Recipe 已儲存：{recipe_path}")
        self.preview_status.setStyleSheet(f"color: {COLORS['accent_text']}; font-size: 9pt;")


def _wrap_layout(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget
