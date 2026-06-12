from __future__ import annotations

import datetime
from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from core.recipe_manager import RecipeError, RecipeManager
from gui import theme
from gui.screens.designer_screen import DesignerScreen
from gui.screens.results_screen import ResultsScreen, flatten_defects, flatten_viewer_overlays
from gui.screens.run_screen import RunScreen
from gui.widgets.common import Toggle
from gui.widgets.drawer import Drawer
from gui.widgets.rail import NavRail
from gui.widgets.topbar import TopBar
from gui.workers import ImagePreviewWorker, InspectionWorker, TilePreviewWorker

# ============================================================
# AOI Console — main window shell (rail + topbar + screens + status bar)
# ============================================================

SCREEN_INDEX = {"run": 0, "designer": 1, "results": 2}
HISTORY_LIMIT = 6

OUTPUT_TOGGLE_LABELS = {
    "save_overlay": "儲存 overlay 影像",
    "save_ng_tiles": "儲存 NG tiles",
    "save_csv": "輸出 CSV 報表",
    "save_json": "輸出 JSON 報表",
}


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "panel-title")
    return label


class _DetectorListCompatibility:
    def __init__(self, window: "MainWindow"):
        self._window = window

    def count(self) -> int:
        return len((self._window.recipe or {}).get("detectors", {}))


class _RecipePanelCompatibility:
    def __init__(self, window: "MainWindow"):
        self.detector_list = _DetectorListCompatibility(window)
        self._window = window

    def load_recipe(self, path: Path) -> None:
        self._window._load_recipe(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            theme.install_application_font(app)
        self.setWindowTitle("AOI 視覺檢測系統")
        self.resize(1440, 900)

        # ---- state ----
        self.mode = "eng"
        self.image_path: Path | None = None
        self.recipe_path: Path | None = None
        self.recipe: dict | None = None
        self.running = False
        self.result: dict | None = None
        self.selected_defect_id = None
        self.show_overlay = True
        self.output_dir = "outputs"
        self.output_opts = {"save_overlay": True, "save_ng_tiles": True, "save_csv": True, "save_json": True}
        self.history: list[dict] = []

        self._defects: list[dict] = []
        self._current_image = None
        self._run_started_at: datetime.datetime | None = None

        self._preview_thread: QThread | None = None
        self._preview_worker: ImagePreviewWorker | None = None
        self._preview_updates_current_image = False
        self._inspection_thread: QThread | None = None
        self._inspection_worker: InspectionWorker | None = None
        self._tile_preview_thread: QThread | None = None
        self._tile_preview_worker: TilePreviewWorker | None = None

        self.recipe_manager = RecipeManager()
        self.recipe_panel = _RecipePanelCompatibility(self)

        self._build_ui()
        self._connect_signals()

        self._set_screen("run")
        self.topbar.set_mode(self.mode)
        self.run_screen.set_mode(self.mode)
        self.rail.set_designer_visible(self.mode == "eng")
        self._update_mode_status_label()
        self._refresh_image_chip()
        self._update_run_ready()
        self.statusBar().showMessage("就緒")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.shell = QWidget()
        self.shell.setObjectName("shell")
        shell_layout = QHBoxLayout(self.shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self.rail = NavRail()
        shell_layout.addWidget(self.rail)

        main_col = QWidget()
        main_col_layout = QVBoxLayout(main_col)
        main_col_layout.setContentsMargins(0, 0, 0, 0)
        main_col_layout.setSpacing(0)

        self.topbar = TopBar()
        main_col_layout.addWidget(self.topbar)

        self.run_screen = RunScreen()
        self.designer_screen = DesignerScreen()
        self.results_screen = ResultsScreen()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.run_screen)
        self.stack.addWidget(self.designer_screen)
        self.stack.addWidget(self.results_screen)

        content_wrap = QWidget()
        content_layout = QVBoxLayout(content_wrap)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.addWidget(self.stack)
        main_col_layout.addWidget(content_wrap, 1)

        shell_layout.addWidget(main_col, 1)
        self.setCentralWidget(self.shell)

        status_bar = QStatusBar()
        status_bar.setFixedHeight(26)
        self.mode_status_label = QLabel()
        self.mode_status_label.setProperty("mono", "true")
        status_bar.addPermanentWidget(self.mode_status_label)
        self.setStatusBar(status_bar)

        self.settings_drawer = self._build_settings_drawer()

    def _build_settings_drawer(self) -> Drawer:
        drawer = Drawer("設定", self.shell)

        drawer.add_widget(_section_label("輸出"))

        output_form = QFormLayout()
        output_form.setHorizontalSpacing(12)
        output_form.setVerticalSpacing(10)

        output_dir_row = QWidget()
        output_dir_layout = QHBoxLayout(output_dir_row)
        output_dir_layout.setContentsMargins(0, 0, 0, 0)
        output_dir_layout.setSpacing(6)
        self.output_dir_edit = QLineEdit(self.output_dir)
        self.output_dir_edit.setProperty("mono", "true")
        self.output_dir_edit.editingFinished.connect(self._on_output_dir_changed)
        output_dir_browse = QPushButton("瀏覽")
        output_dir_browse.setProperty("variant", "secondary")
        output_dir_browse.setProperty("size", "sm")
        output_dir_browse.clicked.connect(self._choose_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit, 1)
        output_dir_layout.addWidget(output_dir_browse)
        output_form.addRow("輸出目錄", output_dir_row)

        self.output_toggles: dict[str, Toggle] = {}
        for key, label in OUTPUT_TOGGLE_LABELS.items():
            toggle = Toggle(checked=self.output_opts[key])
            toggle.toggled.connect(lambda checked, k=key: self._on_output_opt_toggled(k, checked))
            self.output_toggles[key] = toggle
            output_form.addRow(label, toggle)

        drawer.add_layout(output_form)

        drawer.add_widget(_section_label("機台"))

        machine_form = QFormLayout()
        machine_form.setHorizontalSpacing(12)
        machine_form.setVerticalSpacing(10)

        machine_id_edit = QLineEdit("AOI_01")
        machine_id_edit.setProperty("mono", "true")
        machine_id_edit.setReadOnly(True)
        machine_form.addRow("Machine ID", machine_id_edit)

        pipeline_version_edit = QLineEdit("0.4.2 (MVP)")
        pipeline_version_edit.setProperty("mono", "true")
        pipeline_version_edit.setReadOnly(True)
        machine_form.addRow("Pipeline 版本", pipeline_version_edit)

        drawer.add_layout(machine_form)

        return drawer

    def _connect_signals(self) -> None:
        self.rail.screen_changed.connect(self._set_screen)
        self.rail.settings_clicked.connect(self.settings_drawer.open_drawer)

        self.topbar.image_chip_clicked.connect(self._choose_image)
        self.topbar.recipe_chip_clicked.connect(self._choose_recipe)
        self.topbar.mode_changed.connect(self._on_mode_changed)

        self.run_screen.start_requested.connect(self._run_inspection)
        self.run_screen.open_recipe_requested.connect(self._choose_recipe)
        self.run_screen.view_results_requested.connect(lambda: self._set_screen("results"))
        self.run_screen.image_viewer.defect_clicked.connect(self._on_defect_selected)
        self.run_screen.image_viewer.overlay_toggled.connect(self._on_overlay_toggled)

        self.designer_screen.preview_requested.connect(self._preview_contour_tiles)
        self.designer_screen.recipe_saved.connect(self._on_designed_recipe_saved)

        self.results_screen.defect_selected.connect(self._on_defect_selected)
        self.results_screen.view_requested.connect(self._on_view_defect)
        self.results_screen.go_to_run_requested.connect(lambda: self._set_screen("run"))

    # ------------------------------------------------------------------
    # screen / mode switching
    # ------------------------------------------------------------------
    def _set_screen(self, screen_id: str) -> None:
        self.stack.setCurrentIndex(SCREEN_INDEX[screen_id])
        self.rail.set_active(screen_id)
        self.topbar.set_screen(screen_id)

    def _on_mode_changed(self, mode: str) -> None:
        self.mode = mode
        self.rail.set_designer_visible(mode == "eng")
        self.run_screen.set_mode(mode)
        self._update_mode_status_label()

    def _update_mode_status_label(self) -> None:
        mode_text = "操作員模式" if self.mode == "op" else "工程師模式"
        self.mode_status_label.setText(f"AOI_01 · {mode_text}")

    def _on_overlay_toggled(self, checked: bool) -> None:
        self.show_overlay = checked

    # ------------------------------------------------------------------
    # defect selection sync
    # ------------------------------------------------------------------
    def _on_defect_selected(self, defect_id) -> None:
        self.selected_defect_id = defect_id
        self.run_screen.image_viewer.set_selected_defect(defect_id)
        self.results_screen.set_selected(defect_id)

    def _on_view_defect(self, defect_id) -> None:
        self._on_defect_selected(defect_id)
        self._set_screen("run")

    # ------------------------------------------------------------------
    # settings drawer
    # ------------------------------------------------------------------
    def _on_output_opt_toggled(self, key: str, checked: bool) -> None:
        self.output_opts[key] = checked

    def _on_output_dir_changed(self) -> None:
        self.output_dir = self.output_dir_edit.text() or "outputs"

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出目錄", self.output_dir_edit.text())
        if path:
            self.output_dir_edit.setText(path)
            self.output_dir = path

    # ------------------------------------------------------------------
    # image loading
    # ------------------------------------------------------------------
    def _choose_image(self) -> None:
        if self.running:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "載入檢測影像",
            "",
            "圖片檔案 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        self._start_preview_load(path, update_current_image=True)

    def _start_preview_load(self, path: Path, update_current_image: bool) -> None:
        if self._preview_thread and self._preview_thread.isRunning():
            QMessageBox.information(self, "載入影像", "影像仍在載入中，請稍候。")
            return

        self._preview_updates_current_image = update_current_image
        if update_current_image:
            self.topbar.image_chip.set_value("", loading=True)
        self.statusBar().showMessage(f"影像載入中：{path}")
        self._preview_thread = QThread(self)
        self._preview_worker = ImagePreviewWorker(path)
        self._preview_worker.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.progress.connect(self._on_status_progress)
        self._preview_worker.loaded.connect(self._on_preview_loaded)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.loaded.connect(self._preview_thread.quit)
        self._preview_worker.failed.connect(self._preview_thread.quit)
        self._preview_thread.finished.connect(self._preview_worker.deleteLater)
        self._preview_thread.finished.connect(self._on_preview_thread_finished)
        self._preview_thread.start()

    def _on_preview_loaded(self, path: Path, image) -> None:
        self.run_screen.image_viewer.set_qimage(image, name=Path(path).name)
        if self.run_screen.image_viewer.last_error:
            QMessageBox.warning(self, "載入影像", self.run_screen.image_viewer.last_error)
            return
        if self._preview_updates_current_image:
            self.image_path = Path(path)
            self._current_image = image
            self.result = None
            self._defects = []
            self.selected_defect_id = None
            self.run_screen.image_viewer.set_defects([])
            self.run_screen.run_control_panel.clear_result()
            self.results_screen.set_result(None, None)
            self.designer_screen.set_image_path(self.image_path)
        self.statusBar().showMessage(f"影像已載入：{path}")
        self._update_run_ready()

    def _on_preview_failed(self, path: Path, message: str) -> None:
        QMessageBox.warning(self, "載入影像", f"影像載入失敗：\n{path}\n\n{message}")
        self.statusBar().showMessage("影像載入失敗")

    def _on_preview_thread_finished(self) -> None:
        self._preview_updates_current_image = False
        self._preview_thread = None
        self._preview_worker = None
        self._refresh_image_chip()

    def _refresh_image_chip(self) -> None:
        if self.image_path:
            self.topbar.image_chip.set_value(self.image_path.name)
        else:
            self.topbar.image_chip.set_value("", empty=True)

    # ------------------------------------------------------------------
    # recipe loading
    # ------------------------------------------------------------------
    def _choose_recipe(self) -> None:
        if self.running:
            return
        start_dir = str(Path(self.recipe_path).parent) if self.recipe_path else "recipes"
        path, _ = QFileDialog.getOpenFileName(self, "載入 Recipe", start_dir, "Recipe 檔案 (*.yaml *.yml)")
        if path:
            self._load_recipe(Path(path))

    def _load_recipe(self, path: Path) -> None:
        try:
            recipe = self.recipe_manager.load(path)
        except RecipeError as exc:
            QMessageBox.warning(self, "載入 Recipe", str(exc))
            return
        self.recipe_path = path
        self.recipe = recipe
        self.topbar.recipe_chip.set_value(path.name)
        self.run_screen.recipe_info_panel.set_recipe(recipe)
        self.statusBar().showMessage(f"Recipe 已載入：{path}")
        self._update_run_ready()

    def _on_designed_recipe_saved(self, path: Path) -> None:
        self._load_recipe(path)
        self.statusBar().showMessage(f"設計 Recipe 已儲存並載入：{path}")

    # ------------------------------------------------------------------
    # inspection run
    # ------------------------------------------------------------------
    def _is_ready(self) -> bool:
        return self.image_path is not None and self.recipe_path is not None

    def _update_run_ready(self) -> None:
        if self.running:
            return
        has_image = self.image_path is not None
        has_recipe = self.recipe_path is not None
        ready = has_image and has_recipe
        self.run_screen.run_control_panel.set_ready(ready, has_image, has_recipe, False)
        self.run_screen.op_panel.set_state(ready, False, 0, "", self.result)

    def _run_inspection(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "執行檢測", "請先載入影像。")
            return
        if not self.recipe_path:
            QMessageBox.warning(self, "執行檢測", "請先載入 Recipe。")
            return
        if self._inspection_thread and self._inspection_thread.isRunning():
            QMessageBox.information(self, "執行檢測", "檢測執行中，請稍候。")
            return

        self._set_inspection_running(True)
        self._run_started_at = datetime.datetime.now()
        self.statusBar().showMessage("檢測執行中...")
        self._inspection_thread = QThread(self)
        self._inspection_worker = InspectionWorker(
            image_path=self.image_path,
            recipe_path=self.recipe_path,
            output_dir=Path(self.output_dir or "outputs"),
            output_overrides=dict(self.output_opts),
        )
        self._inspection_worker.moveToThread(self._inspection_thread)
        self._inspection_thread.started.connect(self._inspection_worker.run)
        self._inspection_worker.progress.connect(self._on_inspection_progress)
        self._inspection_worker.finished.connect(self._on_inspection_finished)
        self._inspection_worker.failed.connect(self._on_inspection_failed)
        self._inspection_worker.finished.connect(self._inspection_thread.quit)
        self._inspection_worker.failed.connect(self._inspection_thread.quit)
        self._inspection_thread.finished.connect(self._inspection_worker.deleteLater)
        self._inspection_thread.finished.connect(self._on_inspection_thread_finished)
        self._inspection_thread.start()

    def _on_inspection_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.topbar.set_running(True, percent)
        self.run_screen.image_viewer.set_running(True, percent)
        self.run_screen.run_control_panel.set_progress(True, self.result is not None, percent, message)
        self.run_screen.op_panel.set_state(False, True, percent, message, self.result)
        self.statusBar().showMessage(f"{message} ({percent}%)")

    def _on_status_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.statusBar().showMessage(f"{message} ({percent}%)")

    def _on_inspection_finished(self, result: dict) -> None:
        self.result = result
        self._defects = flatten_defects(result)
        viewer_overlays = flatten_viewer_overlays(result)
        self.selected_defect_id = None
        self.run_screen.image_viewer.set_defects(viewer_overlays)
        self.run_screen.image_viewer.set_selected_defect(None)

        duration = ""
        if self._run_started_at is not None:
            elapsed = (datetime.datetime.now() - self._run_started_at).total_seconds()
            duration = f"{elapsed:.1f}s"

        self.run_screen.run_control_panel.show_result(result, duration)
        self.results_screen.set_result(result, self._current_image, duration)

        final = result.get("final_result", "-")
        summary = result.get("summary", {})
        self.history.insert(
            0,
            {
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "result": final,
                "defects": summary.get("defect_count", 0),
            },
        )
        self.history = self.history[:HISTORY_LIMIT]
        self.run_screen.op_panel.set_history(self.history)

        self.statusBar().showMessage(f"檢測完成：{final}")

    def _on_inspection_failed(self, message: str) -> None:
        QMessageBox.critical(self, "執行檢測", message)
        self.statusBar().showMessage("檢測失敗")

    def _on_inspection_thread_finished(self) -> None:
        self._set_inspection_running(False)
        self._inspection_thread = None
        self._inspection_worker = None
        self._update_run_ready()
        self.run_screen.run_control_panel.set_progress(False, self.result is not None, 0, "")
        self.run_screen.op_panel.set_state(self._is_ready(), False, 0, "", self.result)

    def _set_inspection_running(self, running: bool) -> None:
        self.running = running
        self.topbar.set_running(running, 0)
        self.run_screen.image_viewer.set_running(running, 0)
        has_image = self.image_path is not None
        has_recipe = self.recipe_path is not None
        ready = has_image and has_recipe and not running
        self.run_screen.run_control_panel.set_ready(ready, has_image, has_recipe, running)

    # ------------------------------------------------------------------
    # tile preview (Recipe designer)
    # ------------------------------------------------------------------
    def _preview_contour_tiles(self, tile_config: dict) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Recipe 設計", "請先載入影像再預覽切圖。")
            return
        if self._tile_preview_thread and self._tile_preview_thread.isRunning():
            QMessageBox.information(self, "Recipe 設計", "切圖預覽執行中，請稍候。")
            return

        self.designer_screen.set_preview_running(True)
        self.statusBar().showMessage("切圖預覽中...")
        self._tile_preview_thread = QThread(self)
        self._tile_preview_worker = TilePreviewWorker(self.image_path, tile_config)
        self._tile_preview_worker.moveToThread(self._tile_preview_thread)
        self._tile_preview_thread.started.connect(self._tile_preview_worker.run)
        self._tile_preview_worker.progress.connect(self._on_status_progress)
        self._tile_preview_worker.finished.connect(self._on_tile_preview_finished)
        self._tile_preview_worker.failed.connect(self._on_tile_preview_failed)
        self._tile_preview_worker.finished.connect(self._tile_preview_thread.quit)
        self._tile_preview_worker.failed.connect(self._tile_preview_thread.quit)
        self._tile_preview_thread.finished.connect(self._on_tile_preview_thread_finished)
        self._tile_preview_thread.start()

    def _on_tile_preview_finished(
        self,
        image_bytes: bytes,
        width: int,
        height: int,
        bytes_per_line: int,
        tile_count: int,
        shape_counts: dict,
    ) -> None:
        self.designer_screen.show_preview_result(image_bytes, width, height, bytes_per_line, tile_count, shape_counts)
        self.statusBar().showMessage(f"切圖預覽完成：{tile_count} 張")

    def _on_tile_preview_failed(self, message: str) -> None:
        self.designer_screen.show_preview_error(message)
        QMessageBox.warning(self, "Recipe 設計", message)
        self.statusBar().showMessage("切圖預覽失敗")

    def _on_tile_preview_thread_finished(self) -> None:
        self.designer_screen.set_preview_running(False)
        self._tile_preview_thread = None
        self._tile_preview_worker = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self._inspection_thread and self._inspection_thread.isRunning():
            QMessageBox.information(self, "背景作業", "檢測仍在執行中，請等待完成後再關閉。")
            event.ignore()
            return
        if self._preview_thread and self._preview_thread.isRunning():
            QMessageBox.information(self, "背景作業", "影像仍在載入中，請等待完成後再關閉。")
            event.ignore()
            return
        if self._tile_preview_thread and self._tile_preview_thread.isRunning():
            QMessageBox.information(self, "背景作業", "切圖預覽仍在執行中，請等待完成後再關閉。")
            event.ignore()
            return
        super().closeEvent(event)


def run_app() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    theme.install_application_font(app)
    app.setStyleSheet(theme.build_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()
