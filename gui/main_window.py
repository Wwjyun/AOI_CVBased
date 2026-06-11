from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from gui.detector_param_panel import DetectorParamPanel
from gui.image_viewer import ImageViewer
from gui.recipe_panel import RecipePanel
from gui.result_panel import ResultPanel
from gui.workers import ImagePreviewWorker, InspectionWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOI 視覺檢測系統")
        self.resize(1280, 820)
        self.image_path: Path | None = None
        self.recipe_path: Path | None = None
        self._preview_thread: QThread | None = None
        self._preview_worker: ImagePreviewWorker | None = None
        self._preview_updates_current_image = False
        self._inspection_thread: QThread | None = None
        self._inspection_worker: InspectionWorker | None = None

        self.image_viewer = ImageViewer()
        self.recipe_panel = RecipePanel()
        self.detector_param_panel = DetectorParamPanel()
        self.result_panel = ResultPanel()

        self.recipe_panel.recipe_loaded.connect(self._on_recipe_loaded)
        self.recipe_panel.detector_selected.connect(self.detector_param_panel.show_detector)

        self.output_edit = QLineEdit("outputs")
        self.output_edit.setMinimumWidth(220)
        self.load_image_button = QPushButton("載入圖片")
        self.load_recipe_button = QPushButton("載入配方")
        self.output_button = QPushButton("瀏覽")
        self.run_button = QPushButton("開始檢測")

        self._build_toolbar()
        self._build_layout()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就緒")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("主工具列")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.load_image_button.clicked.connect(self._choose_image)
        toolbar.addWidget(self.load_image_button)

        self.load_recipe_button.clicked.connect(self.recipe_panel._choose_recipe)
        toolbar.addWidget(self.load_recipe_button)

        toolbar.addWidget(QLabel("輸出目錄"))
        toolbar.addWidget(self.output_edit)

        self.output_button.clicked.connect(self._choose_output_dir)
        toolbar.addWidget(self.output_button)

        self.run_button.clicked.connect(self._run_inspection)
        toolbar.addWidget(self.run_button)

    def _build_layout(self) -> None:
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.recipe_panel, 2)
        left_layout.addWidget(self.detector_param_panel, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.image_viewer, 4)
        right_layout.addWidget(self.result_panel, 2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([360, 920])

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "載入圖片",
            "",
            "圖片檔案 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        self._start_preview_load(path, update_current_image=True)

    def _start_preview_load(self, path: Path, update_current_image: bool) -> None:
        if self._preview_thread and self._preview_thread.isRunning():
            QMessageBox.information(self, "載入圖片", "圖片仍在載入中，請稍候。")
            return

        self._preview_updates_current_image = update_current_image
        self.load_image_button.setEnabled(False)
        self.statusBar().showMessage(f"圖片載入中：{path}")
        self._preview_thread = QThread(self)
        self._preview_worker = ImagePreviewWorker(path)
        self._preview_worker.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.loaded.connect(self._on_preview_loaded)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.loaded.connect(self._preview_thread.quit)
        self._preview_worker.failed.connect(self._preview_thread.quit)
        self._preview_thread.finished.connect(self._preview_worker.deleteLater)
        self._preview_thread.finished.connect(self._on_preview_thread_finished)
        self._preview_thread.start()

    def _on_preview_loaded(self, path: Path, image: object) -> None:
        self.image_viewer.set_qimage(image)
        if self.image_viewer.last_error:
            QMessageBox.warning(self, "載入圖片", self.image_viewer.last_error)
            return
        if self._preview_updates_current_image:
            self.image_path = Path(path)
            self.result_panel.clear()
        self.statusBar().showMessage(f"圖片已載入：{path}")

    def _on_preview_failed(self, path: Path, message: str) -> None:
        QMessageBox.warning(self, "載入圖片", f"圖片載入失敗：\n{path}\n\n{message}")
        self.statusBar().showMessage("圖片載入失敗")

    def _on_preview_thread_finished(self) -> None:
        inspection_running = bool(self._inspection_thread and self._inspection_thread.isRunning())
        self.load_image_button.setEnabled(not inspection_running)
        self._preview_updates_current_image = False
        self._preview_thread = None
        self._preview_worker = None

    def _on_recipe_loaded(self, path: Path, recipe: dict) -> None:
        self.recipe_path = path
        self.statusBar().showMessage(f"配方已載入：{path}")

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出目錄", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _run_inspection(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "執行檢測", "請先載入圖片。")
            return
        if not self.recipe_path:
            QMessageBox.warning(self, "執行檢測", "請先載入配方。")
            return
        if self._inspection_thread and self._inspection_thread.isRunning():
            QMessageBox.information(self, "執行檢測", "檢測執行中，請稍候。")
            return

        self._set_inspection_running(True)
        self.statusBar().showMessage("檢測執行中...")
        self._inspection_thread = QThread(self)
        self._inspection_worker = InspectionWorker(
            image_path=self.image_path,
            recipe_path=self.recipe_path,
            output_dir=Path(self.output_edit.text() or "outputs"),
        )
        self._inspection_worker.moveToThread(self._inspection_thread)
        self._inspection_thread.started.connect(self._inspection_worker.run)
        self._inspection_worker.finished.connect(self._on_inspection_finished)
        self._inspection_worker.failed.connect(self._on_inspection_failed)
        self._inspection_worker.finished.connect(self._inspection_thread.quit)
        self._inspection_worker.failed.connect(self._inspection_thread.quit)
        self._inspection_thread.finished.connect(self._inspection_worker.deleteLater)
        self._inspection_thread.finished.connect(self._on_inspection_thread_finished)
        self._inspection_thread.start()

    def _on_inspection_finished(self, result: dict) -> None:
        overlay_path = result.get("outputs", {}).get("overlay")
        if overlay_path:
            self._start_preview_load(Path(overlay_path), update_current_image=False)
        self.result_panel.show_result(result)
        self.statusBar().showMessage(f"檢測完成：{result.get('final_result', '-')}")

    def _on_inspection_failed(self, message: str) -> None:
        QMessageBox.critical(self, "執行檢測", message)
        self.statusBar().showMessage("檢測失敗")

    def _on_inspection_thread_finished(self) -> None:
        self._set_inspection_running(False)
        self._inspection_thread = None
        self._inspection_worker = None

    def _set_inspection_running(self, running: bool) -> None:
        self.load_image_button.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.load_recipe_button.setEnabled(not running)
        self.output_button.setEnabled(not running)
        self.output_edit.setEnabled(not running)

    def closeEvent(self, event) -> None:
        if self._inspection_thread and self._inspection_thread.isRunning():
            QMessageBox.information(self, "背景作業", "檢測仍在執行中，請等待完成後再關閉。")
            event.ignore()
            return
        if self._preview_thread and self._preview_thread.isRunning():
            QMessageBox.information(self, "背景作業", "圖片仍在載入中，請等待完成後再關閉。")
            event.ignore()
            return
        super().closeEvent(event)


def run_app() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
