from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
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

from core.pipeline import AOIPipeline
from gui.detector_param_panel import DetectorParamPanel
from gui.image_viewer import ImageViewer
from gui.recipe_panel import RecipePanel
from gui.result_panel import ResultPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOI CV Based")
        self.resize(1280, 820)
        self.image_path: Path | None = None
        self.recipe_path: Path | None = None

        self.image_viewer = ImageViewer()
        self.recipe_panel = RecipePanel()
        self.detector_param_panel = DetectorParamPanel()
        self.result_panel = ResultPanel()

        self.recipe_panel.recipe_loaded.connect(self._on_recipe_loaded)
        self.recipe_panel.detector_selected.connect(self.detector_param_panel.show_detector)

        self.output_edit = QLineEdit("outputs")
        self.output_edit.setMinimumWidth(220)

        self._build_toolbar()
        self._build_layout()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        load_image_button = QPushButton("Load Image")
        load_image_button.clicked.connect(self._choose_image)
        toolbar.addWidget(load_image_button)

        load_recipe_button = QPushButton("Load Recipe")
        load_recipe_button.clicked.connect(self.recipe_panel._choose_recipe)
        toolbar.addWidget(load_recipe_button)

        toolbar.addWidget(QLabel("Output"))
        toolbar.addWidget(self.output_edit)

        output_button = QPushButton("Browse")
        output_button.clicked.connect(self._choose_output_dir)
        toolbar.addWidget(output_button)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self._run_inspection)
        toolbar.addWidget(run_button)

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
            "Load Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        if not self.image_viewer.load_image(path):
            message = self.image_viewer.last_error or f"Failed to load image:\n{path}"
            QMessageBox.warning(self, "Load Image", message)
            return
        self.image_path = Path(path)
        self.result_panel.clear()
        self.statusBar().showMessage(f"Image loaded: {path}")

    def _on_recipe_loaded(self, path: Path, recipe: dict) -> None:
        self.recipe_path = path
        self.statusBar().showMessage(f"Recipe loaded: {path}")

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output Directory", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _run_inspection(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Run Inspection", "Load an image first.")
            return
        if not self.recipe_path:
            QMessageBox.warning(self, "Run Inspection", "Load a recipe first.")
            return

        try:
            pipeline = AOIPipeline(
                recipe_path=self.recipe_path,
                output_dir=Path(self.output_edit.text() or "outputs"),
            )
            result = pipeline.run(self.image_path)
        except Exception as exc:
            QMessageBox.critical(self, "Run Inspection", str(exc))
            return

        overlay_path = result.get("outputs", {}).get("overlay")
        if overlay_path:
            self.image_viewer.load_image(Path(overlay_path))
        self.result_panel.show_result(result)
        self.statusBar().showMessage(f"Inspection finished: {result.get('final_result', '-')}")


def run_app() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
