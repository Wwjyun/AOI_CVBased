from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.recipe_manager import RecipeManager


class RecipePanel(QWidget):
    recipe_loaded = Signal(Path, dict)
    detector_selected = Signal(str, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.recipe_path: Path | None = None
        self.recipe: dict | None = None
        self._manager = RecipeManager()

        self.load_button = QPushButton("Load Recipe")
        self.load_button.clicked.connect(self._choose_recipe)

        self.recipe_name = QLabel("-")
        self.product_id = QLabel("-")
        self.machine_id = QLabel("-")
        self.version = QLabel("-")
        form = QFormLayout()
        form.addRow("Recipe", self.recipe_name)
        form.addRow("Product", self.product_id)
        form.addRow("Machine", self.machine_id)
        form.addRow("Version", self.version)

        meta_group = QGroupBox("Recipe")
        meta_group.setLayout(form)

        self.detector_list = QListWidget()
        self.detector_list.currentRowChanged.connect(self._emit_detector)

        layout = QVBoxLayout(self)
        layout.addWidget(self.load_button)
        layout.addWidget(meta_group)
        layout.addWidget(QLabel("Detectors"))
        layout.addWidget(self.detector_list, 1)

    def load_recipe(self, path: Path) -> None:
        recipe = self._manager.load(path)
        self.recipe_path = Path(path)
        self.recipe = recipe
        self.recipe_name.setText(str(recipe.get("recipe_name", "-")))
        self.product_id.setText(str(recipe.get("product_id", "-")))
        self.machine_id.setText(str(recipe.get("machine_id", "-")))
        self.version.setText(str(recipe.get("version", "-")))

        self.detector_list.clear()
        for detector_id, config in recipe.get("detectors", {}).items():
            state = "ON" if config.get("enabled", False) else "OFF"
            display = config.get("display_name", detector_id)
            self.detector_list.addItem(f"{detector_id} [{state}] {display}")

        if self.detector_list.count():
            self.detector_list.setCurrentRow(0)
        self.recipe_loaded.emit(self.recipe_path, recipe)

    def _choose_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Recipe", "recipes", "YAML Files (*.yaml *.yml)")
        if path:
            self.load_recipe(Path(path))

    def _emit_detector(self, row: int) -> None:
        if row < 0 or not self.recipe:
            return
        detector_items = list(self.recipe.get("detectors", {}).items())
        if row >= len(detector_items):
            return
        detector_id, config = detector_items[row]
        self.detector_selected.emit(str(detector_id), config)
