from __future__ import annotations

import os
from pathlib import Path
import sys

from gui.main_window import run_app


def bundled_recipe_path() -> Path:
    """Return a recipe from source checkout or PyInstaller's one-dir bundle."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / "recipes" / "PRODUCT_A_AOI_01.yaml"


def run_packaged_smoke_test() -> int:
    """Exercise bundled Qt startup and recipe loading without entering the event loop."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    recipe_path = bundled_recipe_path()
    if not recipe_path.is_file():
        return 2
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.recipe_panel.load_recipe(recipe_path)
    app.processEvents()
    valid = bool(window.windowTitle()) and window.recipe_panel.detector_list.count() > 0
    window.close()
    app.processEvents()
    return 0 if valid else 3


if __name__ == "__main__":
    if "--smoke-test" in sys.argv[1:]:
        raise SystemExit(run_packaged_smoke_test())
    raise SystemExit(run_app())
