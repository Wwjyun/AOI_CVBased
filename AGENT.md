# AGENT.md

Instructions for future Codex work in this repository.

## Project

This is an AOI computer vision system built around a recipe based OpenCV pipeline and a PySide6 GUI.

Primary entry points:

- CLI: `python main.py --image <image> --recipe recipes/PRODUCT_A_AOI_01.yaml --output outputs`
- GUI: `python main.py --gui`

Use the local virtual environment first:

```powershell
.\env\Scripts\python.exe
```

## Working Rules

- Read `todo.md` before starting implementation work.
- Keep completed work marked in `todo.md`, preferably in the `Progress / Completed Items` section because the original TODO text has encoding issues.
- Preserve existing user changes. Do not revert unrelated edits.
- Keep generated validation files out of git. `.gitignore` already excludes temporary validation images and `outputs_validation/`.
- Use focused changes that match the existing project structure.

## Required End-of-Change Workflow

For every code or documentation change:

1. Run validation before finishing.
2. Commit the completed change.
3. Push to `origin/main`.
4. Report the commit hash and validation result.

Recommended validation commands:

```powershell
.\env\Scripts\python.exe -m compileall main.py core detectors gui
```

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\env\Scripts\python.exe -c "from pathlib import Path; from PySide6.QtWidgets import QApplication; from gui.main_window import MainWindow; app=QApplication([]); w=MainWindow(); w.recipe_panel.load_recipe(Path('recipes/PRODUCT_A_AOI_01.yaml')); print(w.windowTitle(), w.recipe_panel.detector_list.count())"
```

Run a CLI smoke test with a temporary synthetic image when pipeline behavior may be affected.

## Git

Default branch is `main`.

```powershell
git status --short --branch
git add .
git commit -m "<concise message>"
git push origin main
```

Only skip commit or push if the user explicitly asks not to, or if validation fails and the user wants the broken state preserved.
