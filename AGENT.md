# Repository Agent Instructions

These rules apply to all future Codex work in this repository.

## Project and environment

VisionFlow AOI is a recipe-driven OpenCV inspection system with a PySide6 GUI and an optional CUDA DLL backend.

Primary entry points:

- CLI: `python main.py --image <image> --recipe <recipe.yaml> --output <directory>`
- GUI: `python main.py --gui`
- CUDA build: `gpu/build_cuda_dll.ps1`
- CUDA validation: `gpu/validate_cuda_dll.py`

Use the workspace virtual environment for every Python command:

```powershell
.\env\Scripts\python.exe
```

The normal development machine may not have `nvcc`, CMake, or an NVIDIA GPU. Never claim that CUDA source compiled or passed runtime validation unless those commands actually ran. Record outstanding RTX 3090 validation in `Todo.md`.

## Canonical roadmap

- `Todo.md` is the only project task list. Read it before implementation work.
- Do not create separate CPU, GPU, CUDA, GUI, release, or feature Todo files.
- Mark only work that is genuinely complete. Hardware-dependent tasks remain unchecked until tested on the target machine.
- After a completed change, update the relevant checkbox and append a dated entry under `完成紀錄`.
- Keep CPU correctness, GPU optimization, deployment, and acceptance criteria in the same roadmap.

## Module ownership

- `core/`: pipeline, recipe loading, tiling, reporting, profiling, GPU bridge, preprocessing plans and executors.
- `detectors/`: detector-specific feature extraction, geometry, filtering, and result metadata.
- `gpu/`: CUDA C ABI, kernels, persistent contexts, build scripts, native smoke tests, and CPU/GPU validation.
- `gui/`: PySide6 screens, widgets, workers, status, and preview behavior.
- `recipes/`: YAML configuration and production defaults.
- `tests/`: automated correctness, fallback, routing, and regression tests.
- `.github/workflows/`: CI only; keep GPU runtime jobs isolated from ordinary hosted runners.

Put behavior in the narrowest appropriate module. Do not duplicate pipeline or fallback policy inside individual detectors.

## CPU/GPU architecture contract

- CPU-only operation is a fully supported product mode and the correctness reference.
- Missing GPU, missing/old DLL, unsupported operator, CUDA initialization failure, kernel error, or OOM must not break CPU execution when fallback is enabled.
- A failed GPU step must restart the entire detector on CPU. Never combine partial GPU intermediate results with a CPU continuation.
- Preserve recipe semantics, PASS/NG, coordinates, defect metadata, output formats, and ordering. Define and test any allowed numerical tolerance.
- Do not create one CUDA workflow or exported function per detector.
- Detectors declare backend-neutral immutable `PreprocessPlan` objects using shared typed operators.
- `CpuPreprocessExecutor` defines OpenCV fallback semantics. `CudaPreprocessExecutor` selects a generic native plan, compatibility adapter, reusable primitives, or explicit fallback.
- Add a shared operator when an algorithm is reusable. Detector-named native adapters are compatibility code, not the extension model.
- Do not silently substitute a faster operation with different semantics, such as nearest-neighbor for OpenCV `INTER_AREA`.
- Prefer one upload, multiple device operators, and one necessary download. Reuse context buffers across operators, tiles, and images where lifetime permits.
- Keep small contour/geometry work, YAML, aggregation, GUI control, CSV/JSON, PNG encoding, and disk I/O on CPU unless profiling proves otherwise.
- GPU default enablement requires RTX 3090 equivalence, stability, and end-to-end performance evidence.

## Compatibility and OOP rules

- Preserve the public ABI v1 primitive API unless an explicit versioned migration is planned.
- Add native capabilities through optional export probing so old DLLs retain legacy GPU or CPU fallback paths.
- Device pointers belong to native context objects; do not expose ownerless raw device pointers to Python.
- Keep runtime lifecycle explicit with `close()`/context manager behavior and safe cleanup.
- Keep shared runtime calls thread-safe. A single bounded GPU queue is preferred over competing workers.
- Avoid module globals that hold mutable detector, recipe, image, or GPU state.
- Inject runtime/backend dependencies where tests need CPU, fake DLL, legacy DLL, or failing GPU behavior.

## Required workflow

Before editing:

1. Run `git status --short --branch`.
2. Read the relevant `Todo.md` sections and nearby implementation/tests.
3. Identify user-owned or unrelated working-tree changes and preserve them.

While editing:

1. Make focused changes with the existing module boundaries.
2. Add or update automated tests for behavior, CPU equivalence, old-DLL routing, and failure fallback.
3. Update `Todo.md` accurately; do not mark source-only CUDA work as hardware-validated.
4. Keep generated files under ignored validation/output directories.

Before finishing, always run:

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
.\env\Scripts\python.exe -m compileall main.py core detectors gui
git diff --check
```

For pipeline, detector, recipe, tiling, GPU bridge, or reporter changes, also run a CLI smoke test with a synthetic image and write only to `outputs_validation/`.

For GUI changes, also run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\env\Scripts\python.exe -c "from pathlib import Path; from PySide6.QtWidgets import QApplication; from gui.main_window import MainWindow; app=QApplication([]); w=MainWindow(); w.recipe_panel.load_recipe(Path('recipes/PRODUCT_A_AOI_01.yaml')); print(w.windowTitle(), w.recipe_panel.detector_list.count())"
```

For CUDA header/source/API changes:

- Run all available Python/fake-DLL/static checks locally.
- Inspect public declarations, native smoke coverage, validation tooling, and brace/argument consistency.
- If `nvcc` is unavailable, explicitly report that the DLL was not rebuilt.
- Leave RTX 3090 compile, primitive matrix, production recipe equivalence, benchmark, and stress tasks unchecked until executed.

If any required validation fails, fix it and rerun the relevant full set before commit.

## Git and artifacts

- Default branch and push target: `main` → `origin/main`.
- Stage only files that belong to the current task. Do not use `git add .` in a dirty workspace.
- Never commit user-provided release ZIPs, `outputs/logs/`, `outputs_validation/`, temporary images, generated reports, DLL build outputs, or unrelated changes.
- Do not discard, reset, overwrite, or reformat unrelated user changes.
- Use a concise commit message describing the completed outcome.
- Push every completed validated change unless the user explicitly says not to push.

Typical safe sequence:

```powershell
git status --short --branch
git add -- <explicit files>
git diff --cached --check
git commit -m "<concise outcome>"
git push origin main
git status -sb
```

## Final handoff

Report:

- What changed and which roadmap items were marked.
- CPU, fallback, GUI, CUDA, and compatibility impact as applicable.
- Exact validation commands and results.
- Any validation that could not run, especially `nvcc`/RTX 3090 work.
- Commit hash and push result.
- Remaining untracked user artifacts only when relevant.
