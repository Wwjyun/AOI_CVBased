# VisionFlow AOI

Recipe-based AOI computer vision framework built with Python, OpenCV, and PySide6.

This repository is a practical AOI inspection system, not just a detector demo. It contains a reusable inspection pipeline, YAML recipe configuration, multiple tiling strategies, pluggable classical CV detectors, a desktop GUI, batch inspection, folder monitoring, report export, rotating logs, and Windows executable packaging.

The current implementation is aimed at small-to-medium production inspection scenarios where the defect rules can be described by deterministic computer vision logic and tuned through recipes.

## Current Status

The project has moved beyond the original CLI MVP. The current scope includes:

- CLI single-image inspection through `main.py`.
- PySide6 desktop GUI launched by `python main.py --gui`.
- Recipe-driven detector and tiling configuration.
- Grid, template-anchored grid, contour, and pattern-match tiling.
- Detector registry with production-oriented detectors `401`, `401-1`, `401-2`, and `900`.
- Single-image GUI inspection with overlay, result table, thumbnails, and report links.
- Recipe Designer with tiling preview, detector enable/disable, editable parameters, and YAML saving.
- Batch folder inspection with optional recursive scan and parallel workers.
- Batch Dashboard with pass rate, tile statistics, defect ranking, and scatter chart.
- Folder monitor mode that watches for newly added stable images and processes them one by one.
- OP / Engineer / Admin mode separation in the GUI.
- Output toggles for overlay, NG tiles, CSV, matrix CSV, and JSON.
- Rotating file logging for CLI, GUI workers, pipeline, reporter, batch, and monitor processors.
- Windows executable packaging through PyInstaller.
- Release packages named `VisionFlow-AOI-vX.Y.Z-windows-x64.zip`.

Known remaining work:

- Build a formal validation dataset with expected PASS / NG labels.
- Add full per-detector debug image export for all detectors.
- Add AI detector plugin support when model-based inspection is needed.
- Improve long-term production integrations such as MES, operator ID, lot ID, station ID, and recipe history.

## Why This Project Exists

AOI projects often fail when every product variant requires hard-coded inspection logic. This project uses a recipe-based architecture so inspection behavior can be changed by editing YAML instead of rewriting the pipeline.

The design goals are:

- Keep the inspection core independent from the GUI.
- Make detector parameters visible and tunable.
- Preserve traceability through JSON, CSV, images, matrix CSV, and logs.
- Support both engineering workflows and OP production workflows.
- Allow new classical CV detectors to be added without changing the pipeline.
- Keep outputs readable by engineers, production staff, and downstream tools.

## Technology Stack

- Python 3.10 or newer recommended.
- OpenCV for image processing and detector implementation.
- NumPy for numeric operations.
- Pillow for safer large-image loading and preview conversion.
- PyYAML for recipe loading.
- PySide6 for the desktop GUI.
- PyInstaller for Windows executable packaging.

Dependencies are declared in `requirements.txt`:

```text
opencv-python>=4.9.0
numpy>=1.26.0
Pillow>=10.3.0
PyYAML>=6.0.1
PySide6>=6.7.0
PyInstaller>=6.14.0
```

## Repository Layout

```text
.
|-- main.py                         # CLI and GUI entry point
|-- gui_launcher.py                 # GUI launcher helper for packaging
|-- build_exe.ps1                   # PyInstaller build script
|-- VisionFlow AOI.spec             # PyInstaller spec
|-- requirements.txt
|-- README.md
|-- PROJECT_REPORT.md               # project report notes
|-- WEEKLY_UPDATE_2026-06-24_to_2026-07-01.md
|-- export_scatter_plots.py         # standalone scatter plot exporter
|-- export_matrix_summary.py        # standalone matrix summary exporter
|-- core/
|   |-- image_loader.py             # image loading, supported extensions, PIL/OpenCV bridge
|   |-- tiler.py                    # grid, anchored grid, contour, pattern-match tilers
|   |-- pipeline.py                 # inspection orchestration
|   |-- recipe_manager.py           # YAML loading and validation
|   |-- recipe_builder.py           # GUI recipe construction / template path sync
|   |-- detector_manager.py         # detector registry and factory
|   |-- aggregator.py               # tile/final PASS-NG aggregation
|   |-- result_mapper.py            # local bbox to global bbox mapping
|   |-- result_compactor.py         # compact GUI batch/monitor result representation
|   |-- reporter.py                 # overlay, NG tile, CSV, matrix CSV, JSON output
|   |-- logging_system.py           # OOP logging facade and rotating logs
|   |-- batch_processor.py          # parallel batch folder inspection
|   |-- batch_dashboard.py          # dashboard statistics and scatter model
|   `-- monitor_processor.py        # folder monitor processing loop
|-- detectors/
|   |-- base_detector.py            # detector API contract
|   |-- detector_401.py             # negative-pole rotated rectangle NG detector
|   |-- detector_401_1.py           # adaptive circle contour NG detector
|   |-- detector_401_2.py           # adaptive white-ratio contour NG detector
|   `-- detector_900.py             # dual-frame spacing detector
|-- gui/
|   |-- main_window.py              # desktop app shell, state, threads, mode permissions
|   |-- workers.py                  # QThread worker objects
|   |-- image_viewer.py             # image display, overlay interaction
|   |-- detector_labels.py
|   |-- icons.py
|   |-- theme.py
|   |-- screens/
|   |   |-- run_screen.py           # single inspection, OP panel, batch controls
|   |   |-- results_screen.py       # result table, thumbnails, output links
|   |   |-- designer_screen.py      # recipe designer and tiling preview
|   |   |-- batch_dashboard_screen.py
|   |   `-- monitor_screen.py
|   `-- widgets/
|       |-- rail.py
|       |-- topbar.py
|       |-- panel.py
|       |-- drawer.py
|       |-- scatter_chart.py
|       `-- common.py
|-- recipes/
|   |-- PRODUCT_A_AOI_01.yaml
|   |-- PRODUCT_A_CIRCLE_401_1_AOI_01.yaml
|   |-- PRODUCT_A_NEGATIVE_401_AOI_01.yaml
|   |-- PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml
|   `-- PRODUCT_A_FRAME_900_AOI_01.yaml
|-- outputs/
|   |-- overlay/
|   |-- ng_tiles/
|   |-- csv/
|   |-- matrix_csv/
|   |-- json/
|   `-- logs/
`-- outputs_validation/              # local validation output, not product data
```

## System Architecture

The core inspection flow is intentionally separated from the GUI:

```text
Image + Recipe
    |
    v
RecipeManager
    |
    v
ImageLoader
    |
    v
create_tiler(recipe["tile"])
    |
    v
DetectorManager.create_enabled(recipe["detectors"])
    |
    v
for each tile:
    for each detector:
        detector.run(tile.image)
        map bbox_local -> bbox_global
    aggregate tile PASS / NG
    |
    v
Aggregator
    |
    v
Reporter
    |
    v
overlay / NG tiles / CSV / matrix CSV / JSON / logs
```

The GUI does not duplicate inspection logic. It calls the same `AOIPipeline` used by CLI and runs the work in Qt worker threads so the UI remains responsive during image loading, inspection, batch processing, and monitor processing.

## Core Design Concepts

### Recipe-Based Configuration

A recipe is a YAML file that describes:

- Product and machine metadata.
- Tiling strategy.
- Decision rule.
- Enabled detectors.
- Detector display names.
- Detector parameters.
- Output toggles.
- Optional asset/template paths.

The pipeline reads the recipe at runtime, validates required sections, creates the requested tiler, creates enabled detectors, and writes outputs according to the recipe and GUI override options.

### Tiling Strategies

The project supports four practical tiling styles:

- Basic grid tiling: fixed `width`, `height`, `overlap_x`, and `overlap_y`.
- Template-anchored grid: find an anchor by template matching in a search ROI, then create a row/column ROI grid using offsets, ROI size, and gaps.
- Contour tiling: segment the image, find contours, filter by shape/size/circularity, and crop accepted regions as inspection tiles.
- Pattern-match tiling: find repeated template matches, apply local-peak filtering and NMS, sort top-to-bottom / left-to-right, then inspect each match crop.

Each tile carries metadata:

- `tile_id`
- `x`, `y`, `width`, `height`
- `row`, `col`
- `metadata.mode`
- mode-specific details such as template score, match bbox, shape, contour geometry, or anchor ROI.

### Detector Contract

All detectors inherit from `BaseDetector` and expose a shared interface:

- `detector_id`
- `detector_name`
- `display_name`
- `default_params`
- `preprocess(image)`
- `detect(image)`
- `run(image)`

Detector output is normalized so the rest of the pipeline can treat all detectors the same:

```python
{
    "detector_id": "401-1",
    "display_name": "401-1 adaptive circle contour detector",
    "pass": False,
    "score": 0.92,
    "defects": [
        {
            "type": "401_1_circle_detected_ng",
            "bbox_local": [x, y, w, h],
            "bbox_global": [global_x, global_y, w, h],
            "area": 120.0,
            "confidence": 0.92,
            "tile_id": "r0000_c0000",
            "metadata": {}
        }
    ]
}
```

### Local-to-Global Mapping

Detectors work on tile-local images. `core/result_mapper.py` maps every defect from tile-local coordinates back to original-image coordinates:

```text
global_x = tile_x + local_x
global_y = tile_y + local_y
```

This allows overlay images, CSV rows, JSON reports, and GUI interactions to refer back to the original image coordinate system.

### Aggregation

`core/aggregator.py` summarizes detector results at tile and image level:

- A tile is `NG` if any enabled detector returns NG for that tile.
- A tile is `PASS` if all enabled detectors pass.
- `decision.max_ng_count` controls final image tolerance.
- Current final rule is effectively `PASS` when `ng_tile_count <= max_ng_count`, otherwise `NG`.

The summary includes:

- `tile_count`
- `ng_count`
- `defect_count`
- `detector_ng_counts`

## Available Detectors

The active detector registry is defined in `core/detector_manager.py`. Current detectors are:

### Detector `401`: Negative-Pole Rotated Rectangle Detector

File: `detectors/detector_401.py`

Purpose:

- Detect negative-pole NG regions represented by rotated rectangle-like contours.

Main processing:

- Optional ROI inset.
- Gaussian blur.
- Morphological operation, default open.
- Grayscale conversion.
- Adaptive mean threshold.
- Binary INV can be toggled.
- Contour retrieval.
- Rotated rectangle fitting with `cv2.minAreaRect`.
- Area filtering.

Important parameters:

- `roi_inset_px`
- `blur_size`
- `morph_operation`
- `morph_kernel`
- `morph_iterations`
- `adaptive_block_size`
- `adaptive_c`
- `binary_inv`
- `max_value`
- `contour_mode`
- `min_area`
- `max_area`

Defect type:

```text
401_negative_rect_detected_ng
```

Sample recipe:

```text
recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml
```

### Detector `401-1`: Adaptive Circle Contour Detector

File: `detectors/detector_401_1.py`

Purpose:

- Detect circular contour NG regions using adaptive thresholding and circle-like contour filters.

Main processing:

- Grayscale preprocessing.
- Optional ROI inset.
- Optional process downscale.
- Gaussian blur.
- Adaptive mean threshold.
- Optional morphology.
- Contour retrieval.
- Area, circularity, and fill-ratio filtering.
- Circle center/radius metadata.

Important parameters:

- `threshold_method`
- `max_value`
- `invert`
- `blur_size`
- `adaptive_block_size`
- `adaptive_c`
- `roi_inset_px`
- `contour_mode`
- `morph_operation`
- `morph_kernel`
- `morph_iterations`
- `process_scale`
- `min_area`
- `max_area`
- `min_circularity`
- `min_fill_ratio`
- `max_fill_ratio`

Defect type:

```text
401_1_circle_detected_ng
```

Sample recipe:

```text
recipes/PRODUCT_A_CIRCLE_401_1_AOI_01.yaml
```

### Detector `401-2`: Adaptive White-Ratio Contour Detector

File: `detectors/detector_401_2.py`

Purpose:

- Detect NG contours based on the ratio of white pixels inside a contour after adaptive inverse thresholding.

Main processing:

- Grayscale preprocessing.
- Optional ROI inset.
- Gaussian blur.
- Adaptive mean inverse threshold.
- Contour retrieval.
- Optional area filtering.
- Filled contour mask.
- White pixel count divided by contour pixel count.
- NG if ratio is above `white_pixel_ratio_threshold`.

Important parameters:

- `max_value`
- `blur_size`
- `adaptive_block_size`
- `adaptive_c`
- `roi_inset_px`
- `contour_mode`
- `min_area`
- `max_area`
- `white_pixel_ratio_threshold`

Default threshold:

```text
white_pixel_ratio_threshold = 0.625
```

Defect type:

```text
401_2_white_pixel_ratio_ng
```

Sample recipe:

```text
recipes/PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml
```

### Detector `900`: Dual-Frame Spacing Detector

File: `detectors/detector_900.py`

Purpose:

- Inspect a two-frame structure by finding an outer frame and an inner frame, then verifying their relative spacing.

Main processing:

- Grayscale preprocessing.
- Optional ROI inset.
- Outer frame detection with global threshold.
- Inner frame detection with adaptive mean threshold.
- Candidate collection by contour bounding boxes.
- Width/height tolerance filtering for outer and inner candidates.
- Pairing logic that requires the inner frame to sit inside the outer frame.
- Edge gap calculation: left, top, right, bottom.
- PASS only when max edge gap is within `max_edge_gap`.
- NG metadata includes best candidates, rejected candidates, pair debug data, and reason.

Important parameters:

- `outer_threshold`
- `outer_invert`
- `outer_contour_mode`
- `outer_target_width`
- `outer_width_tolerance`
- `outer_target_height`
- `outer_height_tolerance`
- `inner_adaptive_block_size`
- `inner_adaptive_c`
- `inner_invert`
- `inner_contour_mode`
- `inner_target_width`
- `inner_width_tolerance`
- `inner_target_height`
- `inner_height_tolerance`
- `max_edge_gap`
- `roi_inset_px`

Default frame targets:

```text
outer: 1033 +- 33 wide, 1211 +- 33 high
inner: 998 +- 33 wide, 1164 +- 33 high
max_edge_gap: 31 px
```

Defect type:

```text
900_frame_spacing_ng
```

Sample recipe:

```text
recipes/PRODUCT_A_FRAME_900_AOI_01.yaml
```

Special output behavior:

- NG tile output for Detector 900 draws outer candidates, inner candidates, rejected candidates, edge-gap guides, and reason text to help tune the recipe.

## Recipe Format

A recipe must include:

- `recipe_name`
- `product_id`
- `machine_id`
- `version`
- `tile`
- `decision`
- `detectors`
- `output`

Minimal example:

```yaml
recipe_name: "PRODUCT_A_CIRCLE_401_1_AOI_01"
product_id: "PRODUCT_A"
machine_id: "AOI_01"
version: "0.1.0"

tile:
  mode: "grid"
  width: 512
  height: 512
  overlap_x: 64
  overlap_y: 64

decision:
  mode: "all_detectors_must_pass"
  important_detectors:
    - "401-1"
  max_ng_count: 0

detectors:
  "401-1":
    enabled: true
    display_name: "401-1 adaptive circle contour detector"
    params:
      threshold_method: "adaptive_mean"
      max_value: 255
      invert: false
      blur_size: 45
      adaptive_block_size: 33
      adaptive_c: -2.0
      roi_inset_px: 100
      contour_mode: "list"
      morph_operation: "none"
      morph_kernel: 3
      morph_iterations: 1
      process_scale: 1.0
      min_area: 100
      max_area: 1000
      min_circularity: 0.70
      min_fill_ratio: 0.55
      max_fill_ratio: 1.20

output:
  save_overlay: true
  save_ng_tiles: true
  save_csv: true
  save_matrix_csv: true
  save_json: true
```

## Tiling Recipe Examples

### Fixed Grid

```yaml
tile:
  mode: "grid"
  width: 512
  height: 512
  overlap_x: 64
  overlap_y: 64
```

Best for:

- Large images where every region should be inspected.
- Simple fallback inspection.
- Uniform products without strong alignment requirements.

### Template-Anchored Grid

```yaml
tile:
  mode: "grid"
  template_path: "path/to/template.png"
  search_x: 0
  search_y: 0
  search_w: 1200
  search_h: 1200
  offset_x: 10
  offset_y: 20
  rows: 8
  cols: 12
  roi_w: 100
  roi_h: 100
  gap_x: 12
  gap_y: 10
  match_threshold: 0.8
```

Best for:

- Products with repeated units arranged in rows and columns.
- Images with small position drift.
- Inspection where a known anchor must define the ROI grid.

### Contour Tiling

```yaml
tile:
  mode: "contour"
  threshold:
    method: "adaptive_mean"
    max_value: 255
    invert: false
    adaptive_block_size: 31
    adaptive_c: 5.0
    blur_size: 3
    morph_open_kernel: 0
    morph_open_iterations: 1
    morph_close_kernel: 0
    morph_close_iterations: 1
  shapes:
    enabled_shapes: ["rectangle", "circle", "polygon"]
    min_area: 100
    max_area: 0
    min_width: 0
    max_width: 0
    min_height: 0
    max_height: 0
    min_aspect_ratio: 0
    max_aspect_ratio: 0
    min_radius: 0
    max_radius: 0
    min_circularity: 0.75
    polygon_min_vertices: 3
    polygon_max_vertices: 99
    approx_epsilon_ratio: 0.02
    subpixel_enabled: true
    subpixel_window: 5
    crop_padding: 0
```

Best for:

- ROI extraction from visible part contours.
- Products where repeated units are not arranged in a clean grid.
- Engineering exploration and recipe creation.

### Pattern-Match Tiling

```yaml
tile:
  mode: "pattern_match"
  pattern_match:
    template_path: "path/to/template.png"
    match_threshold: 0.8
    max_count: 999
    nms_threshold: 0.3
    crop_padding: 0
    sort_row_tolerance: 20
    max_candidates: 20000
```

Best for:

- Repeated visual structures.
- Multi-match ROI discovery.
- Cases where template matching is more stable than contour segmentation.

## CLI Usage

Install dependencies first:

```powershell
pip install -r requirements.txt
```

Run a single image:

```powershell
python main.py --image C:\path\to\image.png --recipe recipes\PRODUCT_A_CIRCLE_401_1_AOI_01.yaml --output outputs
```

Run with debug flag:

```powershell
python main.py --image C:\path\to\image.png --recipe recipes\PRODUCT_A_FRAME_900_AOI_01.yaml --output outputs --debug
```

Set log options:

```powershell
python main.py --image C:\path\to\image.png --recipe recipes\PRODUCT_A_FRAME_900_AOI_01.yaml --output outputs --log-level DEBUG --log-dir outputs\logs
```

Environment overrides:

```powershell
$env:AOI_LOG_LEVEL='DEBUG'
$env:AOI_LOG_DIR='outputs\logs'
```

CLI output is a JSON summary:

```json
{
  "image_name": "sample.png",
  "recipe_name": "PRODUCT_A_FRAME_900_AOI_01",
  "final_result": "PASS",
  "ng_count": 0,
  "defect_count": 0,
  "duration_sec": 0.423,
  "outputs": {
    "overlay": "outputs\\overlay\\...",
    "ng_tiles_dir": "outputs\\ng_tiles",
    "csv": "outputs\\csv\\...",
    "matrix_csv": "outputs\\matrix_csv\\...",
    "json": "outputs\\json\\..."
  }
}
```

Exit code behavior:

- `0`: final result is `PASS`.
- `2`: final result is `NG`.
- Other non-zero errors may occur for invalid recipe, invalid image, or runtime exceptions.

## GUI Usage

Launch:

```powershell
python main.py --gui
```

The GUI window title is `VisionFlow AOI`.

Main screens:

- Run: load image, load recipe, run single inspection, view overlay, inspect recent history, run batch folder inspection.
- Monitor: watch a folder and process new stable images.
- Recipe Designer: edit recipe metadata, tiling mode, detector enablement, and detector parameters.
- Results: inspect final result, defect table, thumbnails, and output paths.
- Batch Dashboard: analyze batch result summary and per-image tile scatter.

### GUI Modes

The GUI supports three operating modes:

- OP mode: restricted production-facing mode; only monitor workflow is visible.
- Engineer mode: engineering workflow with advanced detector parameters partially hidden in the designer.
- Admin mode: full recipe and detector parameter access.

This is UI-level permission separation for workflow safety. It is not a security boundary.

### Single Inspection Workflow

1. Launch GUI.
2. Load an inspection image.
3. Load a recipe.
4. Confirm output toggles in the settings drawer.
5. Run inspection.
6. Review PASS / NG, tile count, NG count, defect count, and duration.
7. Inspect overlay and defect table.
8. Open generated output files from the result panel.

### Recipe Designer Workflow

1. Load an image.
2. Open Recipe Designer.
3. Choose tiling mode: pattern match, grid, or contour.
4. Configure template path, search ROI, grid rows/cols, contour threshold, or shape filters.
5. Preview tiles on the loaded image.
6. Enable detectors.
7. Tune detector parameters.
8. Save the recipe YAML.
9. The saved recipe is loaded back into the GUI.

### Batch Folder Workflow

1. Load a recipe.
2. Choose a folder of images.
3. Choose whether to scan recursively.
4. Start batch inspection.
5. Each supported image is processed by the same `AOIPipeline`.
6. Results are written under `outputs\batch\<timestamp>\`.
7. Batch Dashboard summarizes total images, PASS, NG, ERROR, defects, tiles, and NG tiles.

Batch worker behavior:

- Supported extensions come from `core.image_loader.SUPPORTED_EXTENSIONS`.
- Worker count defaults to up to 4 workers, bounded by CPU count and image count.
- `AOI_BATCH_WORKERS` can override the worker count.
- Each image result is compacted in memory to reduce long-run GUI slowdown.
- Full detail remains in the JSON report output.

### Monitor Workflow

1. Load a recipe.
2. Choose a monitor input folder.
3. Optionally choose a processed-image move folder.
4. Start monitoring.
5. Existing images are treated as already seen.
6. Newly added images are processed after they pass stable-file checks.
7. Images are processed one by one.
8. Results appear in the monitor table and scatter chart.
9. If a move folder is configured, processed images are moved while preserving subfolders.

Monitor behavior:

- Recursive folder scan is used.
- File stability is checked by size and modified time.
- Poll interval defaults to 1 second.
- Stable checks default to 2.
- Processed image moves use unique filenames when collisions occur.
- The monitor table can open processed/original image paths.

## Outputs

`core/reporter.py` writes output files into the configured output directory.

Default folders:

```text
outputs/
|-- overlay/
|-- ng_tiles/
|-- csv/
|-- matrix_csv/
|-- json/
`-- logs/
```

### Overlay PNG

Path pattern:

```text
outputs/overlay/<image>_<recipe>_<timestamp>_<uuid>_overlay.png
```

Behavior:

- For status-tile modes such as grid and pattern match, overlay draws tile-level OK/NG frames.
- OK tiles are green.
- NG tiles are red.
- For defect-bbox overlays, detected defects are drawn on original image coordinates.
- Circle metadata is drawn with a circle plus bbox when available.

### NG Tiles

Folder:

```text
outputs/ng_tiles/
```

Behavior:

- Only NG tile crops are written.
- Local defect boxes are drawn on the tile crop.
- Detector 900 NG tiles include additional debug overlays for outer/inner candidates, rejected candidates, gap lines, and reason text.

### Defect CSV

Folder:

```text
outputs/csv/
```

Fields:

- `image_name`
- `recipe_name`
- `machine_id`
- `product_id`
- `final_result`
- `detector_id`
- `defect_type`
- `bbox_global`
- `bbox_local`
- `tile_id`
- `score`
- `area`

Encoding:

- UTF-8 with BOM (`utf-8-sig`) for easier Excel opening.

### Matrix CSV

Folder:

```text
outputs/matrix_csv/
```

Purpose:

- Convert row/column tile NG status into a matrix-style CSV.
- Columns are named `c1`, `c2`, `c3`, etc.
- Row IDs include the source image stem and reversed row numbering.
- NG cells are marked with a check mark.

This format is useful when a product has a physical row/column layout and production staff need a compact defect map.

### JSON Report

Folder:

```text
outputs/json/
```

The JSON report contains:

- Image metadata.
- Recipe metadata.
- Final result.
- Runtime duration.
- Summary counts.
- Output paths.
- Tile list.
- Detector results.
- Defect lists.
- Local/global coordinates.
- Detector-specific metadata.

JSON is the richest output and should be used for traceability or downstream processing.

### Logs

Default log paths:

- CLI: `<output>\logs\aoi.log`
- GUI: `outputs\logs\aoi.log`

The logging system is implemented in `core/logging_system.py` and provides:

- OOP logging facade.
- Shared logger access through `LogMixin`.
- Rotating file handler.
- Console handler.
- Environment variable overrides.
- Named loggers for pipeline, reporter, batch, monitor, GUI workers, and main entry point.

## Standalone Export Tools

Two helper tools are included:

```powershell
python export_scatter_plots.py
python export_matrix_summary.py
```

Their purpose:

- Export scatter-style visual summaries from folders containing JSON and/or CSV reports.
- Combine matrix CSV outputs into summary reports.
- Support recursive folder selection and practical post-inspection review.

These are separate from the main GUI pipeline so analysis/reporting utilities can evolve without complicating the inspection core.

## Build Windows EXE

Build from the project virtual environment:

```powershell
.\build_exe.ps1
```

The executable is created under:

```text
dist\VisionFlow AOI\VisionFlow AOI.exe
```

Important:

- Copy or zip the whole `dist\VisionFlow AOI` folder.
- The executable depends on the `_internal` runtime directory next to it.
- Do not distribute only the `.exe`.

Release package naming:

```text
VisionFlow-AOI-vX.Y.Z-windows-x64.zip
```

Packages currently present in the workspace include:

- `VisionFlow-AOI-v1.0.0-windows-x64.zip`
- `VisionFlow-AOI-v1.1.0-windows-x64.zip`

Older pre-product-name packages are also present:

- `AOI_GUI-v0.2.0-windows-x64.zip`
- `AOI_GUI-v0.3.0-windows-x64.zip`

## Validation

Basic compile validation:

```powershell
.\env\Scripts\python.exe -m compileall main.py core detectors gui
```

GUI offscreen smoke test:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\env\Scripts\python.exe -c "from pathlib import Path; from PySide6.QtWidgets import QApplication; from gui.main_window import MainWindow; app=QApplication([]); w=MainWindow(); w.recipe_panel.load_recipe(Path('recipes/PRODUCT_A_AOI_01.yaml')); print(w.windowTitle(), w.recipe_panel.detector_list.count())"
```

CLI smoke testing can be done with a temporary synthetic image and output directed to:

```text
outputs_validation/
```

Recommended validation before release:

- Compile all modules.
- Launch GUI in offscreen mode.
- Run CLI on known PASS image.
- Run CLI on known NG image.
- Verify overlay, NG tiles, CSV, matrix CSV, JSON.
- Verify batch folder run with at least two images.
- Verify monitor mode can process a newly copied image.
- Verify packaged executable opens and can run one recipe.

## Engineering Notes

### Strengths in the Current Codebase

- The pipeline is separated from UI code and can run from CLI, GUI, batch, and monitor workflows.
- Detector outputs are normalized, which keeps reporter and aggregator logic detector-agnostic.
- Recipe files make product tuning possible without editing Python code.
- Tiling is extensible and supports real alignment problems beyond simple fixed grids.
- Batch and monitor processors reuse the same single-image pipeline.
- Output reports include both visual and machine-readable formats.
- Rotating logs make long-running GUI/batch/monitor diagnosis practical.
- GUI workers keep expensive image and inspection work outside the main Qt UI thread.
- Memory compaction was added for batch/monitor result storage to reduce long-run slowdown.

### Current Tradeoffs

- Detectors are currently classical CV rules, so quality depends heavily on lighting, threshold tuning, and fixture consistency.
- The project does not yet include a formal labeled validation dataset, so performance claims should be verified per product.
- GUI mode permissions are workflow restrictions, not authentication/security controls.
- Some legacy notes and reports in the repository have encoding damage; the current README and source code should be treated as the reliable reference.
- `--debug` exists at CLI level, but complete per-detector debug image export is still planned work. Detector 900 already writes rich NG tile debug overlays.

### Good Next Improvements

- Create `tests/` with unit tests for tiler, aggregator, recipe validation, result mapping, and detector edge cases.
- Create a small synthetic validation dataset committed to the repo or generated by test fixtures.
- Add golden-output smoke tests for report files.
- Add per-detector debug artifact export controlled by recipe/output settings.
- Add recipe version history and signed/locked production recipes.
- Add operator/lot/station metadata to reports.
- Add a plugin layer for YOLO/RT-DETR/segmentation detectors while preserving the current detector result format.

## Quick Start

### Optional CUDA DLL

Recipe Designer 可分別勾選切小圖 GPU、GUI 預覽 GPU，以及每一個 detector 的 GPU。設定會保存為：

```yaml
gpu:
  tiling: false
  display: false
  dll_path: "gpu/visionflow_cuda.dll"
  fallback_to_cpu: true

detectors:
  "401-1":
    enabled: true
    use_gpu: false
```

所有開關預設關閉。勾選後會透過額外的 CUDA DLL 執行；DLL 或 CUDA device 不可用時預設回退 CPU，實際 backend 與原因會寫入結果的 `execution.gpu`。

RTX 3090 (`sm_86`) 主機安裝 CUDA Toolkit 後可執行：

```powershell
.\gpu\build_cuda_dll.ps1
```

Qt 的 QImage/QPixmap、overlay、文字與檔案輸出仍由 CPU 處理；GPU 顯示選項加速的是預覽影像的色彩轉換。CPU/GPU 功能進度、CUDA 編譯與完整實機驗收步驟統一維護在 [`Todo.md`](Todo.md)。

```powershell
cd C:\Users\王\Desktop\AOI_CVbased
.\env\Scripts\python.exe -m pip install -r requirements.txt
.\env\Scripts\python.exe main.py --gui
```

Or run CLI:

```powershell
.\env\Scripts\python.exe main.py --image C:\path\to\image.png --recipe recipes\PRODUCT_A_FRAME_900_AOI_01.yaml --output outputs
```

For reviewers evaluating the project, start with:

- `core/pipeline.py` for the inspection orchestration.
- `core/tiler.py` for ROI generation strategies.
- `detectors/` for detector implementations.
- `core/reporter.py` for output generation and traceability.
- `core/batch_processor.py` and `core/monitor_processor.py` for production-style workflows.
- `gui/main_window.py` and `gui/screens/` for the desktop application.
