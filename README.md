# AOI CV Based

Recipe based AOI computer vision framework. The current MVP focuses on a command line inspection pipeline:

- Load `jpg`, `png`, `bmp`, `tif`, `tiff` images with OpenCV.
- Split large images into overlapping tiles.
- Run enabled detectors from a YAML recipe.
- Map local tile defects back to global image coordinates.
- Aggregate PASS / NG decisions.
- Export overlay images, NG tiles, CSV, and JSON reports.

## Project Layout

```text
.
|-- main.py
|-- requirements.txt
|-- README.md
|-- recipes/
|   `-- PRODUCT_A_AOI_01.yaml
|-- core/
|   |-- image_loader.py
|   |-- tiler.py
|   |-- pipeline.py
|   |-- recipe_manager.py
|   |-- detector_manager.py
|   |-- aggregator.py
|   |-- result_mapper.py
|   `-- reporter.py
|-- detectors/
|   |-- base_detector.py
|   |-- detector_401.py
|   `-- detector_401_1.py
`-- outputs/
    |-- overlay/
    |-- ng_tiles/
    |-- csv/
    `-- json/
```

## Environment Setup

Create and activate your own Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Run CLI Inspection

```powershell
python main.py --image C:\path\to\image.png --recipe recipes\PRODUCT_A_AOI_01.yaml --output outputs
```

Optional arguments:

```powershell
python main.py --image C:\path\to\image.png --recipe recipes\PRODUCT_A_AOI_01.yaml --output outputs --debug
```

The command prints the final PASS / NG result and report file paths.

## Run GUI

```powershell
python main.py --gui
```

The GUI supports:

- Loading an inspection image.
- Loading a YAML recipe.
- Viewing recipe metadata and detector parameters.
- Running the current recipe against the loaded image.
- Viewing the generated overlay and defect table.

## Recipe Format

Recipes are YAML files. Example:

```yaml
recipe_name: "PRODUCT_A_AOI_01"
product_id: "PRODUCT_A"
machine_id: "AOI_01"
version: "0.1.0"

tile:
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
      min_area: 100
      max_area: 1000
      min_circularity: 0.70
      min_fill_ratio: 0.55
      max_fill_ratio: 1.20

output:
  save_overlay: true
  save_ng_tiles: true
  save_csv: true
  save_json: true
```

## Detector 401-1

`detector_401_1` is an adaptive mean threshold and circular contour NG detector. Finding a matching circle produces NG; finding none produces PASS. It supports:

- `blur_size`: Gaussian blur kernel size, default `45`.
- `adaptive_block_size`: adaptive mean block size, default `33`.
- `adaptive_c`: adaptive threshold C value, default `-2.0`.
- `roi_inset_px`: pixels to inset from each edge before detection, default `100`.
- `contour_mode`: `external`, `list` / `all`, or `tree`.
- `min_area` / `max_area`: contour area range.
- `min_circularity`: minimum circularity.
- `min_fill_ratio` / `max_fill_ratio`: contour fill ratio range.

Detector output follows the shared detector result format:

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
            "area": 120,
            "confidence": 0.92,
            "metadata": {}
        }
    ]
}
```

## Detector 401

`detector_401` is a negative-pole rotated rectangle NG detector. The image is processed with Gaussian blur kernel `15`, morphology open kernel `5` for `10` iterations, grayscale conversion, adaptive mean inverted threshold block `29` C `5`, then contour list retrieval. Matching rotated rectangles with area from `25` to `10000` produce NG; finding none produces PASS.

The sample recipe is `recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml`.

## Available Detectors

- `401`: negative-pole adaptive mean rotated rectangle NG detector.
- `401-1`: adaptive mean circular contour NG detector.

The sample recipe enables `401-1` by default.

## Outputs

The pipeline can write:

- `outputs/overlay/*.png`: original image with global defect boxes.
- `outputs/ng_tiles/*.png`: NG tile crops.
- `outputs/csv/*.csv`: flat defect table.
- `outputs/json/*.json`: full inspection result.

CSV fields include image name, recipe name, machine ID, product ID, final result, detector ID, defect type, global and local bounding boxes, tile ID, score, and area.

## Current Scope

Implemented MVP scope:

- Image loader
- Tiler
- Base detector API
- Detector manager
- Detector 401-1
- Recipe loader and validator
- Result mapper
- Aggregator
- Reporter
- CLI entry point
- GUI shell with recipe panel, image viewer, detector parameter view, and result table

OP mode, detector debug image export, and editable recipe saving are intentionally left for later phases.
