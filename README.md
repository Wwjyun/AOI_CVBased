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
|   `-- detector_999.py
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
    - "999"
  max_ng_count: 0

detectors:
  "999":
    enabled: true
    display_name: "Dark / bright blob detector"
    params:
      threshold: 45
      min_area: 20
      max_area: 5000
      blur_size: 3
      invert: false
      clahe_enabled: true

output:
  save_overlay: true
  save_ng_tiles: true
  save_csv: true
  save_json: true
```

## Detector 999

`detector_999` is a threshold and contour based blob detector. It supports:

- `threshold`: binary threshold value.
- `min_area`: minimum contour area.
- `max_area`: maximum contour area.
- `blur_size`: Gaussian blur kernel size. Use `0` or `1` to disable.
- `invert`: use inverse binary threshold when `true`.
- `clahe_enabled`: apply CLAHE before thresholding.

Detector output follows the shared detector result format:

```python
{
    "detector_id": "999",
    "display_name": "Dark / bright blob detector",
    "pass": False,
    "score": 0.92,
    "defects": [
        {
            "type": "blob",
            "bbox_local": [x, y, w, h],
            "area": 120,
            "confidence": 0.92,
            "metadata": {}
        }
    ]
}
```

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
- Detector 999
- Recipe loader and validator
- Result mapper
- Aggregator
- Reporter
- CLI entry point

GUI files and additional detectors are intentionally left for later phases.
