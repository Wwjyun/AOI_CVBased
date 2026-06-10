from __future__ import annotations

from copy import deepcopy


def map_tile_result_to_global(tile, detector_result: dict) -> dict:
    mapped = deepcopy(detector_result)
    for defect in mapped.get("defects", []):
        local_x, local_y, width, height = defect["bbox_local"]
        defect["bbox_global"] = [tile.x + local_x, tile.y + local_y, width, height]
        defect["tile_id"] = tile.tile_id
        defect["tile"] = {
            "x": tile.x,
            "y": tile.y,
            "width": tile.width,
            "height": tile.height,
            "row": tile.row,
            "col": tile.col,
        }
    return mapped
