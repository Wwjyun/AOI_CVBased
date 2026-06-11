from __future__ import annotations

from pathlib import Path

from core.aggregator import Aggregator
from core.detector_manager import DetectorManager
from core.image_loader import load_image
from core.recipe_manager import RecipeManager
from core.reporter import Reporter
from core.result_mapper import map_tile_result_to_global
from core.tiler import create_tiler


class AOIPipeline:
    def __init__(self, recipe_path: Path, output_dir: Path, debug: bool = False):
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.recipe_manager = RecipeManager()
        self.detector_manager = DetectorManager()

    def run(self, image_path: Path) -> dict:
        recipe = self.recipe_manager.load(self.recipe_path)
        image = load_image(image_path)
        tile_config = recipe["tile"]
        tiler = create_tiler(tile_config)
        detectors = self.detector_manager.create_enabled(self.recipe_manager.enabled_detectors(recipe))

        tile_results = []
        for tile in tiler.iter_tiles(image):
            detector_results = []
            for detector in detectors:
                detector_result = detector.run(tile.image)
                detector_results.append(map_tile_result_to_global(tile, detector_result))

            tile_results.append(
                {
                    "tile": {
                        "tile_id": tile.tile_id,
                        "x": tile.x,
                        "y": tile.y,
                        "width": tile.width,
                        "height": tile.height,
                        "row": tile.row,
                        "col": tile.col,
                        "metadata": tile.metadata or {},
                    },
                    "detectors": detector_results,
                    "_tile_image": tile.image,
                }
            )

        aggregate = Aggregator(recipe["decision"]).aggregate(tile_results)
        result = {
            "image_name": Path(image_path).name,
            "recipe_name": recipe["recipe_name"],
            "machine_id": recipe["machine_id"],
            "product_id": recipe["product_id"],
            "recipe_version": recipe["version"],
            "final_result": aggregate["final_result"],
            "summary": aggregate["summary"],
            "tiles": tile_results,
            "outputs": {},
        }

        serializable_result = self._without_runtime_images(result)
        outputs = Reporter(self.output_dir, recipe["output"]).write(image, result)
        serializable_result["outputs"] = outputs
        return serializable_result

    @staticmethod
    def _without_runtime_images(result: dict) -> dict:
        cleaned = dict(result)
        cleaned["tiles"] = []
        for tile_result in result["tiles"]:
            cleaned_tile = dict(tile_result)
            cleaned_tile.pop("_tile_image", None)
            cleaned["tiles"].append(cleaned_tile)
        return cleaned
