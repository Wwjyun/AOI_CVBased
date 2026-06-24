from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from core.aggregator import Aggregator
from core.detector_manager import DetectorManager
from core.image_loader import load_image
from core.logging_system import LogMixin
from core.recipe_manager import RecipeManager
from core.recipe_builder import RecipeTemplatePathSync
from core.reporter import Reporter
from core.result_mapper import map_tile_result_to_global
from core.tiler import create_tiler


class AOIPipeline(LogMixin):
    def __init__(
        self,
        recipe_path: Path,
        output_dir: Path,
        debug: bool = False,
        progress_callback: Callable[[int, str], None] | None = None,
        output_overrides: dict | None = None,
    ):
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.progress_callback = progress_callback
        self.output_overrides = output_overrides
        self.recipe_manager = RecipeManager()
        self.detector_manager = DetectorManager()

    def run(self, image_path: Path) -> dict:
        started = time.perf_counter()
        self.logger.info(
            "Inspection started: image=%s recipe=%s output=%s debug=%s",
            image_path,
            self.recipe_path,
            self.output_dir,
            self.debug,
        )
        self._progress(0, "Starting inspection")
        recipe = self.recipe_manager.load(self.recipe_path)
        if self.output_overrides:
            recipe["output"] = {**recipe.get("output", {}), **self.output_overrides}
        recipe = RecipeTemplatePathSync.from_recipe(recipe).apply(recipe)
        self.logger.info("Recipe loaded: name=%s version=%s", recipe.get("recipe_name"), recipe.get("version"))
        self._progress(5, "Recipe loaded")
        image = load_image(image_path)
        self.logger.info("Image loaded: image=%s shape=%s", image_path, getattr(image, "shape", None))
        self._progress(10, "Image loaded")
        tile_config = recipe["tile"]
        tiler = create_tiler(tile_config)
        detectors = self.detector_manager.create_enabled(self.recipe_manager.enabled_detectors(recipe))
        self.logger.info("Detectors initialized: count=%s ids=%s", len(detectors), [d.detector_id for d in detectors])
        self._progress(15, "Detectors initialized")

        tiles = list(tiler.iter_tiles(image))
        self.logger.info("Tiles prepared: count=%s mode=%s", len(tiles), tile_config.get("mode", "grid"))
        self._progress(20, f"Tiles prepared: {len(tiles)}")

        tile_results = []
        total_work = max(len(tiles) * max(len(detectors), 1), 1)
        completed_work = 0
        for tile_index, tile in enumerate(tiles, start=1):
            detector_results = []
            for detector in detectors:
                detector_result = detector.run(tile.image)
                detector_results.append(map_tile_result_to_global(tile, detector_result))
                completed_work += 1
                percent = 20 + int(completed_work / total_work * 60)
                self._progress(
                    min(percent, 80),
                    f"Inspecting tile {tile_index}/{len(tiles)} with detector {detector.detector_id}",
                )

            if not detectors:
                completed_work += 1
                percent = 20 + int(completed_work / total_work * 60)
                self._progress(min(percent, 80), f"Preparing tile {tile_index}/{len(tiles)}")

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

        self._progress(85, "Aggregating PASS / NG result")
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
            "duration_sec": round(time.perf_counter() - started, 3),
        }

        serializable_result = self._without_runtime_images(result)
        self._progress(92, "Writing overlay, CSV, and JSON")
        outputs = Reporter(self.output_dir, recipe["output"]).write(image, result)
        serializable_result["outputs"] = outputs
        self.logger.info(
            "Inspection completed: image=%s final=%s defects=%s ng_tiles=%s duration=%.3fs",
            Path(image_path).name,
            serializable_result["final_result"],
            serializable_result["summary"].get("defect_count", 0),
            serializable_result["summary"].get("ng_count", 0),
            serializable_result["duration_sec"],
        )
        self._progress(100, "Inspection complete")
        return serializable_result

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(max(0, min(100, int(percent))), message)

    @staticmethod
    def _without_runtime_images(result: dict) -> dict:
        cleaned = dict(result)
        cleaned["tiles"] = []
        for tile_result in result["tiles"]:
            cleaned_tile = dict(tile_result)
            cleaned_tile.pop("_tile_image", None)
            cleaned["tiles"].append(cleaned_tile)
        return cleaned
