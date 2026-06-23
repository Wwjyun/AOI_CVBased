from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class RecipeError(RuntimeError):
    pass


class RecipeManager:
    REQUIRED_TOP_LEVEL_KEYS = {"recipe_name", "product_id", "machine_id", "version", "tile", "decision", "detectors", "output"}

    def load(self, path: Path) -> dict[str, Any]:
        recipe_path = Path(path)
        if not recipe_path.exists():
            raise RecipeError(f"Recipe does not exist: {recipe_path}")

        with recipe_path.open("r", encoding="utf-8") as handle:
            recipe = yaml.safe_load(handle) or {}

        self.validate(recipe)
        return recipe

    def validate(self, recipe: dict[str, Any]) -> None:
        missing = self.REQUIRED_TOP_LEVEL_KEYS - set(recipe)
        if missing:
            raise RecipeError(f"Recipe missing required keys: {', '.join(sorted(missing))}")

        tile = recipe["tile"]
        mode = str(tile.get("mode", "grid")).lower()
        if mode == "grid":
            required = ("width", "height", "overlap_x", "overlap_y")
            if str(tile.get("template_path", "")).strip():
                required = (
                    "template_path",
                    "search_x",
                    "search_y",
                    "search_w",
                    "search_h",
                    "offset_x",
                    "offset_y",
                    "rows",
                    "cols",
                    "roi_w",
                    "roi_h",
                    "gap_x",
                    "gap_y",
                )
            for key in required:
                if key not in tile:
                    raise RecipeError(f"Recipe tile section missing: {key}")
        elif mode not in {"contour", "pattern_match"}:
            raise RecipeError(f"Unsupported tile mode: {mode}")

        if not isinstance(recipe["detectors"], dict) or not recipe["detectors"]:
            raise RecipeError("Recipe must define at least one detector.")

    @staticmethod
    def enabled_detectors(recipe: dict[str, Any]) -> dict[str, Any]:
        detectors = recipe.get("detectors", {})
        return {
            detector_id: deepcopy(config)
            for detector_id, config in detectors.items()
            if config.get("enabled", False)
        }
