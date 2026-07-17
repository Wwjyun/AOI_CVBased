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

        gpu = recipe.get("gpu", {})
        if gpu is not None and not isinstance(gpu, dict):
            raise RecipeError("Recipe gpu section must be a mapping.")
        for key in ("tiling", "display", "fallback_to_cpu"):
            if key in (gpu or {}) and not isinstance(gpu[key], bool):
                raise RecipeError(f"Recipe gpu.{key} must be true or false.")
        if "dll_path" in (gpu or {}) and not isinstance(gpu["dll_path"], str):
            raise RecipeError("Recipe gpu.dll_path must be a string.")
        if str((gpu or {}).get("mode", "auto")).lower() not in {"auto", "cpu", "cuda"}:
            raise RecipeError("Recipe gpu.mode must be auto, cpu, or cuda.")
        if "queue_depth" in (gpu or {}) and (
            not isinstance(gpu["queue_depth"], int) or gpu["queue_depth"] <= 0
        ):
            raise RecipeError("Recipe gpu.queue_depth must be a positive integer.")
        for detector_id, config in recipe["detectors"].items():
            if not isinstance(config, dict):
                raise RecipeError(f"Recipe detector {detector_id} must be a mapping.")
            if "use_gpu" in config and not isinstance(config["use_gpu"], bool):
                raise RecipeError(f"Recipe detector {detector_id}.use_gpu must be true or false.")
        self._validate_detector_parameters(recipe["detectors"])

    @staticmethod
    def _validate_detector_parameters(detectors: dict[str, Any]) -> None:
        from core.detector_manager import DetectorManager
        from core.parameter_schema import validate_parameter_mapping

        definitions = DetectorManager().definitions()
        for detector_id, config in detectors.items():
            detector_id = str(detector_id)
            if detector_id not in definitions:
                raise RecipeError(f"Recipe detector is not registered: {detector_id}")
            unknown_config = set(config) - {"enabled", "use_gpu", "display_name", "params"}
            if unknown_config:
                raise RecipeError(
                    f"Recipe detector {detector_id} has unknown keys: {', '.join(sorted(unknown_config))}"
                )
            try:
                validate_parameter_mapping(
                    config.get("params", {}), DetectorManager().parameter_specs(detector_id),
                    f"detectors.{detector_id}.params"
                )
            except ValueError as exc:
                raise RecipeError(str(exc)) from exc

    @staticmethod
    def enabled_detectors(recipe: dict[str, Any]) -> dict[str, Any]:
        detectors = recipe.get("detectors", {})
        return {
            detector_id: deepcopy(config)
            for detector_id, config in detectors.items()
            if config.get("enabled", False)
        }

    @staticmethod
    def gpu_mode(gpu: dict | None) -> str:
        return str((gpu or {}).get("mode", "auto")).lower()

    @classmethod
    def gpu_feature_requested(cls, gpu: dict | None, feature: str) -> bool:
        return cls.gpu_mode(gpu) != "cpu" and bool((gpu or {}).get(feature, False))

    @classmethod
    def gpu_fallback_enabled(cls, gpu: dict | None) -> bool:
        mode = cls.gpu_mode(gpu)
        return mode != "cuda" and bool((gpu or {}).get("fallback_to_cpu", True))
