from __future__ import annotations

from copy import deepcopy


class RecipeTemplatePathSync:
    """Keep template image references consistent across recipe sections."""

    def __init__(self, template_path: str):
        self.template_path = str(template_path or "").strip()

    @classmethod
    def from_recipe(cls, recipe: dict) -> "RecipeTemplatePathSync":
        assets_path = recipe.get("assets", {}).get("template_picture", "")
        tile = recipe.get("tile", {})
        tile_path = tile.get("pattern_match", {}).get("template_path", "") or tile.get("template_path", "")
        return cls(assets_path or tile_path)

    def apply(self, recipe: dict) -> dict:
        synced = deepcopy(recipe)
        if not self.template_path:
            return synced

        synced.setdefault("assets", {})["template_picture"] = self.template_path

        tile = synced.setdefault("tile", {})
        if tile.get("mode") == "pattern_match":
            tile.setdefault("pattern_match", {})["template_path"] = self.template_path
        elif tile.get("mode") == "grid" and str(tile.get("template_path", "")).strip():
            tile["template_path"] = self.template_path

        return synced
