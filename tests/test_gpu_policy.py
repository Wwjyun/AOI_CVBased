from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path

from core.recipe_manager import RecipeManager, RecipeError


class GpuModePolicyTests(unittest.TestCase):
    def test_auto_cpu_and_cuda_modes_have_explicit_request_and_fallback_semantics(self):
        manager = RecipeManager()

        self.assertTrue(manager.gpu_feature_requested({"mode": "auto", "display": True}, "display"))
        self.assertTrue(manager.gpu_fallback_enabled({"mode": "auto", "fallback_to_cpu": True}))
        self.assertFalse(manager.gpu_feature_requested({"mode": "cpu", "display": True}, "display"))
        self.assertTrue(manager.gpu_fallback_enabled({"mode": "cpu"}))
        self.assertTrue(manager.gpu_feature_requested({"mode": "cuda", "display": True}, "display"))
        self.assertFalse(manager.gpu_fallback_enabled({"mode": "cuda", "fallback_to_cpu": True}))

    def test_recipe_validation_rejects_unknown_mode_and_invalid_queue_depth(self):
        manager = RecipeManager()
        base = manager.load(
            Path(__file__).resolve().parents[1] / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        )
        invalid_mode = deepcopy(base)
        invalid_mode["gpu"]["mode"] = "magic"
        with self.assertRaisesRegex(RecipeError, "gpu.mode"):
            manager.validate(invalid_mode)
        invalid_queue = deepcopy(base)
        invalid_queue["gpu"].update(mode="auto", queue_depth=0)
        with self.assertRaisesRegex(RecipeError, "queue_depth"):
            manager.validate(invalid_queue)


if __name__ == "__main__":
    unittest.main()
