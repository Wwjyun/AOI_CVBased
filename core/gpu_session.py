from __future__ import annotations

from contextlib import contextmanager
import threading
from pathlib import Path

from core.gpu_runtime import GpuRuntime, GpuRuntimeError
from core.recipe_manager import RecipeManager


class GpuExecutionSession:
    """Own one long-lived runtime/context shared by compatible pipeline runs."""

    def __init__(self, runtime: GpuRuntime, requested: bool, config: dict, workload: str = "latency"):
        self.runtime = runtime
        self.requested = bool(requested)
        self._dll_path = GpuRuntime._resolve_path(
            str(config.get("dll_path", GpuRuntime.DEFAULT_DLL))
        )
        self._fallback_to_cpu = RecipeManager().gpu_fallback_enabled(config)
        self.workload = workload
        self._closed = False
        self._pipeline_lock = threading.RLock()

    @classmethod
    def from_recipe(cls, recipe: dict, workload: str = "latency") -> "GpuExecutionSession":
        gpu_config = recipe.get("gpu", {}) or {}
        manager = RecipeManager()
        detector_configs = manager.enabled_detectors(recipe)
        requested = manager.gpu_feature_requested(gpu_config, "tiling") or manager.gpu_mode(gpu_config) != "cpu" and any(
            bool(config.get("use_gpu", False)) for config in detector_configs.values()
        )
        runtime = GpuRuntime(
            gpu_config.get("dll_path", GpuRuntime.DEFAULT_DLL),
            fallback_to_cpu=manager.gpu_fallback_enabled(gpu_config),
            enabled=requested,
            queue_depth=(1 if workload == "latency" else int(gpu_config.get("queue_depth", 8))),
            workload=workload,
        )
        return cls(runtime, requested, gpu_config, workload=workload)

    @classmethod
    def from_recipe_path(cls, recipe_path: Path, workload: str = "latency") -> "GpuExecutionSession":
        return cls.from_recipe(RecipeManager().load(Path(recipe_path)), workload=workload)

    def runtime_for(self, gpu_config: dict, requested: bool) -> GpuRuntime:
        if self._closed:
            raise GpuRuntimeError("GPU execution session is already closed")
        requested_path = GpuRuntime._resolve_path(
            str(gpu_config.get("dll_path", GpuRuntime.DEFAULT_DLL))
        )
        fallback_to_cpu = RecipeManager().gpu_fallback_enabled(gpu_config)
        if requested_path != self._dll_path or fallback_to_cpu != self._fallback_to_cpu:
            raise GpuRuntimeError("Injected GPU session is incompatible with the recipe GPU configuration")
        if requested and not self.requested:
            raise GpuRuntimeError("Injected GPU session was created without CUDA enabled")
        return self.runtime

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.runtime.close()

    @contextmanager
    def execution_scope(self):
        if self._closed:
            raise GpuRuntimeError("GPU execution session is already closed")
        with self._pipeline_lock:
            yield

    def __enter__(self) -> "GpuExecutionSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
