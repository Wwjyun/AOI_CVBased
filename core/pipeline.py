from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from core.aggregator import Aggregator
from core.detector_manager import DetectorManager
from core.image_loader import load_image
from core.gpu_runtime import GpuRuntime, GpuRuntimeError
from core.gpu_session import GpuExecutionSession
from core.logging_system import LogMixin
from core.performance import PipelineProfiler
from core.preprocess_cache import TilePreprocessCache
from core.provenance import inspection_provenance
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
        gpu_session: GpuExecutionSession | None = None,
    ):
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.progress_callback = progress_callback
        self.output_overrides = output_overrides
        self.gpu_session = gpu_session
        self.recipe_manager = RecipeManager()
        self.detector_manager = DetectorManager()
        self._active_profiler = None
        self._last_progress_percent = None

    def run(self, image_path: Path) -> dict:
        if self.gpu_session is not None:
            with self.gpu_session.execution_scope():
                return self._run(image_path)
        return self._run(image_path)

    def _run(self, image_path: Path) -> dict:
        started = time.perf_counter()
        profiler = PipelineProfiler()
        self._active_profiler = profiler
        self._last_progress_percent = None
        self.logger.info(
            "Inspection started: image=%s recipe=%s output=%s debug=%s",
            image_path,
            self.recipe_path,
            self.output_dir,
            self.debug,
        )
        self._progress(0, "Starting inspection")
        with profiler.measure("recipe_setup"):
            recipe = self.recipe_manager.load(self.recipe_path)
            if self.output_overrides:
                recipe["output"] = {**recipe.get("output", {}), **self.output_overrides}
            recipe = RecipeTemplatePathSync.from_recipe(recipe).apply(recipe)
            provenance = inspection_provenance(self.recipe_path, recipe)
            gpu_config = recipe.get("gpu", {}) or {}
            detector_configs = self.recipe_manager.enabled_detectors(recipe)
            gpu_mode = self.recipe_manager.gpu_mode(gpu_config)
            tiling_gpu_requested = self.recipe_manager.gpu_feature_requested(gpu_config, "tiling")
            detector_gpu_allowed = gpu_mode != "cpu"
            gpu_requested = tiling_gpu_requested or detector_gpu_allowed and any(
                bool(config.get("use_gpu", False)) for config in detector_configs.values()
            )
            gpu_runtime = (
                self.gpu_session.runtime_for(gpu_config, gpu_requested)
                if self.gpu_session is not None
                else GpuRuntime(
                    gpu_config.get("dll_path", GpuRuntime.DEFAULT_DLL),
                    fallback_to_cpu=self.recipe_manager.gpu_fallback_enabled(gpu_config),
                    enabled=gpu_requested,
                    queue_depth=1,
                    workload="latency",
                )
            )
        if gpu_requested and not gpu_runtime.available and not gpu_runtime.fallback_to_cpu:
            raise GpuRuntimeError(gpu_runtime.unavailable_reason)
        if gpu_requested and gpu_runtime.available:
            self.logger.info(
                "CUDA DLL active: path=%s device=%s capability=%s",
                gpu_runtime.dll_path,
                gpu_runtime.device_name,
                gpu_runtime.compute_capability,
            )
        elif gpu_requested:
            self.logger.warning("CUDA requested; falling back to CPU: %s", gpu_runtime.unavailable_reason)
        self.logger.info("Recipe loaded: name=%s version=%s", recipe.get("recipe_name"), recipe.get("version"))
        self._progress(5, "Recipe loaded")
        with profiler.measure("image_load"):
            image = load_image(image_path)
        self.logger.info("Image loaded: image=%s shape=%s", image_path, getattr(image, "shape", None))
        self._progress(10, "Image loaded")
        with profiler.measure("initialization"):
            tile_config = recipe["tile"]
            resident_image = None
            detector_gpu_requested = detector_gpu_allowed and any(
                bool(config.get("use_gpu", False)) for config in detector_configs.values()
            )
            if (
                detector_gpu_requested
                and gpu_runtime.available
                and gpu_runtime.supports_resident_roi
                and str(tile_config.get("mode", "grid")).lower() == "grid"
            ):
                try:
                    resident_image = gpu_runtime.upload_image(image)
                except Exception as exc:
                    gpu_runtime.fallback_or_raise(exc)
            tiler = create_tiler(
                tile_config,
                gpu_runtime=(gpu_runtime if tiling_gpu_requested and resident_image is None else None),
                resident_image=resident_image,
            )
            if not detector_gpu_allowed:
                for config in detector_configs.values():
                    config["use_gpu"] = False
            detectors = self.detector_manager.create_enabled(detector_configs, gpu_runtime=gpu_runtime)
        self.logger.info("Detectors initialized: count=%s ids=%s", len(detectors), [d.detector_id for d in detectors])
        self._progress(15, "Detectors initialized")

        with profiler.measure("tiling"):
            tiles = list(tiler.iter_tiles(image))
        tiling_gpu_metrics = gpu_runtime.performance_stats()
        crop_metrics = tiling_gpu_metrics.get("functions", {}).get("vf_crop_u8", {})
        if tiling_gpu_requested and crop_metrics.get("calls", 0) > 1:
            self.logger.warning(
                "CUDA tiling performed %s synchronous crop round trips and estimated %s H2D bytes; "
                "keep gpu.tiling disabled for performance until source buffers are reusable",
                crop_metrics["calls"],
                crop_metrics["host_to_device_bytes"],
            )
        self.logger.info("Tiles prepared: count=%s mode=%s", len(tiles), tile_config.get("mode", "grid"))
        self._progress(20, f"Tiles prepared: {len(tiles)}")

        tile_results = []
        total_work = max(len(tiles) * max(len(detectors), 1), 1)
        completed_work = 0
        with profiler.measure("detectors_total"):
            for tile_index, tile in enumerate(tiles, start=1):
                detector_results = []
                preprocess_cache = TilePreprocessCache(tile.image)
                for detector in detectors:
                    with profiler.measure(f"detector:{detector.detector_id}"):
                        detector_result = detector.run(
                            tile.image,
                            device_roi=tile.device_roi,
                            preprocess_cache=preprocess_cache,
                        )
                        for stage, duration in (
                            detector_result.get("execution", {})
                            .get("performance", {})
                            .get("stages_sec", {})
                            .items()
                        ):
                            profiler.add_duration(
                                f"detector_stage:{detector.detector_id}:{stage}", duration
                            )
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

        detector_fallbacks = {
            detector.detector_id: detector.gpu_fallback_reason
            for detector in detectors
            if detector.use_gpu and detector.gpu_fallback_reason
        }
        if detector_fallbacks:
            self.logger.warning("Detector CUDA fallback: %s", detector_fallbacks)
        fallback_message = " (CPU fallback)" if gpu_requested and (
            not gpu_runtime.available or gpu_runtime.last_error or detector_fallbacks
        ) else ""
        self._progress(85, f"Aggregating PASS / NG result{fallback_message}")
        with profiler.measure("aggregation"):
            aggregate = Aggregator(recipe["decision"]).aggregate(tile_results)
        result = {
            "image_name": Path(image_path).name,
            "recipe_name": recipe["recipe_name"],
            "machine_id": recipe["machine_id"],
            "product_id": recipe["product_id"],
            "recipe_version": recipe["version"],
            "provenance": provenance,
            "final_result": aggregate["final_result"],
            "summary": aggregate["summary"],
            "tiles": tile_results,
            "outputs": {},
            "duration_sec": round(time.perf_counter() - started, 3),
            "execution": {
                "gpu": {
                    "mode": gpu_mode,
                    "resident_image": {
                        "active": resident_image is not None,
                        "generation": resident_image.generation if resident_image is not None else 0,
                        "shape": (
                            [resident_image.height, resident_image.width, resident_image.channels]
                            if resident_image is not None else []
                        ),
                    },
                    "tiling": gpu_runtime.status(tiling_gpu_requested),
                    "display_requested": self.recipe_manager.gpu_feature_requested(gpu_config, "display"),
                    "detectors": {
                        detector.detector_id: {
                            "requested": detector.use_gpu,
                            "active": detector.gpu_active,
                            "backend": "cuda_dll" if detector.gpu_active else "cpu",
                            "fallback_reason": detector.gpu_fallback_reason,
                        }
                        for detector in detectors
                    },
                    "metrics": gpu_runtime.performance_stats(),
                },
                "performance": profiler.snapshot(),
            },
        }

        serializable_result = self._without_runtime_images(result)
        self._progress(92, "Writing overlay, CSV, and JSON")
        with profiler.measure("reporting_total"):
            outputs = Reporter(self.output_dir, recipe["output"], profiler=profiler).write(image, result)
        serializable_result["outputs"] = outputs
        serializable_result["execution"]["gpu"]["metrics"] = gpu_runtime.performance_stats()
        serializable_result["execution"]["performance"] = profiler.snapshot()
        self.logger.info(
            "Inspection completed: image=%s final=%s defects=%s ng_tiles=%s duration=%.3fs",
            Path(image_path).name,
            serializable_result["final_result"],
            serializable_result["summary"].get("defect_count", 0),
            serializable_result["summary"].get("ng_count", 0),
            serializable_result["duration_sec"],
        )
        self.logger.info("Inspection performance: %s", serializable_result["execution"]["performance"])
        if gpu_requested:
            self.logger.info("CUDA host metrics: %s", serializable_result["execution"]["gpu"]["metrics"])
        self._progress(100, "Inspection complete")
        return serializable_result

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback is None:
            return
        bounded = max(0, min(100, int(percent)))
        if bounded == self._last_progress_percent:
            return
        self._last_progress_percent = bounded
        started = time.perf_counter()
        self.progress_callback(bounded, message)
        if self._active_profiler is not None:
            self._active_profiler.add_duration(
                "progress_callback", time.perf_counter() - started
            )

    @staticmethod
    def _without_runtime_images(result: dict) -> dict:
        cleaned = dict(result)
        cleaned["tiles"] = []
        for tile_result in result["tiles"]:
            cleaned_tile = dict(tile_result)
            cleaned_tile.pop("_tile_image", None)
            cleaned["tiles"].append(cleaned_tile)
        return cleaned
