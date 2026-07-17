from __future__ import annotations

from copy import deepcopy

from core.preprocess_plan import (
    CpuPreprocessExecutor,
    CpuPreprocessDagExecutor,
    CudaPreprocessExecutor,
    PreprocessPlan,
    PreprocessDagPlan,
    PreprocessPlanCache,
)


class BaseDetector:
    detector_id = ""
    detector_name = ""
    display_name = ""
    default_params: dict = {}

    def __init__(self, display_name: str | None = None, params: dict | None = None, use_gpu: bool = False, gpu_runtime=None):
        self.display_name = display_name or self.display_name or self.detector_name
        self.params = deepcopy(self.default_params)
        self.params.update(params or {})
        self.use_gpu = bool(use_gpu)
        self.gpu_runtime = gpu_runtime
        self.gpu_fallback_reason = ""
        if self.use_gpu and (gpu_runtime is None or not gpu_runtime.available):
            self.gpu_fallback_reason = getattr(gpu_runtime, "unavailable_reason", "CUDA runtime was not created")
        self._cpu_preprocess_executor = CpuPreprocessExecutor()
        self._cpu_preprocess_dag_executor = CpuPreprocessDagExecutor()
        self._cuda_preprocess_executor = CudaPreprocessExecutor(gpu_runtime) if gpu_runtime is not None else None
        self._preprocess_plan_cache = PreprocessPlanCache()
        self.last_preprocess_capability: dict = {}

    @property
    def gpu_active(self) -> bool:
        return bool(self.use_gpu and self.gpu_runtime is not None and self.gpu_runtime.available and not self.gpu_fallback_reason)

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        raise NotImplementedError

    def execute_preprocess_plan(self, image, plan: PreprocessPlan):
        if self.gpu_active and self._cuda_preprocess_executor is not None:
            self.last_preprocess_capability = self._cuda_preprocess_executor.capability_report(
                plan, image
            ).to_dict()
            return self._cuda_preprocess_executor.execute(image, plan)
        report = self._cpu_preprocess_executor.capability_report(plan).to_dict()
        if self.use_gpu and self.gpu_fallback_reason:
            report.update(
                requested_backend="cuda",
                selected_backend="cpu",
                route="fallback",
                reason=self.gpu_fallback_reason,
            )
        self.last_preprocess_capability = report
        return self._cpu_preprocess_executor.execute(image, plan)

    def cached_preprocess_plan(self, image, signature, factory) -> PreprocessPlan | PreprocessDagPlan:
        return self._preprocess_plan_cache.get_or_create(image, signature, factory)

    def execute_preprocess_dag(self, image, plan: PreprocessDagPlan) -> dict:
        if self.gpu_active:
            raise RuntimeError("CUDA DAG executor is not available; full detector CPU fallback required")
        report = self._cpu_preprocess_dag_executor.capability_report(plan).to_dict()
        if self.use_gpu and self.gpu_fallback_reason:
            report.update(
                requested_backend="cuda",
                selected_backend="cpu",
                route="fallback",
                reason=self.gpu_fallback_reason,
            )
        self.last_preprocess_capability = report
        return self._cpu_preprocess_dag_executor.execute(image, plan)

    @property
    def preprocess_plan_cache_size(self) -> int:
        return self._preprocess_plan_cache.size

    def run(self, image) -> dict:
        try:
            processed = self.preprocess(image)
            defects = self.detect(processed)
        except Exception as exc:
            if not self.gpu_active:
                raise
            self.gpu_fallback_reason = str(exc)
            processed = self.preprocess(image)
            defects = self.detect(processed)
        max_confidence = max((defect.get("confidence", 0.0) for defect in defects), default=0.0)
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "display_name": self.display_name,
            "pass": len(defects) == 0,
            "score": float(max_confidence),
            "defects": defects,
            "execution": {
                "gpu_requested": self.use_gpu,
                "gpu_active": self.gpu_active,
                "backend": "cuda_dll" if self.gpu_active else "cpu",
                "fallback_reason": self.gpu_fallback_reason,
                "preprocess_capability": self.last_preprocess_capability,
            },
        }
