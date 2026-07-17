from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import TypeAlias

import cv2
import numpy as np


class UnsupportedPreprocessPlan(RuntimeError):
    """Raised when an executor cannot preserve the requested plan semantics."""


@dataclass(frozen=True, slots=True)
class Gray:
    pass


@dataclass(frozen=True, slots=True)
class Resize:
    width: int
    height: int
    interpolation: str = "area"


@dataclass(frozen=True, slots=True)
class Gaussian:
    kernel_size: int


@dataclass(frozen=True, slots=True)
class Threshold:
    threshold: int
    max_value: int = 255
    invert: bool = False


@dataclass(frozen=True, slots=True)
class AdaptiveMean:
    block_size: int
    c: float
    max_value: int = 255
    invert: bool = False


@dataclass(frozen=True, slots=True)
class Morphology:
    operation: str
    kernel_size: int = 3
    iterations: int = 1


PreprocessOperator: TypeAlias = Gray | Resize | Gaussian | Threshold | AdaptiveMean | Morphology


@dataclass(frozen=True, slots=True)
class PreprocessPlan:
    operations: tuple[PreprocessOperator, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("A preprocessing plan must contain at least one operator")


class PreprocessPlanCache:
    """Small per-detector LRU cache for immutable, shape-aware preprocessing plans."""

    def __init__(self, max_entries: int = 32) -> None:
        if max_entries <= 0:
            raise ValueError("Preprocess plan cache size must be positive")
        self.max_entries = int(max_entries)
        self._plans: OrderedDict[tuple, PreprocessPlan] = OrderedDict()

    def get_or_create(
        self,
        image: np.ndarray,
        signature: Hashable,
        factory: Callable[[], PreprocessPlan],
    ) -> PreprocessPlan:
        source = np.asarray(image)
        key = (tuple(int(value) for value in source.shape), source.dtype.str, signature)
        cached = self._plans.get(key)
        if cached is not None:
            self._plans.move_to_end(key)
            return cached

        plan = factory()
        if not isinstance(plan, PreprocessPlan):
            raise TypeError("Preprocess plan cache factory must return PreprocessPlan")
        self._plans[key] = plan
        if len(self._plans) > self.max_entries:
            self._plans.popitem(last=False)
        return plan

    @property
    def size(self) -> int:
        return len(self._plans)

    def clear(self) -> None:
        self._plans.clear()


class CpuPreprocessExecutor:
    """Reference executor. Its OpenCV result defines fallback semantics."""

    _INTERPOLATIONS = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }

    def execute(self, image: np.ndarray, plan: PreprocessPlan) -> np.ndarray:
        output = np.asarray(image)
        for operator in plan.operations:
            output = self._execute_operator(output, operator)
        return output

    def _execute_operator(self, image: np.ndarray, operator: PreprocessOperator) -> np.ndarray:
        if isinstance(operator, Gray):
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        if isinstance(operator, Resize):
            interpolation = self._INTERPOLATIONS.get(operator.interpolation)
            if interpolation is None:
                raise UnsupportedPreprocessPlan(f"Unsupported CPU resize interpolation: {operator.interpolation}")
            return cv2.resize(image, (operator.width, operator.height), interpolation=interpolation)
        if isinstance(operator, Gaussian):
            return cv2.GaussianBlur(image, (operator.kernel_size, operator.kernel_size), 0)
        if isinstance(operator, Threshold):
            threshold_type = cv2.THRESH_BINARY_INV if operator.invert else cv2.THRESH_BINARY
            return cv2.threshold(image, operator.threshold, operator.max_value, threshold_type)[1]
        if isinstance(operator, AdaptiveMean):
            threshold_type = cv2.THRESH_BINARY_INV if operator.invert else cv2.THRESH_BINARY
            return cv2.adaptiveThreshold(
                image,
                operator.max_value,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                threshold_type,
                operator.block_size,
                operator.c,
            )
        if isinstance(operator, Morphology):
            return self._morphology(image, operator)
        raise UnsupportedPreprocessPlan(f"Unsupported CPU preprocessing operator: {type(operator).__name__}")

    @staticmethod
    def _morphology(image: np.ndarray, operator: Morphology) -> np.ndarray:
        operation = operator.operation.lower()
        if operation in {"", "none"} or operator.iterations <= 0 or operator.kernel_size <= 1:
            return image.copy()
        operations = {
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "dilate": cv2.MORPH_DILATE,
            "erode": cv2.MORPH_ERODE,
        }
        cv_operation = operations.get(operation)
        if cv_operation is None:
            raise UnsupportedPreprocessPlan(f"Unsupported CPU morphology operation: {operator.operation}")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (operator.kernel_size, operator.kernel_size))
        if cv_operation == cv2.MORPH_DILATE:
            return cv2.dilate(image, kernel, iterations=operator.iterations)
        if cv_operation == cv2.MORPH_ERODE:
            return cv2.erode(image, kernel, iterations=operator.iterations)
        return cv2.morphologyEx(image, cv_operation, kernel, iterations=operator.iterations)


class CudaPreprocessExecutor:
    """CUDA executor with a legacy fused adapter and reusable primitive fallback."""

    def __init__(self, runtime):
        self.runtime = runtime

    def execute(self, image: np.ndarray, plan: PreprocessPlan) -> np.ndarray:
        fused = self._execute_legacy_fused(image, plan)
        if fused is not None:
            return fused
        output = np.asarray(image)
        for operator in plan.operations:
            output = self._execute_operator(output, operator)
        return output

    def _execute_legacy_fused(self, image: np.ndarray, plan: PreprocessPlan) -> np.ndarray | None:
        operations = plan.operations
        if (
            len(operations) == 3
            and isinstance(operations[0], Gray)
            and isinstance(operations[1], Gaussian)
            and isinstance(operations[2], AdaptiveMean)
            and self.runtime.supports_fused_401_2
        ):
            gaussian = operations[1]
            adaptive = operations[2]
            return self.runtime.preprocess_401_2(
                image,
                gaussian.kernel_size,
                adaptive.block_size,
                adaptive.c,
                adaptive.max_value,
                adaptive.invert,
            )
        return None

    def _execute_operator(self, image: np.ndarray, operator: PreprocessOperator) -> np.ndarray:
        if isinstance(operator, Gray):
            return self.runtime.bgr_to_gray(image) if image.ndim == 3 else image.copy()
        if isinstance(operator, Resize):
            if operator.interpolation != "nearest":
                raise UnsupportedPreprocessPlan(
                    f"CUDA resize cannot preserve {operator.interpolation} interpolation yet"
                )
            return self.runtime.resize_gray(image, operator.width, operator.height)
        if isinstance(operator, Gaussian):
            return self.runtime.gaussian_blur(image, operator.kernel_size)
        if isinstance(operator, Threshold):
            return self.runtime.threshold(image, operator.threshold, operator.max_value, operator.invert)
        if isinstance(operator, AdaptiveMean):
            return self.runtime.adaptive_threshold(
                image, operator.block_size, operator.c, operator.max_value, operator.invert
            )
        if isinstance(operator, Morphology):
            if operator.operation.lower() in {"", "none"} or operator.iterations <= 0 or operator.kernel_size <= 1:
                return image.copy()
            return self.runtime.morphology(
                image, operator.operation.lower(), operator.kernel_size, operator.iterations
            )
        raise UnsupportedPreprocessPlan(f"Unsupported CUDA preprocessing operator: {type(operator).__name__}")
