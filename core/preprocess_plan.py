from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
import math
from typing import TypeAlias

import cv2
import numpy as np


class UnsupportedPreprocessPlan(RuntimeError):
    """Raised when an executor cannot preserve the requested plan semantics."""


class InvalidPreprocessPlan(ValueError):
    """Raised when a plan or its input violates the shared preprocessing contract."""


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
class PreprocessTensorSpec:
    shape: tuple[int, ...]
    dtype: str
    channels: int


@dataclass(frozen=True, slots=True)
class PreprocessCapabilityReport:
    requested_backend: str
    selected_backend: str
    route: str
    reason: str
    plan_signature: tuple
    unsupported_operators: tuple[str, ...] = ()

    SCHEMA_VERSION = 1

    def to_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "route": self.route,
            "reason": self.reason,
            "plan_signature": self.plan_signature,
            "unsupported_operators": list(self.unsupported_operators),
        }


def operator_signature(operator: PreprocessOperator) -> tuple:
    if isinstance(operator, Gray):
        return ("gray",)
    if isinstance(operator, Resize):
        return ("resize", operator.width, operator.height, operator.interpolation.lower())
    if isinstance(operator, Gaussian):
        return ("gaussian", operator.kernel_size)
    if isinstance(operator, Threshold):
        return ("threshold", operator.threshold, operator.max_value, operator.invert)
    if isinstance(operator, AdaptiveMean):
        return ("adaptive_mean", operator.block_size, operator.c, operator.max_value, operator.invert)
    if isinstance(operator, Morphology):
        return ("morphology", operator.operation.lower(), operator.kernel_size, operator.iterations)
    raise InvalidPreprocessPlan(f"Unsupported preprocessing operator: {type(operator).__name__}")


def _validate_operator(operator: PreprocessOperator) -> None:
    if isinstance(operator, Gray):
        return
    if isinstance(operator, Resize):
        if operator.width <= 0 or operator.height <= 0:
            raise InvalidPreprocessPlan("Resize width and height must be positive")
        if operator.interpolation.lower() not in {"area", "linear", "nearest"}:
            raise InvalidPreprocessPlan(f"Unsupported resize interpolation: {operator.interpolation}")
        return
    if isinstance(operator, Gaussian):
        if operator.kernel_size <= 0 or operator.kernel_size % 2 == 0:
            raise InvalidPreprocessPlan("Gaussian kernel_size must be a positive odd integer")
        return
    if isinstance(operator, Threshold):
        if not 0 <= operator.threshold <= 255 or not 0 <= operator.max_value <= 255:
            raise InvalidPreprocessPlan("Threshold values must be within uint8 range")
        return
    if isinstance(operator, AdaptiveMean):
        if operator.block_size < 3 or operator.block_size % 2 == 0:
            raise InvalidPreprocessPlan("AdaptiveMean block_size must be an odd integer at least 3")
        if not math.isfinite(operator.c):
            raise InvalidPreprocessPlan("AdaptiveMean c must be finite")
        if not 0 <= operator.max_value <= 255:
            raise InvalidPreprocessPlan("AdaptiveMean max_value must be within uint8 range")
        return
    if isinstance(operator, Morphology):
        if operator.operation.lower() not in {"", "none", "open", "close", "dilate", "erode"}:
            raise InvalidPreprocessPlan(f"Unsupported morphology operation: {operator.operation}")
        if operator.kernel_size <= 0 or operator.iterations < 0:
            raise InvalidPreprocessPlan("Morphology kernel_size must be positive and iterations non-negative")
        return
    raise InvalidPreprocessPlan(f"Unsupported preprocessing operator: {type(operator).__name__}")


@dataclass(frozen=True, slots=True)
class PreprocessPlan:
    SCHEMA_VERSION = 1

    operations: tuple[PreprocessOperator, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("A preprocessing plan must contain at least one operator")
        for operator in self.operations:
            _validate_operator(operator)

    @property
    def signature(self) -> tuple:
        return (self.SCHEMA_VERSION, tuple(operator_signature(operator) for operator in self.operations))

    def validate_input(self, image: np.ndarray) -> PreprocessTensorSpec:
        if not isinstance(image, np.ndarray):
            raise InvalidPreprocessPlan("Preprocess input must be a numpy.ndarray")
        if image.dtype != np.uint8:
            raise InvalidPreprocessPlan(f"Preprocess input dtype must be uint8, got {image.dtype}")
        if image.ndim not in {2, 3}:
            raise InvalidPreprocessPlan(f"Preprocess input must have 2 or 3 dimensions, got {image.ndim}")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise InvalidPreprocessPlan("Preprocess input height and width must be positive")
        channels = 1 if image.ndim == 2 else int(image.shape[2])
        if channels not in {1, 3} or (image.ndim == 3 and channels == 1):
            raise InvalidPreprocessPlan("Preprocess input must be 2D gray or 3-channel BGR")

        shape = tuple(int(value) for value in image.shape)
        for operator in self.operations:
            if isinstance(operator, Gray):
                channels = 1
                shape = (shape[0], shape[1])
            elif isinstance(operator, Resize):
                shape = (
                    (operator.height, operator.width)
                    if channels == 1
                    else (operator.height, operator.width, channels)
                )
            elif isinstance(operator, (Threshold, AdaptiveMean, Morphology)) and channels != 1:
                raise InvalidPreprocessPlan(
                    f"{type(operator).__name__} requires single-channel input; add Gray first"
                )
        return PreprocessTensorSpec(shape=shape, dtype="uint8", channels=channels)

    @staticmethod
    def validate_output(output: np.ndarray, expected: PreprocessTensorSpec) -> np.ndarray:
        if not isinstance(output, np.ndarray):
            raise InvalidPreprocessPlan("Preprocess output must be a numpy.ndarray")
        if output.dtype != np.uint8:
            raise InvalidPreprocessPlan(f"Preprocess output dtype must be uint8, got {output.dtype}")
        if tuple(output.shape) != expected.shape:
            raise InvalidPreprocessPlan(
                f"Preprocess output shape mismatch: expected {expected.shape}, got {tuple(output.shape)}"
            )
        channels = 1 if output.ndim == 2 else int(output.shape[2])
        if channels != expected.channels:
            raise InvalidPreprocessPlan(
                f"Preprocess output channel mismatch: expected {expected.channels}, got {channels}"
            )
        return output


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

    @staticmethod
    def capability_report(plan: PreprocessPlan) -> PreprocessCapabilityReport:
        return PreprocessCapabilityReport(
            requested_backend="cpu",
            selected_backend="cpu",
            route="cpu",
            reason="CPU OpenCV reference executor selected",
            plan_signature=plan.signature,
        )

    def execute(self, image: np.ndarray, plan: PreprocessPlan) -> np.ndarray:
        expected = plan.validate_input(image)
        output = np.asarray(image)
        for operator in plan.operations:
            output = self._execute_operator(output, operator)
        return plan.validate_output(output, expected)

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

    def capability_report(self, plan: PreprocessPlan, image: np.ndarray | None = None) -> PreprocessCapabilityReport:
        if self._is_legacy_fused_plan(plan) and bool(getattr(self.runtime, "supports_fused_401_2", False)):
            return PreprocessCapabilityReport(
                requested_backend="cuda",
                selected_backend="cuda",
                route="fused",
                reason="Legacy 401-2 fused export supports the complete plan",
                plan_signature=plan.signature,
            )

        unsupported = []
        method_by_type = {
            Resize: "resize_gray",
            Gaussian: "gaussian_blur",
            Threshold: "threshold",
            AdaptiveMean: "adaptive_threshold",
            Morphology: "morphology",
        }
        for operator in plan.operations:
            if isinstance(operator, Gray):
                if image is not None and image.ndim == 2:
                    continue
                method_name = "bgr_to_gray"
            elif isinstance(operator, Resize) and operator.interpolation != "nearest":
                unsupported.append(f"Resize({operator.interpolation}) semantics unavailable")
                continue
            elif isinstance(operator, Morphology) and (
                operator.operation.lower() in {"", "none"}
                or operator.iterations <= 0
                or operator.kernel_size <= 1
            ):
                continue
            else:
                method_name = method_by_type[type(operator)]
            if not callable(getattr(self.runtime, method_name, None)):
                unsupported.append(f"missing runtime primitive: {method_name}")

        if unsupported:
            return PreprocessCapabilityReport(
                requested_backend="cuda",
                selected_backend="cpu",
                route="fallback",
                reason="; ".join(unsupported),
                plan_signature=plan.signature,
                unsupported_operators=tuple(unsupported),
            )
        return PreprocessCapabilityReport(
            requested_backend="cuda",
            selected_backend="cuda",
            route="primitive",
            reason="All operators are supported by reusable CUDA primitives",
            plan_signature=plan.signature,
        )

    def execute(self, image: np.ndarray, plan: PreprocessPlan) -> np.ndarray:
        expected = plan.validate_input(image)
        fused = self._execute_legacy_fused(image, plan)
        if fused is not None:
            return plan.validate_output(fused, expected)
        output = np.asarray(image)
        for operator in plan.operations:
            output = self._execute_operator(output, operator)
        return plan.validate_output(output, expected)

    def _execute_legacy_fused(self, image: np.ndarray, plan: PreprocessPlan) -> np.ndarray | None:
        operations = plan.operations
        if self._is_legacy_fused_plan(plan) and self.runtime.supports_fused_401_2:
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

    @staticmethod
    def _is_legacy_fused_plan(plan: PreprocessPlan) -> bool:
        operations = plan.operations
        return (
            len(operations) == 3
            and isinstance(operations[0], Gray)
            and isinstance(operations[1], Gaussian)
            and isinstance(operations[2], AdaptiveMean)
        )

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
