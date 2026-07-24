from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import yaml


class AiModelError(RuntimeError):
    pass


class AiBackendUnavailable(AiModelError):
    pass


@dataclass(frozen=True)
class YoloXModelManifest:
    model_id: str
    name: str
    version: str
    model_format: str
    model_path: Path
    sha256: str
    class_names: tuple[str, ...]
    input_name: str
    input_width: int
    input_height: int
    color_order: str
    pixel_scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    letterbox_value: int
    letterbox_placement: str
    output_name: str
    decoder: str
    strides: tuple[int, ...]
    scores_are_probabilities: bool
    allowed_backends: tuple[str, ...]
    allowed_precisions: tuple[str, ...]
    test_only: bool = False

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.input_height, self.input_width)


class YoloXModelRegistry:
    SCHEMA_VERSION = 1

    def __init__(self, root: Path | None = None):
        self.root = Path(root or self.default_root()).resolve()
        self.registry_path = self.root / "registry.yaml"
        self._models = self._load()

    @staticmethod
    def default_root() -> Path:
        configured = os.getenv("VISIONFLOW_YOLOX_MODEL_DIR")
        if configured:
            return Path(configured)
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            return Path(getattr(sys, "_MEIPASS")) / "models" / "yolox"
        return Path(__file__).resolve().parents[1] / "models" / "yolox"

    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def get(self, model_id: str) -> YoloXModelManifest:
        normalized = str(model_id).strip()
        try:
            return self._models[normalized]
        except KeyError as exc:
            available = ", ".join(self.model_ids()) or "(none)"
            raise AiModelError(
                f"Unknown YOLOX model_id {normalized!r}; available: {available}"
            ) from exc

    def _load(self) -> dict[str, YoloXModelManifest]:
        if not self.registry_path.is_file():
            raise AiModelError(f"YOLOX model registry does not exist: {self.registry_path}")
        try:
            document = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise AiModelError(f"Cannot read YOLOX model registry: {exc}") from exc
        if document.get("schema_version") != self.SCHEMA_VERSION:
            raise AiModelError(
                f"Unsupported YOLOX registry schema: {document.get('schema_version')!r}"
            )
        entries = document.get("models")
        if not isinstance(entries, dict) or not entries:
            raise AiModelError("YOLOX model registry must define at least one model")
        models: dict[str, YoloXModelManifest] = {}
        for model_id, config in entries.items():
            models[str(model_id)] = self._parse_model(str(model_id), config)
        return models

    def _parse_model(self, model_id: str, config: Any) -> YoloXModelManifest:
        if not isinstance(config, dict):
            raise AiModelError(f"YOLOX model {model_id} must be a mapping")
        input_config = self._mapping(config, "input", model_id)
        output_config = self._mapping(config, "output", model_id)
        normalization = self._mapping(input_config, "normalization", model_id)
        letterbox = self._mapping(input_config, "letterbox", model_id)

        model_path = (self.root / str(config.get("file", ""))).resolve()
        if self.root != model_path and self.root not in model_path.parents:
            raise AiModelError(f"YOLOX model {model_id} file escapes the registry root")
        if not model_path.is_file():
            raise AiModelError(f"YOLOX model {model_id} file does not exist: {model_path}")

        expected_sha256 = str(config.get("sha256", "")).lower()
        actual_sha256 = _sha256_file(model_path)
        if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
            raise AiModelError(
                f"YOLOX model {model_id} SHA-256 mismatch: "
                f"expected={expected_sha256 or '(missing)'} actual={actual_sha256}"
            )

        class_names = self._string_tuple(config.get("class_names"), "class_names", model_id)
        strides = self._positive_int_tuple(output_config.get("strides"), "strides", model_id)
        allowed_backends = self._string_tuple(
            config.get("allowed_backends", ["onnxruntime_cpu"]),
            "allowed_backends",
            model_id,
        )
        allowed_precisions = self._string_tuple(
            config.get("allowed_precisions", ["fp32"]),
            "allowed_precisions",
            model_id,
        )
        width = self._positive_int(input_config.get("width"), "input.width", model_id)
        height = self._positive_int(input_config.get("height"), "input.height", model_id)
        if any(width % stride or height % stride for stride in strides):
            raise AiModelError(
                f"YOLOX model {model_id} input shape must be divisible by every stride"
            )

        mean = self._float_triplet(normalization.get("mean", [0, 0, 0]), "mean", model_id)
        std = self._float_triplet(normalization.get("std", [1, 1, 1]), "std", model_id)
        if any(value == 0 for value in std):
            raise AiModelError(f"YOLOX model {model_id} normalization std cannot contain zero")

        model_format = str(config.get("format", "")).lower()
        if model_format != "onnx":
            raise AiModelError(f"YOLOX model {model_id} format must be onnx for the CPU reference")
        decoder = str(output_config.get("decoder", "")).lower()
        if decoder != "yolox_raw":
            raise AiModelError(f"YOLOX model {model_id} decoder must be yolox_raw")
        color_order = str(input_config.get("color_order", "BGR")).upper()
        if color_order not in {"BGR", "RGB"}:
            raise AiModelError(f"YOLOX model {model_id} color_order must be BGR or RGB")
        placement = str(letterbox.get("placement", "top_left")).lower()
        if placement not in {"top_left", "center"}:
            raise AiModelError(
                f"YOLOX model {model_id} letterbox placement must be top_left or center"
            )

        return YoloXModelManifest(
            model_id=model_id,
            name=str(config.get("name", model_id)),
            version=str(config.get("version", "")),
            model_format=model_format,
            model_path=model_path,
            sha256=expected_sha256,
            class_names=class_names,
            input_name=str(input_config.get("name", "images")),
            input_width=width,
            input_height=height,
            color_order=color_order,
            pixel_scale=float(normalization.get("pixel_scale", 1.0)),
            mean=mean,
            std=std,
            letterbox_value=int(letterbox.get("value", 114)),
            letterbox_placement=placement,
            output_name=str(output_config.get("name", "output")),
            decoder=decoder,
            strides=strides,
            scores_are_probabilities=bool(
                output_config.get("scores_are_probabilities", True)
            ),
            allowed_backends=allowed_backends,
            allowed_precisions=allowed_precisions,
            test_only=bool(config.get("test_only", False)),
        )

    @staticmethod
    def _mapping(parent: dict, key: str, model_id: str) -> dict:
        value = parent.get(key)
        if not isinstance(value, dict):
            raise AiModelError(f"YOLOX model {model_id} {key} must be a mapping")
        return value

    @staticmethod
    def _string_tuple(value: Any, field: str, model_id: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise AiModelError(
                f"YOLOX model {model_id} {field} must be a non-empty string list"
            )
        return tuple(item.strip() for item in value)

    @staticmethod
    def _positive_int_tuple(value: Any, field: str, model_id: str) -> tuple[int, ...]:
        if not isinstance(value, list) or not value:
            raise AiModelError(f"YOLOX model {model_id} {field} must be a non-empty list")
        return tuple(
            YoloXModelRegistry._positive_int(item, field, model_id) for item in value
        )

    @staticmethod
    def _positive_int(value: Any, field: str, model_id: str) -> int:
        if type(value) is not int or value <= 0:
            raise AiModelError(f"YOLOX model {model_id} {field} must be a positive integer")
        return int(value)

    @staticmethod
    def _float_triplet(value: Any, field: str, model_id: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3 or not all(
            type(item) in {int, float} for item in value
        ):
            raise AiModelError(f"YOLOX model {model_id} {field} must contain three numbers")
        return tuple(float(item) for item in value)


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_shape": [self.source_height, self.source_width],
            "input_shape": [self.input_height, self.input_width],
            "scale": [self.scale_x, self.scale_y],
            "padding": [self.pad_x, self.pad_y],
        }


def prepare_yolox_input(
    image: np.ndarray, manifest: YoloXModelManifest
) -> tuple[np.ndarray, LetterboxTransform]:
    if not isinstance(image, np.ndarray):
        raise AiModelError("YOLOX input must be a numpy.ndarray")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise AiModelError("YOLOX input must be a BGR uint8 image with three channels")
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise AiModelError("YOLOX input cannot be empty")

    ratio = min(
        manifest.input_width / float(source_width),
        manifest.input_height / float(source_height),
    )
    resized_width = max(1, min(manifest.input_width, int(source_width * ratio)))
    resized_height = max(1, min(manifest.input_height, int(source_height * ratio)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    if manifest.letterbox_placement == "center":
        pad_x = (manifest.input_width - resized_width) // 2
        pad_y = (manifest.input_height - resized_height) // 2
    else:
        pad_x = 0
        pad_y = 0
    canvas = np.full(
        (manifest.input_height, manifest.input_width, 3),
        manifest.letterbox_value,
        dtype=np.uint8,
    )
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    if manifest.color_order == "RGB":
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    tensor = canvas.astype(np.float32)
    tensor *= np.float32(manifest.pixel_scale)
    tensor -= np.asarray(manifest.mean, dtype=np.float32)
    tensor /= np.asarray(manifest.std, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])
    transform = LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        input_width=manifest.input_width,
        input_height=manifest.input_height,
        scale_x=resized_width / float(source_width),
        scale_y=resized_height / float(source_height),
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return tensor, transform


def parse_target_class_ids(value: Any, class_count: int) -> tuple[int, ...]:
    if value is None or value == "":
        return tuple(range(class_count))
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        try:
            parsed = [int(piece) for piece in pieces]
        except ValueError as exc:
            raise AiModelError("target_class_ids must be comma-separated integers") from exc
    elif isinstance(value, (list, tuple)):
        if not all(type(item) is int for item in value):
            raise AiModelError("target_class_ids must contain integers")
        parsed = list(value)
    else:
        raise AiModelError("target_class_ids must be a comma-separated string")
    unique = tuple(sorted(set(parsed)))
    if any(item < 0 or item >= class_count for item in unique):
        raise AiModelError(
            f"target_class_ids must be between 0 and {max(0, class_count - 1)}"
        )
    return unique


def validate_yolox_parameters(
    params: dict[str, Any], registry: YoloXModelRegistry
) -> YoloXModelManifest:
    model_id = str(params.get("model_id", "")).strip()
    if not model_id:
        raise AiModelError("YOLOX model_id must be selected")
    manifest = registry.get(model_id)
    parse_target_class_ids(params.get("target_class_ids", ""), len(manifest.class_names))
    backend = str(params.get("inference_backend", "auto")).lower()
    effective_backend = "onnxruntime_cpu" if backend == "auto" else backend
    if effective_backend not in manifest.allowed_backends:
        raise AiBackendUnavailable(
            f"YOLOX model {model_id} does not support backend {effective_backend}"
        )
    precision = str(params.get("precision", "fp32")).lower()
    if precision not in manifest.allowed_precisions:
        raise AiBackendUnavailable(
            f"YOLOX model {model_id} does not support precision {precision}"
        )
    return manifest


class OnnxRuntimeCpuSession:
    backend = "onnxruntime_cpu"
    device = "CPU"
    precision = "fp32"

    def __init__(self, manifest: YoloXModelManifest):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise AiBackendUnavailable(
                "onnxruntime is required for the YOLOX CPU reference backend"
            ) from exc
        started = time.perf_counter()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(manifest.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.load_sec = time.perf_counter() - started
        self.manifest = manifest
        self._lock = threading.RLock()
        self.last_inference_sec = 0.0
        self.inference_count = 0

        input_names = {item.name for item in self._session.get_inputs()}
        output_names = {item.name for item in self._session.get_outputs()}
        if manifest.input_name not in input_names:
            raise AiModelError(
                f"YOLOX model input {manifest.input_name!r} is not present in the ONNX graph"
            )
        if manifest.output_name not in output_names:
            raise AiModelError(
                f"YOLOX model output {manifest.output_name!r} is not present in the ONNX graph"
            )
        warmup_started = time.perf_counter()
        self._session.run(
            [manifest.output_name],
            {manifest.input_name: np.zeros(manifest.input_shape, dtype=np.float32)},
        )
        self.warmup_sec = time.perf_counter() - warmup_started

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        if tuple(tensor.shape) != self.manifest.input_shape or tensor.dtype != np.float32:
            raise AiModelError(
                f"YOLOX tensor must be float32 {self.manifest.input_shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
        with self._lock:
            started = time.perf_counter()
            output = self._session.run(
                [self.manifest.output_name],
                {self.manifest.input_name: tensor},
            )[0]
            self.last_inference_sec = time.perf_counter() - started
            self.inference_count += 1
        return np.asarray(output)


class AiModelSessionManager:
    def __init__(self, registry: YoloXModelRegistry | None = None):
        self.registry = registry or YoloXModelRegistry()
        self._sessions: dict[tuple[Any, ...], OnnxRuntimeCpuSession] = {}
        self._lock = threading.RLock()
        self.load_count = 0

    def session_for(
        self, manifest: YoloXModelManifest, backend: str = "auto", precision: str = "fp32"
    ) -> OnnxRuntimeCpuSession:
        requested_backend = str(backend).lower()
        effective_backend = (
            "onnxruntime_cpu" if requested_backend == "auto" else requested_backend
        )
        if effective_backend != "onnxruntime_cpu":
            raise AiBackendUnavailable(
                f"YOLOX backend {effective_backend} is not implemented in the CPU reference phase"
            )
        if str(precision).lower() != "fp32":
            raise AiBackendUnavailable("YOLOX CPU reference currently supports fp32 only")
        key = (
            manifest.sha256,
            effective_backend,
            "CPU",
            "fp32",
            manifest.input_shape,
        )
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = OnnxRuntimeCpuSession(manifest)
                self._sessions[key] = session
                self.load_count += 1
            return session

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def close(self) -> None:
        with self._lock:
            self._sessions.clear()


def decode_yolox_output(
    output: np.ndarray,
    manifest: YoloXModelManifest,
    transform: LetterboxTransform,
    *,
    confidence_threshold: float,
    nms_iou_threshold: float,
    target_class_ids: Any,
    max_detections: int,
    min_box_area_px: float,
    class_agnostic_nms: bool,
) -> list[dict[str, Any]]:
    predictions = np.asarray(output, dtype=np.float32)
    expected_attributes = 5 + len(manifest.class_names)
    expected_count = sum(
        (manifest.input_width // stride) * (manifest.input_height // stride)
        for stride in manifest.strides
    )
    if predictions.shape != (1, expected_count, expected_attributes):
        raise AiModelError(
            "YOLOX output shape mismatch: "
            f"expected={(1, expected_count, expected_attributes)} got={predictions.shape}"
        )
    if not np.isfinite(predictions).all():
        raise AiModelError("YOLOX output contains NaN or infinity")

    grids = []
    expanded_strides = []
    for stride in manifest.strides:
        grid_x, grid_y = np.meshgrid(
            np.arange(manifest.input_width // stride, dtype=np.float32),
            np.arange(manifest.input_height // stride, dtype=np.float32),
        )
        grids.append(np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2))
        expanded_strides.append(
            np.full((grid_x.size, 1), float(stride), dtype=np.float32)
        )
    grid = np.concatenate(grids, axis=0)
    stride_values = np.concatenate(expanded_strides, axis=0)
    rows = predictions[0]
    centers = (rows[:, :2] + grid) * stride_values
    sizes = np.exp(np.clip(rows[:, 2:4], -20.0, 20.0)) * stride_values
    model_xyxy = np.concatenate((centers - sizes / 2.0, centers + sizes / 2.0), axis=1)

    objectness = rows[:, 4]
    class_probabilities = rows[:, 5:]
    if not manifest.scores_are_probabilities:
        objectness = 1.0 / (1.0 + np.exp(-objectness))
        class_probabilities = 1.0 / (1.0 + np.exp(-class_probabilities))

    allowed_classes = set(
        parse_target_class_ids(target_class_ids, len(manifest.class_names))
    )
    candidates: list[dict[str, Any]] = []
    for index in range(rows.shape[0]):
        class_id = int(np.argmax(class_probabilities[index]))
        if class_id not in allowed_classes:
            continue
        class_probability = float(class_probabilities[index, class_id])
        object_score = float(objectness[index])
        confidence = object_score * class_probability
        if confidence < confidence_threshold:
            continue
        x1 = (float(model_xyxy[index, 0]) - transform.pad_x) / transform.scale_x
        y1 = (float(model_xyxy[index, 1]) - transform.pad_y) / transform.scale_y
        x2 = (float(model_xyxy[index, 2]) - transform.pad_x) / transform.scale_x
        y2 = (float(model_xyxy[index, 3]) - transform.pad_y) / transform.scale_y
        x1 = min(max(x1, 0.0), float(transform.source_width))
        y1 = min(max(y1, 0.0), float(transform.source_height))
        x2 = min(max(x2, 0.0), float(transform.source_width))
        y2 = min(max(y2, 0.0), float(transform.source_height))
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < min_box_area_px:
            continue
        candidates.append(
            {
                "index": index,
                "class_id": class_id,
                "objectness": object_score,
                "class_probability": class_probability,
                "confidence": confidence,
                "xyxy": [x1, y1, x2, y2],
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["confidence"],
            item["class_id"],
            item["xyxy"][1],
            item["xyxy"][0],
            item["index"],
        )
    )
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        suppressed = False
        for existing in selected:
            if not class_agnostic_nms and candidate["class_id"] != existing["class_id"]:
                continue
            if _box_iou(candidate["xyxy"], existing["xyxy"]) > nms_iou_threshold:
                suppressed = True
                break
        if not suppressed:
            selected.append(candidate)
            if len(selected) >= max_detections:
                break

    defects = []
    for item in selected:
        x1, y1, x2, y2 = item["xyxy"]
        left = max(0, min(transform.source_width, int(np.floor(x1))))
        top = max(0, min(transform.source_height, int(np.floor(y1))))
        right = max(left, min(transform.source_width, int(np.ceil(x2))))
        bottom = max(top, min(transform.source_height, int(np.ceil(y2))))
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            continue
        class_id = item["class_id"]
        defects.append(
            {
                "type": manifest.class_names[class_id],
                "bbox_local": [left, top, width, height],
                "area": float(width * height),
                "confidence": float(round(item["confidence"], 6)),
                "metadata": {
                    "class_id": class_id,
                    "class_name": manifest.class_names[class_id],
                    "objectness": float(round(item["objectness"], 6)),
                    "class_probability": float(
                        round(item["class_probability"], 6)
                    ),
                    "model_id": manifest.model_id,
                    "model_version": manifest.version,
                    "model_sha256": manifest.sha256,
                    "confidence_threshold": float(confidence_threshold),
                    "nms_iou_threshold": float(nms_iou_threshold),
                    "bbox_xyxy_float": [round(value, 6) for value in item["xyxy"]],
                    "letterbox": transform.to_dict(),
                },
            }
        )
    defects.sort(
        key=lambda item: (
            -item["confidence"],
            item["metadata"]["class_id"],
            item["bbox_local"][1],
            item["bbox_local"][0],
        )
    )
    return defects


def _box_iou(first: list[float], second: list[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
