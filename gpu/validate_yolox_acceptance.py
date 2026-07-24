from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ai_runtime import AiModelError, YoloXModelRegistry  # noqa: E402
from core.gpu_session import GpuExecutionSession  # noqa: E402
from core.image_loader import load_image  # noqa: E402
from core.pipeline import AOIPipeline  # noqa: E402
from core.recipe_manager import RecipeError, RecipeManager  # noqa: E402


class AcceptanceManifestError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a YOLOX model against a labeled AOI acceptance set."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--backend",
        choices=("onnxruntime_cpu", "onnxruntime_cuda"),
        default="onnxruntime_cpu",
    )
    parser.add_argument(
        "--allow-test-model",
        action="store_true",
        help="Allow a registry model marked test_only. Never use for production sign-off.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve_file(root: Path, value: Any, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise AcceptanceManifestError(f"{field} is required")
    path = Path(text)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_file():
        raise AcceptanceManifestError(f"{field} does not exist: {resolved}")
    return resolved


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if type(value) not in {int, float}:
        raise AcceptanceManifestError(f"{field} must be a number")
    parsed = float(value)
    if parsed < minimum or parsed > maximum:
        raise AcceptanceManifestError(
            f"{field} must be between {minimum} and {maximum}"
        )
    return parsed


def load_acceptance_manifest(
    path: Path,
    *,
    registry: YoloXModelRegistry | None = None,
    allow_test_model: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise AcceptanceManifestError(
            f"Acceptance manifest does not exist: {manifest_path}"
        )
    try:
        document = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise AcceptanceManifestError(f"Cannot read acceptance manifest: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise AcceptanceManifestError("Acceptance manifest schema_version must be 1")

    model_registry = registry or YoloXModelRegistry()
    model_id = str(document.get("model_id", "")).strip()
    if not model_id:
        raise AcceptanceManifestError("model_id is required")
    model = model_registry.get(model_id)
    if model.test_only and not allow_test_model:
        raise AcceptanceManifestError(
            f"YOLOX model {model_id} is marked test_only and cannot pass production acceptance"
        )

    root = manifest_path.parent
    recipe_path = _resolve_file(root, document.get("recipe"), "recipe")
    thresholds = document.get("thresholds")
    if not isinstance(thresholds, dict):
        raise AcceptanceManifestError("thresholds must be a mapping")
    parsed_thresholds = {
        "min_precision": _number(
            thresholds.get("min_precision"), "thresholds.min_precision", minimum=0, maximum=1
        ),
        "min_recall": _number(
            thresholds.get("min_recall"), "thresholds.min_recall", minimum=0, maximum=1
        ),
        "min_map50": _number(
            thresholds.get("min_map50"), "thresholds.min_map50", minimum=0, maximum=1
        ),
        "max_false_kill_rate": _number(
            thresholds.get("max_false_kill_rate"),
            "thresholds.max_false_kill_rate",
            minimum=0,
            maximum=1,
        ),
        "max_miss_rate": _number(
            thresholds.get("max_miss_rate"),
            "thresholds.max_miss_rate",
            minimum=0,
            maximum=1,
        ),
    }
    iou_threshold = _number(
        document.get("iou_match_threshold", 0.5),
        "iou_match_threshold",
        minimum=0,
        maximum=1,
    )

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise AcceptanceManifestError("cases must be a non-empty list")
    cases = []
    case_ids: set[str] = set()
    positive_count = 0
    negative_count = 0
    for index, raw_case in enumerate(raw_cases):
        field = f"cases[{index}]"
        if not isinstance(raw_case, dict):
            raise AcceptanceManifestError(f"{field} must be a mapping")
        case_id = str(raw_case.get("id", "")).strip()
        if not case_id or case_id in case_ids:
            raise AcceptanceManifestError(f"{field}.id must be non-empty and unique")
        case_ids.add(case_id)
        image_path = _resolve_file(root, raw_case.get("image"), f"{field}.image")
        image_height, image_width = load_image(image_path).shape[:2]
        raw_ground_truth = raw_case.get("ground_truth")
        if not isinstance(raw_ground_truth, list):
            raise AcceptanceManifestError(f"{field}.ground_truth must be a list")
        ground_truth = []
        for box_index, raw_box in enumerate(raw_ground_truth):
            box_field = f"{field}.ground_truth[{box_index}]"
            if not isinstance(raw_box, dict):
                raise AcceptanceManifestError(f"{box_field} must be a mapping")
            class_id = raw_box.get("class_id")
            if type(class_id) is not int or not 0 <= class_id < len(model.class_names):
                raise AcceptanceManifestError(
                    f"{box_field}.class_id must be a valid model class id"
                )
            bbox = raw_box.get("bbox_xywh")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(type(value) in {int, float} for value in bbox)
            ):
                raise AcceptanceManifestError(
                    f"{box_field}.bbox_xywh must contain four numbers"
                )
            parsed_bbox = [float(value) for value in bbox]
            if parsed_bbox[0] < 0 or parsed_bbox[1] < 0:
                raise AcceptanceManifestError(
                    f"{box_field}.bbox_xywh x/y must be zero or greater"
                )
            if parsed_bbox[2] <= 0 or parsed_bbox[3] <= 0:
                raise AcceptanceManifestError(
                    f"{box_field}.bbox_xywh width/height must be greater than zero"
                )
            if (
                parsed_bbox[0] + parsed_bbox[2] > image_width
                or parsed_bbox[1] + parsed_bbox[3] > image_height
            ):
                raise AcceptanceManifestError(
                    f"{box_field}.bbox_xywh exceeds image bounds "
                    f"{image_width}x{image_height}"
                )
            ground_truth.append({"class_id": class_id, "bbox": parsed_bbox})
        positive_count += bool(ground_truth)
        negative_count += not ground_truth
        cases.append(
            {
                "id": case_id,
                "image": image_path,
                "ground_truth": ground_truth,
            }
        )
    if positive_count == 0 or negative_count == 0:
        raise AcceptanceManifestError(
            "Acceptance set must contain at least one positive and one negative case"
        )
    return {
        "path": manifest_path,
        "model": model,
        "recipe": recipe_path,
        "iou_match_threshold": iou_threshold,
        "thresholds": parsed_thresholds,
        "cases": cases,
    }


def _iou(first: list[float], second: list[float]) -> float:
    first_x2, first_y2 = first[0] + first[2], first[1] + first[3]
    second_x2, second_y2 = second[0] + second[2], second[1] + second[3]
    width = max(0.0, min(first_x2, second_x2) - max(first[0], second[0]))
    height = max(0.0, min(first_y2, second_y2) - max(first[1], second[1]))
    intersection = width * height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def _match_by_class(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    matched: set[int] = set()
    true_positive = 0
    false_positive = 0
    for prediction in sorted(
        predictions, key=lambda item: (-item["confidence"], item["class_id"])
    ):
        candidates = [
            (index, _iou(prediction["bbox"], truth["bbox"]))
            for index, truth in enumerate(ground_truth)
            if index not in matched and truth["class_id"] == prediction["class_id"]
        ]
        best = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if best[1] >= iou_threshold:
            matched.add(best[0])
            true_positive += 1
        else:
            false_positive += 1
    return true_positive, false_positive, len(ground_truth) - len(matched)


def _average_precision(
    cases: list[dict[str, Any]], class_id: int, iou_threshold: float
) -> float:
    truth_by_case = {
        case["id"]: [
            item for item in case["ground_truth"] if item["class_id"] == class_id
        ]
        for case in cases
    }
    truth_count = sum(len(items) for items in truth_by_case.values())
    if truth_count == 0:
        return 0.0
    predictions = sorted(
        (
            (case["id"], prediction)
            for case in cases
            for prediction in case["predictions"]
            if prediction["class_id"] == class_id
        ),
        key=lambda item: (-item[1]["confidence"], item[0]),
    )
    matched = {case_id: set() for case_id in truth_by_case}
    true_positive = []
    false_positive = []
    for case_id, prediction in predictions:
        candidates = [
            (index, _iou(prediction["bbox"], truth["bbox"]))
            for index, truth in enumerate(truth_by_case[case_id])
            if index not in matched[case_id]
        ]
        best = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        is_match = best[1] >= iou_threshold
        if is_match:
            matched[case_id].add(best[0])
        true_positive.append(1 if is_match else 0)
        false_positive.append(0 if is_match else 1)
    recalls = []
    precisions = []
    cumulative_tp = 0
    cumulative_fp = 0
    for tp, fp in zip(true_positive, false_positive):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / truth_count)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    recall_points = [0.0, *recalls, 1.0]
    precision_points = [0.0, *precisions, 0.0]
    for index in range(len(precision_points) - 2, -1, -1):
        precision_points[index] = max(
            precision_points[index], precision_points[index + 1]
        )
    return sum(
        (recall_points[index] - recall_points[index - 1])
        * precision_points[index]
        for index in range(1, len(recall_points))
        if recall_points[index] != recall_points[index - 1]
    )


def _confusion_matrix(
    cases: list[dict[str, Any]], class_count: int, iou_threshold: float
) -> list[list[int]]:
    background = class_count
    matrix = [[0 for _ in range(class_count + 1)] for _ in range(class_count + 1)]
    for case in cases:
        matched_truth: set[int] = set()
        matched_predictions: set[int] = set()
        candidates = sorted(
            (
                (_iou(truth["bbox"], prediction["bbox"]), truth_index, prediction_index)
                for truth_index, truth in enumerate(case["ground_truth"])
                for prediction_index, prediction in enumerate(case["predictions"])
            ),
            reverse=True,
        )
        for overlap, truth_index, prediction_index in candidates:
            if overlap < iou_threshold:
                break
            if truth_index in matched_truth or prediction_index in matched_predictions:
                continue
            matched_truth.add(truth_index)
            matched_predictions.add(prediction_index)
            matrix[case["ground_truth"][truth_index]["class_id"]][
                case["predictions"][prediction_index]["class_id"]
            ] += 1
        for truth_index, truth in enumerate(case["ground_truth"]):
            if truth_index not in matched_truth:
                matrix[truth["class_id"]][background] += 1
        for prediction_index, prediction in enumerate(case["predictions"]):
            if prediction_index not in matched_predictions:
                matrix[background][prediction["class_id"]] += 1
    return matrix


def evaluate_predictions(
    cases: list[dict[str, Any]],
    *,
    class_names: tuple[str, ...],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    per_class = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in range(len(class_names))
    }
    negative_cases = false_kill_cases = 0
    for case in cases:
        case_tp, case_fp, case_fn = _match_by_class(
            case["ground_truth"], case["predictions"], iou_threshold
        )
        true_positive += case_tp
        false_positive += case_fp
        false_negative += case_fn
        if not case["ground_truth"]:
            negative_cases += 1
            false_kill_cases += bool(case["predictions"])
        for class_id in per_class:
            class_tp, class_fp, class_fn = _match_by_class(
                [
                    item
                    for item in case["ground_truth"]
                    if item["class_id"] == class_id
                ],
                [
                    item
                    for item in case["predictions"]
                    if item["class_id"] == class_id
                ],
                iou_threshold,
            )
            per_class[class_id]["tp"] += class_tp
            per_class[class_id]["fp"] += class_fp
            per_class[class_id]["fn"] += class_fn
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    ground_truth_count = true_positive + false_negative
    ap50 = {
        class_names[class_id]: _average_precision(cases, class_id, iou_threshold)
        for class_id in range(len(class_names))
        if any(
            item["class_id"] == class_id
            for case in cases
            for item in case["ground_truth"]
        )
    }
    labels = [*class_names, "__background__"]
    return {
        "counts": {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "ground_truth": ground_truth_count,
            "predictions": true_positive + false_positive,
        },
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "map50": round(sum(ap50.values()) / len(ap50) if ap50 else 0.0, 6),
        "false_kill_rate": round(
            false_kill_cases / negative_cases if negative_cases else 0.0, 6
        ),
        "miss_rate": round(
            false_negative / ground_truth_count if ground_truth_count else 0.0, 6
        ),
        "ap50_per_class": {
            name: round(value, 6) for name, value in ap50.items()
        },
        "per_class": {
            class_names[class_id]: counts for class_id, counts in per_class.items()
        },
        "confusion_matrix": {
            "rows_ground_truth": labels,
            "columns_prediction": labels,
            "values": _confusion_matrix(cases, len(class_names), iou_threshold),
        },
    }


def run_validation(
    manifest_path: Path,
    *,
    backend: str = "onnxruntime_cpu",
    allow_test_model: bool = False,
) -> dict[str, Any]:
    registry = YoloXModelRegistry()
    acceptance = load_acceptance_manifest(
        manifest_path,
        registry=registry,
        allow_test_model=allow_test_model,
    )
    recipe = deepcopy(RecipeManager().load(acceptance["recipe"]))
    detector = recipe.get("detectors", {}).get("yolox")
    if not isinstance(detector, dict) or not detector.get("enabled", False):
        raise AcceptanceManifestError("recipe must enable the yolox detector")
    detector.setdefault("params", {})["model_id"] = acceptance["model"].model_id
    detector["params"]["inference_backend"] = backend
    detector["use_gpu"] = backend == "onnxruntime_cuda"
    recipe.setdefault("gpu", {})["mode"] = (
        "cuda" if backend == "onnxruntime_cuda" else "cpu"
    )
    recipe["gpu"]["fallback_to_cpu"] = False
    recipe["output"] = {
        **recipe.get("output", {}),
        "save_overlay": False,
        "save_ng_tiles": False,
        "save_csv": False,
        "save_matrix_csv": False,
        "save_json": False,
        "save_debug_images": False,
    }

    evaluated_cases = []
    with tempfile.TemporaryDirectory(prefix="visionflow_yolox_acceptance_") as temporary:
        root = Path(temporary)
        recipe_path = root / "acceptance_recipe.yaml"
        recipe_path.write_text(
            yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        with GpuExecutionSession.from_recipe(recipe, workload="throughput") as session:
            for index, case in enumerate(acceptance["cases"]):
                result = AOIPipeline(
                    recipe_path,
                    root / f"case_{index}",
                    gpu_session=session,
                ).run(case["image"])
                predictions = []
                actual_backends = set()
                for tile in result["tiles"]:
                    for detector_result in tile["detectors"]:
                        if detector_result["detector_id"] != "yolox":
                            continue
                        ai = detector_result.get("execution", {}).get("ai", {})
                        if ai.get("actual_backend"):
                            actual_backends.add(ai["actual_backend"])
                        for defect in detector_result["defects"]:
                            predictions.append(
                                {
                                    "class_id": int(defect["metadata"]["class_id"]),
                                    "bbox": [
                                        float(value)
                                        for value in defect["bbox_global"]
                                    ],
                                    "confidence": float(defect["confidence"]),
                                }
                            )
                if actual_backends != {backend}:
                    raise AiModelError(
                        f"Case {case['id']} used unexpected backends: "
                        f"{sorted(actual_backends) or ['(none)']}"
                    )
                evaluated_cases.append(
                    {
                        "id": case["id"],
                        "image": str(case["image"]),
                        "ground_truth": case["ground_truth"],
                        "predictions": predictions,
                        "pipeline_final": result["final_result"],
                    }
                )
            session_stats = session.ai_session_manager.performance_stats()

    metrics = evaluate_predictions(
        evaluated_cases,
        class_names=acceptance["model"].class_names,
        iou_threshold=acceptance["iou_match_threshold"],
    )
    thresholds = acceptance["thresholds"]
    checks = {
        "precision": metrics["precision"] >= thresholds["min_precision"],
        "recall": metrics["recall"] >= thresholds["min_recall"],
        "map50": metrics["map50"] >= thresholds["min_map50"],
        "false_kill_rate": (
            metrics["false_kill_rate"] <= thresholds["max_false_kill_rate"]
        ),
        "miss_rate": metrics["miss_rate"] <= thresholds["max_miss_rate"],
        "single_session": (
            session_stats["load_count"] == 1
            and session_stats["session_count"] == 1
        ),
    }
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "manifest": str(acceptance["path"]),
        "model": {
            "id": acceptance["model"].model_id,
            "version": acceptance["model"].version,
            "sha256": acceptance["model"].sha256,
            "test_only": acceptance["model"].test_only,
        },
        "backend": backend,
        "iou_match_threshold": acceptance["iou_match_threshold"],
        "thresholds": thresholds,
        "checks": checks,
        "metrics": metrics,
        "session": session_stats,
        "cases": evaluated_cases,
    }


def main() -> int:
    args = parse_args()
    try:
        report = run_validation(
            args.manifest,
            backend=args.backend,
            allow_test_model=args.allow_test_model,
        )
        exit_code = 0 if report["passed"] else 1
    except (
        AcceptanceManifestError,
        AiModelError,
        RecipeError,
        OSError,
        ValueError,
    ) as exc:
        report = {"schema_version": 1, "passed": False, "error": str(exc)}
        exit_code = 2
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
