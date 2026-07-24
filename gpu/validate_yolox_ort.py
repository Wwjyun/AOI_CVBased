from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ai_runtime import (  # noqa: E402
    AiModelError,
    AiModelSessionManager,
    YoloXModelRegistry,
    compare_yolox_backend_results,
    decode_yolox_output,
    prepare_yolox_input,
)
from core.image_loader import load_image  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare YOLOX ONNX Runtime CPU and CUDA outputs."
    )
    parser.add_argument("--model-id", default="yolox_tiny_fixture")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--raw-atol", type=float, default=1e-5)
    parser.add_argument("--raw-rtol", type=float, default=1e-5)
    parser.add_argument("--bbox-tolerance-px", type=int, default=1)
    parser.add_argument("--confidence-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _timed_inference(session, tensor: np.ndarray, iterations: int):
    timings = []
    output = None
    for _ in range(iterations):
        started = time.perf_counter()
        output = session.infer(tensor)
        timings.append((time.perf_counter() - started) * 1000.0)
    return output, {
        "iterations": iterations,
        "median_ms": round(statistics.median(timings), 6),
        "p95_ms": round(float(np.percentile(timings, 95)), 6),
    }


def run_validation(args: argparse.Namespace) -> dict:
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than 0")
    registry = YoloXModelRegistry()
    manifest = registry.get(args.model_id)
    manager = AiModelSessionManager(
        registry,
        gpu_mode="cuda",
        fallback_to_cpu=False,
    )
    try:
        image = (
            load_image(args.image)
            if args.image
            else np.zeros(
                (manifest.input_height, manifest.input_width, 3),
                dtype=np.uint8,
            )
        )
        tensor, transform = prepare_yolox_input(image, manifest)
        cpu = manager.session_for(manifest, backend="onnxruntime_cpu")
        cuda = manager.session_for(manifest, backend="onnxruntime_cuda")
        cpu_output, cpu_timing = _timed_inference(cpu, tensor, args.iterations)
        cuda_output, cuda_timing = _timed_inference(cuda, tensor, args.iterations)
        decode_options = {
            "confidence_threshold": 0.25,
            "nms_iou_threshold": 0.45,
            "target_class_ids": "",
            "max_detections": 300,
            "min_box_area_px": 0.0,
            "class_agnostic_nms": False,
        }
        cpu_defects = decode_yolox_output(
            cpu_output, manifest, transform, **decode_options
        )
        cuda_defects = decode_yolox_output(
            cuda_output, manifest, transform, **decode_options
        )
        comparison = compare_yolox_backend_results(
            cpu_output,
            cuda_output,
            cpu_defects,
            cuda_defects,
            raw_atol=args.raw_atol,
            raw_rtol=args.raw_rtol,
            bbox_abs_tolerance_px=args.bbox_tolerance_px,
            confidence_abs_tolerance=args.confidence_tolerance,
        )
        return {
            "schema_version": 1,
            "model_id": manifest.model_id,
            "model_version": manifest.version,
            "model_sha256": manifest.sha256,
            "image": str(args.image.resolve()) if args.image else "synthetic_zero",
            "providers": list(manager.available_providers()),
            "cpu": {
                "backend": cpu.backend,
                "device": cpu.device,
                "load_sec": round(cpu.load_sec, 6),
                "warmup_sec": round(cpu.warmup_sec, 6),
                **cpu_timing,
            },
            "cuda": {
                "backend": cuda.backend,
                "device": cuda.device,
                "load_sec": round(cuda.load_sec, 6),
                "warmup_sec": round(cuda.warmup_sec, 6),
                **cuda_timing,
            },
            "comparison": comparison,
        }
    finally:
        manager.close()


def main() -> int:
    args = _arguments()
    try:
        report = run_validation(args)
        exit_code = 0 if report["comparison"]["passed"] else 1
    except (AiModelError, OSError, ValueError) as exc:
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
