from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ai_runtime import (  # noqa: E402
    AiBackendUnavailable,
    AiModelError,
    AiModelSessionManager,
    YoloXModelRegistry,
    prepare_yolox_input,
)
from core.image_loader import load_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress one persistent YOLOX session and report memory/latency checkpoints."
    )
    parser.add_argument("--model-id", default="yolox_tiny_fixture")
    parser.add_argument(
        "--backend",
        choices=("onnxruntime_cpu", "onnxruntime_cuda"),
        default="onnxruntime_cpu",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoints", default="10,100,1000")
    parser.add_argument("--max-rss-growth-mb", type=float, default=64.0)
    parser.add_argument("--allow-test-model", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _process_rss_bytes() -> int | None:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        if not get_process_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.WorkingSetSize)
    statm = Path("/proc/self/statm")
    if statm.is_file():
        try:
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None
    return None


def _gpu_memory_mb() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    current_pid = str(os.getpid())
    total = 0.0
    for line in completed.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) == 2 and pieces[0] == current_pid:
            try:
                total += float(pieces[1])
            except ValueError:
                continue
    return total


def _parse_checkpoints(value: str, iterations: int) -> tuple[int, ...]:
    try:
        parsed = sorted(
            {
                int(piece.strip())
                for piece in str(value).split(",")
                if piece.strip()
            }
        )
    except ValueError as exc:
        raise ValueError("--checkpoints must contain comma-separated integers") from exc
    if not parsed or parsed[0] <= 0 or parsed[-1] > iterations:
        raise ValueError(
            "--checkpoints must be positive and cannot exceed --iterations"
        )
    if parsed[-1] != iterations:
        parsed.append(iterations)
    return tuple(parsed)


def run_stability(
    *,
    model_id: str,
    backend: str,
    image_path: Path | None,
    warmup: int,
    iterations: int,
    checkpoints: tuple[int, ...],
    max_rss_growth_mb: float,
    allow_test_model: bool,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("--warmup must be zero or greater")
    if iterations <= 0:
        raise ValueError("--iterations must be greater than zero")
    if max_rss_growth_mb < 0:
        raise ValueError("--max-rss-growth-mb must be zero or greater")
    if not checkpoints or checkpoints[-1] != iterations:
        raise ValueError("checkpoints must include the final iteration")

    registry = YoloXModelRegistry()
    manifest = registry.get(model_id)
    if manifest.test_only and not allow_test_model:
        raise AiModelError(
            f"YOLOX model {model_id} is marked test_only; use --allow-test-model "
            "only for software validation"
        )
    image = (
        load_image(image_path)
        if image_path
        else np.zeros(
            (manifest.input_height, manifest.input_width, 3), dtype=np.uint8
        )
    )
    tensor, _ = prepare_yolox_input(image, manifest)
    manager = AiModelSessionManager(
        registry,
        gpu_mode="cuda" if backend == "onnxruntime_cuda" else "cpu",
        fallback_to_cpu=False,
        queue_depth=1,
        max_cached_sessions=1,
    )
    checkpoint_reports = []
    timings_ms = []
    output_digest = ""
    deterministic = True
    try:
        session = manager.session_for(manifest, backend=backend)
        for _ in range(warmup):
            session.infer(tensor)
        baseline_rss = _process_rss_bytes()
        baseline_vram = _gpu_memory_mb() if backend == "onnxruntime_cuda" else None
        for index in range(1, iterations + 1):
            started = time.perf_counter()
            output = session.infer(tensor)
            timings_ms.append((time.perf_counter() - started) * 1000.0)
            digest = hashlib.sha256(
                np.ascontiguousarray(output).tobytes()
            ).hexdigest()
            if output_digest and digest != output_digest:
                deterministic = False
            output_digest = output_digest or digest
            if index in checkpoints:
                checkpoint_reports.append(
                    {
                        "iteration": index,
                        "rss_bytes": _process_rss_bytes(),
                        "gpu_process_memory_mb": (
                            _gpu_memory_mb()
                            if backend == "onnxruntime_cuda"
                            else None
                        ),
                        "session_count": manager.session_count,
                        "load_count": manager.load_count,
                        "inference_count": session.inference_count,
                    }
                )
        final_rss = checkpoint_reports[-1]["rss_bytes"]
        first_rss = checkpoint_reports[0]["rss_bytes"]
        rss_growth = (
            max(0, final_rss - first_rss)
            if final_rss is not None and first_rss is not None
            else None
        )
        final_vram = checkpoint_reports[-1]["gpu_process_memory_mb"]
        first_vram = checkpoint_reports[0]["gpu_process_memory_mb"]
        vram_growth = (
            max(0.0, final_vram - first_vram)
            if final_vram is not None and first_vram is not None
            else None
        )
        session_stats = session.performance_stats()
        checks = {
            "deterministic_output": deterministic,
            "single_session": manager.session_count == 1 and manager.load_count == 1,
            "no_inference_failures": session_stats["inference_failures"] == 0,
            "no_queue_rejections": session_stats["queue_rejections"] == 0,
            "queue_drained": (
                session_stats["active"] == 0
                and session_stats["queue_waiting"] == 0
            ),
            "rss_growth_within_limit": (
                rss_growth is not None
                and rss_growth <= max_rss_growth_mb * 1024 * 1024
            ),
            "vram_observed": (
                vram_growth is not None if backend == "onnxruntime_cuda" else True
            ),
        }
        return {
            "schema_version": 1,
            "passed": all(checks.values()),
            "model": {
                "id": manifest.model_id,
                "version": manifest.version,
                "sha256": manifest.sha256,
                "test_only": manifest.test_only,
            },
            "backend": backend,
            "providers": list(manager.available_providers()),
            "input": str(image_path.resolve()) if image_path else "synthetic_zero",
            "warmup": warmup,
            "iterations": iterations,
            "checkpoints": checkpoint_reports,
            "latency": {
                "median_ms": round(statistics.median(timings_ms), 6),
                "p95_ms": round(float(np.percentile(timings_ms, 95)), 6),
                "min_ms": round(min(timings_ms), 6),
                "max_ms": round(max(timings_ms), 6),
            },
            "memory": {
                "baseline_rss_bytes": baseline_rss,
                "rss_growth_after_first_checkpoint_bytes": rss_growth,
                "max_rss_growth_mb": max_rss_growth_mb,
                "baseline_gpu_process_memory_mb": baseline_vram,
                "gpu_process_memory_growth_after_first_checkpoint_mb": vram_growth,
            },
            "output_sha256": output_digest,
            "checks": checks,
            "session": manager.performance_stats(),
        }
    finally:
        manager.close()


def main() -> int:
    args = parse_args()
    try:
        checkpoints = _parse_checkpoints(args.checkpoints, args.iterations)
        report = run_stability(
            model_id=args.model_id,
            backend=args.backend,
            image_path=args.image,
            warmup=args.warmup,
            iterations=args.iterations,
            checkpoints=checkpoints,
            max_rss_growth_mb=args.max_rss_growth_mb,
            allow_test_model=args.allow_test_model,
        )
        exit_code = 0 if report["passed"] else 1
    except (AiBackendUnavailable, AiModelError, OSError, ValueError) as exc:
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
