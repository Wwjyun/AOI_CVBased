from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.image_loader import SUPPORTED_EXTENSIONS
from core.logging_system import LogMixin
from core.pipeline import AOIPipeline


MonitorProgressCallback = Callable[[int, str], None]
MonitorItemCallback = Callable[[dict], None]
MonitorStopCallback = Callable[[], bool]


@dataclass(frozen=True)
class MonitorImageResult:
    image_path: Path
    final_result: str
    defect_count: int
    ng_count: int
    tile_count: int
    duration_sec: float
    outputs: dict
    detail: dict
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "image_name": self.image_path.name,
            "final_result": self.final_result,
            "defect_count": self.defect_count,
            "ng_count": self.ng_count,
            "tile_count": self.tile_count,
            "duration_sec": self.duration_sec,
            "outputs": dict(self.outputs),
            "detail": dict(self.detail),
            "error": self.error,
        }


class FolderMonitorProcessor(LogMixin):
    """Watch a folder tree and process newly added images one at a time."""

    def __init__(
        self,
        input_dir: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        poll_interval_sec: float = 1.0,
        stable_checks: int = 2,
        progress_callback: MonitorProgressCallback | None = None,
        item_callback: MonitorItemCallback | None = None,
        stop_callback: MonitorStopCallback | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.poll_interval_sec = max(0.2, float(poll_interval_sec))
        self.stable_checks = max(1, int(stable_checks))
        self.progress_callback = progress_callback
        self.item_callback = item_callback
        self.stop_callback = stop_callback
        self._seen: set[Path] = set()
        self._pending: list[Path] = []
        self._file_states: dict[Path, tuple[int, int, int]] = {}
        self._processed_count = 0

    def run(self) -> dict:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Monitor folder does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Monitor input is not a folder: {self.input_dir}")

        started_at = datetime.datetime.now()
        monitor_output_dir = self.output_dir / "monitor" / started_at.strftime("%Y%m%d_%H%M%S")
        monitor_output_dir.mkdir(parents=True, exist_ok=True)
        self._seen = set(self._discover_images())
        self.logger.info(
            "Folder monitor started: input=%s recipe=%s output=%s initial_seen=%s",
            self.input_dir,
            self.recipe_path,
            monitor_output_dir,
            len(self._seen),
        )
        self._progress(0, f"Monitoring {self.input_dir}")

        while not self._should_stop():
            self._enqueue_new_stable_images()
            while self._pending and not self._should_stop():
                image_path = self._pending.pop(0)
                result = self._process_image(image_path, monitor_output_dir)
                self._processed_count += 1
                if self.item_callback is not None:
                    self.item_callback(result.to_dict())
                self._progress(100, f"Processed {image_path.name}")
            self._sleep_interval()

        finished_at = datetime.datetime.now()
        summary = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": round((finished_at - started_at).total_seconds(), 2),
            "output_dir": str(monitor_output_dir),
            "processed": self._processed_count,
        }
        self.logger.info("Folder monitor stopped: summary=%s", summary)
        return summary

    def _enqueue_new_stable_images(self) -> None:
        for image_path in self._discover_images():
            if image_path in self._seen or image_path in self._pending:
                continue
            if not self._is_stable(image_path):
                continue
            self._seen.add(image_path)
            self._pending.append(image_path)
            self.logger.info("Monitor queued image: %s", image_path)
            self._progress(5, f"Queued {image_path.name}")

    def _process_image(self, image_path: Path, monitor_output_dir: Path) -> MonitorImageResult:
        try:
            self.logger.info("Monitor image started: image=%s", image_path)
            pipeline = AOIPipeline(
                recipe_path=self.recipe_path,
                output_dir=monitor_output_dir,
                output_overrides=self.output_overrides,
                progress_callback=lambda pct, msg: self._progress(pct, f"{image_path.name}: {msg}"),
            )
            result = pipeline.run(image_path)
            summary = result.get("summary", {})
            return MonitorImageResult(
                image_path=image_path,
                final_result=str(result.get("final_result", "-")),
                defect_count=int(summary.get("defect_count", 0)),
                ng_count=int(summary.get("ng_count", 0)),
                tile_count=int(summary.get("tile_count", 0)),
                duration_sec=float(result.get("duration_sec", 0) or 0),
                outputs=result.get("outputs", {}),
                detail=result,
            )
        except Exception as exc:
            self.logger.exception("Monitor image failed: image=%s", image_path)
            return MonitorImageResult(
                image_path=image_path,
                final_result="ERROR",
                defect_count=0,
                ng_count=0,
                tile_count=0,
                duration_sec=0.0,
                outputs={},
                detail={},
                error=str(exc),
            )

    def _discover_images(self) -> list[Path]:
        return sorted(
            path
            for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _is_stable(self, image_path: Path) -> bool:
        try:
            stat = image_path.stat()
        except OSError:
            return False
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
        last_size, last_mtime_ns, count = self._file_states.get(image_path, (-1, -1, 0))
        count = count + 1 if size == last_size and mtime_ns == last_mtime_ns else 1
        self._file_states[image_path] = (size, mtime_ns, count)
        return count >= self.stable_checks

    def _sleep_interval(self) -> None:
        remaining = self.poll_interval_sec
        while remaining > 0 and not self._should_stop():
            step = min(0.1, remaining)
            time.sleep(step)
            remaining -= step

    def _should_stop(self) -> bool:
        return bool(self.stop_callback and self.stop_callback())

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(max(0, min(100, int(percent))), message)
