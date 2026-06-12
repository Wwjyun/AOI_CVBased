from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.image_loader import SUPPORTED_EXTENSIONS
from core.pipeline import AOIPipeline


BatchProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class BatchImageResult:
    image_path: Path
    final_result: str
    defect_count: int
    ng_count: int
    tile_count: int
    outputs: dict
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "image_name": self.image_path.name,
            "final_result": self.final_result,
            "defect_count": self.defect_count,
            "ng_count": self.ng_count,
            "tile_count": self.tile_count,
            "outputs": dict(self.outputs),
            "error": self.error,
        }


class BatchInspectionProcessor:
    """Run the existing AOI pipeline over every supported image in a folder."""

    def __init__(
        self,
        input_dir: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        recursive: bool = False,
        progress_callback: BatchProgressCallback | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.recursive = recursive
        self.progress_callback = progress_callback

    def run(self) -> dict:
        image_paths = self.discover_images()
        started_at = datetime.datetime.now()
        batch_output_dir = self.output_dir / "batch" / started_at.strftime("%Y%m%d_%H%M%S")
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        if not image_paths:
            return self._build_summary(started_at, batch_output_dir, [])

        results: list[BatchImageResult] = []
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            self._progress_for_image(index, total, 0, f"Batch {index}/{total}: starting {image_path.name}")
            try:
                pipeline = AOIPipeline(
                    recipe_path=self.recipe_path,
                    output_dir=batch_output_dir,
                    progress_callback=lambda pct, msg, i=index, t=total: self._progress_for_image(i, t, pct, msg),
                    output_overrides=self.output_overrides,
                )
                result = pipeline.run(image_path)
                summary = result.get("summary", {})
                results.append(
                    BatchImageResult(
                        image_path=image_path,
                        final_result=str(result.get("final_result", "-")),
                        defect_count=int(summary.get("defect_count", 0)),
                        ng_count=int(summary.get("ng_count", 0)),
                        tile_count=int(summary.get("tile_count", 0)),
                        outputs=result.get("outputs", {}),
                    )
                )
            except Exception as exc:
                results.append(
                    BatchImageResult(
                        image_path=image_path,
                        final_result="ERROR",
                        defect_count=0,
                        ng_count=0,
                        tile_count=0,
                        outputs={},
                        error=str(exc),
                    )
                )
            self._progress_for_image(index, total, 100, f"Batch {index}/{total}: finished {image_path.name}")

        return self._build_summary(started_at, batch_output_dir, results)

    def discover_images(self) -> list[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Batch folder does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Batch input is not a folder: {self.input_dir}")

        iterator = self.input_dir.rglob("*") if self.recursive else self.input_dir.iterdir()
        return sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _progress_for_image(self, image_index: int, total_images: int, image_percent: int, message: str) -> None:
        if self.progress_callback is None:
            return
        total_images = max(total_images, 1)
        image_percent = max(0, min(100, int(image_percent)))
        overall = int(((image_index - 1) + image_percent / 100.0) / total_images * 100)
        self.progress_callback(max(0, min(100, overall)), message)

    @staticmethod
    def _build_summary(started_at: datetime.datetime, batch_output_dir: Path, results: list[BatchImageResult]) -> dict:
        finished_at = datetime.datetime.now()
        rows = [result.to_dict() for result in results]
        total = len(rows)
        pass_count = sum(1 for row in rows if row["final_result"] == "PASS")
        ng_count = sum(1 for row in rows if row["final_result"] == "NG")
        error_count = sum(1 for row in rows if row["final_result"] == "ERROR")
        return {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": round((finished_at - started_at).total_seconds(), 2),
            "output_dir": str(batch_output_dir),
            "summary": {
                "total": total,
                "pass": pass_count,
                "ng": ng_count,
                "error": error_count,
                "defects": sum(int(row.get("defect_count", 0)) for row in rows),
            },
            "items": rows,
        }
