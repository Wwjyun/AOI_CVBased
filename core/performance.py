from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator


class PipelineProfiler:
    """Collect additive host wall-clock timings without changing pipeline behavior."""

    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._durations: defaultdict[str, float] = defaultdict(float)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._durations[str(name)] += time.perf_counter() - started

    def add_duration(self, name: str, duration: float) -> None:
        self._durations[str(name)] += max(0.0, float(duration))

    def snapshot(self) -> dict:
        stages: dict[str, float] = {}
        detectors: dict[str, float] = {}
        reporting: dict[str, float] = {}
        detector_stages: dict[str, dict[str, float]] = {}
        for name, duration in sorted(self._durations.items()):
            rounded = round(duration, 6)
            if name.startswith("detector:"):
                detectors[name.split(":", 1)[1]] = rounded
            elif name.startswith("detector_stage:"):
                _, detector_id, stage = name.split(":", 2)
                detector_stages.setdefault(detector_id, {})[stage] = rounded
            elif name.startswith("report:"):
                reporting[name.split(":", 1)[1]] = rounded
            else:
                stages[name] = rounded
        if "detectors_total" in self._durations:
            detector_total = self._durations["detectors_total"]
            measured_detectors = sum(
                duration for name, duration in self._durations.items()
                if name.startswith("detector:")
            )
            stages["python_tile_detector_loop"] = round(
                max(0.0, detector_total - measured_detectors), 6
            )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "measurement_scope": "host_wall_clock",
            "stages_sec": stages,
            "detectors_sec": detectors,
            "detector_stages_sec": detector_stages,
            "reporting_sec": reporting,
            "end_to_end_sec": round(time.perf_counter() - self._started, 6),
        }
