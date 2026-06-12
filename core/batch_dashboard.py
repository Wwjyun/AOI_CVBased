from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchDashboardModel:
    total: int
    pass_count: int
    ng_count: int
    error_count: int
    defect_count: int
    duration_sec: float
    output_dir: str
    pass_rate: float
    ng_rate: float
    avg_defects: float
    result_distribution: list[tuple[str, int]]
    top_defect_images: list[dict]
    rows: list[dict]


class BatchDashboardBuilder:
    """Build chart-ready dashboard data from a batch inspection result."""

    def __init__(self, batch_result: dict | None):
        self.batch_result = batch_result or {}

    def build(self) -> BatchDashboardModel:
        summary = self.batch_result.get("summary", {})
        rows = list(self.batch_result.get("items", []))
        total = int(summary.get("total", len(rows)) or 0)
        pass_count = int(summary.get("pass", self._count_result(rows, "PASS")) or 0)
        ng_count = int(summary.get("ng", self._count_result(rows, "NG")) or 0)
        error_count = int(summary.get("error", self._count_result(rows, "ERROR")) or 0)
        defect_count = int(summary.get("defects", sum(int(row.get("defect_count", 0) or 0) for row in rows)) or 0)
        duration_sec = float(self.batch_result.get("duration_sec", 0) or 0)

        return BatchDashboardModel(
            total=total,
            pass_count=pass_count,
            ng_count=ng_count,
            error_count=error_count,
            defect_count=defect_count,
            duration_sec=duration_sec,
            output_dir=str(self.batch_result.get("output_dir", "")),
            pass_rate=self._rate(pass_count, total),
            ng_rate=self._rate(ng_count, total),
            avg_defects=round(defect_count / total, 2) if total else 0.0,
            result_distribution=[
                ("PASS", pass_count),
                ("NG", ng_count),
                ("ERROR", error_count),
            ],
            top_defect_images=self._top_defect_images(rows),
            rows=rows,
        )

    @staticmethod
    def _count_result(rows: list[dict], result: str) -> int:
        return sum(1 for row in rows if row.get("final_result") == result)

    @staticmethod
    def _rate(value: int, total: int) -> float:
        if not total:
            return 0.0
        return round(value / total * 100.0, 1)

    @staticmethod
    def _top_defect_images(rows: list[dict], limit: int = 8) -> list[dict]:
        ranked = sorted(
            rows,
            key=lambda row: (int(row.get("defect_count", 0) or 0), int(row.get("ng_count", 0) or 0)),
            reverse=True,
        )
        return ranked[:limit]
