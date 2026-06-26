from __future__ import annotations

# detector_id -> Chinese display label.
DETECTOR_ZH = {
    "401": "401_ negative",
    "401-1": "401-1 圓形 NG 檢測",
    "401-2": "401-2 白色比例 NG 檢測",
}


def detector_zh_name(detector_id: str) -> str:
    return DETECTOR_ZH.get(str(detector_id), "")
