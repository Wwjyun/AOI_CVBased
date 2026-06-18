from __future__ import annotations

# detector_id -> 中文名稱對照（design_handoff_aoi_gui/app/data.js DETECTOR_DEFS.zh）
DETECTOR_ZH = {
    "401-1": "401-1 圓形 NG 檢測",
}


def detector_zh_name(detector_id: str) -> str:
    return DETECTOR_ZH.get(str(detector_id), "")
