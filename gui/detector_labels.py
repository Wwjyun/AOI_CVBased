from __future__ import annotations

# detector_id -> 中文名稱對照（design_handoff_aoi_gui/app/data.js DETECTOR_DEFS.zh）
DETECTOR_ZH = {
    "000": "二值化輪廓面積檢查",
    "001": "圓形閾值 NG 檢測",
    "102": "刮痕／細線檢測",
    "305": "亮度／均勻度檢測",
    "401-1": "401-1 圓形 NG 檢測",
    "777": "Pattern 計數檢測",
    "888": "紋理／模糊檢測",
    "999": "暗點／亮點 blob 檢測",
}


def detector_zh_name(detector_id: str) -> str:
    return DETECTOR_ZH.get(str(detector_id), "")
