from __future__ import annotations

from copy import deepcopy


class BaseDetector:
    detector_id = ""
    detector_name = ""
    display_name = ""
    default_params: dict = {}

    def __init__(self, display_name: str | None = None, params: dict | None = None):
        self.display_name = display_name or self.display_name or self.detector_name
        self.params = deepcopy(self.default_params)
        self.params.update(params or {})

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        raise NotImplementedError

    def run(self, image) -> dict:
        processed = self.preprocess(image)
        defects = self.detect(processed)
        max_confidence = max((defect.get("confidence", 0.0) for defect in defects), default=0.0)
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "display_name": self.display_name,
            "pass": len(defects) == 0,
            "score": float(max_confidence),
            "defects": defects,
        }
