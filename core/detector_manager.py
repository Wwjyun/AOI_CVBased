from __future__ import annotations

from detectors.detector_999 import Detector999


class DetectorManager:
    def __init__(self):
        self._registry = {
            Detector999.detector_id: Detector999,
        }

    def create(self, detector_id: str, display_name: str | None = None, params: dict | None = None):
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        return detector_cls(display_name=display_name, params=params or {})

    def create_enabled(self, detector_configs: dict):
        detectors = []
        for detector_id, config in detector_configs.items():
            detectors.append(
                self.create(
                    detector_id=str(detector_id),
                    display_name=config.get("display_name"),
                    params=config.get("params", {}),
                )
            )
        return detectors
