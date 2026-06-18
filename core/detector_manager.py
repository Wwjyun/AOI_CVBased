from __future__ import annotations

from copy import deepcopy

from detectors.detector_401_1 import Detector401_1


class DetectorManager:
    def __init__(self):
        self._registry = {
            Detector401_1.detector_id: Detector401_1,
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

    def definitions(self) -> dict[str, dict]:
        return {
            detector_id: {
                "display_name": detector_cls.display_name,
                "detector_name": detector_cls.detector_name,
                "default_params": deepcopy(detector_cls.default_params),
            }
            for detector_id, detector_cls in self._registry.items()
        }
