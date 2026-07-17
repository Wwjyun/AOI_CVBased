from __future__ import annotations

from copy import deepcopy

from detectors.detector_401 import Detector401
from detectors.detector_401_1 import Detector401_1
from detectors.detector_401_2 import Detector401_2
from detectors.detector_900 import Detector900


class DetectorManager:
    def __init__(self):
        self._registry = {
            Detector401.detector_id: Detector401,
            Detector401_1.detector_id: Detector401_1,
            Detector401_2.detector_id: Detector401_2,
            Detector900.detector_id: Detector900,
        }

    def create(self, detector_id: str, display_name: str | None = None, params: dict | None = None, use_gpu: bool = False, gpu_runtime=None):
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        return detector_cls(display_name=display_name, params=params or {}, use_gpu=use_gpu, gpu_runtime=gpu_runtime)

    def create_enabled(self, detector_configs: dict, gpu_runtime=None):
        detectors = []
        for detector_id, config in detector_configs.items():
            detectors.append(
                self.create(
                    detector_id=str(detector_id),
                    display_name=config.get("display_name"),
                    params=config.get("params", {}),
                    use_gpu=bool(config.get("use_gpu", False)),
                    gpu_runtime=gpu_runtime,
                )
            )
        return detectors

    @staticmethod
    def run_batch(detectors, images, rois=None) -> dict[str, list[dict]]:
        return {
            detector.detector_id: detector.run_batch(images, rois=rois)
            for detector in detectors
        }

    def definitions(self) -> dict[str, dict]:
        return {
            detector_id: {
                "display_name": detector_cls.display_name,
                "detector_name": detector_cls.detector_name,
                "default_params": deepcopy(detector_cls.default_params),
                "param_spec": {
                    key: spec.to_dict() for key, spec in detector_cls.PARAM_SPEC.items()
                },
            }
            for detector_id, detector_cls in self._registry.items()
        }

    def parameter_specs(self, detector_id: str):
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        return detector_cls.PARAM_SPEC
