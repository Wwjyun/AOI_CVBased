from __future__ import annotations

from copy import deepcopy

from detectors.detector_401 import Detector401
from detectors.detector_401_1 import Detector401_1
from detectors.detector_401_2 import Detector401_2
from detectors.detector_900 import Detector900
from detectors.detector_yolox import DetectorYolox


class DetectorManager:
    def __init__(self, ai_session_manager=None):
        self._registry = {
            Detector401.detector_id: Detector401,
            Detector401_1.detector_id: Detector401_1,
            Detector401_2.detector_id: Detector401_2,
            Detector900.detector_id: Detector900,
            DetectorYolox.detector_id: DetectorYolox,
        }
        self._ai_session_manager = ai_session_manager

    def create(
        self,
        detector_id: str,
        display_name: str | None = None,
        params: dict | None = None,
        use_gpu: bool = False,
        gpu_runtime=None,
    ):
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        return detector_cls(
            display_name=display_name,
            params=params or {},
            use_gpu=use_gpu,
            gpu_runtime=gpu_runtime,
            ai_session_manager=(
                self._ai_manager() if detector_cls is DetectorYolox else None
            ),
        )

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

    def definitions(self, include_runtime_metadata: bool = False) -> dict[str, dict]:
        definitions = {}
        for detector_id, detector_cls in self._registry.items():
            definition = {
                "display_name": detector_cls.display_name,
                "detector_name": detector_cls.detector_name,
                "default_params": deepcopy(detector_cls.default_params),
                "param_spec": {
                    key: spec.to_dict() for key, spec in detector_cls.PARAM_SPEC.items()
                },
            }
            if detector_cls is DetectorYolox and include_runtime_metadata:
                definition.update(self._yolox_definition_metadata())
            definitions[detector_id] = definition
        return definitions

    def parameter_specs(self, detector_id: str):
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        return detector_cls.PARAM_SPEC

    def validate_parameters(self, detector_id: str, params: dict) -> None:
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        validator = getattr(detector_cls, "validate_parameters", None)
        if callable(validator):
            validator(params, self._ai_manager().registry)

    def _ai_manager(self):
        if self._ai_session_manager is None:
            from core.ai_runtime import AiModelSessionManager

            self._ai_session_manager = AiModelSessionManager()
        return self._ai_session_manager

    def _yolox_definition_metadata(self) -> dict:
        try:
            registry = self._ai_manager().registry
            models = []
            for model_id in registry.model_ids():
                manifest = registry.get(model_id)
                models.append(
                    {
                        "model_id": manifest.model_id,
                        "label": (
                            f"{manifest.name} · v{manifest.version}"
                            f"{' · 測試用' if manifest.test_only else ''}"
                        ),
                        "version": manifest.version,
                        "input_size": [manifest.input_width, manifest.input_height],
                        "class_names": list(manifest.class_names),
                        "allowed_backends": list(manifest.allowed_backends),
                        "allowed_precisions": list(manifest.allowed_precisions),
                        "test_only": manifest.test_only,
                    }
                )
            return {"model_options": models, "model_registry_error": ""}
        except RuntimeError as exc:
            return {"model_options": [], "model_registry_error": str(exc)}
