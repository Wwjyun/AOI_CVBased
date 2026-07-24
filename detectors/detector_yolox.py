from __future__ import annotations

from copy import deepcopy

from core.ai_runtime import (
    AiModelSessionManager,
    decode_yolox_output,
    parse_target_class_ids,
    prepare_yolox_input,
    validate_yolox_parameters,
)
from core.parameter_schema import specs_from_defaults
from detectors.base_detector import BaseDetector


class DetectorYolox(BaseDetector):
    detector_id = "yolox"
    detector_name = "yolox_object_detector"
    display_name = "YOLOX object detector"
    requires_serial_inference = True
    default_params = {
        "model_id": "",
        "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.45,
        "target_class_ids": "",
        "max_detections": 300,
        "min_box_area_px": 0.0,
        "inference_backend": "auto",
        "precision": "fp32",
        "class_agnostic_nms": False,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            "model_id": {"label": "模型"},
            "confidence_threshold": {
                "minimum": 0.0,
                "maximum": 1.0,
                "label": "信心門檻",
            },
            "nms_iou_threshold": {
                "minimum": 0.0,
                "maximum": 1.0,
                "label": "NMS 重疊率 (IoU)",
            },
            "target_class_ids": {"label": "NG 類別 ID"},
            "max_detections": {
                "minimum": 1,
                "label": "最大偵測數",
            },
            "min_box_area_px": {
                "minimum": 0.0,
                "label": "最小框面積 (px²)",
            },
            "inference_backend": {
                "choices": (
                    "auto",
                    "onnxruntime_cpu",
                    "onnxruntime_cuda",
                    "tensorrt",
                ),
                "engineer_visible": False,
                "label": "推論後端",
            },
            "precision": {
                "choices": ("fp32", "fp16", "int8"),
                "engineer_visible": False,
                "label": "推論精度",
            },
            "class_agnostic_nms": {
                "engineer_visible": False,
                "label": "跨類別 NMS",
            },
        },
    )

    def __init__(
        self,
        display_name: str | None = None,
        params: dict | None = None,
        use_gpu: bool = False,
        gpu_runtime=None,
        ai_session_manager=None,
    ):
        super().__init__(
            display_name=display_name,
            params=params,
            use_gpu=use_gpu,
            gpu_runtime=gpu_runtime,
            ai_session_manager=ai_session_manager,
        )
        self.ai_session_manager = ai_session_manager or AiModelSessionManager()
        self._ai_execution: dict = {}

    @staticmethod
    def validate_parameters(params: dict, registry) -> None:
        merged = deepcopy(DetectorYolox.default_params)
        merged.update(params or {})
        validate_yolox_parameters(merged, registry)

    def detect(self, image) -> list[dict]:
        manifest = validate_yolox_parameters(
            self.params, self.ai_session_manager.registry
        )
        with self.measure_detection_stage("dl_preprocess"):
            tensor, transform = prepare_yolox_input(image, manifest)
        with self.measure_detection_stage("model_session"):
            session = self.ai_session_manager.session_for(
                manifest,
                backend=str(self.params.get("inference_backend", "auto")),
                precision=str(self.params.get("precision", "fp32")),
            )
        with self.measure_detection_stage("inference"):
            output = session.infer(tensor)
        with self.measure_detection_stage("postprocess"):
            defects = decode_yolox_output(
                output,
                manifest,
                transform,
                confidence_threshold=float(
                    self.params.get("confidence_threshold", 0.25)
                ),
                nms_iou_threshold=float(
                    self.params.get("nms_iou_threshold", 0.45)
                ),
                target_class_ids=parse_target_class_ids(
                    self.params.get("target_class_ids", ""),
                    len(manifest.class_names),
                ),
                max_detections=int(self.params.get("max_detections", 300)),
                min_box_area_px=float(self.params.get("min_box_area_px", 0.0)),
                class_agnostic_nms=bool(
                    self.params.get("class_agnostic_nms", False)
                ),
            )
        self._ai_execution = {
            "requested_backend": str(
                self.params.get("inference_backend", "auto")
            ),
            "actual_backend": session.backend,
            "device": session.device,
            "precision": session.precision,
            "model_id": manifest.model_id,
            "model_version": manifest.version,
            "model_sha256": manifest.sha256,
            "input_shape": list(tensor.shape),
            "output_shape": list(output.shape),
            "batch_size": int(tensor.shape[0]),
            "model_load_sec": round(session.load_sec, 6),
            "warmup_sec": round(session.warmup_sec, 6),
            "last_inference_sec": round(session.last_inference_sec, 6),
            "session_inference_count": session.inference_count,
            "fallback_reason": "",
        }
        return defects

    def run(self, image, device_roi=None, preprocess_cache=None) -> dict:
        self._ai_execution = {}
        result = super().run(
            image, device_roi=device_roi, preprocess_cache=preprocess_cache
        )
        result["execution"]["ai"] = deepcopy(self._ai_execution)
        if self._ai_execution:
            result["execution"]["backend"] = self._ai_execution["actual_backend"]
            result["execution"]["gpu_active"] = False
        return result
