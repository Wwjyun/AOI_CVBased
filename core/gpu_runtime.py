from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path

import numpy as np


class GpuRuntimeError(RuntimeError):
    pass


class GpuRuntime:
    """Thread-safe ctypes bridge for the optional VisionFlow CUDA DLL."""

    DEFAULT_DLL = "gpu/visionflow_cuda.dll"
    ABI_VERSION = 1

    def __init__(self, dll_path: str | Path = DEFAULT_DLL, fallback_to_cpu: bool = True, enabled: bool = True):
        self.requested_path = str(dll_path or self.DEFAULT_DLL)
        self.fallback_to_cpu = bool(fallback_to_cpu)
        self.dll_path = self._resolve_path(self.requested_path)
        self._lock = threading.RLock()
        self._dll = None
        self.device_count = 0
        self.device_name = ""
        self.compute_capability = ""
        self.unavailable_reason = ""
        self.last_error = ""
        if enabled:
            self._load()

    @property
    def available(self) -> bool:
        return self._dll is not None and self.device_count > 0

    @property
    def backend(self) -> str:
        return "cuda_dll" if self.available else "cpu"

    def status(self, requested: bool = False) -> dict:
        active = bool(requested and self.available and not self.last_error)
        return {
            "requested": bool(requested),
            "active": active,
            "backend": "cuda_dll" if active else "cpu",
            "dll_path": str(self.dll_path),
            "device_count": self.device_count,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "fallback_reason": (self.unavailable_reason if not self.available else self.last_error) if requested else "",
        }

    def crop(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        source = self._u8_image(image, channels=(1, 3))
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > source.shape[1] or y + height > source.shape[0]:
            raise GpuRuntimeError(f"Invalid CUDA crop: x={x}, y={y}, width={width}, height={height}, shape={source.shape}")
        output = np.empty((height, width) if source.ndim == 2 else (height, width, source.shape[2]), dtype=np.uint8)
        self._call_image(
            "vf_crop_u8",
            source,
            output,
            int(x),
            int(y),
            int(width),
            int(height),
        )
        return output

    def bgr_to_gray(self, image: np.ndarray) -> np.ndarray:
        source = self._u8_image(image, channels=(3,))
        output = np.empty(source.shape[:2], dtype=np.uint8)
        self._call_image("vf_bgr_to_gray_u8", source, output)
        return output

    def bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        source = self._u8_image(image, channels=(3,))
        output = np.empty_like(source)
        self._call_image("vf_bgr_to_rgb_u8", source, output)
        return output

    def resize_gray(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        source = self._u8_image(image, channels=(1,))
        if width <= 0 or height <= 0:
            raise GpuRuntimeError(f"Invalid CUDA resize target: {width}x{height}")
        output = np.empty((int(height), int(width)), dtype=np.uint8)
        self._call_image("vf_resize_gray_u8", source, output, int(width), int(height))
        return output

    def gaussian_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        source = self._u8_image(image, channels=(1, 3))
        output = np.empty_like(source)
        self._call_image("vf_gaussian_blur_u8", source, output, int(kernel_size))
        return output

    def threshold(self, image: np.ndarray, threshold: int, max_value: int, invert: bool) -> np.ndarray:
        source = self._u8_image(image, channels=(1,))
        output = np.empty_like(source)
        self._call_image("vf_threshold_u8", source, output, int(threshold), int(max_value), int(bool(invert)))
        return output

    def adaptive_threshold(self, image: np.ndarray, block_size: int, c: float, max_value: int, invert: bool) -> np.ndarray:
        source = self._u8_image(image, channels=(1,))
        output = np.empty_like(source)
        self._call_image(
            "vf_adaptive_mean_u8",
            source,
            output,
            int(block_size),
            ctypes.c_float(float(c)),
            int(max_value),
            int(bool(invert)),
        )
        return output

    def morphology(self, image: np.ndarray, operation: str, kernel_size: int, iterations: int) -> np.ndarray:
        operations = {"open": 0, "close": 1, "dilate": 2, "erode": 3}
        if operation not in operations:
            raise GpuRuntimeError(f"Unsupported CUDA morphology operation: {operation}")
        source = self._u8_image(image, channels=(1, 3))
        output = np.empty_like(source)
        self._call_image(
            "vf_morphology_rect_u8",
            source,
            output,
            operations[operation],
            int(kernel_size),
            int(iterations),
        )
        return output

    def _load(self) -> None:
        if not self.dll_path.exists():
            self.unavailable_reason = f"CUDA DLL not found: {self.dll_path}"
            return
        try:
            dll = ctypes.CDLL(str(self.dll_path))
            dll.vf_gpu_abi_version.restype = ctypes.c_int
            abi_version = int(dll.vf_gpu_abi_version())
            if abi_version != self.ABI_VERSION:
                self.unavailable_reason = (
                    f"CUDA DLL ABI mismatch: expected {self.ABI_VERSION}, got {abi_version}"
                )
                return
            dll.vf_gpu_device_count.restype = ctypes.c_int
            dll.vf_gpu_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
            dll.vf_gpu_device_name.restype = ctypes.c_int
            count = int(dll.vf_gpu_device_count())
            if count <= 0:
                self.unavailable_reason = "CUDA DLL loaded but no CUDA device is available"
                return
            buffer = ctypes.create_string_buffer(256)
            if int(dll.vf_gpu_device_name(buffer, len(buffer))) != 0:
                self.unavailable_reason = "CUDA DLL could not query the device name"
                return
            self._dll = dll
            self.device_count = count
            self.device_name = buffer.value.decode("utf-8", errors="replace")
            capability = getattr(dll, "vf_gpu_compute_capability", None)
            if capability is not None:
                encoded = int(capability())
                self.compute_capability = f"{encoded // 10}.{encoded % 10}" if encoded > 0 else ""
        except (OSError, AttributeError) as exc:
            self.unavailable_reason = f"CUDA DLL load failed: {exc}"

    def _call_image(self, function_name: str, source: np.ndarray, output: np.ndarray, *extra) -> None:
        if not self.available:
            raise GpuRuntimeError(self.unavailable_reason or "CUDA runtime is unavailable")
        function = getattr(self._dll, function_name, None)
        if function is None:
            raise GpuRuntimeError(f"CUDA DLL is missing export: {function_name}")
        src_channels = 1 if source.ndim == 2 else source.shape[2]
        dst_channels = 1 if output.ndim == 2 else output.shape[2]
        common = (
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(source.shape[1]),
            int(source.shape[0]),
            int(source.strides[0]),
            int(src_channels),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(output.strides[0]),
            int(dst_channels),
        )
        with self._lock:
            result = int(function(*common, *extra))
        if result != 0:
            raise GpuRuntimeError(
                f"{function_name} failed with CUDA DLL error {result}: {self._error_message(result)}"
            )

    def _error_message(self, error_code: int) -> str:
        function = getattr(self._dll, "vf_gpu_error_message", None)
        if function is None:
            return "unknown error"
        buffer = ctypes.create_string_buffer(512)
        try:
            function(int(error_code), buffer, len(buffer))
            return buffer.value.decode("utf-8", errors="replace") or "unknown error"
        except (OSError, ValueError):
            return "unknown error"

    def fallback_or_raise(self, exc: Exception) -> None:
        self.last_error = str(exc)
        if not self.fallback_to_cpu:
            raise exc

    @staticmethod
    def _u8_image(image: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(image)
        count = 1 if array.ndim == 2 else array.shape[2] if array.ndim == 3 else 0
        if array.dtype != np.uint8 or count not in channels:
            raise GpuRuntimeError(f"CUDA DLL expects uint8 image with channels in {channels}; got {array.dtype}, {array.shape}")
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise GpuRuntimeError(f"CUDA DLL does not accept empty images: {array.shape}")
        return np.ascontiguousarray(array)

    @staticmethod
    def _resolve_path(path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        bases = [Path.cwd()]
        if getattr(sys, "frozen", False):
            bases.insert(0, Path(sys.executable).resolve().parent)
            bundle = getattr(sys, "_MEIPASS", None)
            if bundle:
                bases.insert(0, Path(bundle))
        for base in bases:
            resolved = base / candidate
            if resolved.exists():
                return resolved.resolve()
        return (bases[0] / candidate).resolve()
