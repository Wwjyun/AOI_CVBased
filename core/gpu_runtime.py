from __future__ import annotations

import ctypes
import sys
import threading
import time
from pathlib import Path

import numpy as np


class GpuRuntimeError(RuntimeError):
    pass


class GpuRuntime:
    """Thread-safe ctypes bridge for the optional VisionFlow CUDA DLL."""

    DEFAULT_DLL = "gpu/visionflow_cuda.dll"
    ABI_VERSION = 1

    def __init__(self, dll_path: str | Path = DEFAULT_DLL, fallback_to_cpu: bool = True, enabled: bool = True):
        load_started = time.perf_counter()
        self.requested_path = str(dll_path or self.DEFAULT_DLL)
        self.fallback_to_cpu = bool(fallback_to_cpu)
        self.dll_path = self._resolve_path(self.requested_path)
        self._lock = threading.RLock()
        self._dll = None
        self._context = None
        self.device_count = 0
        self.device_name = ""
        self.compute_capability = ""
        self.unavailable_reason = ""
        self.last_error = ""
        self.fused_unavailable_reason = ""
        self._performance = {
            "load_sec": 0.0,
            "call_count": 0,
            "estimated_round_trips": 0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "wall_sec": 0.0,
            "lock_wait_sec": 0.0,
            "functions": {},
        }
        if enabled:
            self._load()
            self._performance["load_sec"] = time.perf_counter() - load_started

    @property
    def available(self) -> bool:
        return self._dll is not None and self.device_count > 0

    @property
    def backend(self) -> str:
        return "cuda_dll" if self.available else "cpu"

    @property
    def supports_fused_401_2(self) -> bool:
        return bool(
            self.available
            and self._context is not None
            and getattr(self._dll, "vf_preprocess_401_2_u8", None) is not None
        )

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
            "capabilities": {"persistent_context": self._context is not None, "fused_401_2": self.supports_fused_401_2},
            "fallback_reason": (self.unavailable_reason if not self.available else self.last_error) if requested else "",
        }

    def performance_stats(self) -> dict:
        """Return host-observed CUDA call metrics; DLL-internal phases are not separable in ABI v1."""
        with self._lock:
            context_stats = self._context_stats_unlocked()
            functions = {
                name: {
                    "calls": int(values["calls"]),
                    "host_to_device_bytes": int(values["host_to_device_bytes"]),
                    "device_to_host_bytes": int(values["device_to_host_bytes"]),
                    "wall_sec": round(float(values["wall_sec"]), 6),
                    "lock_wait_sec": round(float(values["lock_wait_sec"]), 6),
                }
                for name, values in sorted(self._performance["functions"].items())
            }
            return {
                "measurement_scope": "synchronous_host_wrapper_estimate",
                "note": "Host timing combines H2D, kernels, synchronize and D2H; stateless ABI calls also allocate/free.",
                "load_sec": round(float(self._performance["load_sec"]), 6),
                "call_count": int(self._performance["call_count"]),
                "estimated_round_trips": int(self._performance["estimated_round_trips"]),
                "host_to_device_bytes": int(self._performance["host_to_device_bytes"]),
                "device_to_host_bytes": int(self._performance["device_to_host_bytes"]),
                "wall_sec": round(float(self._performance["wall_sec"]), 6),
                "lock_wait_sec": round(float(self._performance["lock_wait_sec"]), 6),
                "persistent_context": context_stats,
                "functions": functions,
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

    def preprocess_401_2(
        self,
        image: np.ndarray,
        gaussian_kernel_size: int,
        adaptive_block_size: int,
        adaptive_c: float,
        max_value: int,
        invert: bool = True,
    ) -> np.ndarray:
        if not self.supports_fused_401_2:
            raise GpuRuntimeError(self.fused_unavailable_reason or "CUDA DLL does not support fused 401-2 preprocessing")
        source = self._u8_image(image, channels=(1, 3))
        output = np.empty(source.shape[:2], dtype=np.uint8)
        channels = 1 if source.ndim == 2 else source.shape[2]
        function_name = "vf_preprocess_401_2_u8"
        function = getattr(self._dll, function_name)
        arguments = (
            self._context,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(source.shape[1]),
            int(source.shape[0]),
            int(source.strides[0]),
            int(channels),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(output.strides[0]),
            int(gaussian_kernel_size),
            int(adaptive_block_size),
            ctypes.c_float(float(adaptive_c)),
            int(max_value),
            int(bool(invert)),
        )
        queued = time.perf_counter()
        with self._lock:
            lock_acquired = time.perf_counter()
            result = int(function(*arguments))
            completed = time.perf_counter()
            self._record_performance(
                function_name,
                int(source.nbytes),
                int(output.nbytes),
                completed - lock_acquired,
                lock_acquired - queued,
            )
        if result != 0:
            raise GpuRuntimeError(
                f"{function_name} failed with CUDA DLL error {result}: {self._error_message(result)}"
            )
        return output

    def close(self) -> None:
        with self._lock:
            context = self._context
            self._context = None
            if context is None or self._dll is None:
                return
            destroy = getattr(self._dll, "vf_context_destroy", None)
            if destroy is None:
                return
            result = int(destroy(context))
            if result != 0:
                self.last_error = f"vf_context_destroy failed with CUDA DLL error {result}: {self._error_message(result)}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except (AttributeError, OSError, TypeError, ValueError):
            pass

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
            self._load_optional_context()
        except (OSError, AttributeError) as exc:
            self.unavailable_reason = f"CUDA DLL load failed: {exc}"

    def _load_optional_context(self) -> None:
        create = getattr(self._dll, "vf_context_create", None)
        destroy = getattr(self._dll, "vf_context_destroy", None)
        fused = getattr(self._dll, "vf_preprocess_401_2_u8", None)
        if create is None or destroy is None or fused is None:
            self.fused_unavailable_reason = "CUDA DLL uses legacy stateless ABI without fused 401-2 exports"
            return
        create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = ctypes.c_int
        stats = getattr(self._dll, "vf_context_stats", None)
        if stats is not None:
            stats.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            stats.restype = ctypes.c_int
        context = ctypes.c_void_p()
        result = int(create(ctypes.byref(context)))
        if result != 0 or not context.value:
            self.fused_unavailable_reason = (
                f"CUDA persistent context creation failed with error {result}: {self._error_message(result)}"
            )
            return
        self._context = context

    def _context_stats_unlocked(self) -> dict:
        if self._context is None or self._dll is None:
            return {"active": False, "reserved_bytes": 0, "allocation_count": 0}
        stats = getattr(self._dll, "vf_context_stats", None)
        if stats is None:
            return {"active": True, "reserved_bytes": None, "allocation_count": None}
        reserved_bytes = ctypes.c_uint64()
        allocation_count = ctypes.c_uint64()
        result = int(stats(self._context, ctypes.byref(reserved_bytes), ctypes.byref(allocation_count)))
        if result != 0:
            return {"active": True, "reserved_bytes": None, "allocation_count": None, "error_code": result}
        return {
            "active": True,
            "reserved_bytes": int(reserved_bytes.value),
            "allocation_count": int(allocation_count.value),
        }

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
        queued = time.perf_counter()
        with self._lock:
            lock_acquired = time.perf_counter()
            result = int(function(*common, *extra))
            completed = time.perf_counter()
            self._record_performance(
                function_name,
                int(source.nbytes),
                int(output.nbytes),
                completed - lock_acquired,
                lock_acquired - queued,
            )
        if result != 0:
            raise GpuRuntimeError(
                f"{function_name} failed with CUDA DLL error {result}: {self._error_message(result)}"
            )

    def _record_performance(
        self,
        function_name: str,
        host_to_device_bytes: int,
        device_to_host_bytes: int,
        wall_sec: float,
        lock_wait_sec: float,
    ) -> None:
        functions = self._performance["functions"]
        values = functions.setdefault(
            function_name,
            {
                "calls": 0,
                "host_to_device_bytes": 0,
                "device_to_host_bytes": 0,
                "wall_sec": 0.0,
                "lock_wait_sec": 0.0,
            },
        )
        values["calls"] += 1
        values["host_to_device_bytes"] += host_to_device_bytes
        values["device_to_host_bytes"] += device_to_host_bytes
        values["wall_sec"] += wall_sec
        values["lock_wait_sec"] += lock_wait_sec
        self._performance["call_count"] += 1
        self._performance["estimated_round_trips"] += 1
        self._performance["host_to_device_bytes"] += host_to_device_bytes
        self._performance["device_to_host_bytes"] += device_to_host_bytes
        self._performance["wall_sec"] += wall_sec
        self._performance["lock_wait_sec"] += lock_wait_sec

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
