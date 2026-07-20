from __future__ import annotations


class GpuPerformanceRecorder:
    def __init__(self) -> None:
        self.values = {
            "load_sec": 0.0, "call_count": 0, "estimated_round_trips": 0,
            "host_to_device_bytes": 0, "device_to_host_bytes": 0,
            "wall_sec": 0.0, "lock_wait_sec": 0.0, "functions": {},
            "native_cumulative_ms": {}, "kernel_launch_count": 0,
            "peak_vram_bytes": 0,
        }

    def record(self, function_name: str, h2d: int, d2h: int, wall: float, wait: float) -> None:
        function = self.values["functions"].setdefault(function_name, {
            "calls": 0, "host_to_device_bytes": 0, "device_to_host_bytes": 0,
            "wall_sec": 0.0, "lock_wait_sec": 0.0,
        })
        function["calls"] += 1
        function["host_to_device_bytes"] += h2d
        function["device_to_host_bytes"] += d2h
        function["wall_sec"] += wall
        function["lock_wait_sec"] += wait
        self.values["call_count"] += 1
        self.values["estimated_round_trips"] += 1
        self.values["host_to_device_bytes"] += h2d
        self.values["device_to_host_bytes"] += d2h
        self.values["wall_sec"] += wall
        self.values["lock_wait_sec"] += wait

    def record_native(
        self,
        timings: dict | None,
        *,
        kernel_launch_count: int = 0,
        reserved_bytes: int = 0,
    ) -> None:
        if timings:
            cumulative = self.values["native_cumulative_ms"]
            for name, value in timings.items():
                if not isinstance(value, (int, float)):
                    continue
                if name == "context_create_ms":
                    cumulative[name] = max(float(cumulative.get(name, 0.0)), float(value))
                else:
                    cumulative[name] = float(cumulative.get(name, 0.0)) + float(value)
        self.values["kernel_launch_count"] += max(0, int(kernel_launch_count))
        self.values["peak_vram_bytes"] = max(
            int(self.values["peak_vram_bytes"]), max(0, int(reserved_bytes))
        )

    def snapshot(self) -> dict:
        values = self.values
        return {
            "load_sec": round(float(values["load_sec"]), 6),
            "call_count": int(values["call_count"]),
            "estimated_round_trips": int(values["estimated_round_trips"]),
            "host_to_device_bytes": int(values["host_to_device_bytes"]),
            "device_to_host_bytes": int(values["device_to_host_bytes"]),
            "wall_sec": round(float(values["wall_sec"]), 6),
            "lock_wait_sec": round(float(values["lock_wait_sec"]), 6),
            "native_cumulative_ms": {
                name: round(float(value), 6)
                for name, value in sorted(values["native_cumulative_ms"].items())
            },
            "kernel_launch_count": int(values["kernel_launch_count"]),
            "peak_vram_bytes": int(values["peak_vram_bytes"]),
            "functions": {
                name: {
                    "calls": int(item["calls"]),
                    "host_to_device_bytes": int(item["host_to_device_bytes"]),
                    "device_to_host_bytes": int(item["device_to_host_bytes"]),
                    "wall_sec": round(float(item["wall_sec"]), 6),
                    "lock_wait_sec": round(float(item["lock_wait_sec"]), 6),
                }
                for name, item in sorted(values["functions"].items())
            },
        }
