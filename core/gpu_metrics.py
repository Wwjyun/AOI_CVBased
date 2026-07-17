from __future__ import annotations


class GpuPerformanceRecorder:
    def __init__(self) -> None:
        self.values = {
            "load_sec": 0.0, "call_count": 0, "estimated_round_trips": 0,
            "host_to_device_bytes": 0, "device_to_host_bytes": 0,
            "wall_sec": 0.0, "lock_wait_sec": 0.0, "functions": {},
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
