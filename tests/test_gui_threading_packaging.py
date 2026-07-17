from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GuiThreadingPackagingContractTests(unittest.TestCase):
    def test_cuda_pipeline_workers_are_moved_to_qthreads_before_start(self):
        source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        for worker in (
            "_preview_worker",
            "_inspection_worker",
            "_batch_worker",
            "_monitor_worker",
            "_tile_preview_worker",
        ):
            move = f"self.{worker}.moveToThread"
            started = f"started.connect(self.{worker}.run)"
            self.assertIn(move, source)
            self.assertIn(started, source)
            self.assertLess(source.index(move), source.index(started))
        self.assertNotIn(".wait(", source)

    def test_worker_error_progress_and_monitor_cancel_use_signals_or_callback(self):
        workers = (ROOT / "gui" / "workers.py").read_text(encoding="utf-8")

        self.assertIn("failed = Signal(str)", workers)
        self.assertIn("progress = Signal(int, str)", workers)
        self.assertIn("stop_callback=lambda: self._stop_requested", workers)
        self.assertIn("self.failed.emit(str(exc))", workers)

    def test_pyinstaller_cuda_dll_is_optional_and_keeps_gpu_relative_path(self):
        spec = (ROOT / "VisionFlow AOI.spec").read_text(encoding="utf-8")
        build = (ROOT / "build_exe.ps1").read_text(encoding="utf-8")

        self.assertIn("if cuda_dll.exists() else []", spec)
        self.assertIn("(str(cuda_dll), 'gpu')", spec)
        self.assertIn("if (Test-Path $cudaDll)", build)
        self.assertIn('"--add-binary", "$cudaDll;gpu"', build)
        self.assertIn("CPU-compatible package", build)


if __name__ == "__main__":
    unittest.main()
