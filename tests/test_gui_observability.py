from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from gui.image_viewer import ImageViewer
from gui.workers import ImagePreviewWorker


class GuiDisplayObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_worker_reports_qimage_and_conversion_timings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview.png"
            encoded, buffer = cv2.imencode(".png", np.zeros((12, 16, 3), dtype=np.uint8))
            self.assertTrue(encoded)
            image_path.write_bytes(buffer.tobytes())
            loaded = []
            worker = ImagePreviewWorker(image_path, gpu_config={"display": False})
            worker.loaded.connect(lambda path, image, status: loaded.append((path, image, status)))

            worker.run()

        self.assertEqual(len(loaded), 1)
        path, image, status = loaded[0]
        self.assertEqual(path, image_path)
        self.assertFalse(image.isNull())
        snapshot = status["display_performance"]["worker"]
        self.assertIn("image_load", snapshot["stages_sec"])
        self.assertIn("color_conversion", snapshot["stages_sec"])
        self.assertIn("qimage_copy", snapshot["stages_sec"])
        self.assertGreaterEqual(snapshot["end_to_end_sec"], 0.0)

    def test_viewer_reports_qpixmap_and_scene_timings_in_tooltip(self):
        image = QImage(16, 12, QImage.Format.Format_RGB888)
        image.fill(0)
        viewer = ImageViewer()

        snapshot = viewer.set_qimage(image, name="preview.png")
        status = {
            "requested": False,
            "active": False,
            "display_performance": {
                "worker": {"end_to_end_sec": 0.001},
                "viewer": snapshot,
                "user_wait_sec": 0.003,
            },
        }
        viewer.set_backend_status(status)

        self.assertIn("qpixmap_conversion", snapshot["stages_sec"])
        self.assertIn("scene_update", snapshot["stages_sec"])
        self.assertIn("fit_to_view", snapshot["stages_sec"])
        self.assertIn("QImage worker", viewer.backend_label.toolTip())
        self.assertIn("QPixmap/viewer", viewer.backend_label.toolTip())
        self.assertIn("User wait", viewer.backend_label.toolTip())


if __name__ == "__main__":
    unittest.main()
