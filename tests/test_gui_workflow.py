from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from core.recipe_manager import RecipeManager
from gui.main_window import MainWindow, _backend_status_from_result
from gui.permission_manager import ModePasswordPrompt, PermissionManager
from gui.workers import InspectionWorker
from gui.preferences import GuiPreferences
from gui.screens.designer_screen import DesignerScreen
from gui.screens.results_screen import ResultsScreen
from gui.table_models import RowTableModel, StatusFilterProxyModel, TableColumn, deterministic_sample
from gui.widgets.topbar import TopBar


class GuiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_topbar_reports_actual_backend_and_fallback_reason(self):
        topbar = TopBar()

        topbar.set_backend_status({"requested": False, "active": False})
        self.assertEqual(topbar.backend_badge.text(), "CPU")

        topbar.set_backend_status({"requested": True, "active": True, "device_name": "RTX 3090"})
        self.assertEqual(topbar.backend_badge.text(), "CUDA · RTX 3090")
        self.assertIn("RTX 3090", topbar.backend_badge.toolTip())

        topbar.set_backend_status({"requested": True, "active": False, "fallback_reason": "DLL missing"})
        self.assertEqual(topbar.backend_badge.text(), "CPU FALLBACK")
        self.assertIn("DLL missing", topbar.backend_badge.toolTip())

        status = _backend_status_from_result(
            {
                "execution": {
                    "gpu": {
                        "tiling": {"requested": False, "active": False, "device_name": "RTX 3090"},
                        "detectors": {"401": {"requested": True, "active": True, "fallback_reason": ""}},
                    }
                }
            }
        )
        self.assertTrue(status["active"])
        self.assertEqual(status["device_name"], "RTX 3090")

    def test_permission_manager_defaults_to_op_and_checks_each_privileged_password(self):
        permissions = PermissionManager()

        self.assertEqual(permissions.current_mode, "op")
        self.assertFalse(permissions.switch_mode("eng", "wrong"))
        self.assertEqual(permissions.current_mode, "op")
        self.assertTrue(permissions.switch_mode("eng", "1234"))
        self.assertEqual(permissions.current_mode, "eng")
        self.assertFalse(permissions.switch_mode("admin", "1234"))
        self.assertEqual(permissions.current_mode, "eng")
        self.assertTrue(permissions.switch_mode("admin", "5678"))
        self.assertTrue(permissions.switch_mode("op"))
        self.assertEqual(permissions.current_mode, "op")

    def test_main_window_requires_password_for_privileged_modes(self):
        prompt = Mock(spec=ModePasswordPrompt)
        prompt.request_password.side_effect = [
            ("", False),
            ("wrong", True),
            ("1234", True),
            ("5678", True),
        ]
        window = MainWindow(password_prompt=prompt)

        self.assertEqual(window.mode, "op")
        window._on_mode_changed("eng")
        self.assertEqual(window.mode, "op")
        window._on_mode_changed("eng")
        self.assertEqual(window.mode, "op")
        self.assertEqual(window.topbar.mode_switch.value(), "op")
        window._on_mode_changed("eng")
        self.assertEqual(window.mode, "eng")
        self.assertIn("designer", window._visible_screens_for_mode())
        window._on_mode_changed("admin")
        self.assertEqual(window.mode, "admin")
        window._on_mode_changed("op")
        self.assertEqual(window.mode, "op")
        self.assertEqual(prompt.request_password.call_count, 4)
        window._inspection_gpu_sessions.close()
        window.deleteLater()

    def test_single_inspection_displays_user_wait_and_preserves_pipeline_duration(self):
        window = MainWindow()
        result = {
            "duration_sec": 1.4,
            "final_result": "PASS",
            "summary": {"tile_count": 0, "ng_count": 0, "defect_count": 0},
            "tiles": [],
            "outputs": {},
            "execution": {"gpu": {}},
        }
        window._run_started_at = 10.0
        with patch("gui.main_window.time.perf_counter", return_value=13.0):
            window._on_inspection_finished(result)

        self.assertEqual(result["duration_sec"], 1.4)
        self.assertEqual(result["execution"]["performance"]["gui_user_wait_sec"], 3.0)
        window._inspection_gpu_sessions.close()
        window.deleteLater()

    def test_single_inspection_worker_injects_cached_gpu_session(self):
        session = Mock()
        cache = Mock()
        cache.session_for.return_value = session
        pipeline = Mock()
        pipeline.run.return_value = {"final_result": "PASS"}
        worker = InspectionWorker(
            Path("input.png"), Path("recipe.yaml"), Path("outputs"),
            gpu_session_cache=cache,
        )

        with patch("gui.workers.AOIPipeline", return_value=pipeline) as pipeline_type:
            worker.run()

        cache.session_for.assert_called_once_with(Path("recipe.yaml"))
        self.assertIs(pipeline_type.call_args.kwargs["gpu_session"], session)
        pipeline.run.assert_called_once_with(Path("input.png"))

    def test_results_keyboard_navigation_and_focus_signal(self):
        screen = ResultsScreen()
        image = QImage(160, 120, QImage.Format.Format_RGB888)
        image.fill(0)
        result = {
            "final_result": "NG",
            "summary": {"tile_count": 1, "ng_count": 1, "defect_count": 2},
            "tiles": [
                {
                    "tile": {"tile_id": "T1"},
                    "detectors": [
                        {
                            "detector_id": "scratch",
                            "score": 0.8,
                            "defects": [
                                {"type": "scratch", "bbox_global": [10, 10, 8, 8], "area": 64},
                                {"type": "scratch", "bbox_global": [80, 50, 10, 12], "area": 120},
                            ],
                        }
                    ],
                }
            ],
            "outputs": {},
        }
        selected = []
        focused = []
        screen.defect_selected.connect(selected.append)
        screen.view_requested.connect(focused.append)

        screen.set_result(result, image)
        screen._next_shortcut.activated.emit()
        screen._next_shortcut.activated.emit()
        screen._focus_shortcut.activated.emit()

        self.assertEqual(selected, [1, 2])
        self.assertEqual(focused, [2])

    def test_designer_tracks_dirty_and_invalid_states(self):
        screen = DesignerScreen()
        recipe = RecipeManager().load(Path("recipes/PRODUCT_A_AOI_01.yaml"))
        screen.set_recipe(recipe)
        self.assertFalse(screen.is_dirty())
        self.assertEqual(screen.editor_state_badge.text(), "已儲存")

        screen.recipe_name_edit.setText("changed-recipe")
        self.assertTrue(screen.is_dirty())
        self.assertEqual(screen.editor_state_badge.text(), "未儲存")

        screen._enabled = {key: False for key in screen._enabled}
        screen._save_recipe()
        self.assertIn("驗證失敗", screen.editor_state_badge.text())

    def test_preferences_ignore_stale_paths_and_round_trip_typed_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "input"
            existing.mkdir()
            settings = QSettings(str(root / "gui.ini"), QSettings.Format.IniFormat)
            preferences = GuiPreferences(settings)

            preferences.set_value("paths/image", str(existing))
            preferences.set_value("paths/recipe", str(root / "missing.yaml"))
            preferences.save_output_options({"save_csv": False, "save_json": True})
            preferences.set_value("ui/splitter", [100, 200])
            settings.sync()

            self.assertEqual(preferences.existing_path("paths/image"), existing)
            self.assertIsNone(preferences.existing_path("paths/recipe"))
            self.assertEqual(
                preferences.output_options({"save_csv": True, "save_json": False}),
                {"save_csv": False, "save_json": True},
            )
            self.assertEqual(preferences.splitter_sizes("ui/splitter", [1, 1]), [100, 200])
            preferences.set_value("output/options", "[]")
            self.assertEqual(preferences.output_options({"save_csv": True}), {"save_csv": True})

    def test_main_window_restores_and_saves_working_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = QSettings(str(Path(temp_dir) / "window.ini"), QSettings.Format.IniFormat)
            settings.setValue("ui/mode", "admin")
            settings.setValue("ui/last_screen", "results")
            settings.setValue("output/directory", "custom-output")
            settings.setValue("ui/monitor_splitter", [600, 400, 300])
            settings.sync()

            window = MainWindow(settings=settings)
            window.show()
            self.app.processEvents()
            self.assertEqual(window.mode, "op")
            self.assertEqual(window._current_screen, "monitor")
            self.assertEqual(window.output_dir, "custom-output")
            window._set_screen("monitor")
            self.app.processEvents()
            window.monitor_screen.add_item(
                {
                    "processed_at": "now",
                    "image_name": "sample.png",
                    "final_result": "PASS",
                    "defect_count": 0,
                    "ng_count": 0,
                    "duration_sec": 0.1,
                    "tiles": [],
                }
            )
            self.app.processEvents()
            splitter_sizes = window.monitor_screen.data_splitter.sizes()
            self.assertAlmostEqual(splitter_sizes[0] / splitter_sizes[1], 1.5, delta=0.05)
            self.assertAlmostEqual(splitter_sizes[1] / splitter_sizes[2], 4 / 3, delta=0.05)

            window._save_preferences()
            self.assertEqual(settings.value("ui/last_screen"), "monitor")
            self.assertTrue(settings.value("ui/geometry"))
            window.deleteLater()
            self.app.processEvents()

    def test_incremental_model_filter_and_deterministic_sampling(self):
        model = RowTableModel([TableColumn("結果", "final_result"), TableColumn("名稱", "name")])
        model.set_rows([{"final_result": "PASS", "name": "a"}, {"final_result": "NG", "name": "b"}])
        model.prepend({"final_result": "ERROR", "name": "c"}, limit=2)
        self.assertEqual([row["name"] for row in model.rows], ["c", "a"])

        proxy = StatusFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.set_status("error")
        self.assertEqual(proxy.rowCount(), 1)
        self.assertEqual(proxy.row_dict(0)["name"], "c")

        sampled = deterministic_sample(range(10_000), 1_000)
        self.assertEqual(len(sampled), 1_000)
        self.assertEqual((sampled[0], sampled[-1]), (0, 9_999))
        self.assertEqual(sampled, deterministic_sample(range(10_000), 1_000))


if __name__ == "__main__":
    unittest.main()
