from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.batch_processor import BatchImageResult, BatchInspectionProcessor
from core.gpu_runtime import GpuRuntimeError
from core.gpu_session import GpuExecutionSession
from core.monitor_processor import FolderMonitorProcessor
from core.pipeline import AOIPipeline


ROOT = Path(__file__).resolve().parents[1]


class _CloseTrackingRuntime:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class GpuExecutionSessionTests(unittest.TestCase):
    def test_session_rejects_incompatible_config_and_closes_once(self):
        runtime = _CloseTrackingRuntime()
        config = {"dll_path": "gpu/visionflow_cuda.dll", "fallback_to_cpu": True}
        session = GpuExecutionSession(runtime, requested=True, config=config)

        self.assertIs(session.runtime_for(config, requested=True), runtime)
        with self.assertRaisesRegex(GpuRuntimeError, "incompatible"):
            session.runtime_for({**config, "fallback_to_cpu": False}, requested=True)

        session.close()
        session.close()
        self.assertEqual(runtime.close_calls, 1)
        with self.assertRaisesRegex(GpuRuntimeError, "already closed"):
            session.runtime_for(config, requested=True)

    def test_two_pipeline_runs_share_one_injected_runtime_until_session_close(self):
        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        image_path = ROOT / "tmp_validation_input.png"
        output_overrides = {
            "save_overlay": False,
            "save_ng_tiles": False,
            "save_csv": False,
            "save_matrix_csv": False,
            "save_json": False,
        }
        with tempfile.TemporaryDirectory(prefix="visionflow_gpu_session_") as temporary:
            session = GpuExecutionSession.from_recipe_path(recipe_path)
            runtime = session.runtime
            try:
                first = AOIPipeline(
                    recipe_path,
                    Path(temporary) / "first",
                    output_overrides=output_overrides,
                    gpu_session=session,
                ).run(image_path)
                second = AOIPipeline(
                    recipe_path,
                    Path(temporary) / "second",
                    output_overrides=output_overrides,
                    gpu_session=session,
                ).run(image_path)

                self.assertIs(session.runtime, runtime)
                self.assertFalse(session._closed)
                self.assertEqual(first["execution"]["gpu"]["metrics"]["call_count"], 0)
                self.assertEqual(second["execution"]["gpu"]["metrics"]["call_count"], 0)
            finally:
                session.close()
            self.assertTrue(session._closed)

    def test_batch_workers_receive_one_shared_session(self):
        fake_session = Mock()
        fake_session.__enter__ = Mock(return_value=fake_session)
        fake_session.__exit__ = Mock(return_value=None)
        captured_sessions = []

        def process(image_path, _output_dir, gpu_session):
            captured_sessions.append(gpu_session)
            return BatchImageResult(
                image_path=image_path,
                final_result="PASS",
                defect_count=0,
                ng_count=0,
                tile_count=1,
                duration_sec=0.01,
                outputs={},
                detail={},
            )

        with tempfile.TemporaryDirectory(prefix="visionflow_batch_session_") as temporary:
            processor = BatchInspectionProcessor(
                Path(temporary),
                ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml",
                Path(temporary) / "output",
                max_workers=2,
            )
            processor.discover_images = Mock(
                return_value=[Path(temporary) / "one.png", Path(temporary) / "two.png"]
            )
            processor._process_image = Mock(side_effect=process)
            with patch(
                "core.batch_processor.GpuExecutionSession.from_recipe_path",
                return_value=fake_session,
            ):
                result = processor.run()

        self.assertEqual(result["summary"]["total"], 2)
        self.assertEqual(captured_sessions, [fake_session, fake_session])

    def test_monitor_pipeline_receives_existing_session(self):
        fake_session = Mock()
        pipeline = Mock()
        pipeline.run.return_value = {
            "final_result": "PASS",
            "summary": {"defect_count": 0, "ng_count": 0, "tile_count": 1},
            "duration_sec": 0.01,
            "outputs": {},
            "tiles": [],
        }
        with tempfile.TemporaryDirectory(prefix="visionflow_monitor_session_") as temporary:
            processor = FolderMonitorProcessor(
                Path(temporary),
                ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml",
                Path(temporary) / "output",
            )
            with patch("core.monitor_processor.AOIPipeline", return_value=pipeline) as pipeline_type:
                result = processor._process_image(
                    Path(temporary) / "image.png",
                    Path(temporary) / "monitor_output",
                    fake_session,
                )

        self.assertEqual(result.final_result, "PASS")
        self.assertIs(pipeline_type.call_args.kwargs["gpu_session"], fake_session)


if __name__ == "__main__":
    unittest.main()
