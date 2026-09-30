"""Tests: pipeline integration (with injected fake detector) and invalid-input handling."""

from __future__ import annotations


import pytest

from src.behavior.engine import BehaviorEngine, PerBehaviorConfig
from src.detection.detector import BoundingBox, DetectionResult
from src.pipeline import VideoPipeline
from src.preprocessing.video import VideoCaptureError
from src.tracking.tracker import Tracker
from src.utils.config import load_config
from tests.conftest import make_pose


# ---------------------------------------------------------------------------
# Fake detector: no ML model downloads required for unit tests.
# ---------------------------------------------------------------------------


class FakeDetector:
    """Deterministic stand-in for MultiDetector."""

    def __init__(self, persons=None, objects=None):
        self.persons = persons if persons is not None else [make_pose()]
        self.objects = objects if objects is not None else []
        self.calls = 0

    def detect(self, frame) -> DetectionResult:
        self.calls += 1
        return DetectionResult(
            frame_number=self.calls,
            timestamp=0.0,
            persons=self.persons,
            objects=self.objects,
        )

    def warmup(self, *args, **kwargs):
        return None


def make_pipeline(detector=None):
    config = load_config()
    # Fast, permissive behavior settings for testing.
    for section in (config.behavior.drinking, config.behavior.sleeping, config.behavior.phone_usage):
        section.min_consecutive_frames = 1
    config.behavior.drinking.min_duration_seconds = 0.1
    config.behavior.sleeping.min_duration_seconds = 0.1
    config.behavior.phone_usage.min_duration_seconds = 0.1
    config.reporting.min_event_duration_seconds = 0.0
    engine = BehaviorEngine(
        configs={
            "drinking": PerBehaviorConfig(enabled=True, min_duration_seconds=0.1,
                                          confidence_threshold=0.6, cooldown_seconds=5.0,
                                          min_consecutive_frames=1),
            "sleeping": PerBehaviorConfig(enabled=True, min_duration_seconds=0.1,
                                          confidence_threshold=0.65, cooldown_seconds=5.0,
                                          min_consecutive_frames=1),
            "phone_usage": PerBehaviorConfig(enabled=True, min_duration_seconds=0.1,
                                             confidence_threshold=0.6, cooldown_seconds=5.0,
                                             min_consecutive_frames=1),
        },
        fps=10.0,
        min_event_duration_seconds=0.0,
    )
    return VideoPipeline(
        config=config,
        detector=detector or FakeDetector(),
        tracker=Tracker(min_hits=1),
        engine=engine,
    )


class TestPipelineInvalidInput:
    def test_missing_video_raises(self, tmp_path):
        pipeline = make_pipeline()
        with pytest.raises(VideoCaptureError, match="not found"):
            pipeline.process_video(str(tmp_path / "ghost.mp4"))

    def test_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "clip.gif"
        bad.write_bytes(b"GIF89a not supported")
        pipeline = make_pipeline()
        with pytest.raises(VideoCaptureError, match="Unsupported"):
            pipeline.process_video(str(bad))

    def test_corrupt_video_raises(self, tmp_path):
        bad = tmp_path / "corrupt.mp4"
        bad.write_bytes(b"\x00" * 5000)
        pipeline = make_pipeline()
        with pytest.raises(VideoCaptureError):
            pipeline.process_video(str(bad))

    def test_empty_file_raises(self, tmp_path):
        bad = tmp_path / "empty.mp4"
        bad.write_bytes(b"")
        pipeline = make_pipeline()
        with pytest.raises(VideoCaptureError, match="empty"):
            pipeline.process_video(str(bad))


class TestPipelineProcessing:
    def test_process_synthetic_video(self, synthetic_video, tmp_path):
        pipeline = make_pipeline()
        out = tmp_path / "annotated.mp4"
        result = pipeline.process_video(synthetic_video, output_path=str(out))
        assert result.frames_processed == 20
        assert result.processing_fps > 0
        assert out.exists()
        assert result.event_report is not None

    def test_process_max_frames(self, synthetic_video):
        pipeline = make_pipeline()
        result = pipeline.process_video(synthetic_video, max_frames=4)
        assert result.frames_processed == 4

    def test_process_frame_shape_preserved(self, blank_frame):
        pipeline = make_pipeline()
        frame_result = pipeline.process_frame(blank_frame, 0, 0.0, 10.0)
        assert frame_result.annotated_frame.shape == blank_frame.shape
        assert frame_result.person_count >= 1

    def test_no_objects_means_no_events(self, synthetic_video):
        """Empty detections must not crash and must produce zero events."""
        pipeline = make_pipeline(detector=FakeDetector(persons=[], objects=[]))
        result = pipeline.process_video(synthetic_video)
        assert result.events == []

    def test_drinking_sequence_produces_event(self, synthetic_video):
        """A hand-to-face bottle pose held for the whole video -> drinking event."""
        drinking_pose = make_pose(
            nose=(0.25, 0.18),
            left_wrist=(0.26, 0.20),
            right_wrist=(0.35, 0.40),
        )
        bottle = BoundingBox(0.21, 0.14, 0.29, 0.22, 0.95, 39, "bottle")
        pipeline = make_pipeline(
            detector=FakeDetector(persons=[drinking_pose], objects=[bottle])
        )
        result = pipeline.process_video(synthetic_video)
        types = {e.event_type for e in result.events}
        assert "drinking" in types

    def test_progress_callback_called(self, synthetic_video):
        pipeline = make_pipeline()
        seen = []

        def progress(idx, total, frame_result):
            seen.append(idx)

        pipeline.process_video(synthetic_video, progress_callback=progress)
        assert len(seen) == 20

    def test_result_report_json_serializable(self, synthetic_video):
        import json

        pipeline = make_pipeline()
        result = pipeline.process_video(synthetic_video)
        payload = result.event_report.to_dict()
        json.dumps(payload)  # must not raise
        assert "events" in payload


class TestEndToEndReporting:
    def test_full_export_flow(self, synthetic_video, tmp_path):
        from src.reporting.events import EventLog
        from src.reporting.exporters import export_csv, export_json

        pipeline = make_pipeline(
            detector=FakeDetector(
                persons=[make_pose(nose=(0.25, 0.18), left_wrist=(0.26, 0.20))],
                objects=[BoundingBox(0.21, 0.14, 0.29, 0.22, 0.95, 39, "bottle")],
            )
        )
        result = pipeline.process_video(synthetic_video)

        log = EventLog(source=result.source)
        log.extend(result.events)
        report = log.build_report(
            video_duration=result.video_duration,
            video_fps=result.video_fps,
            video_frames=result.video_frames,
            frames_processed=result.frames_processed,
            processing_fps=result.processing_fps,
        )
        json_path = export_json(report, tmp_path / "events.json")
        csv_path = export_csv(result.events, tmp_path / "events.csv")
        assert json_path.exists() and json_path.stat().st_size > 0
        assert csv_path.exists() and csv_path.stat().st_size > 0