"""Tests: reporting output schemas (JSON/CSV), annotation, evaluation metrics."""

from __future__ import annotations

import csv
import json

import pytest

from src.behavior.engine import BehaviorEvent
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_event_metrics,
    confusion_matrix,
    fps_benchmark,
)
from src.reporting.annotate import (
    AnnotationConfig,
    ObjectOverlay,
    TrackOverlay,
    annotate_frame,
)
from src.reporting.events import EventLog
from src.reporting.exporters import CSV_COLUMNS, export_csv, export_events_json, export_json


def make_event(event_type="drinking", person_id=1, confidence=0.91, start=134.0, duration=3.2):
    return BehaviorEvent(
        event_type=event_type,
        person_id=person_id,
        confidence=confidence,
        duration=duration,
        frame_start=int(start * 25),
        frame_end=int((start + duration) * 25),
        time_start=start,
        time_end=start + duration,
    )


class TestEventReportSchema:
    def test_report_json_schema(self, tmp_path):
        log = EventLog(source="sample.mp4", config_snapshot={"k": 1})
        log.add(make_event())
        log.add(make_event(event_type="sleeping", person_id=7, confidence=0.84, start=227.0))
        report = log.build_report(video_duration=300.0, video_fps=25.0, video_frames=7500,
                                  frames_processed=3000, processing_fps=12.5)
        path = export_json(report, tmp_path / "report.json")

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        assert data["report_version"] == "1.0"
        assert data["event_count"] == 2
        assert data["events_by_type"] == {"drinking": 1, "sleeping": 1}
        assert data["video"]["fps"] == 25.0
        assert data["processing"]["processing_fps"] == 12.5
        assert "disclaimer" in data

        event = data["events"][0]
        for key in ("timestamp", "event", "person_id", "confidence", "duration",
                    "frame_start", "frame_end"):
            assert key in event
        assert event["confidence"] == 0.91
        assert event["duration"] == 3.2

    def test_csv_schema(self, tmp_path):
        events = [make_event(), make_event(event_type="phone_usage", person_id=2)]
        path = export_csv(events, tmp_path / "events.csv")

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        assert reader.fieldnames == CSV_COLUMNS
        assert len(rows) == 2
        assert rows[0]["event"] == "drinking"
        assert float(rows[0]["confidence"]) == 0.91

    def test_empty_events_export(self, tmp_path):
        path = export_csv([], tmp_path / "empty.csv")
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert rows == []
        assert reader.fieldnames == CSV_COLUMNS

    def test_events_json_export(self, tmp_path):
        path = export_events_json([make_event()], tmp_path / "e.json")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        assert data["event_count"] == 1
        assert data["events"][0]["event"] == "drinking"

    def test_report_contains_no_personal_info(self):
        log = EventLog(source="x.mp4")
        log.add(make_event())
        report = log.build_report()
        data = json.dumps(report.to_dict())
        for forbidden in ("name", "face_embedding", "student_id", "phone_number", "email"):
            assert forbidden not in data.lower() or forbidden == "phone"  # 'phone_usage' ok
        # person_id is the only identifier and must be an integer track id.
        assert isinstance(report.events[0].person_id, int)


class TestAnnotation:
    def test_annotate_frame_returns_same_size(self, blank_frame):
        tracks = [
            TrackOverlay(track_id=3, bbox=(0.1, 0.1, 0.5, 0.8), score=0.9, statuses=[]),
        ]
        out = annotate_frame(blank_frame, tracks, frame_index=10, timestamp=1.5, fps=24.0)
        assert out.shape == blank_frame.shape
        # Annotation must actually draw (frame no longer all zeros).
        assert out.sum() > blank_frame.sum()

    def test_annotate_with_objects_and_faces(self, blank_frame):
        tracks = [
            TrackOverlay(
                track_id=1,
                bbox=(0.1, 0.1, 0.5, 0.8),
                statuses=[],
                face_box=(0.2, 0.1, 0.4, 0.3),
            )
        ]
        objects = [ObjectOverlay(bbox=(0.6, 0.6, 0.8, 0.9), class_name="bottle", score=0.8)]
        out = annotate_frame(blank_frame, tracks, objects, config=AnnotationConfig(blur_faces=True))
        assert out.shape == blank_frame.shape

    def test_annotate_ignores_none_frames_not_supported(self, blank_frame):
        out = annotate_frame(blank_frame, [], objects=[])
        assert out.shape == blank_frame.shape


class TestMetrics:
    def test_classification_metrics_perfect(self):
        report = compute_classification_metrics(
            y_true=[0, 1, 2, 0, 1, 2],
            y_pred=[0, 1, 2, 0, 1, 2],
            labels=["drinking", "sleeping", "phone_usage"],
        )
        assert report.accuracy == 1.0
        assert report.macro_f1 == 1.0
        for label in report.labels:
            assert report.precision[label] == 1.0
            assert report.recall[label] == 1.0

    def test_classification_metrics_with_errors(self):
        report = compute_classification_metrics(
            y_true=[0, 0, 1, 1, 2, 2],
            y_pred=[0, 1, 1, 1, 2, 0],
            labels=["drinking", "sleeping", "phone_usage"],
        )
        assert 0 < report.accuracy < 1
        assert report.confusion is not None
        assert report.confusion.sum() == 6
        assert report.false_positive_rate["sleeping"] > 0

    def test_confusion_matrix_counts(self):
        matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], n_classes=3)
        assert matrix[0, 0] == 1
        assert matrix[0, 1] == 1
        assert matrix[1, 1] == 1
        assert matrix[2, 2] == 1

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError):
            compute_classification_metrics([], [], ["a"])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_classification_metrics([0, 1], [0], ["a", "b"])

    def test_event_metrics_perfect_match(self):
        gt = [
            {"event": "drinking", "time_start": 10.0, "time_end": 13.0},
            {"event": "sleeping", "time_start": 30.0, "time_end": 45.0},
        ]
        pred = [
            {"event": "drinking", "time_start": 10.5, "time_end": 13.2, "confidence": 0.9},
            {"event": "sleeping", "time_start": 31.0, "time_end": 44.0, "confidence": 0.8},
        ]
        metrics = compute_event_metrics(pred, gt, iou_threshold=0.3)
        assert metrics.precision["drinking"] == 1.0
        assert metrics.recall["sleeping"] == 1.0

    def test_event_metrics_false_positive_and_negative(self):
        gt = [{"event": "drinking", "time_start": 10.0, "time_end": 13.0}]
        pred = [
            {"event": "drinking", "time_start": 100.0, "time_end": 103.0, "confidence": 0.9},
            {"event": "drinking", "time_start": 200.0, "time_end": 203.0, "confidence": 0.7},
        ]
        metrics = compute_event_metrics(pred, gt, iou_threshold=0.3)
        assert metrics.matches["drinking"] == 0
        assert metrics.false_positives["drinking"] == 2
        assert metrics.false_negatives["drinking"] == 1

    def test_event_metrics_duplicate_predictions(self):
        gt = [{"event": "phone_usage", "time_start": 5.0, "time_end": 8.0}]
        pred = [
            {"event": "phone_usage", "time_start": 5.0, "time_end": 8.0, "confidence": 0.95},
            {"event": "phone_usage", "time_start": 5.5, "time_end": 7.5, "confidence": 0.6},
        ]
        metrics = compute_event_metrics(pred, gt, iou_threshold=0.3)
        assert metrics.matches["phone_usage"] == 1
        assert metrics.false_positives["phone_usage"] == 1

    def test_fps_benchmark(self):
        result = fps_benchmark(lambda x: sum(range(100)), 0, warmup=1, runs=5)
        assert result["mean_fps"] > 0
        assert result["mean_latency_ms"] > 0
        assert result["runs"] == 5