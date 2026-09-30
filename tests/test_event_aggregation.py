"""Tests: event aggregation, confidence thresholding, duration filtering, cooldowns."""

from __future__ import annotations


from src.behavior.base import (
    BehaviorDetector,
    BehaviorObservation,
    BehaviorScore,
    evidence_label,
)
from src.behavior.engine import (
    BehaviorEngine,
    PerBehaviorConfig,
    default_configs,
    default_detectors,
    detector_registry,
)
from tests.conftest import make_observation


class ConstantDetector(BehaviorDetector):
    """Test double: always returns a fixed confidence."""

    key = "drinking"
    display_name = "Drinking"

    def __init__(self, confidence: float):
        self.confidence = confidence

    def score(self, observation: BehaviorObservation) -> BehaviorScore:
        return BehaviorScore(self.key, self.confidence)


class ScriptedDetector(BehaviorDetector):
    """Test double: returns a scripted confidence sequence."""

    key = "drinking"
    display_name = "Drinking"

    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def score(self, observation: BehaviorObservation) -> BehaviorScore:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return BehaviorScore(self.key, value)


def make_engine(confidence, min_duration=0.4, cooldown=1.0, min_frames=2, fps=10.0):
    return BehaviorEngine(
        detectors=[ConstantDetector(confidence)],
        configs={
            "drinking": PerBehaviorConfig(
                enabled=True,
                min_duration_seconds=min_duration,
                confidence_threshold=0.6,
                cooldown_seconds=cooldown,
                min_consecutive_frames=min_frames,
            )
        },
        fps=fps,
        min_event_duration_seconds=0.0,
    )


def run_engine(engine, seconds, fps=10.0):
    statuses = []
    for i in range(int(seconds * fps)):
        obs = make_observation(frame_index=i, timestamp=i / fps)
        statuses.append(engine.update(i, i / fps, [obs]))
    return statuses


# ---------------------------------------------------------------------------
# Duration & confidence gating
# ---------------------------------------------------------------------------


def test_high_confidence_long_event_is_detected():
    engine = make_engine(0.95, min_duration=0.4)
    run_engine(engine, 2.0)
    events = engine.finalize()
    assert len(events) == 1
    assert events[0].event_type == "drinking"
    assert events[0].confidence >= 0.6
    assert events[0].duration >= 0.4


def test_short_blip_is_filtered_by_min_duration():
    """A 0.2 s strong blip must NOT produce an event when min_duration=0.5 s."""
    detector = ScriptedDetector([0.95] * 2 + [0.0] * 50)  # 0.2 s at 10 fps
    engine = BehaviorEngine(
        detectors=[detector],
        configs={
            "drinking": PerBehaviorConfig(
                enabled=True,
                min_duration_seconds=0.5,
                confidence_threshold=0.6,
                cooldown_seconds=1.0,
                min_consecutive_frames=1,
            )
        },
        fps=10.0,
        min_event_duration_seconds=0.0,
    )
    for i in range(52):
        engine.update(i, i / 10.0, [make_observation(frame_index=i, timestamp=i / 10.0)])
    assert engine.finalize() == []


def test_low_confidence_never_triggers():
    engine = make_engine(0.2, min_duration=0.1, min_frames=1)
    run_engine(engine, 5.0)
    assert engine.finalize() == []


def test_boundary_confidence_is_smoothed():
    """Raw values exactly at threshold trigger only after smoothing/EMA settles."""
    engine = make_engine(0.6, min_duration=0.2, min_frames=2)
    run_engine(engine, 1.5)
    events = engine.finalize()
    assert len(events) == 1


def test_min_consecutive_frames_respected():
    """Sporadic positives below min_consecutive_frames never open a candidate."""
    detector = ScriptedDetector([0.95, 0.1] * 20)  # alternating
    engine = BehaviorEngine(
        detectors=[detector],
        configs={
            "drinking": PerBehaviorConfig(
                enabled=True,
                min_duration_seconds=0.1,
                confidence_threshold=0.6,
                cooldown_seconds=0.0,
                min_consecutive_frames=5,
            )
        },
        fps=10.0,
        min_event_duration_seconds=0.0,
    )
    for i in range(40):
        engine.update(i, i / 10.0, [make_observation(frame_index=i, timestamp=i / 10.0)])
    # EMA will hover around 0.5, never sustaining 5 consecutive positives.
    for event in engine.finalize():
        assert event.confidence < 0.75


# ---------------------------------------------------------------------------
# Cooldown behavior
# ---------------------------------------------------------------------------


def test_cooldown_suppresses_repeats():
    detector = ScriptedDetector([0.95] * 15 + [0.0] * 5 + [0.95] * 15 + [0.0] * 30)
    engine = BehaviorEngine(
        detectors=[detector],
        configs={
            "drinking": PerBehaviorConfig(
                enabled=True,
                min_duration_seconds=0.3,
                confidence_threshold=0.6,
                cooldown_seconds=5.0,
                min_consecutive_frames=1,
            )
        },
        fps=10.0,
        min_event_duration_seconds=0.0,
    )
    for i in range(65):
        engine.update(i, i / 10.0, [make_observation(frame_index=i, timestamp=i / 10.0)])
    events = engine.finalize()
    # Second burst happens ~2 s after the first ends -> within 5 s cooldown.
    assert len(events) == 1


def test_no_cooldown_allows_two_events():
    detector = ScriptedDetector([0.95] * 15 + [0.0] * 15 + [0.95] * 15 + [0.0] * 15)
    engine = BehaviorEngine(
        detectors=[detector],
        configs={
            "drinking": PerBehaviorConfig(
                enabled=True,
                min_duration_seconds=0.3,
                confidence_threshold=0.6,
                cooldown_seconds=0.5,
                min_consecutive_frames=1,
            )
        },
        fps=10.0,
        min_event_duration_seconds=0.0,
    )
    for i in range(60):
        engine.update(i, i / 10.0, [make_observation(frame_index=i, timestamp=i / 10.0)])
    events = engine.finalize()
    assert len(events) == 2


# ---------------------------------------------------------------------------
# Multiple tracks
# ---------------------------------------------------------------------------


def test_events_are_tracked_per_person():
    engine = BehaviorEngine(
        detectors=[ConstantDetector(0.95)],
        configs={
            "drinking": PerBehaviorConfig(
                enabled=True,
                min_duration_seconds=0.3,
                confidence_threshold=0.6,
                cooldown_seconds=1.0,
                min_consecutive_frames=1,
            )
        },
        fps=10.0,
        min_event_duration_seconds=0.0,
    )
    for i in range(20):
        obs1 = make_observation(track_id=1, frame_index=i, timestamp=i / 10.0)
        obs2 = make_observation(track_id=2, frame_index=i, timestamp=i / 10.0)
        engine.update(i, i / 10.0, [obs1, obs2])
    events = engine.finalize()
    assert len(events) == 2
    assert {e.person_id for e in events} == {1, 2}


def test_disabled_behavior_never_fires():
    engine = BehaviorEngine(
        detectors=[ConstantDetector(0.95)],
        configs={
            "drinking": PerBehaviorConfig(enabled=False, min_consecutive_frames=1)
        },
        fps=10.0,
        min_event_duration_seconds=0.0,
    )
    run_engine(engine, 3.0)
    assert engine.finalize() == []


# ---------------------------------------------------------------------------
# Labels & schema
# ---------------------------------------------------------------------------


def test_evidence_labels():
    assert evidence_label(0.95) == "HIGH CONFIDENCE"
    assert evidence_label(0.7) == "POSSIBLE"
    assert evidence_label(0.3) == "DETECTED"


def test_event_dict_schema():
    engine = make_engine(0.95, min_duration=0.3)
    run_engine(engine, 1.5)
    events = engine.finalize()
    assert events
    data = events[0].to_dict()
    for key in ("timestamp", "event", "person_id", "confidence", "duration",
                "frame_start", "frame_end", "label", "timestamp_end"):
        assert key in data
    assert data["event"] == "drinking"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["frame_end"] >= data["frame_start"]


def test_registry_contains_default_behaviors():
    registry = detector_registry()
    assert set(registry) >= {"drinking", "sleeping", "phone_usage"}
    defaults = default_configs()
    assert set(defaults) == set(registry)
    assert len(default_detectors()) == len(registry)