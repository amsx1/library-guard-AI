"""Tests: tracking logic (Kalman, association, ID persistence, lifecycle)."""

from __future__ import annotations

import pytest

from src.tracking.tracker import (
    BoTSORTTracker,
    KalmanBBoxFilter,
    TrackState,
    Tracker,
    create_tracker,
)
from src.utils.constants import Box


def box(x1, y1, x2, y2) -> Box:
    return Box(x1, y1, x2, y2)


class TestKalman:
    def test_predict_moves_toward_velocity(self):
        kf = KalmanBBoxFilter(box(0.10, 0.10, 0.20, 0.30))
        # Feed a couple of updates moving right.
        kf.update(box(0.11, 0.10, 0.21, 0.30))
        kf.update(box(0.12, 0.10, 0.22, 0.30))
        predicted = kf.predict()
        assert predicted.x1 > 0.11  # moved in +x direction

    def test_update_pulls_prediction_to_measurement(self):
        kf = KalmanBBoxFilter(box(0.10, 0.10, 0.20, 0.30))
        for _ in range(5):
            kf.update(box(0.50, 0.50, 0.60, 0.70))
        result = kf.to_box()
        assert abs(result.center.x - 0.55) < 0.1

    def test_to_box_round_trip(self):
        original = box(0.2, 0.3, 0.4, 0.7)
        kf = KalmanBBoxFilter(original)
        recovered = kf.to_box()
        assert abs(recovered.x1 - original.x1) < 1e-6
        assert abs(recovered.y2 - original.y2) < 1e-6


class TestTrackerBasics:
    def test_new_track_is_tentative_then_confirmed(self):
        tracker = Tracker(max_age=30, min_hits=3, iou_threshold=0.3)
        b = box(0.1, 0.1, 0.3, 0.5)

        active = tracker.update([b], [0.9])
        assert active == []  # tentative, not exposed yet

        for _ in range(2):
            active = tracker.update([b], [0.9])
        assert len(active) == 1
        assert active[0].state == TrackState.CONFIRMED
        assert active[0].track_id == 1

    def test_ids_are_stable_across_frames(self):
        tracker = Tracker(min_hits=2)
        for i in range(10):
            offset = 0.01 * i
            tracker.update(
                [box(0.10 + offset, 0.1, 0.30 + offset, 0.5),
                 box(0.60, 0.1, 0.80, 0.5)],
                [0.9, 0.9],
            )
        active = tracker.active_tracks()
        assert len(active) == 2
        ids = sorted(t.track_id for t in active)
        assert ids == [1, 2]

    def test_track_reappears_keeps_id_when_lost_briefly(self):
        tracker = Tracker(max_age=5, min_hits=2)
        for _ in range(4):
            tracker.update([box(0.1, 0.1, 0.3, 0.5)], [0.9])
        # Person disappears for 3 frames (within max_age).
        for _ in range(3):
            tracker.update([], [])
        # Person reappears in roughly the same place.
        tracker.update([box(0.12, 0.1, 0.32, 0.5)], [0.9])
        active = tracker.active_tracks()
        assert len(active) == 1
        assert active[0].track_id == 1

    def test_long_disappearance_removes_track(self):
        tracker = Tracker(max_age=3, min_hits=2)
        for _ in range(4):
            tracker.update([box(0.1, 0.1, 0.3, 0.5)], [0.9])
        for _ in range(6):
            tracker.update([], [])
        assert tracker.active_tracks() == []
        assert len(tracker.tracks) == 0

    def test_two_people_crossing_keep_ids(self):
        tracker = Tracker(min_hits=2, max_age=5)
        # Two people approach but don't overlap enough to swap.
        for i in range(8):
            a = box(0.1 + 0.02 * i, 0.1, 0.25 + 0.02 * i, 0.5)
            b = box(0.8 - 0.01 * i, 0.1, 0.95 - 0.01 * i, 0.5)
            tracker.update([a, b], [0.9, 0.9])
        active = tracker.active_tracks()
        assert len(active) == 2
        assert {t.track_id for t in active} == {1, 2}

    def test_low_score_detection_does_not_create_track(self):
        tracker = Tracker(min_hits=1, score_threshold=0.5)
        tracker.update([box(0.1, 0.1, 0.3, 0.5)], [0.2])
        assert tracker.active_tracks() == []

    def test_movement_metric_reflects_motion(self):
        tracker = Tracker(min_hits=1)
        track = None
        for i in range(10):
            active = tracker.update([box(0.1 + 0.05 * i, 0.1, 0.2 + 0.05 * i, 0.4)], [0.9])
            track = active[0] if active else tracker.tracks[0]
        moving = track.movement_over_window(frames=5)

        tracker2 = Tracker(min_hits=1)
        track2 = None
        for _ in range(10):
            active = tracker2.update([box(0.1, 0.1, 0.2, 0.4)], [0.9])
            track2 = active[0] if active else tracker2.tracks[0]
        still = track2.movement_over_window(frames=5)

        assert moving > still
        assert still == pytest.approx(0.0, abs=1e-9)

    def test_update_length_mismatch_raises(self):
        tracker = Tracker()
        with pytest.raises(ValueError):
            tracker.update([box(0.1, 0.1, 0.2, 0.2)], [0.9, 0.8])

    def test_reset_clears_state(self):
        tracker = Tracker(min_hits=1)
        tracker.update([box(0.1, 0.1, 0.3, 0.5)], [0.9])
        tracker.reset()
        assert tracker.tracks == []
        assert tracker.frame_count == 0


class TestFactory:
    def test_create_bytetrack(self):
        tracker = create_tracker("bytetrack", max_age=7)
        assert isinstance(tracker, Tracker)
        assert tracker.max_age == 7

    def test_create_botsort(self):
        tracker = create_tracker("botsort")
        assert isinstance(tracker, BoTSORTTracker)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown tracker"):
            create_tracker("deepsort-magic")

    def test_botsort_assigns_ids(self):
        tracker = create_tracker("botsort", min_hits=2)
        for _ in range(4):
            tracker.update([box(0.1, 0.1, 0.3, 0.5)], [0.9])
        active = tracker.active_tracks()
        assert len(active) == 1