"""
LibraryGuard - Person tracking.

Implements a ByteTrack-style multi-object tracker:

* Kalman filter for constant-velocity bbox motion prediction
* Hungarian assignment (scipy) with IoU cost
* Two-stage association (high-score then low-score detections) following
  the ByteTrack idea, so briefly-occluded people keep their IDs
* Track lifecycle management (tentative -> confirmed -> lost -> removed)

Tracks are anonymous: they only carry numeric IDs. The tracker never
performs re-identification or any form of biometric matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.constants import Box, Point, iou

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalman filter (self-contained, no external filterpy dependency)
# ---------------------------------------------------------------------------


class KalmanBBoxFilter:
    """
    Constant-velocity Kalman filter over (cx, cy, s, r) where s = area and
    r = aspect ratio. State: [cx, cy, s, r, vcx, vcy, vs].
    """

    def __init__(self, box: Box):
        # State vector
        cx, cy = box.center.x, box.center.y
        w, h = box.width, box.height
        s = w * h
        r = w / h if h > 0 else 1.0
        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float64)

        # Motion model: positions integrate velocities.
        self.F = np.eye(7)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        # Measurement model: we observe (cx, cy, s, r).
        self.H = np.zeros((4, 7))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariances (tuned defaults used by SORT-family trackers)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high initial velocity uncertainty
        self.Q = np.eye(7)
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4)
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0

    def predict(self) -> Box:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.to_box()

    def update(self, box: Box) -> Box:
        z = self._box_to_measurement(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        return self.to_box()

    @staticmethod
    def _box_to_measurement(box: Box) -> np.ndarray:
        w, h = box.width, box.height
        s = w * h
        r = w / h if h > 0 else 1.0
        return np.array([box.center.x, box.center.y, s, r], dtype=np.float64)

    def to_box(self) -> Box:
        cx, cy, s, r = self.x[0], self.x[1], self.x[2], self.x[3]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / w if w > 0 else np.sqrt(s)
        return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def mahalanobis(self, box: Box) -> float:
        """Chi-squared style distance of a measurement from the prediction."""
        z = self._box_to_measurement(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            return float(y.T @ np.linalg.inv(S) @ y)
        except np.linalg.LinAlgError:
            return float("inf")


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Track:
    """A single anonymous person track."""

    track_id: int
    kalman: KalmanBBoxFilter
    state: TrackState = TrackState.TENTATIVE
    score: float = 0.0
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    min_hits: int = 3
    history: List[Box] = field(default_factory=list)
    max_history: int = 64

    def __post_init__(self) -> None:
        # A track born with enough hits (e.g. min_hits=1) is confirmed at once.
        if self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED

    @property
    def bbox(self) -> Box:
        return self.kalman.to_box()

    @property
    def center(self) -> Point:
        return self.bbox.center

    def predict(self) -> Box:
        self.age += 1
        self.time_since_update += 1
        return self.kalman.predict()

    def update(self, box: Box, score: float) -> None:
        self.kalman.update(box)
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED
        self.history.append(box)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def mark_missed(self, max_age: int) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.REMOVED
        elif self.time_since_update > max_age:
            self.state = TrackState.REMOVED
        else:
            self.state = TrackState.LOST

    def movement_over_window(self, frames: int = 10) -> float:
        """
        Normalized centroid movement over the last N frames, used as a
        'stillness' signal by the sleeping behavior detector.
        """
        if len(self.history) < 2:
            return 0.0
        window = self.history[-(frames + 1):]
        if len(window) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(window[:-1], window[1:]):
            ca, cb = a.center, b.center
            total += ((ca.x - cb.x) ** 2 + (ca.y - cb.y) ** 2) ** 0.5
        return total / (len(window) - 1)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


def _iou_matrix(tracks: List[Track], detections: List[Box]) -> np.ndarray:
    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for t_idx, track in enumerate(tracks):
        tb = track.bbox
        for d_idx, det in enumerate(detections):
            matrix[t_idx, d_idx] = iou(tb, det)
    return matrix


class Tracker:
    """
    ByteTrack-style multi-object tracker.

    Parameters
    ----------
    max_age:
        How many consecutive missed frames a confirmed track survives.
    min_hits:
        Hits required before a track is confirmed and exposed to consumers.
    iou_threshold:
        Minimum IoU for an association to be accepted.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        score_threshold: float = 0.5,
        low_score_threshold: float = 0.1,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.low_score_threshold = low_score_threshold

        self._tracks: List[Track] = []
        self._next_id = 1
        self.frame_count = 0

    # -- public API ---------------------------------------------------------

    @property
    def tracks(self) -> List[Track]:
        return self._tracks

    def active_tracks(self, confirmed_only: bool = True) -> List[Track]:
        result = []
        for track in self._tracks:
            if track.state in (TrackState.REMOVED,):
                continue
            if confirmed_only and track.state != TrackState.CONFIRMED:
                continue
            if track.time_since_update > 1 and confirmed_only:
                continue
            result.append(track)
        return result

    def get_track(self, track_id: int) -> Optional[Track]:
        for track in self._tracks:
            if track.track_id == track_id:
                return track
        return None

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.frame_count = 0

    def update(
        self,
        detections: List[Box],
        scores: Optional[List[float]] = None,
    ) -> List[Track]:
        """
        Advance the tracker by one frame.

        Returns the list of confirmed tracks that are currently visible.
        """
        self.frame_count += 1
        if scores is None:
            scores = [1.0] * len(detections)
        if len(detections) != len(scores):
            raise ValueError("detections and scores must have the same length")

        # Predict all existing tracks.
        for track in self._tracks:
            track.predict()

        # Split detections into high and low score groups (ByteTrack style).
        high_idx = [i for i, s in enumerate(scores) if s >= self.score_threshold]
        low_idx = [i for i, s in enumerate(scores) if self.low_score_threshold <= s < self.score_threshold]
        unmatched_after_first: List[int] = []

        matched_pairs: List[Tuple[int, int]] = []

        # Stage 1: associate high-score detections with confirmed tracks first,
        # then with tentative tracks.
        confirmed = [t for t in self._tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self._tracks if t.state == TrackState.TENTATIVE]

        high_dets = [detections[i] for i in high_idx]
        used_det_slots: set = set()

        if confirmed and high_dets:
            pairs, unmatched_t, unmatched_d = self._assign(confirmed, high_dets)
            for t_i, d_i in pairs:
                matched_pairs.append((self._tracks.index(confirmed[t_i]), high_idx[d_i]))
                used_det_slots.add(d_i)
            unmatched_after_first = [high_idx[d_i] for d_i in unmatched_d]
        else:
            unmatched_after_first = list(high_idx)

        # Stage 2: remaining high-score detections vs tentative tracks.
        if tentative and unmatched_after_first:
            rem_dets = [detections[i] for i in unmatched_after_first]
            pairs, unmatched_t, unmatched_d = self._assign(tentative, rem_dets)
            for t_i, d_i in pairs:
                matched_pairs.append((self._tracks.index(tentative[t_i]), unmatched_after_first[d_i]))
            unmatched_after_first = [unmatched_after_first[d_i] for d_i in unmatched_d]

        # Stage 3: unmatched tracks vs low-score detections (ByteTrack second
        # association) -- recovers tracks through brief occlusions.
        matched_det_indices = {d for _, d in matched_pairs}
        unmatched_tracks = [
            t for i, t in enumerate(self._tracks)
            if i not in {t for t, _ in matched_pairs} and t.state in (TrackState.CONFIRMED, TrackState.LOST)
        ]
        if unmatched_tracks and low_idx:
            low_dets = [detections[i] for i in low_idx]
            pairs, unmatched_t, unmatched_d = self._assign(unmatched_tracks, low_dets, iou_floor=self.iou_threshold * 0.5)
            for t_i, d_i in pairs:
                matched_pairs.append((self._tracks.index(unmatched_tracks[t_i]), low_idx[d_i]))
                matched_det_indices.add(low_idx[d_i])

        # Apply updates.
        for t_idx, d_idx in matched_pairs:
            self._tracks[t_idx].update(detections[d_idx], scores[d_idx])
            matched_det_indices.add(d_idx)

        # Unmatched tracks: mark missed.
        matched_track_indices = {t for t, _ in matched_pairs}
        for i, track in enumerate(self._tracks):
            if i not in matched_track_indices:
                track.mark_missed(self.max_age)

        # New tracks from unmatched detections above the score threshold.
        for i in unmatched_after_first:
            if i in matched_det_indices:
                continue
            if scores[i] >= self.score_threshold:
                self._tracks.append(
                    Track(
                        track_id=self._next_id,
                        kalman=KalmanBBoxFilter(detections[i]),
                        score=scores[i],
                        min_hits=self.min_hits,
                    )
                )
                self._next_id += 1

        # Drop removed tracks.
        self._tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]

        return self.active_tracks()

    # -- internals ----------------------------------------------------------

    def _assign(
        self,
        tracks: List[Track],
        detections: List[Box],
        iou_floor: Optional[float] = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian assignment on IoU cost; returns (matches, unmatched tracks, unmatched dets)."""
        floor = self.iou_threshold if iou_floor is None else iou_floor
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost = 1.0 - _iou_matrix(tracks, detections)
        row_idx, col_idx = linear_sum_assignment(cost)

        matches: List[Tuple[int, int]] = []
        matched_rows, matched_cols = set(), set()
        for r, c in zip(row_idx, col_idx):
            if 1.0 - cost[r, c] >= floor:
                matches.append((r, c))
                matched_rows.add(r)
                matched_cols.add(c)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_rows]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_cols]
        return matches, unmatched_tracks, unmatched_dets


# Backwards-friendly alias
ByteTrackTracker = Tracker


class BoTSORTTracker(Tracker):
    """
    Simplified BoT-SORT-style tracker.

    Extends the ByteTrack-style tracker with appearance-free motion gating
    via Kalman Mahalanobis distance. (A full BoT-SORT would add appearance
    features and camera-motion compensation; those are listed as future
    improvements and intentionally not faked here.)
    """

    def __init__(self, mahalanobis_gate: float = 9.4877, **kwargs):
        super().__init__(**kwargs)
        self.mahalanobis_gate = mahalanobis_gate  # ~chi2 95% for 4 dof

    def _assign(self, tracks, detections, iou_floor=None):
        floor = self.iou_threshold if iou_floor is None else iou_floor
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        n_t, n_d = len(tracks), len(detections)
        cost = np.ones((n_t, n_d), dtype=np.float64)
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                overlap = iou(track.bbox, det)
                if overlap < floor:
                    continue
                if track.kalman.mahalanobis(det) > self.mahalanobis_gate:
                    continue
                cost[t_idx, d_idx] = 1.0 - overlap

        row_idx, col_idx = linear_sum_assignment(cost)
        matches, matched_rows, matched_cols = [], set(), set()
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < 1.0:
                matches.append((r, c))
                matched_rows.add(r)
                matched_cols.add(c)

        unmatched_tracks = [i for i in range(n_t) if i not in matched_rows]
        unmatched_dets = [i for i in range(n_d) if i not in matched_cols]
        return matches, unmatched_tracks, unmatched_dets


def create_tracker(
    tracker_type: str = "bytetrack",
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    **kwargs,
):
    """Factory for trackers."""
    if tracker_type in ("bytetrack", "byte-track"):
        return Tracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, **kwargs)
    if tracker_type in ("botsort", "bot-sort"):
        return BoTSORTTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold, **kwargs)
    if tracker_type in ("simple-iou", "simple"):
        return Tracker(max_age=max_age, min_hits=max(min_hits, 1), iou_threshold=iou_threshold, score_threshold=0.0, **kwargs)
    raise ValueError(f"Unknown tracker type: {tracker_type!r}")