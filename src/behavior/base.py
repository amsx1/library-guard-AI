"""
LibraryGuard - Behavior detector interface.

Defines the abstract :class:`BehaviorDetector` contract plus the shared
evidence/state structures. New behaviors (eating, smoking, ...) can be added
by subclassing :class:`BehaviorDetector` and registering the class in
``behavior/engine.py`` -- no other part of the pipeline needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..utils.constants import (
    LABEL_DETECTED,
    LABEL_HIGH_CONFIDENCE,
    LABEL_POSSIBLE,
    clamp01,
)


def evidence_label(confidence: float, high: float = 0.85, low: float = 0.5) -> str:
    """Map a continuous confidence to the public three-level label."""
    if confidence >= high:
        return LABEL_HIGH_CONFIDENCE
    if confidence >= low:
        return LABEL_POSSIBLE
    return LABEL_DETECTED


@dataclass
class BehaviorEvidence:
    """
    Rolling evidence state for one behavior on one track.

    Confidence is smoothed with an exponential moving average so a single
    strong frame cannot instantly create an alert, and hysteresis thresholds
    prevent flicker at the decision boundary.
    """

    ema_confidence: float = 0.0
    raw_confidence: float = 0.0
    positive_frames: int = 0  # consecutive frames above activation threshold
    total_positive_frames: int = 0  # cumulative positive frames in current candidate
    candidate_start_frame: Optional[int] = None
    peak_confidence: float = 0.0
    active: bool = False  # currently emitting an ongoing behavior
    last_event_frame: int = -10 ** 9
    frames_seen: int = 0

    # Smoothing factor for the EMA (0..1). Higher = faster response.
    alpha: float = 0.3

    def update(self, raw_confidence: float, frame_index: int, min_positive: float = 0.35) -> "BehaviorEvidence":
        self.frames_seen += 1
        self.raw_confidence = clamp01(raw_confidence)
        if self.frames_seen == 1:
            self.ema_confidence = self.raw_confidence
        else:
            self.ema_confidence = self.alpha * self.raw_confidence + (1.0 - self.alpha) * self.ema_confidence

        if self.raw_confidence >= min_positive:
            self.positive_frames += 1
            self.total_positive_frames += 1
            if self.candidate_start_frame is None:
                self.candidate_start_frame = frame_index
            self.peak_confidence = max(self.peak_confidence, self.ema_confidence)
        else:
            self.positive_frames = 0
        return self

    def reset_candidate(self) -> None:
        self.positive_frames = 0
        self.total_positive_frames = 0
        self.candidate_start_frame = None
        self.peak_confidence = 0.0

    def smoothed_confidence(self) -> float:
        return clamp01(self.ema_confidence)


@dataclass
class BehaviorObservation:
    """Per-frame observation handed to behavior detectors."""

    frame_index: int
    timestamp: float  # seconds since video start
    track_id: int
    person_box: object  # Box
    keypoints: Optional[object] = None  # Pose or None
    person_score: float = 1.0
    objects: List[object] = field(default_factory=list)  # list of (Box, class_name, score)
    track_movement: float = 0.0  # normalized centroid movement per frame
    frame_width: int = 0
    frame_height: int = 0


@dataclass
class BehaviorScore:
    """Output of a behavior detector for one track in one frame."""

    behavior: str
    confidence: float  # [0, 1]
    components: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


class BehaviorDetector(ABC):
    """
    Abstract base for all behavior detectors.

    Subclasses implement :meth:`score`, which evaluates a single-frame
    :class:`BehaviorObservation` and returns a :class:`BehaviorScore`.
    Temporal aggregation (duration, smoothing, cooldowns) is handled by the
    shared :class:`~libraryguard.src.behavior.engine.BehaviorEngine`, so each
    detector only encodes the *visual evidence* for its behavior.
    """

    #: key used in config and event reports (e.g. "drinking")
    key: str = ""
    #: human-readable name
    display_name: str = ""

    @abstractmethod
    def score(self, observation: BehaviorObservation) -> BehaviorScore:
        """Evaluate visual evidence for one frame of one tracked person."""

    def describe(self) -> Dict[str, str]:
        return {"key": self.key, "display_name": self.display_name}