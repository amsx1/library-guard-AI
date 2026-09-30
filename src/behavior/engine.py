"""
LibraryGuard - Event aggregation engine.

Turns noisy per-frame behavior scores into clean, deduplicated events using:

* exponential confidence smoothing (per track & behavior)
* minimum consecutive-frame requirements
* minimum duration requirements
* hysteresis (separate activation / deactivation thresholds)
* per-behavior event cooldowns
* three-level evidence labels (DETECTED / POSSIBLE / HIGH CONFIDENCE)
* a final minimum-event-duration filter

The engine is behavior-agnostic: it operates on any registered
:class:`~libraryguard.src.behavior.base.BehaviorDetector`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..utils.constants import (
    clamp01,
    format_timestamp,
    format_timestamp_full,
)
from .base import (
    BehaviorDetector,
    BehaviorEvidence,
    BehaviorObservation,
    BehaviorScore,
    evidence_label,
)
from .drinking import DrinkingDetector
from .phone_usage import PhoneUsageDetector
from .sleeping import SleepingDetector

logger = logging.getLogger(__name__)


@dataclass
class BehaviorEvent:
    """A completed (or live-ongoing) behavior event for one tracked person."""

    event_type: str
    person_id: int
    confidence: float
    duration: float  # seconds
    frame_start: int
    frame_end: int
    time_start: float  # seconds
    time_end: float
    label: str = ""
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def timestamp_str(self) -> str:
        return format_timestamp_full(self.time_start)

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": format_timestamp_full(self.time_start),
            "timestamp_end": format_timestamp_full(self.time_end),
            "event": self.event_type,
            "person_id": self.person_id,
            "confidence": round(self.confidence, 4),
            "label": self.label or evidence_label(self.confidence),
            "duration": round(self.duration, 3),
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
        }

    @property
    def summary(self) -> str:
        return (
            f"{format_timestamp(self.time_start)}  {self.event_type}  "
            f"Person #{self.person_id}  {self.confidence * 100:.0f}%  {self.duration:.1f}s"
        )


@dataclass
class PerBehaviorConfig:
    """Runtime thresholds for one behavior (mirrors YAML config)."""

    enabled: bool = True
    min_duration_seconds: float = 2.0
    confidence_threshold: float = 0.6
    cooldown_seconds: float = 5.0
    min_consecutive_frames: int = 5
    # hysteresis: score must fall below this to deactivate an active behavior
    deactivate_ratio: float = 0.75


@dataclass
class LiveStatus:
    """Current real-time status of one behavior on one track (for UI/overlay)."""

    behavior: str
    label: str  # DETECTED / POSSIBLE / HIGH CONFIDENCE
    confidence: float
    elapsed: float  # seconds the behavior has been continuously evident
    active: bool = False


class BehaviorEngine:
    """
    Temporal aggregation of per-frame behavior scores into events.

    Usage::

        engine = BehaviorEngine(behaviors, configs, fps=25.0)
        for frame_idx, observations in video:
            statuses = engine.update(frame_idx, timestamp, observations)
        events = engine.finalize()
    """

    def __init__(
        self,
        detectors: Optional[List[BehaviorDetector]] = None,
        configs: Optional[Dict[str, PerBehaviorConfig]] = None,
        fps: float = 25.0,
        min_event_duration_seconds: float = 1.0,
        ema_alpha: float = 0.3,
    ):
        self.detectors: List[BehaviorDetector] = detectors or default_detectors()
        self.configs: Dict[str, PerBehaviorConfig] = configs or default_configs()
        self.fps = max(fps, 1e-3)
        self.min_event_duration_seconds = min_event_duration_seconds
        self.ema_alpha = ema_alpha

        # state[(track_id, behavior_key)] -> BehaviorEvidence
        self._evidence: Dict[Tuple[int, str], BehaviorEvidence] = {}
        # active candidate event data
        self._active: Dict[Tuple[int, str], Dict[str, float]] = {}
        self._events: List[BehaviorEvent] = []
        self._last_event_frame: Dict[Tuple[int, str], int] = {}
        self._frame_index = 0

        self._detector_map: Dict[str, BehaviorDetector] = {d.key: d for d in self.detectors}

    # -- public API ---------------------------------------------------------

    def update(
        self,
        frame_index: int,
        timestamp: float,
        observations: List[BehaviorObservation],
        frame_fps: Optional[float] = None,
    ) -> List[LiveStatus]:
        """
        Process one frame's worth of per-track observations.

        Returns live statuses suitable for overlay/UI (currently-evident
        behaviors with labels), for all tracks seen this frame.
        """
        self._frame_index = frame_index
        if frame_fps is not None and frame_fps > 0:
            self.fps = frame_fps

        statuses: List[LiveStatus] = []

        for observation in observations:
            for key, detector in self._detector_map.items():
                cfg = self.configs.get(key)
                if cfg is None or not cfg.enabled:
                    continue

                score = detector.score(observation)
                state = self._evidence.get((observation.track_id, key))
                if state is None:
                    state = BehaviorEvidence(alpha=self.ema_alpha)
                    self._evidence[(observation.track_id, key)] = state

                min_positive = cfg.confidence_threshold * 0.5
                state.update(score.confidence, frame_index, min_positive=min_positive)

                status = self._step_track(
                    observation.track_id,
                    key,
                    score,
                    state,
                    cfg,
                    frame_index,
                    timestamp,
                )
                if status is not None:
                    statuses.append(status)

        return statuses

    def finalize(self) -> List[BehaviorEvent]:
        """Close any in-flight candidate events and return all events."""
        for (track_id, key), candidate in list(self._active.items()):
            self._close_candidate(track_id, key, candidate, self._frame_index, candidate.get("last_time", 0.0))
        self._active.clear()
        return self.events()

    def events(self) -> List[BehaviorEvent]:
        return sorted(self._events, key=lambda e: (e.time_start, e.person_id))

    def reset(self) -> None:
        self._evidence.clear()
        self._active.clear()
        self._events.clear()
        self._last_event_frame.clear()
        self._frame_index = 0

    # -- internals ----------------------------------------------------------

    def _step_track(
        self,
        track_id: int,
        key: str,
        score: BehaviorScore,
        state: BehaviorEvidence,
        cfg: PerBehaviorConfig,
        frame_index: int,
        timestamp: float,
    ) -> Optional[LiveStatus]:
        state_key = (track_id, key)
        conf = state.smoothed_confidence()
        activate = cfg.confidence_threshold
        deactivate = activate * cfg.deactivate_ratio

        candidate = self._active.get(state_key)
        elapsed = 0.0

        if candidate is None:
            # Start a candidate when smoothed confidence crosses activation
            # AND enough consecutive positive frames have been observed.
            if conf >= activate and state.positive_frames >= cfg.min_consecutive_frames:
                self._active[state_key] = {
                    "start_frame": float(frame_index),
                    "start_time": timestamp,
                    "peak": conf,
                    "sum_conf": conf,
                    "n": 1.0,
                    "last_time": timestamp,
                }
                candidate = self._active[state_key]
                elapsed = 0.0
            else:
                return LiveStatus(
                    behavior=key,
                    label=evidence_label(conf),
                    confidence=conf,
                    elapsed=0.0,
                    active=False,
                )
        else:
            candidate["last_time"] = timestamp
            if conf >= deactivate:
                candidate["peak"] = max(candidate["peak"], conf)
                candidate["sum_conf"] += conf
                candidate["n"] += 1.0
                elapsed = timestamp - candidate["start_time"]
                # Event becomes official once minimum duration is reached;
                # afterwards it stays open until evidence drops.
                return LiveStatus(
                    behavior=key,
                    label=evidence_label(candidate["peak"]),
                    confidence=candidate["peak"],
                    elapsed=elapsed,
                    active=True,
                )
            else:
                # Evidence dropped -> close the candidate.
                self._close_candidate(track_id, key, candidate, frame_index, timestamp)
                return LiveStatus(
                    behavior=key,
                    label=evidence_label(conf),
                    confidence=conf,
                    elapsed=0.0,
                    active=False,
                )

        return LiveStatus(
            behavior=key,
            label=evidence_label(conf),
            confidence=conf,
            elapsed=elapsed,
            active=True,
        )

    def _close_candidate(
        self,
        track_id: int,
        key: str,
        candidate: Dict[str, float],
        frame_index: int,
        timestamp: float,
    ) -> None:
        self._active.pop((track_id, key), None)

        cfg = self.configs.get(key)
        if cfg is None:
            return

        duration = max(0.0, timestamp - candidate["start_time"])
        n = max(candidate["n"], 1.0)
        mean_conf = candidate["sum_conf"] / n
        # Final confidence: blend of peak and mean evidence.
        confidence = clamp01(0.6 * candidate["peak"] + 0.4 * mean_conf)

        # Duration filter.
        if duration < max(cfg.min_duration_seconds, self.min_event_duration_seconds):
            logger.debug(
                "Discarded short %s candidate for track %d (%.2fs)", key, track_id, duration
            )
            return

        # Confidence gate.
        if confidence < cfg.confidence_threshold:
            logger.debug(
                "Discarded low-confidence %s candidate for track %d (%.2f)",
                key, track_id, confidence,
            )
            return

        # Cooldown gate.
        last = self._last_event_frame.get((track_id, key), -10 ** 9)
        cooldown_frames = int(cfg.cooldown_seconds * self.fps)
        if frame_index - last < cooldown_frames:
            logger.debug("Suppressed %s for track %d by cooldown", key, track_id)
            return

        event = BehaviorEvent(
            event_type=key,
            person_id=track_id,
            confidence=confidence,
            duration=duration,
            frame_start=int(candidate["start_frame"]),
            frame_end=frame_index,
            time_start=candidate["start_time"],
            time_end=timestamp,
            label=evidence_label(confidence),
            components={},
        )
        self._events.append(event)
        self._last_event_frame[(track_id, key)] = frame_index
        logger.info("Event: %s", event.summary)


# ---------------------------------------------------------------------------
# Defaults / factory helpers
# ---------------------------------------------------------------------------


def default_detectors() -> List[BehaviorDetector]:
    return [DrinkingDetector(), SleepingDetector(), PhoneUsageDetector()]


def default_configs() -> Dict[str, PerBehaviorConfig]:
    return {
        "drinking": PerBehaviorConfig(
            enabled=True,
            min_duration_seconds=1.5,
            confidence_threshold=0.6,
            cooldown_seconds=5.0,
            min_consecutive_frames=5,
        ),
        "sleeping": PerBehaviorConfig(
            enabled=True,
            min_duration_seconds=10.0,
            confidence_threshold=0.65,
            cooldown_seconds=15.0,
            min_consecutive_frames=30,
        ),
        "phone_usage": PerBehaviorConfig(
            enabled=True,
            min_duration_seconds=2.0,
            confidence_threshold=0.6,
            cooldown_seconds=5.0,
            min_consecutive_frames=8,
        ),
    }


def detector_registry() -> Dict[str, Callable[[], BehaviorDetector]]:
    """
    Registry of all behavior detector factories.

    To add a new behavior: implement a BehaviorDetector subclass and register
    it here (plus add a PerBehaviorConfig entry). Nothing else changes.
    """
    return {
        "drinking": DrinkingDetector,
        "sleeping": SleepingDetector,
        "phone_usage": PhoneUsageDetector,
    }


def build_engine_from_config(app_config, fps: float) -> BehaviorEngine:
    """Construct a BehaviorEngine from an AppConfig (src.utils.config)."""
    b = app_config.behavior
    configs = {
        "drinking": PerBehaviorConfig(
            enabled=b.drinking.enabled,
            min_duration_seconds=b.drinking.min_duration_seconds,
            confidence_threshold=b.drinking.confidence_threshold,
            cooldown_seconds=b.drinking.cooldown_seconds,
            min_consecutive_frames=b.drinking.min_consecutive_frames,
        ),
        "sleeping": PerBehaviorConfig(
            enabled=b.sleeping.enabled,
            min_duration_seconds=b.sleeping.min_duration_seconds,
            confidence_threshold=b.sleeping.confidence_threshold,
            cooldown_seconds=b.sleeping.cooldown_seconds,
            min_consecutive_frames=b.sleeping.min_consecutive_frames,
        ),
        "phone_usage": PerBehaviorConfig(
            enabled=b.phone_usage.enabled,
            min_duration_seconds=b.phone_usage.min_duration_seconds,
            confidence_threshold=b.phone_usage.confidence_threshold,
            cooldown_seconds=b.phone_usage.cooldown_seconds,
            min_consecutive_frames=b.phone_usage.min_consecutive_frames,
        ),
    }
    detectors = [
        DrinkingDetector(
            hand_to_mouth_distance_threshold=b.drinking.hand_to_mouth_distance_threshold,
            object_near_face_threshold=b.drinking.object_near_face_threshold,
        ),
        SleepingDetector(
            head_down_angle_threshold=b.sleeping.head_down_angle_threshold,
            low_movement_threshold=b.sleeping.low_movement_threshold,
        ),
        PhoneUsageDetector(
            phone_near_hand_threshold=b.phone_usage.phone_near_hand_threshold,
            phone_near_face_threshold=b.phone_usage.phone_near_face_threshold,
            looking_down_angle_threshold=b.phone_usage.looking_down_angle_threshold,
        ),
    ]
    return BehaviorEngine(
        detectors=detectors,
        configs=configs,
        fps=fps,
        min_event_duration_seconds=app_config.reporting.min_event_duration_seconds,
    )