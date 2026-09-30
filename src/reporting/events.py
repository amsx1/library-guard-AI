"""
LibraryGuard - Event log and report generation.

Events contain only behavioral information (type, anonymous track ID,
confidence, timing). They deliberately contain NO personal information:
no names, faces, biometrics, or identity data.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..behavior.engine import BehaviorEvent


@dataclass
class EventReport:
    """Complete report for one processed video."""

    source: str
    processed_at: str
    video_duration: float
    video_fps: float
    video_frames: int
    frames_processed: int
    processing_fps: float
    events: List[BehaviorEvent] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_version": "1.0",
            "source": self.source,
            "processed_at": self.processed_at,
            "video": {
                "duration_seconds": round(self.video_duration, 3),
                "fps": round(self.video_fps, 3),
                "frames": self.video_frames,
            },
            "processing": {
                "frames_processed": self.frames_processed,
                "processing_fps": round(self.processing_fps, 2),
            },
            "event_count": len(self.events),
            "events_by_type": self.counts_by_type(),
            "events": [e.to_dict() for e in self.events],
            "config": self.config_snapshot,
            "notes": self.notes,
            "disclaimer": (
                "Computer-vision predictions are probabilistic and can be wrong. "
                "Events are behavioral observations only and do not identify people."
            ),
        }

    def counts_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in self.events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts


class EventLog:
    """Accumulates events and produces :class:`EventReport` objects."""

    def __init__(self, source: str = "", config_snapshot: Optional[Dict[str, Any]] = None):
        self.source = source
        self._events: List[BehaviorEvent] = []
        self._config_snapshot = config_snapshot or {}
        self._notes: List[str] = []

    def add(self, event: BehaviorEvent) -> None:
        self._events.append(event)

    def extend(self, events: Iterable[BehaviorEvent]) -> None:
        self._events.extend(events)

    def add_note(self, note: str) -> None:
        self._notes.append(note)

    @property
    def events(self) -> List[BehaviorEvent]:
        return sorted(self._events, key=lambda e: (e.time_start, e.person_id, e.event_type))

    def __len__(self) -> int:
        return len(self._events)

    def build_report(
        self,
        video_duration: float = 0.0,
        video_fps: float = 0.0,
        video_frames: int = 0,
        frames_processed: int = 0,
        processing_fps: float = 0.0,
    ) -> EventReport:
        return EventReport(
            source=self.source,
            processed_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            video_duration=video_duration,
            video_fps=video_fps,
            video_frames=video_frames,
            frames_processed=frames_processed,
            processing_fps=processing_fps,
            events=self.events,
            config_snapshot=self._config_snapshot,
            notes=list(self._notes),
        )