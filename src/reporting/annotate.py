"""
LibraryGuard - Video annotation.

Draws tracking boxes, anonymous IDs, behavior labels, confidence values,
and timestamps onto frames. Designed for debugging the model as much as
for presenting results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..behavior.engine import LiveStatus
from ..utils.constants import (
    BEHAVIOR_EMOJI,
    EVENT_DRINKING,
    EVENT_PHONE,
    EVENT_SLEEPING,
    format_timestamp,
)

# BGR colors
COLOR_PERSON = (80, 200, 120)        # green
COLOR_PERSON_TENTATIVE = (160, 160, 160)
COLOR_OBJECT = (255, 178, 102)       # light blue
COLOR_DRINKING = (255, 178, 102)     # light blue-ish
COLOR_SLEEPING = (255, 153, 255)     # pink
COLOR_PHONE = (102, 204, 255)        # amber-ish
COLOR_TEXT_BG = (30, 30, 30)
COLOR_TEXT = (245, 245, 245)
COLOR_TIMESTAMP = (230, 230, 230)

BEHAVIOR_COLORS = {
    EVENT_DRINKING: COLOR_DRINKING,
    EVENT_SLEEPING: COLOR_SLEEPING,
    EVENT_PHONE: COLOR_PHONE,
}


@dataclass
class AnnotationConfig:
    draw_bboxes: bool = True
    draw_tracking_ids: bool = True
    draw_behavior_labels: bool = True
    draw_confidence: bool = True
    draw_timestamp: bool = True
    draw_object_boxes: bool = True
    font_scale: float = 0.6
    line_thickness: int = 2
    blur_faces: bool = True


@dataclass
class TrackOverlay:
    """Information needed to annotate one tracked person on a frame."""

    track_id: int
    bbox: Tuple[float, float, float, float]  # normalized xyxy
    score: float = 1.0
    statuses: Optional[List[LiveStatus]] = None
    face_box: Optional[Tuple[float, float, float, float]] = None  # normalized


@dataclass
class ObjectOverlay:
    bbox: Tuple[float, float, float, float]
    class_name: str
    score: float


def _put_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    bg_color: Optional[Tuple[int, int, int]] = COLOR_TEXT_BG,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    if bg_color is not None:
        cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _blur_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = max(9, (min(x2 - x1, y2 - y1) // 3) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def annotate_frame(
    frame: np.ndarray,
    tracks: Sequence[TrackOverlay],
    objects: Optional[Sequence[ObjectOverlay]] = None,
    frame_index: int = 0,
    timestamp: float = 0.0,
    fps: Optional[float] = None,
    config: Optional[AnnotationConfig] = None,
) -> np.ndarray:
    """Return an annotated copy of ``frame``."""
    cfg = config or AnnotationConfig()
    out = frame.copy()
    height, width = out.shape[:2]

    # Object boxes (draw first, underneath people).
    if cfg.draw_object_boxes and objects:
        for obj in objects:
            x1, y1, x2, y2 = (int(v * width) if i % 2 == 0 else int(v * height) for i, v in enumerate(obj.bbox))
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_OBJECT, max(1, cfg.line_thickness - 1))
            if cfg.draw_confidence:
                _put_text_with_bg(
                    out,
                    f"{obj.class_name} {obj.score:.2f}",
                    (x1, max(15, y1 - 6)),
                    cfg.font_scale * 0.8,
                    COLOR_OBJECT,
                )

    for track in tracks:
        x1, y1, x2, y2 = (int(v * width) if i % 2 == 0 else int(v * height) for i, v in enumerate(track.bbox))

        # Optional face blur (privacy feature).
        if cfg.blur_faces and track.face_box is not None:
            fx1, fy1, fx2, fy2 = (
                int(v * width) if i % 2 == 0 else int(v * height) for i, v in enumerate(track.face_box)
            )
            _blur_region(out, fx1, fy1, fx2, fy2)

        # Person box + ID.
        if cfg.draw_bboxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_PERSON, cfg.line_thickness)

        label_y = y1 - 8
        if cfg.draw_tracking_ids:
            _put_text_with_bg(out, f"Person #{track.track_id}", (x1, max(18, label_y)), cfg.font_scale, COLOR_PERSON)
            label_y -= 24

        # Behavior labels under the box.
        if cfg.draw_behavior_labels and track.statuses:
            active = [s for s in track.statuses if s.active]
            shown = active if active else [s for s in track.statuses if s.confidence > 0.25]
            for status in shown[:3]:
                color = BEHAVIOR_COLORS.get(status.behavior, COLOR_PERSON)
                emoji = BEHAVIOR_EMOJI.get(status.behavior, "")
                conf_txt = f" {status.confidence * 100:.0f}%" if cfg.draw_confidence else ""
                elapsed_txt = f" ({status.elapsed:.1f}s)" if status.active and status.elapsed > 0 else ""
                text = f"{emoji} {status.behavior.replace('_', ' ').upper()}{conf_txt}{elapsed_txt}"
                y = min(height - 10, y2 + 22 + 20 * shown.index(status))
                _put_text_with_bg(out, text, (x1, y), cfg.font_scale, color)

    # Timestamp / frame / fps banner.
    if cfg.draw_timestamp:
        parts = [f"t={format_timestamp(timestamp)}", f"frame {frame_index}"]
        if fps is not None:
            parts.append(f"{fps:.1f} FPS")
        banner = "  |  ".join(parts)
        _put_text_with_bg(out, banner, (10, 26), cfg.font_scale, COLOR_TIMESTAMP)

    return out