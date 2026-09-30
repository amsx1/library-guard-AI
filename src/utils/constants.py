"""
LibraryGuard - Shared constants and geometry utilities.

Single source of truth for COCO class IDs, default thresholds,
and small geometric helpers used across detection, tracking, and
behavior modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------------
# COCO class IDs used by the project
# ---------------------------------------------------------------------------

COCO_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}

# Objects relevant to the supported behaviors.
DRINKING_OBJECT_CLASSES = {
    COCO_CLASSES["bottle"]: "bottle",
    COCO_CLASSES["cup"]: "cup",
    COCO_CLASSES["wine glass"]: "wine glass",
    COCO_CLASSES["bowl"]: "bowl",
}
PHONE_OBJECT_CLASSES = {
    COCO_CLASSES["cell phone"]: "cell phone",
}

# All object classes the object detector keeps by default.
TARGET_OBJECT_CLASSES = sorted(
    set(DRINKING_OBJECT_CLASSES) | set(PHONE_OBJECT_CLASSES) | {COCO_CLASSES["book"], COCO_CLASSES["laptop"]}
)

# COCO-pose keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Behavior event types
EVENT_DRINKING = "drinking"
EVENT_SLEEPING = "sleeping"
EVENT_PHONE = "phone_usage"
EVENT_TYPES = (EVENT_DRINKING, EVENT_SLEEPING, EVENT_PHONE)

# Evidence labels
LABEL_DETECTED = "DETECTED"
LABEL_POSSIBLE = "POSSIBLE"
LABEL_HIGH_CONFIDENCE = "HIGH CONFIDENCE"

BEHAVIOR_EMOJI = {
    EVENT_DRINKING: "\U0001F964",  # 🥤
    EVENT_SLEEPING: "\U0001F634",  # 😴
    EVENT_PHONE: "\U0001F4F1",  # 📱
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """A 2D point in normalized [0, 1] image coordinates."""

    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class Box:
    """Axis-aligned box, normalized [0, 1] xyxy coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def scale(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Return pixel-coordinate xyxy tuple for a frame of given size."""
        return (
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
            int(round(self.x2 * width)),
            int(round(self.y2 * height)),
        )

    def contains_point(self, point: Point) -> bool:
        return self.x1 <= point.x <= self.x2 and self.y1 <= point.y <= self.y2

    def distance_to_point(self, point: Point) -> float:
        """Normalized distance from the box to a point (0 if inside)."""
        dx = max(self.x1 - point.x, 0.0, point.x - self.x2)
        dy = max(self.y1 - point.y, 0.0, point.y - self.y2)
        return math.hypot(dx, dy)


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two normalized boxes."""
    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_a.area + box_b.area - inter
    return inter / union if union > 0 else 0.0


def center_distance(box_a: Box, box_b: Box) -> float:
    """Euclidean distance between the centers of two normalized boxes."""
    ca, cb = box_a.center, box_b.center
    return math.hypot(ca.x - cb.x, ca.y - cb.y)


def point_distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def format_timestamp(seconds: float) -> str:
    """Format a duration as HH:MM:SS (or MM:SS below one hour)."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def format_timestamp_full(seconds: float) -> str:
    """Format a duration as HH:MM:SS.mmm for logs/reports."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"
