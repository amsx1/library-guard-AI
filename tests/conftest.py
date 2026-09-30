"""
Shared pytest fixtures for LibraryGuard tests.

All test data is synthetic (generated at test time) so it is trivially
redistributable and carries no licensing constraints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.behavior.base import BehaviorObservation  # noqa: E402
from src.detection.detector import BoundingBox, Keypoint, Pose  # noqa: E402
from src.utils.constants import Box  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic video fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_video(tmp_path) -> str:
    """Generate a small synthetic video (2 s, 64x48, 10 fps) with OpenCV."""
    import cv2

    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for i in range(20):
        frame = np.full((48, 64, 3), 30, dtype=np.uint8)
        # moving square so the frame is not constant
        x = (i * 3) % 50
        frame[10:20, x:x + 10] = (200, 180, 100)
        noise = rng.integers(0, 12, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Synthetic detection fixtures
# ---------------------------------------------------------------------------


def make_box(x1=0.1, y1=0.1, x2=0.4, y2=0.6) -> Box:
    return Box(x1, y1, x2, y2)


def make_bbox(x1=0.1, y1=0.1, x2=0.4, y2=0.6, conf=0.9, cls=0, name="person") -> BoundingBox:
    return BoundingBox(x1, y1, x2, y2, conf, cls, name)


def make_pose(
    nose=(0.25, 0.18),
    left_wrist=(0.30, 0.35),
    right_wrist=(0.20, 0.35),
    left_shoulder=(0.30, 0.30),
    right_shoulder=(0.20, 0.30),
    confidence=0.9,
) -> Pose:
    """Build a Pose with the keypoints the behavior detectors use."""
    kps = [Keypoint(0.0, 0.0, 0.0)] * 17
    kps[0] = Keypoint(nose[0], nose[1], confidence)  # nose
    kps[1] = Keypoint(nose[0] + 0.02, nose[1] - 0.02, confidence)  # left eye
    kps[2] = Keypoint(nose[0] - 0.02, nose[1] - 0.02, confidence)  # right eye
    kps[5] = Keypoint(left_shoulder[0], left_shoulder[1], confidence)
    kps[6] = Keypoint(right_shoulder[0], right_shoulder[1], confidence)
    kps[7] = Keypoint(left_wrist[0], left_wrist[1] - 0.06, confidence)  # left elbow
    kps[8] = Keypoint(right_wrist[0], right_wrist[1] - 0.06, confidence)  # right elbow
    kps[9] = Keypoint(left_wrist[0], left_wrist[1], confidence)
    kps[10] = Keypoint(right_wrist[0], right_wrist[1], confidence)
    kps[11] = Keypoint(left_shoulder[0], left_shoulder[1] + 0.25, confidence)  # hips
    kps[12] = Keypoint(right_shoulder[0], right_shoulder[1] + 0.25, confidence)
    return Pose(keypoints=kps, bbox=make_bbox())


def make_observation(
    track_id=1,
    frame_index=0,
    timestamp=0.0,
    objects=None,
    pose=None,
    movement=0.0,
    person_box=None,
) -> BehaviorObservation:
    return BehaviorObservation(
        frame_index=frame_index,
        timestamp=timestamp,
        track_id=track_id,
        person_box=person_box or make_box(),
        keypoints=pose,
        person_score=0.9,
        objects=objects or [],
        track_movement=movement,
        frame_width=64,
        frame_height=48,
    )


@pytest.fixture
def observation_factory():
    return make_observation


@pytest.fixture
def pose_factory():
    return make_pose


@pytest.fixture
def box_factory():
    return make_box


def object_tuple(box, name, score=0.9):
    return (box, name, score)


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_dict() -> dict:
    return {
        "models": {
            "person_detector": {"name": "yolov8n", "confidence_threshold": 0.5, "device": "cpu"},
            "object_detector": {"name": "yolov8n", "confidence_threshold": 0.4, "device": "cpu",
                                "target_classes": [39, 41, 67]},
            "pose_estimator": {"name": "yolov8n-pose", "confidence_threshold": 0.5, "device": "cpu"},
        },
        "tracking": {"tracker": "bytetrack", "max_age": 30, "min_hits": 3, "iou_threshold": 0.3},
        "behavior": {
            "drinking": {
                "enabled": True,
                "min_duration_seconds": 1.5,
                "confidence_threshold": 0.6,
                "cooldown_seconds": 5.0,
                "min_consecutive_frames": 5,
            },
            "sleeping": {
                "enabled": True,
                "min_duration_seconds": 10.0,
                "confidence_threshold": 0.65,
                "cooldown_seconds": 15.0,
                "min_consecutive_frames": 30,
            },
            "phone_usage": {
                "enabled": True,
                "min_duration_seconds": 2.0,
                "confidence_threshold": 0.6,
                "cooldown_seconds": 5.0,
                "min_consecutive_frames": 8,
            },
        },
        "detection": {"global_confidence_threshold": 0.5, "frame_skip": 1},
        "reporting": {"event_cooldown_seconds": 5.0, "min_event_duration_seconds": 1.0},
    }


@pytest.fixture
def config_file(tmp_path, config_dict):
    import yaml

    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_dict, handle)
    return str(path)