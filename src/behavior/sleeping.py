"""
LibraryGuard - Sleeping behavior detector.

Sleeping is a *temporal-postural* behavior: a single frame of a lowered head
is not enough. This detector scores per-frame postural evidence (head-down
angle, slumped shoulders, low motion) and the shared BehaviorEngine requires
the evidence to persist for a configurable duration (default 10 s) before an
event is emitted.

CCTV footage rarely permits reliable eye-closure analysis, so eye state is
explicitly NOT used as a primary cue (documented limitation).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from ..utils.constants import (
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_NOSE,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    clamp01,
)
from .base import BehaviorDetector, BehaviorObservation, BehaviorScore


class SleepingDetector(BehaviorDetector):
    key = "sleeping"
    display_name = "Possible Sleeping"

    def __init__(
        self,
        head_down_angle_threshold: float = 45.0,
        low_movement_threshold: float = 0.02,
    ):
        self.head_down_angle_threshold = head_down_angle_threshold
        self.low_movement_threshold = low_movement_threshold

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _neck(observation: BehaviorObservation) -> Optional[Tuple[float, float]]:
        pose = observation.keypoints
        if pose is None:
            return None
        ls = pose.get_keypoint(KP_LEFT_SHOULDER)
        rs = pose.get_keypoint(KP_RIGHT_SHOULDER)
        if ls and rs:
            return ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)
        if ls is not None:
            return (ls.x, ls.y)
        if rs is not None:
            return (rs.x, rs.y)
        return None

    @staticmethod
    def _head(observation: BehaviorObservation) -> Optional[Tuple[float, float]]:
        pose = observation.keypoints
        if pose is None:
            return None
        nose = pose.get_keypoint(KP_NOSE)
        if nose is None:
            return None
        return (nose.x, nose.y)

    def _head_tilt_degrees(self, observation: BehaviorObservation) -> Optional[float]:
        """
        Angle between the neck->head vector and the vertical axis.

        ~0 deg = head upright; approaching 90 deg = head resting sideways or
        fully down on a surface.
        """
        head = self._head(observation)
        neck = self._neck(observation)
        if head is None or neck is None:
            return None
        dx = head[0] - neck[0]
        dy = head[1] - neck[1]
        # Image coordinates: +y is downward. Vertical (up) = -y.
        angle = math.degrees(math.atan2(abs(dx), max(-dy, 1e-6) if dy < 0 else abs(dy) + 1e-6))
        if dy > 0:  # head BELOW shoulders -> strongly folded forward
            angle = max(angle, 60.0)
        return angle

    def _head_below_shoulders(self, observation: BehaviorObservation) -> bool:
        head = self._head(observation)
        neck = self._neck(observation)
        if head is None or neck is None:
            return False
        return head[1] > neck[1]  # nose lower than shoulder line (y grows downward)

    @staticmethod
    def _torso_slump(observation: BehaviorObservation) -> float:
        """Ratio of head-to-hip vertical distance vs shoulder span (slumped = small)."""
        pose = observation.keypoints
        if pose is None:
            return 0.0
        nose = pose.get_keypoint(KP_NOSE)
        ls = pose.get_keypoint(KP_LEFT_SHOULDER)
        rs = pose.get_keypoint(KP_RIGHT_SHOULDER)
        lh = pose.get_keypoint(KP_LEFT_HIP)
        rh = pose.get_keypoint(KP_RIGHT_HIP)
        if not nose or not (ls and rs) or not (lh and rh):
            return 0.0
        shoulder_y = (ls.y + rs.y) / 2.0
        hip_y = (lh.y + rh.y) / 2.0
        torso = abs(hip_y - shoulder_y)
        if torso < 1e-6:
            return 0.0
        head_drop = clamp01((nose.y - shoulder_y) / torso)
        return head_drop

    # -- main scoring -------------------------------------------------------

    def score(self, observation: BehaviorObservation) -> BehaviorScore:
        angle = self._head_tilt_degrees(observation)

        # Postural evidence: head tilt toward/past the threshold.
        if angle is None:
            posture = 0.2  # weak prior when pose unavailable
            angle_component = 0.2
        else:
            angle_component = clamp01((angle - 15.0) / max(self.head_down_angle_threshold, 1e-6))
            posture = angle_component

        head_low = 1.0 if self._head_below_shoulders(observation) else 0.0
        slump = self._torso_slump(observation)

        # Movement evidence: sleeping implies stillness. The tracker measures
        # normalized centroid movement over the recent window.
        movement = clamp01(observation.track_movement / max(self.low_movement_threshold * 4.0, 1e-6))
        stillness = clamp01(1.0 - movement)

        # Combine. Posture dominates; stillness amplifies but cannot rescue a
        # clearly awake upright posture.
        combined = clamp01(
            0.40 * posture
            + 0.25 * head_low
            + 0.15 * slump
            + 0.20 * stillness
        )
        if angle is not None and angle < 20.0 and head_low == 0.0:
            combined = min(combined, 0.25)  # upright person is not sleeping

        return BehaviorScore(
            self.key,
            combined,
            {
                "posture": posture,
                "head_tilt_deg": -1.0 if angle is None else angle,
                "head_below_shoulders": head_low,
                "torso_slump": slump,
                "stillness": stillness,
            },
        )