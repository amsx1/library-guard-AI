"""
LibraryGuard - Phone usage behavior detector.

Phone usage requires joint evidence:

1. a cell phone is detected and associated with the person
2. the phone is near the person's hands or face (not lying on a table)
3. pose is consistent with phone use (hand raised / head angled down)
4. evidence persists across frames (handled by the BehaviorEngine)

A phone merely visible in the scene -- on a desk, in a bag -- scores near
zero. This is the core anti-false-positive rule for this behavior.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ..utils.constants import (
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_WRIST,
    PHONE_OBJECT_CLASSES,
    clamp01,
)
from .base import BehaviorDetector, BehaviorObservation, BehaviorScore


class PhoneUsageDetector(BehaviorDetector):
    key = "phone_usage"
    display_name = "Phone Usage"

    def __init__(
        self,
        phone_near_hand_threshold: float = 0.15,
        phone_near_face_threshold: float = 0.25,
        looking_down_angle_threshold: float = 30.0,
    ):
        self.phone_near_hand_threshold = phone_near_hand_threshold
        self.phone_near_face_threshold = phone_near_face_threshold
        self.looking_down_angle_threshold = looking_down_angle_threshold

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _phones(observation: BehaviorObservation):
        result = []
        for box, class_name, score in observation.objects:
            if class_name in PHONE_OBJECT_CLASSES.values():
                result.append((box, class_name, score))
        return result

    @staticmethod
    def _face(observation: BehaviorObservation) -> Optional[Tuple[float, float]]:
        pose = observation.keypoints
        if pose is None:
            return None
        nose = pose.get_keypoint(KP_NOSE)
        return (nose.x, nose.y) if nose else None

    @staticmethod
    def _wrists(observation: BehaviorObservation) -> List[Tuple[float, float]]:
        pose = observation.keypoints
        if pose is None:
            return []
        result = []
        for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
            kp = pose.get_keypoint(idx)
            if kp is not None:
                result.append((kp.x, kp.y))
        return result

    def _head_down_score(self, observation: BehaviorObservation) -> float:
        """Head angled down toward a phone held at chest height.

        Uses the neck->nose tilt relative to vertical: upright => ~0 deg,
        reading a phone at chest height => tens of degrees forward tilt.
        """
        pose = observation.keypoints
        if pose is None:
            return 0.3  # weak prior without pose
        nose = pose.get_keypoint(KP_NOSE)
        ls = pose.get_keypoint(5)  # KP_LEFT_SHOULDER
        rs = pose.get_keypoint(6)  # KP_RIGHT_SHOULDER
        if nose is None or not (ls and rs):
            return 0.3
        neck_x = (ls.x + rs.x) / 2.0
        neck_y = (ls.y + rs.y) / 2.0
        dx = nose.x - neck_x
        dy_up = neck_y - nose.y  # positive when nose sits above the shoulders
        if dy_up <= 0:
            # Nose at/below shoulder line: strongly folded forward.
            tilt_deg = 75.0
        else:
            tilt_deg = math.degrees(math.atan2(abs(dx), dy_up))
        return clamp01(tilt_deg / max(self.looking_down_angle_threshold, 1e-6))

    # -- main scoring -------------------------------------------------------

    def score(self, observation: BehaviorObservation) -> BehaviorScore:
        phones = self._phones(observation)
        if not phones:
            return BehaviorScore(self.key, 0.0, {"phone_present": 0.0}, "no phone detected")

        person_box = observation.person_box
        face = self._face(observation)
        wrists = self._wrists(observation)

        best_conf = 0.0
        best_hand = 0.0
        best_face = 0.0
        best_owned = 0.0

        for box, class_name, obj_score in phones:
            center = box.center
            # Is the phone held/owned by this person?
            inside = person_box.contains_point(center)
            near_person = person_box.distance_to_point(center) < 0.05
            owned = 1.0 if (inside or near_person) else 0.0

            # Distance to hands.
            if wrists:
                min_hand = min(math.hypot(center.x - wx, center.y - wy) for wx, wy in wrists)
                hand_prox = clamp01(1.0 - min_hand / max(self.phone_near_hand_threshold * 3.0, 1e-6))
            else:
                # Without pose, being inside the person box is weak evidence of
                # being held.
                hand_prox = 0.5 if inside else 0.0

            # Distance to face.
            if face is not None:
                min_face = math.hypot(center.x - face[0], center.y - face[1])
                face_prox = clamp01(1.0 - min_face / max(self.phone_near_face_threshold * 3.0, 1e-6))
            else:
                upper = center.y < person_box.y1 + 0.55 * (person_box.y2 - person_box.y1)
                face_prox = 0.5 if (inside and upper) else 0.1

            conf = obj_score * (0.35 * owned + 0.40 * hand_prox + 0.25 * face_prox)
            if conf > best_conf:
                best_conf = conf
                best_hand = hand_prox
                best_face = face_prox
                best_owned = owned

        head_down = self._head_down_score(observation)

        # Anti-false-positive gate: a phone lying around (not held, not near
        # face) must not trigger phone usage even if clearly detected.
        if best_owned < 0.5 or (best_hand < 0.1 and best_face < 0.1):
            combined = min(best_conf, 0.2)
        else:
            combined = clamp01(0.55 * best_conf + 0.25 * head_down + 0.20 * max(best_hand, best_face))

        return BehaviorScore(
            self.key,
            combined,
            {
                "phone_present": best_conf,
                "phone_held": best_owned,
                "hand_proximity": best_hand,
                "face_proximity": best_face,
                "head_down": head_down,
            },
        )