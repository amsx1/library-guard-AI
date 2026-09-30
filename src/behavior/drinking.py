"""
LibraryGuard - Drinking behavior detector.

Drinking requires *joint* evidence, never a single cue:

1. a bottle/cup-like object is detected and associated with the person
2. a hand (wrist keypoint) or the object itself is near the face/mouth
3. the object is roughly within the person's bounding box region
4. pose evidence is consistent (hand-to-face), not just object presence

A person simply holding a bottle scores low; only a bottle/cup brought to the
face persists into a high-confidence score.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from ..utils.constants import (
    DRINKING_OBJECT_CLASSES,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_WRIST,
    clamp01,
)
from .base import BehaviorDetector, BehaviorObservation, BehaviorScore

# COCO-pose has no explicit mouth landmark; the nose is used as the
# mouth/face proxy for hand-to-face distance checks.


class DrinkingDetector(BehaviorDetector):
    key = "drinking"
    display_name = "Drinking"

    def __init__(
        self,
        hand_to_mouth_distance_threshold: float = 0.15,
        object_near_face_threshold: float = 0.2,
    ):
        self.hand_to_mouth_distance_threshold = hand_to_mouth_distance_threshold
        self.object_near_face_threshold = object_near_face_threshold

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _face_point(observation: BehaviorObservation) -> Optional[Tuple[float, float]]:
        pose = observation.keypoints
        if pose is None:
            return None
        nose = pose.get_keypoint(KP_NOSE)
        if nose is None:
            return None
        return (nose.x, nose.y)

    @staticmethod
    def _wrist_points(observation: BehaviorObservation):
        pose = observation.keypoints
        if pose is None:
            return []
        wrists = []
        for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
            kp = pose.get_keypoint(idx)
            if kp is not None:
                wrists.append((kp.x, kp.y))
        return wrists

    @staticmethod
    def _drinking_objects(observation: BehaviorObservation):
        result = []
        for obj in observation.objects:
            box, class_name, score = obj
            if class_name in DRINKING_OBJECT_CLASSES.values():
                result.append((box, class_name, score))
        return result

    def _mouth_distance(self, face, wrist) -> float:
        return math.hypot(face[0] - wrist[0], face[1] - wrist[1])

    # -- main scoring -------------------------------------------------------

    def score(self, observation: BehaviorObservation) -> BehaviorScore:
        objects = self._drinking_objects(observation)
        if not objects:
            return BehaviorScore(self.key, 0.0, {"object_present": 0.0}, "no drinking vessel detected")

        # Prefer the drinking object closest to the face.
        face = self._face_point(observation)
        person_box = observation.person_box

        best_object_conf = 0.0
        best_object_near_face = 0.0
        best_object_in_person = 0.0

        for box, class_name, obj_score in objects:
            center = box.center
            # Objects should belong to the person's region.
            inside = person_box.contains_point(center) or person_box.distance_to_point(center) < 0.1
            in_person = 1.0 if inside else 0.0

            if face is not None:
                dist_face = math.hypot(face[0] - center.x, face[1] - center.y)
                near_face = clamp01(1.0 - dist_face / max(self.object_near_face_threshold * 3.0, 1e-6))
            else:
                # No pose available: rely on object being inside person box and
                # positioned in the upper half of it (head height).
                upper = center.y < person_box.y1 + 0.6 * (person_box.y2 - person_box.y1)
                near_face = 0.55 if upper else 0.15

            conf = obj_score * (0.4 + 0.3 * in_person + 0.3 * near_face)
            if conf > best_object_conf:
                best_object_conf = conf
                best_object_near_face = near_face
                best_object_in_person = in_person

        # Hand-to-face evidence from pose.
        hand_to_face = 0.0
        wrists = self._wrist_points(observation)
        if face is not None and wrists:
            min_dist = min(self._mouth_distance(face, w) for w in wrists)
            hand_to_face = clamp01(1.0 - min_dist / max(self.hand_to_mouth_distance_threshold * 2.5, 1e-6))
        elif face is None:
            hand_to_face = 0.25  # weak prior without pose

        # Combine: both object evidence AND hand/face proximity must hold.
        # Multiplicative blend punishes missing pillars instead of averaging
        # them away (a bottle on a table must NOT trigger drinking).
        object_component = clamp01(best_object_conf)
        if object_component <= 0.0 or (hand_to_face <= 0.05 and best_object_near_face <= 0.1):
            combined = 0.0
        else:
            combined = clamp01(
                0.45 * object_component
                + 0.35 * hand_to_face
                + 0.20 * best_object_near_face
            )
            # Hard gate: if the vessel is nowhere near the face/hand, cap low.
            if hand_to_face < 0.15 and best_object_near_face < 0.3:
                combined = min(combined, 0.3)

        return BehaviorScore(
            self.key,
            combined,
            {
                "object_present": object_component,
                "hand_to_face": hand_to_face,
                "object_near_face": best_object_near_face,
                "object_in_person": best_object_in_person,
            },
        )