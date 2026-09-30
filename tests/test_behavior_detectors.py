"""Tests: behavior detectors (drinking / sleeping / phone usage evidence logic)."""

from __future__ import annotations



from src.behavior.drinking import DrinkingDetector
from src.behavior.phone_usage import PhoneUsageDetector
from src.behavior.sleeping import SleepingDetector
from tests.conftest import (
    make_box,
    make_observation,
    make_pose,
    object_tuple,
)


# ---------------------------------------------------------------------------
# Drinking
# ---------------------------------------------------------------------------


def test_drinking_no_object_scores_zero():
    detector = DrinkingDetector()
    obs = make_observation(objects=[], pose=make_pose())
    assert detector.score(obs).confidence == 0.0


def test_drinking_bottle_on_table_scores_low():
    """A bottle far from face/hands (e.g. on a table) must NOT look like drinking."""
    detector = DrinkingDetector()
    bottle = make_box(0.60, 0.75, 0.68, 0.90)  # bottom of frame, away from person
    obs = make_observation(
        objects=[object_tuple(bottle, "bottle", 0.9)],
        pose=make_pose(),  # hands at chest height, face at top
        person_box=make_box(0.1, 0.1, 0.5, 0.9),
    )
    assert detector.score(obs).confidence <= 0.35


def test_drinking_hand_to_face_with_cup_scores_high():
    """Cup at the mouth with a hand there -> strong drinking evidence."""
    detector = DrinkingDetector()
    # Face (nose) at (0.25, 0.18); put cup center just below it and wrist next to it.
    cup = make_box(0.21, 0.14, 0.29, 0.22)
    obs = make_observation(
        objects=[object_tuple(cup, "cup", 0.95)],
        pose=make_pose(
            nose=(0.25, 0.18),
            left_wrist=(0.26, 0.20),
            right_wrist=(0.35, 0.40),
        ),
        person_box=make_box(0.15, 0.05, 0.45, 0.95),
    )
    score = detector.score(obs)
    assert score.confidence >= 0.6
    assert score.components["hand_to_face"] > 0.5
    assert score.components["object_present"] > 0.5


def test_drinking_person_holding_bottle_not_at_face_is_low():
    """Holding a bottle at chest height is NOT drinking."""
    detector = DrinkingDetector()
    bottle = make_box(0.22, 0.38, 0.30, 0.52)  # at wrist height, away from face
    obs = make_observation(
        objects=[object_tuple(bottle, "bottle", 0.9)],
        pose=make_pose(
            nose=(0.25, 0.10),
            left_wrist=(0.26, 0.45),
            right_wrist=(0.24, 0.45),
        ),
        person_box=make_box(0.15, 0.02, 0.45, 0.95),
    )
    assert detector.score(obs).confidence <= 0.45


# ---------------------------------------------------------------------------
# Sleeping
# ---------------------------------------------------------------------------


def test_sleeping_upright_moving_person_scores_low():
    detector = SleepingDetector()
    obs = make_observation(
        pose=make_pose(nose=(0.25, 0.10)),  # head well above shoulders
        movement=0.05,
    )
    assert detector.score(obs).confidence <= 0.3


def test_sleeping_head_down_still_person_scores_high():
    """Head tilted down + no movement -> strong sleeping evidence."""
    detector = SleepingDetector()
    # Nose BELOW shoulder line and displaced sideways (resting head).
    obs = make_observation(
        pose=make_pose(
            nose=(0.38, 0.36),
            left_shoulder=(0.30, 0.30),
            right_shoulder=(0.20, 0.30),
        ),
        movement=0.001,
    )
    score = detector.score(obs)
    assert score.confidence >= 0.5
    assert score.components["head_below_shoulders"] == 1.0
    assert score.components["stillness"] > 0.8


def test_sleeping_still_but_upright_is_not_sleeping():
    """A motionless but upright person is not sleeping."""
    detector = SleepingDetector()
    obs = make_observation(
        pose=make_pose(nose=(0.25, 0.10)),
        movement=0.0,
    )
    assert detector.score(obs).confidence <= 0.3


def test_sleeping_without_pose_uses_weak_prior():
    detector = SleepingDetector()
    obs = make_observation(pose=None, movement=0.0)
    score = detector.score(obs)
    assert 0.0 < score.confidence < 0.55  # weak prior, never confident alone


# ---------------------------------------------------------------------------
# Phone usage
# ---------------------------------------------------------------------------


def test_phone_absent_scores_zero():
    detector = PhoneUsageDetector()
    obs = make_observation(objects=[])
    assert detector.score(obs).confidence == 0.0


def test_phone_on_desk_scores_low():
    """A phone lying on a table (outside person box) must NOT trigger."""
    detector = PhoneUsageDetector()
    phone = make_box(0.70, 0.80, 0.78, 0.92)
    obs = make_observation(
        objects=[object_tuple(phone, "cell phone", 0.95)],
        pose=make_pose(),
        person_box=make_box(0.1, 0.1, 0.5, 0.9),
    )
    assert detector.score(obs).confidence <= 0.25


def test_phone_near_face_scores_high():
    """Phone held at the face -> strong phone-usage evidence."""
    detector = PhoneUsageDetector()
    phone = make_box(0.22, 0.14, 0.30, 0.22)
    obs = make_observation(
        objects=[object_tuple(phone, "cell phone", 0.95)],
        pose=make_pose(
            nose=(0.25, 0.18),
            left_wrist=(0.26, 0.22),
            right_wrist=(0.35, 0.45),
        ),
        person_box=make_box(0.15, 0.05, 0.45, 0.95),
    )
    score = detector.score(obs)
    assert score.confidence >= 0.55
    assert score.components["phone_held"] > 0.5


def test_phone_in_hand_at_waist_scores_moderate():
    """Phone held low is likely phone usage but weaker than at face level."""
    detector = PhoneUsageDetector()
    phone = make_box(0.22, 0.55, 0.30, 0.65)
    obs = make_observation(
        objects=[object_tuple(phone, "cell phone", 0.95)],
        pose=make_pose(
            nose=(0.25, 0.12),
            left_wrist=(0.26, 0.60),
            right_wrist=(0.35, 0.55),
        ),
        person_box=make_box(0.15, 0.05, 0.45, 0.95),
    )
    score = detector.score(obs)
    assert 0.3 <= score.confidence <= 0.85  # real signal, but not maxed out