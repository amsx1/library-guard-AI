"""
LibraryGuard - Behavior recognition package.

Modular temporal behavior detectors. To add a new behavior, subclass
:class:`~libraryguard.src.behavior.base.BehaviorDetector` and register it in
:func:`detector_registry` in ``engine.py``.
"""

from .base import (
    BehaviorDetector,
    BehaviorEvidence,
    BehaviorObservation,
    BehaviorScore,
    evidence_label,
)
from .drinking import DrinkingDetector
from .engine import (
    BehaviorEngine,
    BehaviorEvent,
    LiveStatus,
    PerBehaviorConfig,
    build_engine_from_config,
    default_configs,
    default_detectors,
    detector_registry,
)
from .phone_usage import PhoneUsageDetector
from .sleeping import SleepingDetector

__all__ = [
    "BehaviorDetector",
    "BehaviorEvidence",
    "BehaviorObservation",
    "BehaviorScore",
    "BehaviorEngine",
    "BehaviorEvent",
    "LiveStatus",
    "PerBehaviorConfig",
    "DrinkingDetector",
    "SleepingDetector",
    "PhoneUsageDetector",
    "evidence_label",
    "default_detectors",
    "default_configs",
    "detector_registry",
    "build_engine_from_config",
]