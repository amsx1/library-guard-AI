"""
LibraryGuard - Models package.

Trainable temporal behavior classifier (PyTorch GRU over per-person
feature sequences). See ``docs/architecture.md`` for how this complements
the rule-based temporal evidence engine.
"""

from .temporal import (
    BehaviorSequenceDataset,
    FeatureExtractor,
    TemporalBehaviorClassifier,
    extract_feature_sequence,
)

__all__ = [
    "TemporalBehaviorClassifier",
    "BehaviorSequenceDataset",
    "FeatureExtractor",
    "extract_feature_sequence",
]