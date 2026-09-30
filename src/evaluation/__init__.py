"""
LibraryGuard - Evaluation package.
"""

from .metrics import (
    ClassificationReport,
    EventLevelMetrics,
    compute_classification_metrics,
    compute_event_metrics,
    confusion_matrix,
    fps_benchmark,
)
from .plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_event_timeline,
)

__all__ = [
    "ClassificationReport",
    "EventLevelMetrics",
    "compute_classification_metrics",
    "compute_event_metrics",
    "confusion_matrix",
    "fps_benchmark",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_event_timeline",
]