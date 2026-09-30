"""
LibraryGuard - Evaluation metrics.

Frame-level classification metrics (precision/recall/F1/confusion matrix)
plus event-level detection metrics (event matching by temporal IoU), and
inference-speed benchmarking.

These functions compute metrics from supplied predictions/labels -- they
never invent numbers. Reported metrics in the docs clearly state whether
they come from training runs or are unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Frame-level classification metrics
# ---------------------------------------------------------------------------


@dataclass
class ClassificationReport:
    labels: List[str]
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1: Dict[str, float] = field(default_factory=dict)
    support: Dict[str, int] = field(default_factory=dict)
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    false_positive_rate: Dict[str, float] = field(default_factory=dict)
    false_negative_rate: Dict[str, float] = field(default_factory=dict)
    confusion: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "per_class": {
                label: {
                    "precision": round(self.precision.get(label, 0.0), 4),
                    "recall": round(self.recall.get(label, 0.0), 4),
                    "f1": round(self.f1.get(label, 0.0), 4),
                    "support": int(self.support.get(label, 0)),
                    "false_positive_rate": round(self.false_positive_rate.get(label, 0.0), 4),
                    "false_negative_rate": round(self.false_negative_rate.get(label, 0.0), 4),
                }
                for label in self.labels
            },
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
        }


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """Confusion matrix where rows = true, cols = predicted."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            matrix[t, p] += 1
    return matrix


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
) -> ClassificationReport:
    """Compute per-class precision/recall/F1/FPR/FNR plus aggregates."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    n = len(labels)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same length")
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute metrics on empty inputs")

    matrix = confusion_matrix(y_true_arr, y_pred_arr, n)
    report = ClassificationReport(labels=list(labels), confusion=matrix)
    total = int(matrix.sum())
    report.accuracy = float(np.trace(matrix) / total) if total else 0.0

    f1s, supports_weighted = [], []
    for idx, label in enumerate(labels):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        tn = float(matrix.sum() - tp - fp - fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

        report.precision[label] = precision
        report.recall[label] = recall
        report.f1[label] = f1
        report.support[label] = int(matrix[idx, :].sum())
        report.false_positive_rate[label] = fpr
        report.false_negative_rate[label] = fnr

        f1s.append(f1)
        supports_weighted.append(report.support[label])

    report.macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    total_support = sum(supports_weighted)
    if total_support > 0:
        report.weighted_f1 = float(
            sum(f * s for f, s in zip(f1s, supports_weighted)) / total_support
        )
    return report


# ---------------------------------------------------------------------------
# Event-level metrics (temporal detection)
# ---------------------------------------------------------------------------


@dataclass
class EventLevelMetrics:
    event_types: List[str]
    matches: Dict[str, int] = field(default_factory=dict)
    false_positives: Dict[str, int] = field(default_factory=dict)
    false_negatives: Dict[str, int] = field(default_factory=dict)
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1: Dict[str, float] = field(default_factory=dict)
    mean_temporal_iou: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_event_type": {
                t: {
                    "matches": self.matches.get(t, 0),
                    "false_positives": self.false_positives.get(t, 0),
                    "false_negatives": self.false_negatives.get(t, 0),
                    "precision": round(self.precision.get(t, 0.0), 4),
                    "recall": round(self.recall.get(t, 0.0), 4),
                    "f1": round(self.f1.get(t, 0.0), 4),
                    "mean_temporal_iou": round(self.mean_temporal_iou.get(t, 0.0), 4),
                }
                for t in self.event_types
            },
            "micro_precision": round(
                sum(self.matches.values())
                / max(1, sum(self.matches.values()) + sum(self.false_positives.values())),
                4,
            ),
            "micro_recall": round(
                sum(self.matches.values())
                / max(1, sum(self.matches.values()) + sum(self.false_negatives.values())),
                4,
            ),
        }


def _temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def compute_event_metrics(
    pred_events: Sequence[Dict[str, Any]],
    gt_events: Sequence[Dict[str, Any]],
    iou_threshold: float = 0.3,
    event_types: Optional[Sequence[str]] = None,
) -> EventLevelMetrics:
    """
    Match predicted events to ground-truth events by temporal IoU.

    Each event dict must contain: ``event`` (type), ``time_start``, ``time_end``
    (seconds), optionally ``person_id`` (ignored for matching, since ground
    truth track IDs are not comparable across runs).
    """
    if event_types is None:
        event_types = sorted(
            {e.get("event", "") for e in pred_events} | {e.get("event", "") for e in gt_events}
        )
    metrics = EventLevelMetrics(event_types=list(event_types))

    for etype in metrics.event_types:
        preds = [e for e in pred_events if e.get("event") == etype]
        gts = [e for e in gt_events if e.get("event") == etype]

        used_gt = set()
        matches = 0
        iou_sum = 0.0
        for pred in sorted(preds, key=lambda e: -float(e.get("confidence", 0.0))):
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(gts):
                if idx in used_gt:
                    continue
                overlap = _temporal_iou(
                    (float(pred["time_start"]), float(pred["time_end"])),
                    (float(gt["time_start"]), float(gt["time_end"])),
                )
                if overlap > best_iou:
                    best_iou, best_idx = overlap, idx
            if best_iou >= iou_threshold and best_idx >= 0:
                used_gt.add(best_idx)
                matches += 1
                iou_sum += best_iou

        fp = len(preds) - matches
        fn = len(gts) - matches
        precision = matches / (matches + fp) if (matches + fp) > 0 else 0.0
        recall = matches / (matches + fn) if (matches + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.matches[etype] = matches
        metrics.false_positives[etype] = fp
        metrics.false_negatives[etype] = fn
        metrics.precision[etype] = precision
        metrics.recall[etype] = recall
        metrics.f1[etype] = f1
        metrics.mean_temporal_iou[etype] = iou_sum / matches if matches > 0 else 0.0

    return metrics


# ---------------------------------------------------------------------------
# Speed benchmarking
# ---------------------------------------------------------------------------


def fps_benchmark(
    fn: Callable[[Any], Any],
    sample_input: Any,
    warmup: int = 3,
    runs: int = 20,
) -> Dict[str, float]:
    """
    Benchmark a callable (e.g. model inference) and return timing statistics.

    Returns dict with mean_fps, std_fps, mean_latency_ms, p95_latency_ms.
    """
    for _ in range(max(0, warmup)):
        fn(sample_input)

    latencies: List[float] = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        fn(sample_input)
        latencies.append(time.perf_counter() - start)

    arr = np.asarray(latencies, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "mean_fps": 1.0 / mean if mean > 0 else 0.0,
        "std_fps": float((1.0 / arr).std()) if arr.size > 1 else 0.0,
        "mean_latency_ms": mean * 1000.0,
        "p95_latency_ms": float(np.percentile(arr, 95)) * 1000.0,
        "runs": float(arr.size),
    }