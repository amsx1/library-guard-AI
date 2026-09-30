"""
LibraryGuard - Evaluation visualizations.

Matplotlib-based plots for confusion matrices, precision-recall curves,
and event timelines. All functions save to disk AND return the figure so
they work both in scripts and notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: Optional[Union[str, Path]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
):
    plt = _get_pyplot()
    mat = np.asarray(matrix, dtype=np.float64)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat = mat / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = mat.max() / 2.0 if mat.size else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if value > thresh else "black", fontsize=9)

    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_precision_recall_curve(
    precision: Sequence[float],
    recall: Sequence[float],
    output_path: Optional[Union[str, Path]] = None,
    label: str = "model",
    title: str = "Precision-Recall Curve",
):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, marker=".", label=label)
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=[0, 1], ylim=[0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_event_timeline(
    events: Sequence[Dict[str, Any]],
    video_duration: float,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Detected Event Timeline",
):
    """
    Horizontal timeline of detected events.

    Each event dict needs: ``event``, ``time_start``, ``time_end``,
    optionally ``person_id`` and ``confidence``.
    """
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.6 * max(1, len(events)))))

    color_map = {"drinking": "#4FC3F7", "sleeping": "#F48FB1", "phone_usage": "#FFB74D"}
    y_positions = []
    labels_y = []
    for idx, event in enumerate(events):
        start = float(event.get("time_start", 0.0))
        end = float(event.get("time_end", start))
        etype = event.get("event", "?")
        pid = event.get("person_id", "?")
        conf = float(event.get("confidence", 0.0))
        ax.barh(
            idx,
            max(end - start, 0.1),
            left=start,
            height=0.55,
            color=color_map.get(etype, "#90A4AE"),
            edgecolor="black",
            linewidth=0.5,
        )
        ax.text(
            start,
            idx,
            f" {etype} P#{pid} {conf * 100:.0f}%",
            va="center",
            fontsize=8,
        )
        y_positions.append(idx)
        labels_y.append(f"{event.get('timestamp', '')}")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set(xlabel="Time (s)", title=title, xlim=[0, max(video_duration, 1.0)])
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Training History",
):
    """Plot loss/accuracy curves from a training run (dict of metric -> values)."""
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in history.items():
        ax.plot(values, label=name)
    ax.set(xlabel="Epoch", title=title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig