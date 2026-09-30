#!/usr/bin/env python3
"""
LibraryGuard - Evaluate a model or an event stream.

Two modes:

1. ``--checkpoint``: evaluate a trained temporal classifier on a directory
   of .npz sequences -> frame/clip-level classification metrics.

2. ``--predictions/--ground-truth``: evaluate predicted events against
   ground-truth event JSON -> event-level precision/recall/F1.

    python scripts/evaluate.py --checkpoint models/temporal_classifier.pt \
                               --data-dir data/sequences/test

    python scripts/evaluate.py --predictions outputs/events.json \
                               --ground-truth data/gt/events.json

All numbers produced are computed from the supplied data. If no trained
model or ground truth exists, this script reports that clearly instead of
inventing metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

EVENT_LABELS = ["drinking", "sleeping", "phone_usage"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LibraryGuard models/events")
    parser.add_argument("--checkpoint", default=None, help="Path to temporal classifier checkpoint")
    parser.add_argument("--data-dir", default=None, help="Directory of .npz sequences to evaluate")
    parser.add_argument("--predictions", default=None, help="Predicted events JSON")
    parser.add_argument("--ground-truth", default=None, help="Ground-truth events JSON")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--out", default="outputs/evaluation")
    parser.add_argument("--plots", action="store_true")
    return parser


def evaluate_checkpoint(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required for checkpoint evaluation.")
        return 1

    from src.evaluation.metrics import compute_classification_metrics
    from src.models.temporal import TemporalBehaviorClassifier, BehaviorSequenceDataset

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        logger.error("Train a model first with scripts/train.py, or omit --checkpoint.")
        return 1

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    model = TemporalBehaviorClassifier(
        hidden_dim=cfg.get("hidden_dim", 96),
        num_layers=cfg.get("num_layers", 2),
        num_classes=cfg.get("num_classes", 3),
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    if not args.data_dir:
        logger.error("--data-dir is required with --checkpoint")
        return 1

    dataset = BehaviorSequenceDataset(args.data_dir, seq_len=cfg.get("seq_len", 64))
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, label in dataset:
            logits = model(features.unsqueeze(0))
            pred = int(logits.argmax(dim=1).item())
            y_true.append(int(label.item()))
            y_pred.append(pred)

    labels = EVENT_LABELS + ["other"][: max(0, max(y_true + y_pred) + 1 - len(EVENT_LABELS))]
    report = compute_classification_metrics(y_true, y_pred, labels[: max(y_true + y_pred) + 1])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "classification_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)

    print(json.dumps(report.to_dict(), indent=2))

    if args.plots:
        from src.evaluation.plots import plot_confusion_matrix

        n = len(labels[: max(y_true + y_pred) + 1])
        plot_confusion_matrix(
            report.confusion,
            labels[:n],
            output_path=out_dir / "confusion_matrix.png",
            normalize=True,
        )
        logger.info("Confusion matrix saved to %s", out_dir / "confusion_matrix.png")
    return 0


def evaluate_events(args: argparse.Namespace) -> int:
    from src.evaluation.metrics import compute_event_metrics

    pred_path = Path(args.predictions) if args.predictions else None
    gt_path = Path(args.ground_truth) if args.ground_truth else None

    if pred_path is None or gt_path is None:
        logger.error("Both --predictions and --ground-truth are required for event evaluation")
        return 1
    for p in (pred_path, gt_path):
        if not p.exists():
            logger.error("File not found: %s", p)
            return 1

    with open(pred_path, "r", encoding="utf-8") as handle:
        pred_payload = json.load(handle)
    with open(gt_path, "r", encoding="utf-8") as handle:
        gt_payload = json.load(handle)

    pred_events = pred_payload.get("events", pred_payload if isinstance(pred_payload, list) else [])
    gt_events = gt_payload.get("events", gt_payload if isinstance(gt_payload, list) else [])

    # Normalize event dicts to include time_start/time_end.
    def normalize(events):
        out = []
        for e in events:
            if "time_start" not in e:
                # accept "timestamp" + "duration"
                e = dict(e)
                e["time_start"] = _timestamp_to_seconds(e.get("timestamp", "00:00:00"))
                e["time_end"] = e["time_start"] + float(e.get("duration", 0.0))
            out.append(e)
        return out

    pred_events = normalize(pred_events)
    gt_events = normalize(gt_events)

    metrics = compute_event_metrics(
        pred_events, gt_events, iou_threshold=args.iou_threshold, event_types=EVENT_LABELS
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "event_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)
    print(json.dumps(metrics.to_dict(), indent=2))
    return 0


def _timestamp_to_seconds(ts: str) -> float:
    parts = str(ts).split(":")
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(parts_f) == 3:
        return parts_f[0] * 3600 + parts_f[1] * 60 + parts_f[2]
    if len(parts_f) == 2:
        return parts_f[0] * 60 + parts_f[1]
    return parts_f[0] if parts_f else 0.0


def main() -> int:
    args = build_parser().parse_args()
    if args.checkpoint:
        return evaluate_checkpoint(args)
    if args.predictions or args.ground_truth:
        return evaluate_events(args)
    logger.error("Specify --checkpoint or --predictions/--ground-truth")
    build_parser().print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())