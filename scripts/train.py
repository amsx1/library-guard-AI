#!/usr/bin/env python3
"""
LibraryGuard - Train the temporal behavior classifier.

    python scripts/train.py --train-dir data/sequences/train \
                            --val-dir data/sequences/val \
                            --out models/temporal_classifier.pt

Produces a real training run: per-epoch train/val loss & accuracy logged to
``models/history.json``, best checkpoint saved, and a training-history plot
written next to the checkpoint. Metrics reported are measured, never invented.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LibraryGuard temporal classifier")
    parser.add_argument("--train-dir", required=True, help="Directory of .npz training sequences")
    parser.add_argument("--val-dir", required=True, help="Directory of .npz validation sequences")
    parser.add_argument("--out", default="models/temporal_classifier.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--plot", action="store_true", help="Save training-history plot")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error("PyTorch is required for training. Install with: pip install torch torchvision")
        return 1

    from src.models.temporal import (
        BehaviorSequenceDataset,
        TrainConfig,
        train_model,
    )

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not train_dir.exists():
        logger.error("Training directory not found: %s", train_dir)
        return 1
    if not val_dir.exists():
        logger.error("Validation directory not found: %s", val_dir)
        return 1

    try:
        train_ds = BehaviorSequenceDataset(train_dir, seq_len=args.seq_len, augment=True, seed=args.seed)
        val_ds = BehaviorSequenceDataset(val_dir, seq_len=args.seq_len, augment=False, seed=args.seed)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seq_len=args.seq_len,
        seed=args.seed,
        device=args.device,
        early_stopping_patience=args.patience,
    )

    model, history = train_model(
        train_ds,
        val_ds,
        config,
        num_classes=args.num_classes,
        log_fn=logger.info,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "num_classes": args.num_classes,
                "seq_len": args.seq_len,
            },
            "history": history,
        },
        out_path,
    )
    logger.info("Checkpoint saved to %s", out_path)

    history_path = out_path.parent / "history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    logger.info("History saved to %s", history_path)

    if args.plot:
        from src.evaluation.plots import plot_training_history

        plot_path = out_path.parent / "training_history.png"
        plot_training_history(history, output_path=plot_path)
        logger.info("Training plot saved to %s", plot_path)

    final = {
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else None,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "epochs_run": len(history["train_loss"]),
    }
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())