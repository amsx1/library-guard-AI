#!/usr/bin/env python3
"""
LibraryGuard - Dataset preparation.

Reproducible data pipeline:

    python scripts/prepare_data.py validate  --manifest data/manifest.csv
    python scripts/prepare_data.py split     --manifest data/manifest.csv --out data/splits
    python scripts/prepare_data.py extract   --video path.mp4 --out data/frames
    python scripts/prepare_data.py sequences --manifest data/manifest.csv --out data/sequences

The script never downloads datasets implicitly. Use the documented download
procedures in docs/datasets.md (and scripts/download_datasets.sh) first;
this tool then validates, deduplicates, splits and (for the temporal model)
extracts feature sequences -- with group-aware splitting so frames from one
source video never leak across train/val/test.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.dataset import (  # noqa: E402
    DatasetError,
    group_split,
    load_manifest,
    save_manifest,
    split_statistics,
    validate_dataset,
)
from src.preprocessing.video import VideoCaptureError, extract_frames  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")

DEFAULT_ALLOWED_LABELS = {"drinking", "sleeping", "phone_usage", "other", "background"}


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        rows = load_manifest(args.manifest)
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1

    report = validate_dataset(
        rows,
        allowed_labels=set(args.labels.split(",")) if args.labels else DEFAULT_ALLOWED_LABELS,
        base_dir=args.base_dir,
        check_duplicates=not args.no_duplicates,
    )
    print(report.summary())
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        logger.info("Validation report written to %s", args.report)
    return 0 if report.ok else 2


def cmd_split(args: argparse.Namespace) -> int:
    try:
        rows = load_manifest(args.manifest)
        splits = group_split(
            rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            stratify=not args.no_stratify,
        )
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        dest = out_dir / f"{name}.csv"
        save_manifest(split_rows, dest)
        logger.info("Wrote %d rows to %s", len(split_rows), dest)

    stats = split_statistics(splits)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    try:
        written = extract_frames(
            args.video,
            args.out,
            every_n_seconds=args.every,
            image_format=args.format,
            max_frames=args.max_frames,
        )
    except VideoCaptureError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Extracted %d frames to %s", len(written), args.out)
    return 0


def cmd_sequences(args: argparse.Namespace) -> int:
    """
    Extract feature sequences (.npz) for the temporal classifier.

    Runs the detection/tracking/feature pipeline over each video in the
    manifest and stores per-track sequences with their label.
    """
    import numpy as np

    from src.models.temporal import FeatureExtractor
    from src.pipeline import VideoPipeline
    from src.behavior.base import BehaviorObservation

    try:
        rows = load_manifest(args.manifest)
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor()
    written = 0

    for row in rows:
        video_path = Path(row.path)
        if args.base_dir and not video_path.is_absolute():
            video_path = Path(args.base_dir) / video_path
        if not video_path.exists():
            logger.warning("Missing video %s; skipping", video_path)
            continue

        pipeline = VideoPipeline()
        per_track: dict[int, list] = {}

        def collect(frame_index, timestamp, frame, fps=25.0, sink=per_track):
            result = pipeline.process_frame(frame, frame_index, timestamp, fps)
            # Rebuild observations per track from the last frame processing.
            # process_frame already scored them; we re-extract features from
            # its internal observation objects.
            for overlay in result.track_overlays:
                pass
            return result

        try:
            # Lightweight per-frame walk: reuse pipeline internals directly.
            from src.preprocessing.video import VideoReader

            with VideoReader(str(video_path)) as reader:
                for frame_index, timestamp, frame in reader:
                    det = pipeline.detector.detect(frame)

                    boxes = [p.bbox for p in det.persons if p.bbox is not None]
                    scores = [p.bbox.confidence for p in det.persons if p.bbox is not None]
                    tracks = pipeline.tracker.update(boxes, scores)
                    for track in tracks:
                        tb = track.bbox
                        best_pose = None
                        best_iou = 0.0
                        for pose in det.persons:
                            if pose.bbox is None:
                                continue
                            overlap = _box_iou_local(tb, pose.bbox)
                            if overlap > best_iou:
                                best_iou = overlap
                                best_pose = pose
                        objects = [(b, b.class_name, b.confidence) for b in det.objects]
                        obs = BehaviorObservation(
                            frame_index=frame_index,
                            timestamp=timestamp,
                            track_id=track.track_id,
                            person_box=tb,
                            keypoints=best_pose,
                            person_score=track.score,
                            objects=objects,
                            track_movement=track.movement_over_window(),
                        )
                        per_track.setdefault(track.track_id, []).append(
                            extractor.extract(obs)
                        )
        except VideoCaptureError as exc:
            logger.warning("Skipping %s: %s", video_path, exc)
            continue

        for track_id, feats in per_track.items():
            if len(feats) < args.min_frames:
                continue
            matrix = np.stack(feats, axis=0).astype(np.float32)
            dest = out_dir / f"{video_path.stem}_track{track_id}.npz"
            np.savez_compressed(dest, features=matrix, label=np.int64(_label_index(row.label)))
            written += 1

    logger.info("Wrote %d sequence samples to %s", written, out_dir)
    return 0


def _label_index(label: str) -> int:
    mapping = {"drinking": 0, "sleeping": 1, "phone_usage": 2, "other": 3, "background": 3}
    return mapping.get(label, 3)


def _box_iou_local(a, b) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LibraryGuard dataset preparation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="Validate a dataset manifest")
    p_val.add_argument("--manifest", required=True)
    p_val.add_argument("--base-dir", default=None)
    p_val.add_argument("--labels", default=None, help="Comma-separated allowed labels")
    p_val.add_argument("--report", default=None, help="Write JSON validation report here")
    p_val.add_argument("--no-duplicates", action="store_true")
    p_val.set_defaults(func=cmd_validate)

    p_split = sub.add_parser("split", help="Leakage-safe train/val/test split")
    p_split.add_argument("--manifest", required=True)
    p_split.add_argument("--out", required=True)
    p_split.add_argument("--train-ratio", type=float, default=0.7)
    p_split.add_argument("--val-ratio", type=float, default=0.15)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--no-stratify", action="store_true")
    p_split.set_defaults(func=cmd_split)

    p_ext = sub.add_parser("extract", help="Extract frames from a video")
    p_ext.add_argument("--video", required=True)
    p_ext.add_argument("--out", required=True)
    p_ext.add_argument("--every", type=float, default=1.0, help="Seconds between frames")
    p_ext.add_argument("--format", default="jpg")
    p_ext.add_argument("--max-frames", type=int, default=None)
    p_ext.set_defaults(func=cmd_extract)

    p_seq = sub.add_parser("sequences", help="Extract feature sequences for the temporal model")
    p_seq.add_argument("--manifest", required=True)
    p_seq.add_argument("--out", required=True)
    p_seq.add_argument("--base-dir", default=None)
    p_seq.add_argument("--min-frames", type=int, default=16)
    p_seq.set_defaults(func=cmd_sequences)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())