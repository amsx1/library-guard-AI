#!/usr/bin/env python3
"""
LibraryGuard - Run behavior detection on a video.

    python scripts/predict.py --source video.mp4 --output outputs/annotated.mp4

Writes an annotated video plus event reports (JSON + CSV). Prints a summary
of detected events with confidence labels (DETECTED / POSSIBLE /
HIGH CONFIDENCE). Webcam mode: ``--source webcam``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("predict")

BEHAVIOR_EMOJI = {"drinking": "\U0001F964", "sleeping": "\U0001F634", "phone_usage": "\U0001F4F1"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LibraryGuard video behavior detection")
    parser.add_argument("--source", required=True, help="Video file path or 'webcam'")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="Annotated video output")
    parser.add_argument("--report", default="outputs/events.json", help="Event report JSON output")
    parser.add_argument("--csv", default="outputs/events.csv", help="Event report CSV output")
    parser.add_argument("--config", default="configs/config.yaml", help="Config YAML")
    parser.add_argument("--confidence", type=float, default=None, help="Override global confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--no-video", action="store_true", help="Skip annotated video writing")
    parser.add_argument("--behaviors", default=None, help="Comma list: drinking,sleeping,phone_usage")
    parser.add_argument("--duration", type=float, default=None, help="Webcam capture duration (s)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        from src.pipeline import VideoPipeline, WebcamPipeline
        from src.reporting.exporters import export_csv, export_json
        from src.preprocessing.video import VideoCaptureError
        from src.utils.config import ConfigError, load_config
    except ImportError as exc:
        logger.error("Missing dependency: %s. Install requirements: pip install -r requirements.txt", exc)
        return 1

    try:
        config = load_config(args.config if Path(args.config).exists() else None)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    # CLI overrides
    if args.confidence is not None:
        config.detection.global_confidence_threshold = args.confidence
    if args.frame_skip is not None:
        config.detection.frame_skip = args.frame_skip
    if args.behaviors:
        enabled = {b.strip() for b in args.behaviors.split(",")}
        config.behavior.drinking.enabled = "drinking" in enabled
        config.behavior.sleeping.enabled = "sleeping" in enabled
        config.behavior.phone_usage.enabled = "phone_usage" in enabled

    output_path = None if args.no_video else args.output

    try:
        if args.source.lower() == "webcam":
            pipeline = WebcamPipeline(config=config)
            result = pipeline.process_camera(
                duration_seconds=args.duration,
                max_frames=args.max_frames,
                display=False,
            )
        else:
            source_path = Path(args.source)
            if not source_path.exists():
                logger.error("Video not found: %s", source_path)
                return 1
            pipeline = VideoPipeline(config=config)

            last_print = [time.time()]

            def progress(frame_index, total, frame_result):
                now = time.time()
                if now - last_print[0] >= 2.0:
                    last_print[0] = now
                    pct = (frame_index / total * 100) if total else 0
                    logger.info(
                        "Processing frame %d/%d (%.0f%%) - %d persons - %.1f ms/frame",
                        frame_index, total, pct, frame_result.person_count, frame_result.inference_ms,
                    )

            result = pipeline.process_video(
                source_path,
                output_path=output_path,
                progress_callback=progress,
                max_frames=args.max_frames,
            )
    except VideoCaptureError as exc:
        logger.error("Video error: %s", exc)
        return 1
    except Exception as exc:  # defensive: friendly message instead of raw traceback
        logger.error("Processing failed: %s", exc)
        return 1

    # Export reports
    from src.reporting.events import EventLog

    log = EventLog(source=result.source, config_snapshot=config.to_dict())
    log.extend(result.events)
    report = log.build_report(
        video_duration=result.video_duration,
        video_fps=result.video_fps,
        video_frames=result.video_frames,
        frames_processed=result.frames_processed,
        processing_fps=result.processing_fps,
    )
    export_json(report, args.report)
    export_csv(result.events, args.csv)

    # Summary
    print()
    print("=" * 62)
    print("  LibraryGuard - detection summary")
    print("=" * 62)
    print(f"  Source           : {result.source}")
    print(f"  Frames processed : {result.frames_processed} @ {result.processing_fps:.1f} FPS")
    print(f"  Events detected  : {len(result.events)}")
    for event in result.events:
        emoji = BEHAVIOR_EMOJI.get(event.event_type, "")
        print(
            f"    {emoji} {event.timestamp_str:>8}  {event.label:<15} "
            f"Person #{event.person_id}  {event.confidence * 100:5.1f}%  {event.duration:5.1f}s"
        )
    print(f"  Report (JSON)    : {args.report}")
    print(f"  Report (CSV)     : {args.csv}")
    if output_path:
        print(f"  Annotated video  : {output_path}")
    print()
    print("  Note: predictions are probabilistic and may be wrong.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())