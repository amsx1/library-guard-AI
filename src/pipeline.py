"""
LibraryGuard - End-to-end video analysis pipeline.

VIDEO INPUT -> frame extraction -> person/object/pose detection
-> person tracking -> per-frame behavior scoring -> temporal event
aggregation -> annotated video + event report.

The pipeline is source-agnostic: it consumes frames from any iterator
(:class:`VideoReader`, webcam capture, or a future RTSP reader) so new
video sources can be added without touching the ML logic.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .behavior.base import BehaviorObservation
from .behavior.engine import BehaviorEngine, BehaviorEvent, LiveStatus, build_engine_from_config
from .detection.detector import DetectionResult, MultiDetector, Pose
from .preprocessing.video import VideoCaptureError, VideoReader, VideoWriter, probe_video
from .reporting.annotate import (
    AnnotationConfig,
    ObjectOverlay,
    TrackOverlay,
    annotate_frame,
)
from .reporting.events import EventLog
from .tracking.tracker import Tracker, create_tracker
from .utils.config import AppConfig, load_config
from .utils.constants import Box

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Per-frame processing result (used by the UI for live overlays)."""

    frame_index: int
    timestamp: float
    annotated_frame: Optional[np.ndarray] = None
    track_overlays: List[TrackOverlay] = field(default_factory=list)
    object_overlays: List[ObjectOverlay] = field(default_factory=list)
    live_statuses: List[LiveStatus] = field(default_factory=list)
    person_count: int = 0
    inference_ms: float = 0.0


@dataclass
class PipelineResult:
    """Final result of processing one video."""

    source: str
    events: List[BehaviorEvent]
    frames_processed: int
    processing_fps: float
    video_fps: float
    video_duration: float
    video_frames: int
    output_video_path: Optional[str] = None
    event_report: Optional[Any] = None  # EventReport
    warnings: List[str] = field(default_factory=list)


class VideoPipeline:
    """
    Orchestrates detection, tracking, and behavior recognition for one video.

    Parameters
    ----------
    config:
        Loaded :class:`AppConfig`. If omitted, defaults from ``configs/config.yaml``
        are used.
    write_video:
        If True and ``output_path`` is given to :meth:`process_video`, annotated
        frames are encoded to a video file.
    detector:
        Optional pre-built :class:`MultiDetector` (injected in tests).
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        detector: Optional[MultiDetector] = None,
        tracker: Optional[Tracker] = None,
        engine: Optional[BehaviorEngine] = None,
    ):
        self.config = config or load_config()
        self._detector = detector
        self._tracker = tracker
        self._engine = engine
        self.warnings: List[str] = []

    # -- lazy component construction ---------------------------------------

    @property
    def detector(self) -> MultiDetector:
        if self._detector is None:
            cfg = self.config.models
            self._detector = MultiDetector(
                person_model=cfg.person_detector.name,
                object_model=cfg.object_detector.name,
                pose_model=cfg.pose_estimator.name,
                device=cfg.pose_estimator.device,
                confidence_threshold=self.config.detection.global_confidence_threshold,
                iou_threshold=self.config.tracking.iou_threshold,
                half_precision=self.config.performance.half_precision,
            )
        return self._detector

    @property
    def tracker(self) -> Tracker:
        if self._tracker is None:
            tcfg = self.config.tracking
            self._tracker = create_tracker(
                tracker_type=tcfg.tracker,
                max_age=tcfg.max_age,
                min_hits=tcfg.min_hits,
                iou_threshold=tcfg.iou_threshold,
            )
        return self._tracker

    def _build_engine(self, fps: float) -> BehaviorEngine:
        if self._engine is None:
            self._engine = build_engine_from_config(self.config, fps=fps)
        return self._engine

    # -- public API ---------------------------------------------------------

    def process_video(
        self,
        source: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[int, int, FrameResult], None]] = None,
        collect_frames: bool = False,
        max_frames: Optional[int] = None,
    ) -> PipelineResult:
        """
        Process a full video file.

        Returns a :class:`PipelineResult` with events, timing stats and
        (optionally) the path to the annotated output video.
        """
        self.warnings = []
        source = str(source)

        try:
            info = probe_video(source)
        except VideoCaptureError as exc:
            raise VideoCaptureError(str(exc)) from exc

        engine = self._build_engine(info.fps)
        engine.reset()
        self.tracker.reset()

        total_frames = info.frame_count
        if max_frames is not None:
            total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames
        dcfg = self.config.detection
        effective_max = max_frames if max_frames is not None else dcfg.max_frames

        writer: Optional[VideoWriter] = None
        out_path: Optional[str] = None
        if output_path is not None:
            out_path = str(output_path)
            fps_out = self.config.video.output_fps or info.fps
            writer = VideoWriter(
                out_path,
                fps=fps_out,
                size=(info.width, info.height),
                codec=self.config.video.output_codec,
            )

        event_log = EventLog(
            source=source,
            config_snapshot=self.config.to_dict(),
        )

        frames_processed = 0
        start_time = time.perf_counter()
        try:
            with VideoReader(
                source,
                resize=tuple(dcfg.input_resize) if dcfg.input_resize else None,
                max_frames=effective_max,
                frame_skip=dcfg.frame_skip,
            ) as reader:
                for frame_index, timestamp, frame in reader:
                    result = self.process_frame(frame, frame_index, timestamp, info.fps, engine)
                    frames_processed += 1

                    if writer is not None and result.annotated_frame is not None:
                        # Annotated frames may be resized; write at source size.
                        writer.write(result.annotated_frame)

                    if progress_callback is not None:
                        progress_callback(frame_index, total_frames or 0, result)

        finally:
            if writer is not None:
                writer.release()

        elapsed = time.perf_counter() - start_time
        processing_fps = frames_processed / elapsed if elapsed > 0 else 0.0

        events = engine.finalize()
        event_log.extend(events)
        if self.config.privacy.store_raw_video is False and out_path:
            self.warnings.append(
                "Annotated output video is generated for review; raw input video is not stored by the pipeline."
            )

        report = event_log.build_report(
            video_duration=info.duration,
            video_fps=info.fps,
            video_frames=info.frame_count,
            frames_processed=frames_processed,
            processing_fps=processing_fps,
        )

        return PipelineResult(
            source=source,
            events=events,
            frames_processed=frames_processed,
            processing_fps=processing_fps,
            video_fps=info.fps,
            video_duration=info.duration,
            video_frames=info.frame_count,
            output_video_path=out_path,
            event_report=report,
            warnings=list(self.warnings),
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        video_fps: float,
        engine: Optional[BehaviorEngine] = None,
    ) -> FrameResult:
        """Run detection -> tracking -> behavior scoring -> annotation on one frame."""
        if engine is None:
            engine = self._build_engine(video_fps)

        t0 = time.perf_counter()
        detection: DetectionResult = self.detector.detect(frame)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        height, width = frame.shape[:2]

        # --- tracking ---
        person_boxes: List[Box] = []
        person_scores: List[float] = []
        pose_by_box: List[Pose] = []
        for pose in detection.persons:
            if pose.bbox is not None:
                person_boxes.append(pose.bbox)
                person_scores.append(pose.bbox.confidence)
                pose_by_box.append(pose)

        active_tracks = self.tracker.update(person_boxes, person_scores)

        # --- build behavior observations per track ---
        observations: List[BehaviorObservation] = []
        track_overlays: List[TrackOverlay] = []
        for track in active_tracks:
            tb = track.bbox
            # Find nearest pose for this track (match by IoU of boxes).
            best_pose: Optional[Pose] = None
            best_iou = 0.0
            for pose in pose_by_box:
                if pose.bbox is None:
                    continue
                overlap = _box_iou(tb, pose.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_pose = pose

            objects = [(box, box.class_name, box.confidence) for box in detection.objects]

            observation = BehaviorObservation(
                frame_index=frame_index,
                timestamp=timestamp,
                track_id=track.track_id,
                person_box=tb,
                keypoints=best_pose,
                person_score=track.score,
                objects=objects,
                track_movement=track.movement_over_window(frames=max(2, int(video_fps * 0.5))),
                frame_width=width,
                frame_height=height,
            )
            observations.append(observation)

            # Face box for privacy blur: upper portion of person bbox
            face_box = (tb.x1 + 0.2 * tb.width, tb.y1, tb.x2 - 0.2 * tb.width, tb.y1 + 0.28 * tb.height)

            track_overlays.append(
                TrackOverlay(
                    track_id=track.track_id,
                    bbox=tb.to_tuple(),
                    score=track.score,
                    statuses=[],
                    face_box=face_box,
                )
            )

        # --- behavior engine ---
        engine.update(frame_index, timestamp, observations, frame_fps=video_fps)

        # The engine aggregates across tracks; overlays need per-person labels,
        # so derive per-track statuses from its evidence state.
        per_track_statuses = self._per_track_statuses(engine, observations)

        for overlay in track_overlays:
            overlay.statuses = per_track_statuses.get(overlay.track_id, [])

        object_overlays = [
            ObjectOverlay(bbox=box.to_tuple(), class_name=box.class_name, score=box.confidence)
            for box in detection.objects
        ]

        vcfg = self.config.video
        annotated = annotate_frame(
            frame,
            tracks=track_overlays,
            objects=object_overlays,
            frame_index=frame_index,
            timestamp=timestamp,
            fps=self._rolling_fps(inference_ms),
            config=AnnotationConfig(
                draw_bboxes=vcfg.draw_bboxes,
                draw_tracking_ids=vcfg.draw_tracking_ids,
                draw_behavior_labels=vcfg.draw_behavior_labels,
                draw_confidence=vcfg.draw_confidence,
                draw_timestamp=vcfg.draw_timestamp,
                font_scale=vcfg.font_scale,
                line_thickness=vcfg.line_thickness,
                blur_faces=self.config.privacy.blur_faces,
            ),
        )

        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            annotated_frame=annotated,
            track_overlays=track_overlays,
            object_overlays=object_overlays,
            live_statuses=[s for sts in per_track_statuses.values() for s in sts],
            person_count=len(active_tracks),
            inference_ms=inference_ms,
        )

    # -- internals ----------------------------------------------------------

    def _per_track_statuses(
        self,
        engine: BehaviorEngine,
        observations: Sequence[BehaviorObservation],
    ) -> Dict[int, List[LiveStatus]]:
        """
        Derive per-track live statuses from the engine's evidence state.

        The engine aggregates globally; the overlay needs per-person labels,
        so we read the per-(track, behavior) evidence directly.
        """
        result: Dict[int, List[LiveStatus]] = {}
        from .behavior.base import evidence_label

        for obs in observations:
            statuses: List[LiveStatus] = []
            for key in ("drinking", "sleeping", "phone_usage"):
                state = engine._evidence.get((obs.track_id, key))
                cfg = engine.configs.get(key)
                if state is None or cfg is None or not cfg.enabled:
                    continue
                conf = state.smoothed_confidence()
                candidate = engine._active.get((obs.track_id, key))
                elapsed = 0.0
                active = False
                if candidate is not None:
                    active = True
                    elapsed = max(0.0, obs.timestamp - candidate["start_time"])
                if conf > 0.12 or active:
                    statuses.append(
                        LiveStatus(
                            behavior=key,
                            label=evidence_label(conf),
                            confidence=conf,
                            elapsed=elapsed,
                            active=active,
                        )
                    )
            if statuses:
                result[obs.track_id] = statuses
        return result

    def _rolling_fps(self, inference_ms: float) -> float:
        if inference_ms <= 0:
            return 0.0
        return 1000.0 / inference_ms


def _box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class WebcamPipeline(VideoPipeline):
    """
    Webcam variant of the pipeline.

    Demonstrates that the source layer is decoupled: frames come from a
    camera instead of a file, everything else is identical. An RTSP/IP
    camera source would subclass the same way.
    """

    def __init__(self, camera_index: int = 0, config: Optional[AppConfig] = None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.camera_index = camera_index

    def process_camera(
        self,
        duration_seconds: Optional[float] = None,
        max_frames: Optional[int] = None,
        display: bool = False,
    ) -> PipelineResult:
        import cv2

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise VideoCaptureError(f"Could not open webcam (index {self.camera_index}).")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        engine = self._build_engine(fps)
        engine.reset()
        self.tracker.reset()

        frames_processed = 0
        start = time.perf_counter()
        try:
            while True:
                if max_frames is not None and frames_processed >= max_frames:
                    break
                if duration_seconds is not None and (time.perf_counter() - start) >= duration_seconds:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.warnings.append("Webcam frame grab failed; stopping capture.")
                    break
                timestamp = frames_processed / fps
                result = self.process_frame(frame, frames_processed, timestamp, fps, engine)
                frames_processed += 1
                if display:
                    cv2.imshow("LibraryGuard", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start
        events = engine.finalize()
        report = EventLog(source=f"webcam:{self.camera_index}").build_report(
            video_duration=elapsed,
            video_fps=fps,
            video_frames=frames_processed,
            frames_processed=frames_processed,
            processing_fps=frames_processed / elapsed if elapsed > 0 else 0.0,
        )
        return PipelineResult(
            source=f"webcam:{self.camera_index}",
            events=events,
            frames_processed=frames_processed,
            processing_fps=frames_processed / elapsed if elapsed > 0 else 0.0,
            video_fps=fps,
            video_duration=elapsed,
            video_frames=frames_processed,
            event_report=report,
            warnings=list(self.warnings),
        )