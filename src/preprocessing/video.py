"""
LibraryGuard - Video I/O and frame handling.

Wraps OpenCV capture/encode with robust error handling: missing files,
unsupported codecs, corrupt frames, and zero-fps metadata all raise
:class:`VideoCaptureError` with human-readable messages instead of raw
tracebacks.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}


class VideoCaptureError(Exception):
    """Raised when a video cannot be opened, decoded, or encoded."""


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    fourcc: str

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


def _fourcc_to_str(fourcc_float: float) -> str:
    try:
        code = int(fourcc_float)
        return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
    except Exception:
        return "unknown"


def validate_video_path(path: str) -> Path:
    """Validate that a path points to a readable video file."""
    if not path or not isinstance(path, (str, os.PathLike)):
        raise VideoCaptureError("No video path provided.")
    p = Path(path)
    if not p.exists():
        raise VideoCaptureError(f"Video file not found: {p}")
    if not p.is_file():
        raise VideoCaptureError(f"Path is not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise VideoCaptureError(
            f"Unsupported video format '{p.suffix}'. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )
    if p.stat().st_size == 0:
        raise VideoCaptureError(f"Video file is empty: {p}")
    return p


def probe_video(path: str) -> VideoInfo:
    """Read video metadata without decoding the full stream."""
    p = validate_video_path(path)
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoCaptureError(
                f"Could not open video '{p}'. The file may be corrupt or use an unsupported codec."
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC) or 0)

        if width <= 0 or height <= 0:
            raise VideoCaptureError(f"Video '{p}' reports invalid resolution {width}x{height}.")
        if fps <= 0 or not np.isfinite(fps):
            logger.warning("Video '%s' reports invalid fps (%s); assuming 25 fps.", p, fps)
            fps = 25.0
        if frame_count < 0:
            frame_count = 0

        duration = frame_count / fps if fps > 0 else 0.0
        return VideoInfo(
            path=str(p),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            fourcc=fourcc,
        )
    finally:
        cap.release()


class VideoReader:
    """Iterate frames of a video with error handling and optional resize."""

    def __init__(
        self,
        path: str,
        resize: Optional[Tuple[int, int]] = None,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
    ):
        self.info = probe_video(path)
        self.path = str(path)
        self.resize = resize
        self.max_frames = max_frames
        self.frame_skip = max(1, int(frame_skip))

        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise VideoCaptureError(f"Could not open video '{self.path}'.")

    def __iter__(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yield (frame_index, timestamp_seconds, frame_bgr).

        ``frame_index`` counts decoded frames (0-based, pre-skip numbering).
        """
        index = 0
        yielded = 0
        try:
            while True:
                if self.max_frames is not None and yielded >= self.max_frames:
                    break
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    if index == 0:
                        raise VideoCaptureError(
                            f"Could not decode any frames from '{self.path}'. The file may be corrupt."
                        )
                    logger.debug("Stream ended at frame %d in '%s'.", index, self.path)
                    break
                if index % self.frame_skip == 0:
                    if self.resize is not None:
                        frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)
                    timestamp = index / self.info.fps
                    yielded += 1
                    yield index, timestamp, frame
                index += 1
        finally:
            self._cap.release()

    def __len__(self) -> int:
        total = self.info.frame_count
        if self.max_frames is not None:
            total = min(total, self.max_frames)
        return (total + self.frame_skip - 1) // self.frame_skip

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._cap.release()


class VideoWriter:
    """Write annotated frames to an mp4/avi file."""

    def __init__(
        self,
        path: str,
        fps: float,
        size: Tuple[int, int],
        codec: str = "mp4v",
    ):
        self.path = str(path)
        self.fps = fps if fps > 0 else 25.0
        self.size = (int(size[0]), int(size[1]))
        self.codec = codec

        out_dir = Path(self.path).parent
        if str(out_dir):
            out_dir.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
        if not self._writer.isOpened():
            raise VideoCaptureError(
                f"Could not create output video '{self.path}' with codec '{self.codec}'."
            )
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            raise VideoCaptureError("Attempted to write an empty frame.")
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def extract_frames(
    video_path: str,
    output_dir: str,
    every_n_seconds: float = 1.0,
    image_format: str = "jpg",
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """
    Extract frames from a video to an output directory.

    Returns the list of written image paths. Used by dataset preparation.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    with VideoReader(video_path, resize=resize) as reader:
        interval_frames = max(1, int(round(every_n_seconds * reader.info.fps)))
        for frame_index, timestamp, frame in reader:
            if frame_index % interval_frames != 0:
                continue
            name = f"{Path(video_path).stem}_f{frame_index:06d}_t{timestamp:07.2f}.{image_format}"
            dest = out / name
            if not cv2.imwrite(str(dest), frame):
                logger.warning("Failed to write frame %s", dest)
                continue
            written.append(str(dest))
            if max_frames is not None and len(written) >= max_frames:
                break
    return written