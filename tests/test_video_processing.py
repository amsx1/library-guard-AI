"""Tests: video processing functions (probe, read, write, frame extraction)."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.video import (
    VideoCaptureError,
    VideoReader,
    VideoWriter,
    extract_frames,
    probe_video,
    validate_video_path,
)


class TestValidatePath:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(VideoCaptureError, match="not found"):
            validate_video_path(str(tmp_path / "missing.mp4"))

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(VideoCaptureError, match="Unsupported video format"):
            validate_video_path(str(path))

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.mp4"
        path.write_bytes(b"")
        with pytest.raises(VideoCaptureError, match="empty"):
            validate_video_path(str(path))

    def test_directory_raises(self, tmp_path):
        with pytest.raises(VideoCaptureError):
            validate_video_path(str(tmp_path))

    def test_none_raises(self):
        with pytest.raises(VideoCaptureError, match="No video path"):
            validate_video_path("")


class TestProbeVideo:
    def test_probe_synthetic_video(self, synthetic_video):
        info = probe_video(synthetic_video)
        assert info.width == 64
        assert info.height == 48
        assert info.fps == pytest.approx(10.0, abs=0.5)
        assert info.frame_count == 20
        assert info.duration == pytest.approx(2.0, abs=0.3)

    def test_probe_corrupt_file_raises(self, tmp_path):
        path = tmp_path / "corrupt.mp4"
        path.write_bytes(b"\x00\x01\x02 not a real video " * 100)
        with pytest.raises(VideoCaptureError):
            probe_video(str(path))


class TestVideoReader:
    def test_iterate_all_frames(self, synthetic_video):
        frames = list(VideoReader(synthetic_video))
        assert len(frames) == 20
        index, timestamp, frame = frames[0]
        assert index == 0
        assert timestamp == pytest.approx(0.0)
        assert frame.shape == (48, 64, 3)

    def test_timestamps_increase(self, synthetic_video):
        timestamps = [ts for _, ts, _ in VideoReader(synthetic_video)]
        assert timestamps == sorted(timestamps)
        assert timestamps[1] - timestamps[0] == pytest.approx(0.1, abs=0.01)

    def test_frame_skip(self, synthetic_video):
        frames = list(VideoReader(synthetic_video, frame_skip=2))
        assert len(frames) == 10

    def test_max_frames(self, synthetic_video):
        frames = list(VideoReader(synthetic_video, max_frames=5))
        assert len(frames) == 5

    def test_resize(self, synthetic_video):
        frames = list(VideoReader(synthetic_video, resize=(32, 24)))
        assert frames[0][2].shape == (24, 32, 3)

    def test_context_manager(self, synthetic_video):
        with VideoReader(synthetic_video) as reader:
            assert sum(1 for _ in reader) == 20


class TestVideoWriter:
    def test_write_and_reprobe(self, synthetic_video, tmp_path):
        out = tmp_path / "out.mp4"
        with VideoWriter(str(out), fps=10.0, size=(64, 48)) as writer:
            for _ in range(10):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            assert writer.frames_written == 10
        info = probe_video(str(out))
        assert info.width == 64
        assert info.frame_count == 10

    def test_write_resizes_mismatched_frame(self, tmp_path):
        out = tmp_path / "out2.mp4"
        with VideoWriter(str(out), fps=10.0, size=(64, 48)) as writer:
            writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
        info = probe_video(str(out))
        assert info.width == 64

    def test_write_empty_frame_raises(self, tmp_path):
        out = tmp_path / "out3.mp4"
        with VideoWriter(str(out), fps=10.0, size=(64, 48)) as writer:
            with pytest.raises(VideoCaptureError, match="empty frame"):
                writer.write(np.zeros((0, 0, 3), dtype=np.uint8))


class TestExtractFrames:
    def test_extract_frames(self, synthetic_video, tmp_path):
        written = extract_frames(synthetic_video, str(tmp_path / "frames"), every_n_seconds=0.5)
        # 2 s video, one frame every 0.5 s -> ~4-5 frames
        assert 3 <= len(written) <= 6
        for path in written:
            from pathlib import Path

            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_extract_max_frames(self, synthetic_video, tmp_path):
        written = extract_frames(
            synthetic_video, str(tmp_path / "frames2"), every_n_seconds=0.1, max_frames=3
        )
        assert len(written) == 3

    def test_extract_missing_video_raises(self, tmp_path):
        with pytest.raises(VideoCaptureError):
            extract_frames(str(tmp_path / "ghost.mp4"), str(tmp_path / "out"))