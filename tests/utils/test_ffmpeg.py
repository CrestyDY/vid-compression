from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_ffmpeg
from src.models.video import VideoInfo
from src.utils.errors import FFmpegError
from src.utils.ffmpeg import probe


@requires_ffmpeg
def test_probe_reads_dimensions_and_rate(write_clip, moving_frames):
    info = probe(write_clip(moving_frames(), fps=25))
    assert (info.width, info.height) == (32, 24)
    assert (info.fps_num, info.fps_den) == (25, 1)
    assert info.fps == 25.0


def test_probe_rejects_missing_file(tmp_path):
    with pytest.raises(FFmpegError, match="no such file"):
        probe(tmp_path / "nope.mp4")


@requires_ffmpeg
def test_probe_rejects_a_non_video(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(FFmpegError):
        probe(junk)


def test_video_info_frame_count_is_optional():
    info = VideoInfo(width=2, height=2, fps_num=30000, fps_den=1001)
    assert info.num_frames is None
    assert abs(info.fps - 29.97) < 0.01


@requires_ffmpeg
def test_decode_frames_round_trips_exactly(write_clip, moving_frames):
    from src.utils.ffmpeg import decode_frames

    original = moving_frames()
    with decode_frames(write_clip(original)) as frames:
        decoded = list(frames)

    assert len(decoded) == len(original)
    for want, got in zip(original, decoded):
        assert got.shape == (24, 32, 3)
        assert got.dtype == np.uint8
        assert np.array_equal(want, got)


@requires_ffmpeg
def test_decode_frames_survives_early_abandonment(write_clip, moving_frames):
    """Stopping after one frame must not raise: we asked ffmpeg to stop."""
    from src.utils.ffmpeg import decode_frames

    with decode_frames(write_clip(moving_frames())) as frames:
        first = next(iter(frames))
    assert first.shape == (24, 32, 3)


@requires_ffmpeg
def test_decoded_frames_are_independently_retainable(write_clip, moving_frames):
    """The encoder holds frame N-1 while reading N, so frames must not share a buffer."""
    from src.utils.ffmpeg import decode_frames

    with decode_frames(write_clip(moving_frames())) as frames:
        held = list(frames)
    assert not np.array_equal(held[0], held[-1])
