from __future__ import annotations

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
