"""Shared test fixtures.

``write_clip`` shells out to ffmpeg directly rather than through ``src.utils.ffmpeg``.
Fixtures must not depend on the code under test, or a bug in the encoder silently becomes a
bug in every fixture.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

WIDTH, HEIGHT = 32, 24

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def solid_frame():
    def make(color=(10, 20, 30), height=HEIGHT, width=WIDTH) -> np.ndarray:
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[:, :] = color
        return frame

    return make


@pytest.fixture
def moving_frames(solid_frame):
    """Four frames: a 4x4 white square stepping right across a dark background."""

    def make(count: int = 4) -> list[np.ndarray]:
        out = []
        for i in range(count):
            frame = solid_frame()
            frame[8:12, 2 + i * 4 : 6 + i * 4] = 255
            out.append(frame)
        return out

    return make


@pytest.fixture
def write_clip(tmp_path):
    """Write frames to a lossless RGB MP4 with ffmpeg and return its path.

    libx264rgb, not libx264 -qp 0: the latter converts RGB->YUV before encoding and that
    conversion rounds, so the clip would not decode back to the frames we wrote.
    """

    def make(frames: list[np.ndarray], fps: int = 25, name: str = "clip.mp4") -> Path:
        path = tmp_path / name
        height, width = frames[0].shape[:2]
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast",
            "-pix_fmt", "rgb24",
            str(path),
        ]
        payload = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
        result = subprocess.run(cmd, input=payload, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return path

    return make
