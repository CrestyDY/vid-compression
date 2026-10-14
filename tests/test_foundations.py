from __future__ import annotations

import numpy as np

from conftest import requires_ffmpeg


def test_moving_frames_differ(moving_frames):
    frames = moving_frames()
    assert len(frames) == 4
    assert all(f.shape == (24, 32, 3) and f.dtype == np.uint8 for f in frames)
    assert not np.array_equal(frames[0], frames[1])


@requires_ffmpeg
def test_write_clip_produces_a_file(write_clip, moving_frames):
    path = write_clip(moving_frames())
    assert path.exists() and path.stat().st_size > 0
