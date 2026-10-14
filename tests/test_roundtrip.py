from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_ffmpeg
from src.decoder import decode_to_list, read_header
from src.encoder import encode
from src.models.frame import FrameType


@requires_ffmpeg
@pytest.mark.parametrize("use_entropy", [True, False])
def test_round_trip_is_lossless(tmp_path, write_clip, moving_frames, use_entropy):
    original = moving_frames()
    src = write_clip(original)
    dst = tmp_path / "out.myv"

    stats = encode(src, dst, use_entropy=use_entropy)
    assert stats.frames == len(original)

    decoded = decode_to_list(dst)
    assert len(decoded) == len(original)
    for want, got in zip(original, decoded):
        assert np.array_equal(want, got)


@requires_ffmpeg
def test_header_records_the_source_properties(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames(), fps=25)
    dst = tmp_path / "out.myv"
    encode(src, dst)

    header = read_header(dst)
    assert (header.width, header.height) == (32, 24)
    assert (header.fps_num, header.fps_den) == (25, 1)
    assert header.frame_count == 4


@requires_ffmpeg
def test_frame_zero_is_full_and_the_rest_are_deltas(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    stats = encode(src, dst)
    assert stats.mode_counts[FrameType.FULL] == 1
    assert stats.mode_counts[FrameType.PIXEL_DELTA] == 3


@requires_ffmpeg
def test_decoder_rejects_a_file_not_starting_with_full(tmp_path, write_clip, moving_frames):
    from src.format import HEADER_SIZE
    from src.utils.errors import BadMyvFile

    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)

    data = bytearray(dst.read_bytes())
    data[HEADER_SIZE] = int(FrameType.PIXEL_DELTA)
    broken = tmp_path / "broken.myv"
    broken.write_bytes(bytes(data))

    with pytest.raises(BadMyvFile, match="nothing for it to predict from"):
        decode_to_list(broken)
