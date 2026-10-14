from __future__ import annotations

from conftest import requires_ffmpeg
from src.encoder import encode
from src.verify import verify


@requires_ffmpeg
def test_verify_passes_on_a_good_file(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)

    result = verify(src, dst)
    assert result.ok
    assert result.frames_checked == 4
    assert result.mismatch is None


@requires_ffmpeg
def test_verify_fails_on_a_corrupted_payload(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst, use_entropy=False)

    data = bytearray(dst.read_bytes())
    data[-1] ^= 0xFF  # flip the last payload byte
    corrupted = tmp_path / "bad.myv"
    corrupted.write_bytes(bytes(data))

    result = verify(src, corrupted)
    assert not result.ok
    assert result.mismatch is not None
    assert "pixels differ" in result.mismatch.describe()


@requires_ffmpeg
def test_verify_reports_a_frame_count_difference(tmp_path, write_clip, moving_frames):
    src = write_clip(moving_frames())
    longer = write_clip(moving_frames(count=6), name="longer.mp4")
    dst = tmp_path / "out.myv"
    encode(src, dst)

    result = verify(longer, dst)
    assert not result.ok
    assert result.source_frames != result.decoded_frames
