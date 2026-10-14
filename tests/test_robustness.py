"""Behaviour when the world misbehaves: wrong files, corrupt bytes, dying subprocesses.

Every test here reproduces a finding from the M1 whole-branch review. The codec being
lossless is established in test_roundtrip.py; this file is about the perimeter around it.
"""

from __future__ import annotations

import subprocess

import pytest

from conftest import requires_ffmpeg
from src.cli import main
from src.decoder import decode_to_list, read_header
from src.encoder import encode
from src.format import FRAME_HEADER_SIZE, HEADER_SIZE
from src.utils.errors import BadMyvFile, FFmpegError
from src.verify import verify

# ---------------------------------------------------------------- verify


@requires_ffmpeg
def test_verify_reports_a_dimension_mismatch_instead_of_crashing(
    tmp_path, write_clip, moving_frames, solid_frame
):
    """Verifying against the wrong source file is an ordinary slip, not a crash."""
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)

    other = write_clip(
        [solid_frame(height=8, width=8) for _ in range(4)], name="other.mp4"
    )
    result = verify(other, dst)

    assert not result.ok
    assert result.mismatch is not None
    assert "dimension" in result.mismatch.describe().lower()


@requires_ffmpeg
def test_cli_verify_against_wrong_source_exits_cleanly(
    tmp_path, write_clip, moving_frames, solid_frame, capsys
):
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)
    other = write_clip(
        [solid_frame(height=8, width=8) for _ in range(4)], name="other.mp4"
    )

    assert main(["verify", str(other), str(dst)]) == 1
    assert "not lossless" in capsys.readouterr().out.lower()


# ---------------------------------------------- corrupt files raise BadMyvFile


@requires_ffmpeg
def test_corrupt_entropy_payload_raises_bad_myv(tmp_path, write_clip, moving_frames):
    """A flipped bit in a zstd payload must not escape as zstandard.ZstdError."""
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst, use_entropy=True)

    data = bytearray(dst.read_bytes())
    start = HEADER_SIZE + FRAME_HEADER_SIZE
    data[start : start + 4] = b"junk"  # not a zstd frame
    bad = tmp_path / "bad.myv"
    bad.write_bytes(bytes(data))

    with pytest.raises(BadMyvFile):
        decode_to_list(bad)


@requires_ffmpeg
def test_corrupt_delta_coordinate_raises_bad_myv(tmp_path, write_clip, moving_frames):
    """A corrupt x coordinate indexes outside the frame; IndexError must be wrapped."""
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst, use_entropy=False)

    full_payload = 24 * 32 * 3
    frame1 = HEADER_SIZE + FRAME_HEADER_SIZE + full_payload
    first_x = frame1 + FRAME_HEADER_SIZE + 4  # past the uint32 record count

    data = bytearray(dst.read_bytes())
    data[first_x : first_x + 2] = (65535).to_bytes(2, "little")
    bad = tmp_path / "bad.myv"
    bad.write_bytes(bytes(data))

    with pytest.raises(BadMyvFile):
        decode_to_list(bad)


@requires_ffmpeg
def test_header_with_zero_width_raises_bad_myv(tmp_path, write_clip, moving_frames):
    """A zeroed dimension must be a BadMyvFile, not a pydantic ValidationError."""
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)

    data = bytearray(dst.read_bytes())
    data[8:12] = (0).to_bytes(4, "little")  # width, after magic+version+fmt+flags+reserved
    bad = tmp_path / "bad.myv"
    bad.write_bytes(bytes(data))

    with pytest.raises(BadMyvFile):
        read_header(bad)


# ---------------------------------------------------------------- the CLI


@pytest.mark.parametrize(
    "argv",
    [
        ["info", "missing.myv"],
        ["decode", "missing.myv", "out.mp4"],
        ["verify", "missing.mp4", "missing.myv"],
    ],
    ids=["info", "decode", "verify"],
)
def test_cli_missing_file_exits_2(argv, tmp_path, capsys):
    """encode already exits 2 on a missing file; the others must agree."""
    resolved = [argv[0]] + [str(tmp_path / a) for a in argv[1:]]
    assert main(resolved) == 2
    assert "error:" in capsys.readouterr().err.lower()


# ------------------------------------------------- a failed encode leaves nothing


@requires_ffmpeg
def test_failed_encode_leaves_no_partial_file(
    tmp_path, write_clip, moving_frames, monkeypatch
):
    """A truncated .myv that still parses is worse than no file at all."""
    from contextlib import contextmanager

    from src.utils import ffmpeg as ffmpeg_mod

    real = ffmpeg_mod.decode_frames

    @contextmanager
    def dying(path, pix_fmt=None, info=None):
        with real(path, pix_fmt, info) if pix_fmt else real(path) as frames:

            def partial():
                for i, frame in enumerate(frames):
                    if i == 2:
                        raise FFmpegError("frame 2 is short — got 1000 of 2304 bytes")
                    yield frame

            yield partial()

    monkeypatch.setattr("src.encoder.ffmpeg.decode_frames", dying)

    src = write_clip(moving_frames())
    dst = tmp_path / "partial.myv"
    with pytest.raises(FFmpegError):
        encode(src, dst)

    assert not dst.exists(), "a failed encode must not leave a parseable .myv behind"


# ------------------------------------------------------- odd dimensions on export


@requires_ffmpeg
def test_decode_of_odd_dimensions_explains_itself(
    tmp_path, write_clip, solid_frame, capsys
):
    """libx264 rejects odd dimensions; say so instead of dumping ffmpeg internals."""
    frames = [solid_frame(height=3, width=5) for _ in range(3)]
    src = write_clip(frames, name="odd.mp4")
    myv = tmp_path / "odd.myv"
    encode(src, myv)

    assert main(["decode", str(myv), str(tmp_path / "odd_back.mp4")]) == 2
    err = capsys.readouterr().err.lower()
    assert "odd" in err and "--lossless" in err


@requires_ffmpeg
def test_decode_of_odd_dimensions_works_with_lossless(tmp_path, write_clip, solid_frame):
    frames = [solid_frame(height=3, width=5) for _ in range(3)]
    src = write_clip(frames, name="odd.mp4")
    myv = tmp_path / "odd.myv"
    encode(src, myv)

    out = tmp_path / "odd_back.mp4"
    assert main(["decode", str(myv), str(out), "--lossless"]) == 0
    assert out.exists() and out.stat().st_size > 0


# ------------------------------------------------------- subprocess reaping


@requires_ffmpeg
def test_encode_frames_reaps_ffmpeg_when_the_frame_source_raises(tmp_path, monkeypatch):
    """A generator raising mid-write must not leave ffmpeg alive on the pipe."""
    import numpy as np

    from src.utils import ffmpeg as ffmpeg_mod

    spawned = []
    real_popen = subprocess.Popen

    def tracking_popen(*a, **kw):
        proc = real_popen(*a, **kw)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(ffmpeg_mod.subprocess, "Popen", tracking_popen)

    def exploding():
        yield np.zeros((24, 32, 3), np.uint8)
        raise RuntimeError("the frame source failed")

    with pytest.raises(RuntimeError, match="the frame source failed"):
        ffmpeg_mod.encode_frames(
            tmp_path / "out.mp4", exploding(),
            width=32, height=24, fps_num=25, fps_den=1, lossless=True,
        )

    assert spawned, "expected an ffmpeg process to have been spawned"
    for proc in spawned:
        assert proc.poll() is not None, "ffmpeg was left running"
