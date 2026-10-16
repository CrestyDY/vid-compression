"""The source-native pixel format: planar frames, chosen per plane.

The claim under test is measured, not stylistic — coding a yuv420p source as rgb24 makes
ffmpeg interpolate three quarters of the chroma, and preserving the interpolation costs
about 1.4x (docs/next-lever.md). So these tests check both that the planar path is lossless
and that it is actually smaller than the path it replaces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_ffmpeg
from src import pixfmt
from src.decoder import decode_to_list, read_header
from src.encoder import encode
from src.models import pixel_format as pf
from src.models.frame import FrameType
from src.modes import planes
from src.utils.errors import BadMyvFile
from src.verify import verify

REAL_YUV = Path(__file__).resolve().parent.parent / "samples" / "real_yuv.mp4"
needs_real_yuv = pytest.mark.skipif(
    not REAL_YUV.exists(), reason="samples/real_yuv.mp4 is not present"
)

# ------------------------------------------------------------------ geometry


def test_rgb24_is_one_plane_of_three_channels():
    assert not pf.RGB24.is_planar
    assert pf.RGB24.plane_shapes(32, 24) == ((24, 32, 3),)
    assert pf.RGB24.frame_nbytes(32, 24) == 32 * 24 * 3


def test_yuv420p_is_three_planes_with_quarter_size_chroma():
    assert pf.YUV420P.is_planar
    assert pf.YUV420P.plane_shapes(1920, 1080) == (
        (1080, 1920, 1), (540, 960, 1), (540, 960, 1),
    )
    assert pf.YUV420P.frame_nbytes(1920, 1080) == 1920 * 1080 * 3 // 2


def test_chroma_dimensions_round_up_like_ffmpeg():
    """A 5x3 frame has a 3x2 chroma plane. Rounding down would drop a column."""
    assert pf.YUV420P.plane_shapes(5, 3) == ((3, 5, 1), (2, 3, 1), (2, 3, 1))
    assert pf.YUV420P.frame_nbytes(5, 3) == 15 + 6 + 6


def test_422_halves_width_only_and_444_halves_nothing():
    assert pf.YUV422P.plane_shapes(32, 24) == ((24, 32, 1), (24, 16, 1), (24, 16, 1))
    assert pf.YUV444P.plane_shapes(32, 24) == ((24, 32, 1),) * 3


def test_format_ids_are_stable():
    """Stored in the header. Reassigning one silently misreads every existing file."""
    assert pf.RGB24.id == 0
    assert pf.YUV420P.id == 1
    assert [f.id for f in pf.ALL] == list(range(len(pf.ALL)))
    for fmt in pf.ALL:
        assert pf.by_id(fmt.id) is fmt
        assert pf.by_name(fmt.name) is fmt


def test_unknown_format_id_and_name_are_rejected():
    with pytest.raises(ValueError, match="unknown pixel format id"):
        pf.by_id(200)
    with pytest.raises(ValueError, match="unknown pixel format"):
        pf.by_name("rgb48")


# ------------------------------------------------------------------ splitting


def test_split_and_join_round_trip():
    buf = bytes(range(256)) * 60
    nbytes = pf.YUV420P.frame_nbytes(16, 12)
    got = pixfmt.split(buf, pf.YUV420P, 16, 12)
    assert [p.shape for p in got] == [(12, 16, 1), (6, 8, 1), (6, 8, 1)]
    assert pixfmt.join(got) == buf[:nbytes]


def test_split_is_a_view_not_a_copy():
    """At 1080p a copy per plane per frame would be megabytes of pointless memcpy."""
    buf = bytes(pf.YUV420P.frame_nbytes(16, 12))
    for plane in pixfmt.split(buf, pf.YUV420P, 16, 12):
        assert plane.base is not None


# ------------------------------------------------------------------ source mapping


@pytest.mark.parametrize(
    "name,expected",
    [
        ("yuv420p", pf.YUV420P),
        ("yuvj420p", pf.YUVJ420P),
        ("yuv422p", pf.YUV422P),
        ("yuv444p", pf.YUV444P),
        ("gbrp", pf.RGB24),
        ("rgb24", pf.RGB24),
        ("gray", pf.GRAY),
    ],
)
def test_known_sources_are_coded_natively_and_silently(name, expected):
    fmt, note = pixfmt.for_source(name)
    assert fmt is expected
    assert note is None


def test_full_range_yuv_keeps_its_own_format():
    """yuvj420p and yuv420p differ only by a range tag, but asking ffmpeg for the wrong
    one makes swscale convert, and a conversion rounds. Same geometry, separate ids."""
    assert pf.YUVJ420P.ffmpeg_name == "yuvj420p"
    assert pf.YUVJ420P.planes == pf.YUV420P.planes
    assert pf.YUVJ420P.id != pf.YUV420P.id


@pytest.mark.parametrize("name", ["rgba", "yuv420p10le", "bogus"])
def test_unsupported_sources_fall_back_loudly(name):
    fmt, note = pixfmt.for_source(name)
    assert fmt is pf.RGB24
    assert note and name in note


def test_missing_source_format_falls_back_loudly():
    fmt, note = pixfmt.for_source(None)
    assert fmt is pf.RGB24
    assert note


# ------------------------------------------------------------------ the PLANES payload


def test_planes_payload_round_trips():
    packed = planes.pack(
        [(FrameType.FULL, b"luma"), (FrameType.MOTION, b"u"), (FrameType.TILED, b"vv")]
    )
    assert planes.unpack(packed) == [
        (FrameType.FULL, b"luma"),
        (FrameType.MOTION, b"u"),
        (FrameType.TILED, b"vv"),
    ]


def test_planes_payload_states_its_own_plane_count():
    """Self-describing like every other mode's payload, for one byte a frame."""
    assert planes.pack([(FrameType.FULL, b"")])[0] == 1
    assert planes.pack([(FrameType.FULL, b"")] * 3)[0] == 3


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"", "truncated"),
        (b"\x00", "out of range"),
        (b"\x09", "out of range"),
        (b"\x01\x63\x00", "unknown frame type"),
        (b"\x01\x00\x02abjunk", "trailing bytes"),
        (b"\x01\x00\x09ab", "truncated"),
    ],
)
def test_corrupt_planes_payloads_are_rejected(payload, match):
    with pytest.raises(ValueError, match=match):
        planes.unpack(payload)


def test_planes_may_not_nest():
    packed = planes.pack([(FrameType.PLANES, b"")])
    with pytest.raises(ValueError, match="no recursion"):
        planes.decode(packed, None, ((4, 4, 1),))


def test_planes_count_must_match_the_pixel_format():
    packed = planes.pack([(FrameType.FULL, b"")])
    with pytest.raises(ValueError, match="holds 1 planes but this pixel format has 3"):
        planes.decode(packed, None, ((4, 4, 1), (2, 2, 1), (2, 2, 1)))


# ------------------------------------------------------------------ end to end


@requires_ffmpeg
def test_a_yuv_source_is_coded_in_its_own_planes(
    tmp_path, write_yuv_clip, moving_yuv_frames
):
    original = moving_yuv_frames()
    src = write_yuv_clip(original)
    dst = tmp_path / "out.myv"

    stats = encode(src, dst)

    assert stats.pix_fmt == "yuv420p"
    assert stats.pix_fmt_note is None, "a yuv420p source needs no apology"
    assert read_header(dst).pix_fmt is pf.YUV420P
    # Every frame record is PLANES; the per-plane modes are counted separately.
    assert stats.mode_counts == {FrameType.PLANES: len(original)}
    assert sum(stats.plane_mode_counts.values()) == len(original) * 3


@requires_ffmpeg
def test_the_planar_round_trip_is_lossless(tmp_path, write_yuv_clip, moving_yuv_frames):
    original = moving_yuv_frames()
    src = write_yuv_clip(original)
    dst = tmp_path / "out.myv"
    encode(src, dst)

    decoded = decode_to_list(dst)
    assert len(decoded) == len(original)
    for want, got in zip(original, decoded):
        assert len(got) == 3
        for index, (a, b) in enumerate(zip(want, got)):
            assert np.array_equal(a, b), f"plane {index}"


@requires_ffmpeg
def test_verify_confirms_a_planar_file(tmp_path, write_yuv_clip, moving_yuv_frames):
    src = write_yuv_clip(moving_yuv_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)
    result = verify(src, dst)
    assert result.mismatch is None
    assert result.frames_checked == 4


@requires_ffmpeg
def test_source_native_reads_two_thirds_as_much_data(
    tmp_path, write_yuv_clip, moving_yuv_frames
):
    """Where the win comes from, before any coding: yuv420p is 1.5 bytes per pixel and
    rgb24 is 3, and the extra 1.5 are interpolated rather than read."""
    src = write_yuv_clip(moving_yuv_frames())
    native = encode(src, tmp_path / "native.myv")
    forced = encode(src, tmp_path / "forced.myv", pix_fmt=pf.RGB24)

    assert native.pix_fmt == "yuv420p"
    assert forced.pix_fmt == "rgb24"
    assert native.raw_frame_bytes * 2 == forced.raw_frame_bytes


@requires_ffmpeg
def test_per_plane_overhead_can_lose_on_a_tiny_flat_clip(
    tmp_path, write_yuv_clip, moving_yuv_frames
):
    """The honest limit of the idea, pinned so nobody has to rediscover it.

    A PLANES frame costs 1 byte plus a mode and a length per plane, and the fixtures'
    chroma is flat enough to compress to almost nothing either way. At 32x24 that
    overhead is most of the file, so the native path comes out *larger*. The 1.4x is a
    claim about real footage, which the next test measures.
    """
    src = write_yuv_clip(moving_yuv_frames())
    native = encode(src, tmp_path / "native.myv")
    forced = encode(src, tmp_path / "forced.myv", pix_fmt=pf.RGB24)
    assert native.total_bytes > forced.total_bytes


@requires_ffmpeg
@needs_real_yuv
@pytest.mark.samples
def test_source_native_beats_forcing_rgb_on_real_footage(tmp_path):
    """The whole point of M12, as a number rather than an argument."""
    native = encode(REAL_YUV, tmp_path / "native.myv")
    forced = encode(REAL_YUV, tmp_path / "forced.myv", pix_fmt=pf.RGB24)

    assert native.pix_fmt == "yuv420p"
    assert native.total_bytes < forced.total_bytes
    ratio = forced.total_bytes / native.total_bytes
    assert ratio > 1.25, f"expected a clear win on real footage, got {ratio:.2f}x"


@requires_ffmpeg
@needs_real_yuv
@pytest.mark.samples
def test_real_footage_round_trips_losslessly_in_its_own_planes(tmp_path):
    dst = tmp_path / "real.myv"
    encode(REAL_YUV, dst)
    result = verify(REAL_YUV, dst)
    assert result.mismatch is None
    assert result.frames_checked == 4


@requires_ffmpeg
def test_forcing_rgb_on_a_yuv_source_is_still_lossless_against_rgb(
    tmp_path, write_yuv_clip, moving_yuv_frames
):
    """--pix-fmt rgb24 stays exact with respect to the rgb24 frames ffmpeg produces.
    That is a weaker claim than the native path's, and it is the one being tested."""
    src = write_yuv_clip(moving_yuv_frames())
    dst = tmp_path / "forced.myv"
    encode(src, dst, pix_fmt=pf.RGB24)
    assert verify(src, dst).mismatch is None


@requires_ffmpeg
def test_an_rgb_source_is_unchanged_by_any_of_this(tmp_path, write_clip, moving_frames):
    """A single-plane format must still store a bare mode per frame, not a PLANES
    wrapper: existing .myv files stay readable and their sizes stay comparable."""
    src = write_clip(moving_frames())
    dst = tmp_path / "out.myv"
    stats = encode(src, dst)

    assert stats.pix_fmt == "rgb24"
    assert FrameType.PLANES not in stats.mode_counts
    assert stats.plane_mode_counts == {}


@requires_ffmpeg
def test_raw_frame_bytes_follows_the_pixel_format(
    tmp_path, write_yuv_clip, moving_yuv_frames
):
    """It used to hardcode a times-three that only RGB24 ever justified."""
    src = write_yuv_clip(moving_yuv_frames())
    stats = encode(src, tmp_path / "out.myv")
    assert stats.frame_nbytes == pf.YUV420P.frame_nbytes(32, 24)
    assert stats.raw_frame_bytes == stats.frame_nbytes * stats.frames


@requires_ffmpeg
def test_a_planar_file_rejects_a_bare_mode_frame(
    tmp_path, write_yuv_clip, moving_yuv_frames
):
    """Hand-corrupt the first frame's type byte to FULL. A yuv420p frame cannot be one
    plane, and saying so beats reconstructing a frame of the wrong shape."""
    from src.format import HEADER_SIZE

    src = write_yuv_clip(moving_yuv_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)

    data = bytearray(dst.read_bytes())
    assert data[HEADER_SIZE] == int(FrameType.PLANES)
    data[HEADER_SIZE] = int(FrameType.FULL)
    dst.write_bytes(bytes(data))

    # Frame 0 has its own, more specific complaint: nothing to predict from.
    with pytest.raises(BadMyvFile, match="must start with PLANES"):
        decode_to_list(dst)


@requires_ffmpeg
def test_a_planar_file_rejects_a_bare_mode_on_a_later_frame(
    tmp_path, write_yuv_clip, moving_yuv_frames
):
    """Same corruption past frame 0, where the frame-0 guard does not cover it."""
    import struct

    from src.format import FRAME_HEADER_SIZE, HEADER_SIZE

    src = write_yuv_clip(moving_yuv_frames())
    dst = tmp_path / "out.myv"
    encode(src, dst)

    data = bytearray(dst.read_bytes())
    first_len = struct.unpack_from("<I", data, HEADER_SIZE + 1)[0]
    second = HEADER_SIZE + FRAME_HEADER_SIZE + first_len
    assert data[second] == int(FrameType.PLANES)
    data[second] = int(FrameType.MOTION)
    dst.write_bytes(bytes(data))

    with pytest.raises(BadMyvFile, match="must be PLANES"):
        decode_to_list(dst)
