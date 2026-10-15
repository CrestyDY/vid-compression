from __future__ import annotations

import io

import pytest

from src.format import (
    HEADER_SIZE,
    MyvReader,
    MyvWriter,
    pack_header,
    unpack_header,
)
from src.models.frame import FrameType
from src.models.header import Header
from src.utils.errors import BadMyvFile


def a_header(**kw) -> Header:
    base = dict(width=32, height=24, fps_num=25, fps_den=1, frame_count=0)
    return Header(**{**base, **kw})


def test_header_round_trips():
    header = a_header(frame_count=7)
    restored = unpack_header(pack_header(header))
    assert restored == header
    assert restored.fps == 25.0
    assert restored.frame_shape == (24, 32, 3)


def test_header_is_a_fixed_size():
    assert len(pack_header(a_header())) == HEADER_SIZE == 28


def test_unpack_rejects_bad_magic():
    with pytest.raises(BadMyvFile, match="not a .myv"):
        unpack_header(b"XXXX" + bytes(HEADER_SIZE - 4))


def test_unpack_rejects_a_short_header():
    with pytest.raises(BadMyvFile, match="too short"):
        unpack_header(b"MYV1")


def test_writer_back_patches_the_frame_count():
    stream = io.BytesIO()
    with MyvWriter(stream, a_header()) as writer:
        writer.frame(FrameType.FULL, b"abc")
        writer.frame(FrameType.PIXEL_DELTA, b"de")

    stream.seek(0)
    reader = MyvReader(stream)
    assert reader.header.frame_count == 2
    assert [(f.type, f.payload) for f in reader.frames()] == [
        (FrameType.FULL, b"abc"),
        (FrameType.PIXEL_DELTA, b"de"),
    ]


def test_reader_rejects_an_unknown_frame_type():
    stream = io.BytesIO()
    with MyvWriter(stream, a_header()) as writer:
        writer.frame(FrameType.FULL, b"x")
    data = bytearray(stream.getvalue())
    data[HEADER_SIZE] = 99  # the frame type byte
    with pytest.raises(BadMyvFile, match="unknown frame type 99"):
        list(MyvReader(io.BytesIO(bytes(data))).frames())


def test_reader_rejects_a_truncated_payload():
    stream = io.BytesIO()
    with MyvWriter(stream, a_header()) as writer:
        writer.frame(FrameType.FULL, b"abcdef")
    truncated = stream.getvalue()[:-3]
    with pytest.raises(BadMyvFile, match="truncated"):
        list(MyvReader(io.BytesIO(truncated)).frames())


def test_entropy_flag_survives_the_round_trip():
    assert unpack_header(pack_header(a_header(entropy=False))).entropy is False
    assert unpack_header(pack_header(a_header(entropy=True))).entropy is True


def test_unpack_rejects_an_older_format_version():
    """M8 added an intra filter byte to three payloads, so v1 files cannot be read.

    Rejecting them is the point of the version byte: a v1 payload would otherwise decode
    with its first data byte misread as a filter id, silently and wrongly.
    """
    from src.format import VERSION

    assert VERSION == 2
    data = bytearray(pack_header(a_header()))
    data[4] = 1  # the version byte, after the 4-byte magic
    with pytest.raises(BadMyvFile, match="version 1"):
        unpack_header(bytes(data))
