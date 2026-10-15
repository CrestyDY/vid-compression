from __future__ import annotations

import pytest

from src.utils.bitstream import BitReader, BitWriter, TruncatedStream


def test_u8_and_u32_round_trip():
    w = BitWriter()
    w.u8(7)
    w.u32(70000)
    r = BitReader(w.getvalue())
    assert r.u8() == 7
    assert r.u32() == 70000


def test_blob_round_trips():
    w = BitWriter()
    w.blob(b"hello")
    assert BitReader(w.getvalue()).blob(5) == b"hello"


@pytest.mark.parametrize("value", [0, 1, 127, 128, 300, 16383, 16384, 2**31])
def test_uvarint_round_trips(value):
    w = BitWriter()
    w.uvarint(value)
    assert BitReader(w.getvalue()).uvarint() == value


def test_uvarint_is_one_byte_for_small_values():
    """The whole reason it exists: a u32 would waste 3 bytes per tile per frame."""
    w = BitWriter()
    w.uvarint(5)
    assert len(w.getvalue()) == 1


def test_uvarint_rejects_a_negative():
    with pytest.raises(ValueError, match="negative"):
        BitWriter().uvarint(-1)


def test_reader_reports_the_shortfall():
    """A truncated payload should say where it ran out and by how much."""
    r = BitReader(b"\x01")
    with pytest.raises(TruncatedStream, match="needed 4 bytes at offset 0, 3 short"):
        r.u32()


def test_tell_tracks_position():
    w = BitWriter()
    w.u8(1)
    w.u32(2)
    r = BitReader(w.getvalue())
    assert r.tell() == 0
    r.u8()
    assert r.tell() == 1
    r.u32()
    assert r.tell() == 5


def test_uvarint_rejects_an_unterminated_sequence():
    with pytest.raises(TruncatedStream):
        BitReader(b"\x80\x80\x80").uvarint()
