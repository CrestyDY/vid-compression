"""PIXEL_DELTA — store only the pixels that changed, as coordinates plus RGB offsets.

This is the idea the project starts from: if a frame is mostly unchanged, describe the
handful of pixels that moved rather than the millions that didn't.

Payload layout::

    count : uint32
    count records of 4 + c bytes each, where c is the plane's channel count:
        x  : uint16
        y  : uint16
        d  : uint8 x c      one modular delta per channel

For RGB24 that is the original 7-byte record. A plane of a subsampled format has one
channel and so a 5-byte record; the count is taken from the plane's shape, which the
decoder already has from the header, so nothing extra is stored and existing files read
unchanged.

**The deltas are modular, and that is what makes it lossless.** A channel going from 2 to
250 is a change of -8 mod 256, not +248, so one byte covers every possible transition. No
clamping, no int16, no escape hatch for large jumps. numpy's uint8 arithmetic already wraps
this way, so ``cur - prev`` and ``prev + delta`` are exactly inverse operations for free.

**Four of the seven bytes are coordinates**, on RGB. That overhead is this mode's
defining weakness and is left in on purpose — a changed pixel costs 7 bytes no matter how
small its colour change, so a frame where 40% of pixels shifted by 1 costs nearly 2x what
simply storing the whole frame would. Fixing it by spatial grouping is what BLOCK_DELTA is
for. On a one-channel plane the ratio is worse still: 4 of 5 bytes.
"""

from __future__ import annotations

import numpy as np

MAX_COORD = 0xFFFF
_HEADER = 4  # the uint32 count

_CACHE: dict[int, np.dtype] = {}


def record_dtype(channels: int) -> np.dtype:
    """One changed pixel: 4 bytes of coordinates plus one delta per channel.

    Derived from the plane's channel count rather than fixed at three. A plane of a
    subsampled format has one channel, and the old fixed 3-byte delta field broadcast
    silently on the way in and then refused to broadcast on the way out — which produced
    a payload that encoded cleanly and could not be decoded at all.

    The channel count is not stored. It does not need to be: the decoder gets the plane's
    shape from the header's pixel format, exactly as every other mode does. So an RGB24
    payload is still 4 + 3 bytes per record and every existing file reads unchanged.
    """
    if channels < 1:
        raise ValueError(f"a plane needs at least one channel, got {channels}")
    dtype = _CACHE.get(channels)
    if dtype is None:
        dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("d", "u1", (channels,))])
        _CACHE[channels] = dtype
    return dtype


#: The RGB24 record, kept as a name because the format's 7-byte layout is referred to
#: throughout the documents and the tests.
RECORD = record_dtype(3)
assert RECORD.itemsize == 7, "the RGB record must stay 7 bytes; the format depends on it"


def encode(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> bytes:
    if prev is None:
        raise ValueError("PIXEL_DELTA needs a reference frame to diff against")
    if cur.shape != prev.shape:
        raise ValueError(f"shape mismatch: current {cur.shape} vs reference {prev.shape}")

    height, width = cur.shape[:2]
    if width - 1 > MAX_COORD or height - 1 > MAX_COORD:
        raise ValueError(
            f"frame is {width}x{height}; PIXEL_DELTA coordinates are uint16 and cannot "
            f"address beyond {MAX_COORD}"
        )

    ys, xs = np.nonzero(np.any(cur != prev, axis=2))
    # uint8 subtraction wraps, which is precisely the mod-256 delta we want.
    deltas = cur[ys, xs] - prev[ys, xs]

    records = np.empty(ys.size, dtype=record_dtype(cur.shape[2]))
    records["x"] = xs.astype(np.uint16, copy=False)
    records["y"] = ys.astype(np.uint16, copy=False)
    records["d"] = deltas

    return int(ys.size).to_bytes(_HEADER, "little") + records.tobytes()


def decode(payload: bytes, prev: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    record = record_dtype(shape[2])
    n = count(payload)
    expected = _HEADER + n * record.itemsize
    if len(payload) != expected:
        raise ValueError(
            f"PIXEL_DELTA claims {n} records ({expected} bytes) but payload is "
            f"{len(payload)} bytes"
        )
    if prev is None:
        raise ValueError("PIXEL_DELTA needs a reference frame to reconstruct against")

    out = prev.copy()
    if n:
        records = np.frombuffer(payload, dtype=record, count=n, offset=_HEADER)
        # Each pixel appears at most once, so this read-modify-write is unambiguous.
        out[records["y"], records["x"]] += records["d"]
    return out


def count(payload: bytes) -> int:
    """How many pixels this payload says changed."""
    if len(payload) < _HEADER:
        raise ValueError(f"PIXEL_DELTA payload is {len(payload)} bytes; needs at least 4")
    return int.from_bytes(payload[:_HEADER], "little")


def coordinates(payload: bytes, channels: int = 3) -> list[tuple[int, int]]:
    """The ``(y, x)`` of every changed pixel, in stored order. For tests and debugging."""
    n = count(payload)
    records = np.frombuffer(payload, dtype=record_dtype(channels), count=n, offset=_HEADER)
    return [(int(y), int(x)) for y, x in zip(records["y"], records["x"])]


def cost(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> int:
    """Encoded size in bytes, without building the payload. Exact, not an estimate."""
    if prev is None:
        raise ValueError("PIXEL_DELTA needs a reference frame")
    changed = int(np.count_nonzero(np.any(cur != prev, axis=2)))
    return _HEADER + changed * record_dtype(cur.shape[2]).itemsize
