"""Intra prediction - making a block of pixels compressible before the entropy stage.

Until now every pixel-carrying payload wrote raw interleaved RGB and hoped zstd could do
something with it. It mostly cannot: neighbouring pixels in a photograph are *similar* but
rarely *identical*, and an LZ compressor needs exact repeats. The information is there, it is
just in the wrong shape.

M1 measured the cost of that precisely. On `cut`, whose first frame is a smooth gradient,
raw interleaved RGB compressed to 123,806 bytes; the same frame intra-predicted compresses
to **71 bytes**. And M5's comparison against FFV1 - an intra-only codec with no temporal
prediction at all - found it beating this codec 1.81x on real footage, because its intra
modelling is simply stronger. That is the gap this module closes.

Two transforms fix it, and they compose:

**De-interleaving.** ``R G B R G B ...`` puts three weakly-related channels next to each
other. Splitting into ``RRR... GGG... BBB...`` puts each channel next to itself, where the
correlation actually is.

**Spatial prediction.** Store each pixel as its difference from a neighbour rather than its
value. A smooth gradient becomes a run of near-identical small numbers; a flat region becomes
zeros. This is what PNG does, and it is the single reason PNG beats raw-deflate so heavily.

Both are exactly invertible via modular arithmetic, so nothing is lost. The prediction
inverses are cumulative sums, which numpy does in one vectorised pass - a per-pixel Python
loop would have made this unusable at 1080p.

The filter id is stored in the payload, so the decoder is never configured to match the
encoder; it reads what was done and undoes it.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Filter(IntEnum):
    """Stored as a byte in the payload - values are part of the file format."""

    RAW = 0          #: interleaved, untouched
    PLANAR = 1       #: de-interleaved only
    PLANAR_SUB = 2   #: de-interleaved, then horizontal differences
    PLANAR_UP = 3    #: de-interleaved, then vertical differences
    PLANAR_XY = 4    #: de-interleaved, then horizontal *and* vertical differences


def _to_planes(frame: np.ndarray) -> np.ndarray:
    """(h, w, c) -> (c, h, w), contiguous."""
    return np.ascontiguousarray(frame.transpose(2, 0, 1))


def _from_planes(planes: np.ndarray) -> np.ndarray:
    """(c, h, w) -> (h, w, c), contiguous."""
    return np.ascontiguousarray(planes.transpose(1, 2, 0))


def _diff(planes: np.ndarray, axis: int) -> np.ndarray:
    """Replace each value with its difference from the previous one along ``axis``.

    uint8 subtraction wraps, so this is exactly invertible by a cumulative sum - the same
    modular trick the delta modes use, applied spatially instead of temporally.
    """
    out = planes.copy()
    if axis == 2:
        out[:, :, 1:] = planes[:, :, 1:] - planes[:, :, :-1]
    else:
        out[:, 1:, :] = planes[:, 1:, :] - planes[:, :-1, :]
    return out


def _undiff(planes: np.ndarray, axis: int) -> np.ndarray:
    """Inverse of :func:`_diff`. ``cumsum`` in uint8 wraps to match."""
    return np.cumsum(planes, axis=axis, dtype=np.uint8)


def apply(frame: np.ndarray, filter_id: Filter) -> bytes:
    """Transform a frame into bytes under the given filter."""
    if filter_id == Filter.RAW:
        return np.ascontiguousarray(frame).tobytes()

    planes = _to_planes(frame)
    if filter_id == Filter.PLANAR_SUB:
        planes = _diff(planes, axis=2)
    elif filter_id == Filter.PLANAR_UP:
        planes = _diff(planes, axis=1)
    elif filter_id == Filter.PLANAR_XY:
        planes = _diff(_diff(planes, axis=2), axis=1)
    return planes.tobytes()


def invert(data: bytes, filter_id: int, shape: tuple[int, ...]) -> np.ndarray:
    """Undo :func:`apply`, returning an ``(h, w, c)`` frame."""
    try:
        filter_id = Filter(filter_id)
    except ValueError:
        raise ValueError(f"unknown intra filter {filter_id}") from None

    height, width, channels = shape
    expected = height * width * channels
    if len(data) < expected:
        raise ValueError(
            f"filtered body is {len(data)} bytes, expected {expected} for shape {shape}"
        )

    if filter_id == Filter.RAW:
        return (
            np.frombuffer(data, np.uint8, count=expected).reshape(shape).copy()
        )

    planes = np.frombuffer(data, np.uint8, count=expected).reshape(
        channels, height, width
    )
    if filter_id == Filter.PLANAR_SUB:
        planes = _undiff(planes, axis=2)
    elif filter_id == Filter.PLANAR_UP:
        planes = _undiff(planes, axis=1)
    elif filter_id == Filter.PLANAR_XY:
        # Reverse order: the vertical difference was applied last, so undo it first.
        planes = _undiff(_undiff(planes, axis=1), axis=2)
    return _from_planes(planes)


def _entropy(data: np.ndarray) -> float:
    """Order-0 entropy of the byte histogram, in bits per byte.

    A cheap stand-in for "how well will zstd do on this". It ignores the LZ side of the
    compressor, so it is a proxy rather than an oracle - but it costs one ``bincount`` over
    the plane and ranks these five filters correctly, which is all it has to do.

    Choosing by proxy rather than by trial-compressing all five matters: at 1080p a zstd
    pass is tens of milliseconds, and FULL is priced on every single frame.
    """
    counts = np.bincount(np.asarray(data).ravel(), minlength=256).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


#: Filters worth trying. RAW is included so a frame that genuinely does not benefit pays
#: nothing but the one filter byte.
CANDIDATES = (
    Filter.RAW,
    Filter.PLANAR,
    Filter.PLANAR_SUB,
    Filter.PLANAR_UP,
    Filter.PLANAR_XY,
)

#: A filter has to beat the simplest option by this many bits per byte to be worth using.
#: On incompressible content every filter scores within a hair of 8 bits, and picking the
#: nominal winner there means doing a transform for a gain indistinguishable from noise.
MARGIN = 0.02


def choose(frame: np.ndarray) -> Filter:
    """Pick the filter whose output looks most compressible.

    Biased toward doing less: a more elaborate filter is only chosen when it beats the
    simplest acceptable one by a real margin, not by rounding.
    """
    planes = _to_planes(frame)
    scores = {
        Filter.RAW: _entropy(frame),
        Filter.PLANAR: _entropy(planes),
        Filter.PLANAR_SUB: _entropy(_diff(planes, axis=2)),
        Filter.PLANAR_UP: _entropy(_diff(planes, axis=1)),
        Filter.PLANAR_XY: _entropy(_diff(_diff(planes, axis=2), axis=1)),
    }
    best = min(scores.values())
    # Among everything within MARGIN of the best score, take the cheapest filter.
    return min(f for f, s in scores.items() if s <= best + MARGIN)
