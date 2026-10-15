"""FULL - store the whole frame, predicting nothing from any other frame.

The baseline every other mode is measured against, and the only mode needing no reference
frame - so it is what frame 0 must be, what keyframes will be, and what wins whenever
prediction isn't worth it. On real footage that turns out to be very often, which is why it
is worth making good rather than merely simple.

"Predicting nothing" means nothing *temporal*. Within the frame there is plenty to predict
from: neighbouring pixels are similar, and channels are correlated. So the payload is::

    filter : u8    see src.transform.Filter
    data   : h*w*c bytes, transformed under that filter

The filter de-interleaves the channels and stores spatial differences, which is what turns a
photograph from "incompressible to an LZ coder" into "runs of small numbers". It is exactly
invertible, so FULL remains lossless and still costs exactly one byte more than the raw
pixels in the worst case.
"""

from __future__ import annotations

import numpy as np

from src.transform import Filter, apply, choose, invert

_HEADER = 1  # the filter byte


def encode(cur: np.ndarray, prev: np.ndarray | None = None, opts=None) -> bytes:
    """Serialise the frame. ``prev`` and ``opts`` exist for interface symmetry only."""
    filter_id = choose(cur)
    return bytes([int(filter_id)]) + apply(cur, filter_id)


def decode(payload: bytes, prev: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    expected = _HEADER + int(np.prod(shape))
    if len(payload) != expected:
        raise ValueError(
            f"FULL payload is {len(payload)} bytes, expected {expected} for shape {shape}"
        )
    return invert(payload[_HEADER:], payload[0], shape)


def filter_of(payload: bytes) -> Filter:
    """Which intra filter this payload used. For tests, `info`, and debugging."""
    if not payload:
        raise ValueError("empty FULL payload")
    return Filter(payload[0])


def cost(cur: np.ndarray, prev: np.ndarray | None = None, opts=None) -> int:
    """Encoded size in bytes, without doing the encoding. Exact, not an estimate.

    Intra prediction changes the *content* of the payload, never its length - every filter
    is a permutation-and-difference of the same bytes. So the closed form survives, and the
    cheap analytic cost the selector depends on is still available.
    """
    return _HEADER + int(cur.size)
