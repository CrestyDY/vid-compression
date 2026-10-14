"""FULL — store the whole frame, predicting nothing from any other frame.

The baseline every other mode is measured against, and the only mode needing no reference
frame — so it is what frame 0 must be, what keyframes will be, and what wins whenever
prediction isn't worth it. On real footage that turns out to be very often, which is why it
is eventually worth making good rather than merely simple.

At M1 the payload is the raw pixels. M8 adds intra prediction here, which changes the
payload's *content* but not its length — so the closed-form ``cost`` below survives that
change untouched.
"""

from __future__ import annotations

import numpy as np


def encode(cur: np.ndarray, prev: np.ndarray | None = None, opts=None) -> bytes:
    """Serialise the frame. ``prev`` and ``opts`` exist for interface symmetry only."""
    return np.ascontiguousarray(cur).tobytes()


def decode(payload: bytes, prev: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    expected = int(np.prod(shape))
    if len(payload) != expected:
        raise ValueError(
            f"FULL payload is {len(payload)} bytes, expected {expected} for shape {shape}"
        )
    return np.frombuffer(payload, dtype=np.uint8, count=expected).reshape(shape).copy()


def cost(cur: np.ndarray, prev: np.ndarray | None = None, opts=None) -> int:
    """Encoded size in bytes, without doing the encoding. Exact, not an estimate."""
    return int(cur.size)
