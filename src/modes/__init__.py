"""Encoding modes — the competing representations of "what changed".

No single representation is always best. Each mode here is one answer to "how do I describe
this frame most cheaply", and they are kept deliberately interchangeable so the encoder can
price several and keep the smallest.

Every mode module exposes the same three functions::

    encode(cur, prev, opts) -> bytes
    decode(payload, prev, shape) -> ndarray
    cost(cur, prev, opts) -> int          # encoded size, ideally without encoding

``prev`` is ``None`` for modes that don't predict from a reference frame.

``decode`` takes no options: every mode's payload carries whatever parameters it needs in
its own bytes. That keeps the container format stable as modes gain knobs, and means a
decoder never has to be configured to match an encoder.
"""

from __future__ import annotations

from src.models.frame import FrameType
from src.modes import block_delta, full, motion, pixel_delta, tiled

MODES = {
    FrameType.FULL: full,
    FrameType.PIXEL_DELTA: pixel_delta,
    FrameType.BLOCK_DELTA: block_delta,
    FrameType.MOTION: motion,
    FrameType.TILED: tiled,
}

#: Modes that reconstruct from a reference frame, so are unavailable for frame 0.
PREDICTED = frozenset(
    {
        FrameType.PIXEL_DELTA,
        FrameType.BLOCK_DELTA,
        FrameType.MOTION,
        FrameType.TILED,
    }
)

#: Modes a TILED tile may use. TILED itself is excluded: recursion would be a quadtree,
#: which needs a depth limit and a stopping rule this codec does not have yet.
TILEABLE = frozenset(
    {
        FrameType.FULL,
        FrameType.PIXEL_DELTA,
        FrameType.BLOCK_DELTA,
        FrameType.MOTION,
    }
)

__all__ = [
    "MODES",
    "PREDICTED",
    "TILEABLE",
    "FrameType",
    "block_delta",
    "full",
    "motion",
    "pixel_delta",
    "tiled",
]
