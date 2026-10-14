"""The entropy-coding stage — the last box in the pipeline.

Separate from everything else so reports can show pre-entropy and post-entropy sizes
independently: what the codec's representation bought and what generic compression bought
are different questions, and conflating them hides which one is doing the work.

This is the file that eventually becomes a hand-written range coder.
"""

from __future__ import annotations

import zstandard

#: Fixed, so size comparisons across runs and versions stay meaningful.
LEVEL = 10


def compress(payload: bytes, *, enabled: bool) -> bytes:
    """``enabled`` is keyword-only so a bare ``compress(data, True)`` can't happen."""
    if not enabled:
        return payload  # identity, not a copy
    return zstandard.ZstdCompressor(level=LEVEL).compress(payload)


def decompress(payload: bytes, *, enabled: bool) -> bytes:
    if not enabled:
        return payload
    return zstandard.ZstdDecompressor().decompress(payload)
