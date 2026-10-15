"""Encoder-side knobs.

Never stored in the file - payloads are self-describing, so a decoder reads what was done
rather than being configured to match the encoder. That is what keeps the container format
stable as modes gain knobs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModeOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Side length of the square grid cells used by BLOCK_DELTA and MOTION.
    #:
    #: The two modes want opposite things, and docs/m4-results.md measures it: BLOCK_DELTA
    #: wants large cells, because the unchanged pixels a big cell drags in are delta zeros
    #: that zstd erases almost for free. MOTION wants small cells, because a big cell
    #: straddling a moving object's edge cannot be described by one vector and pays a full
    #: residual. 16 is the compromise; `slide` swings 2.3x across 8/16/32.
    block_size: int = Field(default=16, ge=1, le=255)

    #: How far MOTION searches for a match, in pixels, in each direction.
    #:
    #: Bounded by 127 because displacements are stored as int8. Cost is quadratic in this:
    #: the search visits (2r+1)^2 candidate displacements, so 7 is 225 candidates and 15
    #: is 961. On `slide`, which moves 3px/frame, range 3 finds everything and anything
    #: beyond it costs time for nothing - see docs/m4-results.md.
    search_range: int = Field(default=7, ge=0, le=127)

    #: TILED grid: N means an NxN arrangement of tiles, each picking its own mode.
    #:
    #: Raising it is safe for size - TILED only gets written when it actually wins - but it
    #: costs encode time linearly in the tile count.
    grid: int = Field(default=4, ge=1, le=64)


DEFAULT_OPTIONS = ModeOptions()
