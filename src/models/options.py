"""Encoder-side knobs.

Never stored in the file — payloads are self-describing, so a decoder reads what was done
rather than being configured to match the encoder. That is what keeps the container format
stable as modes gain knobs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModeOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Side length of the square grid cells used by BLOCK_DELTA.
    #:
    #: The whole tuning story: small blocks track the changed region tightly but pay 4
    #: bytes of index each and need many of them; large blocks amortise the index but drag
    #: in a lot of unchanged pixels. 16 is the reference's compromise and the starting
    #: point here — see docs/m3-results.md for the sweep.
    block_size: int = Field(default=16, ge=1, le=255)

    #: How far MOTION searches for a match, in pixels, in each direction.
    #:
    #: Bounded by 127 because displacements are stored as int8. Cost is quadratic in this:
    #: the search visits (2r+1)^2 candidate displacements, so 7 is 225 candidates and 15
    #: is 961.
    search_range: int = Field(default=7, ge=0, le=127)


DEFAULT_OPTIONS = ModeOptions()
