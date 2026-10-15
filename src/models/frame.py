"""FrameType enum and Frame: type byte plus payload bytes."""

from enum import IntEnum

from pydantic import BaseModel, ConfigDict


class FrameType(IntEnum):
    """Stored as a byte in each frame record — values are part of the file format.

    Gaps are reserved for the modes the roadmap plans, so their eventual arrival never
    renumbers anything already written to disk.
    """

    FULL = 0
    PIXEL_DELTA = 1
    BLOCK_DELTA = 2
    MOTION = 3
    # 4 reserved for TILED (M5)


class Frame(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: FrameType
    payload: bytes
