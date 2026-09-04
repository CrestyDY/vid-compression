"""FrameType enum and Frame: type byte plus payload bytes."""

from enum import IntEnum

from pydantic import BaseModel, ConfigDict

class FrameType(IntEnum):

    FULL = 0  # for now, only full frames are supported


class Frame(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: FrameType
    payload: bytes
