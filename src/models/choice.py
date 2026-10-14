"""The winning mode for one frame, plus what every candidate would have cost."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.models.frame import FrameType


class Choice(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_type: FrameType
    payload: bytes
    #: Pre-entropy size of every candidate considered, keyed by mode.
    candidate_raw: dict[FrameType, int] = Field(default_factory=dict)
    #: Post-entropy size of every candidate. Only the ``trial`` strategy fills this in,
    #: because it is the only one that runs the entropy stage.
    candidate_stored: dict[FrameType, int] = Field(default_factory=dict)
