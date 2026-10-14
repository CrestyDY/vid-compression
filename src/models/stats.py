"""Size accounting for one encode run.

Deliberately NOT frozen, unlike every other model here: this one accumulates as the encoder
runs. It still belongs in models/ because it is a data object — rendering it as a report is
src/stats.py's job.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.frame import FrameType


class CodecStats(BaseModel):
    width: int
    height: int
    fps: float
    source_bytes: int
    entropy_enabled: bool
    #: Header plus per-frame record overhead, so the report matches the file on disk.
    container_bytes: int = 0

    frames: int = 0
    #: Payload sizes before the entropy stage.
    raw_bytes: int = 0
    #: Payload sizes as stored.
    stored_bytes: int = 0
    changed_pixels: int = 0
    mode_counts: dict[FrameType, int] = Field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return self.stored_bytes + self.container_bytes

    @property
    def raw_frame_bytes(self) -> int:
        """What the frames would cost uncompressed — the baseline worth beating."""
        return self.width * self.height * 3 * self.frames

    def record(
        self,
        frame_type: FrameType,
        raw: int,
        stored: int,
        changed: int | None = None,
    ) -> None:
        self.frames += 1
        self.raw_bytes += raw
        self.stored_bytes += stored
        if changed is not None:
            self.changed_pixels += changed
        self.mode_counts[frame_type] = self.mode_counts.get(frame_type, 0) + 1
