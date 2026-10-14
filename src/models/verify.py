"""Results of a losslessness check."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Mismatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame: int
    differing_pixels: int
    first_at: tuple[int, int] | None
    expected: tuple[int, ...] | None
    got: tuple[int, ...] | None
    max_abs_error: int

    def describe(self) -> str:
        if self.first_at is None:
            return f"frame {self.frame}: frame counts differ"
        y, x = self.first_at
        return (
            f"frame {self.frame}: {self.differing_pixels:,} pixels differ "
            f"(max channel error {self.max_abs_error}); "
            f"first at (x={x}, y={y}) expected {self.expected}, got {self.got}"
        )


class VerifyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    frames_checked: int
    source_frames: int
    decoded_frames: int
    mismatch: Mismatch | None = None

    @property
    def ok(self) -> bool:
        return self.mismatch is None and self.source_frames == self.decoded_frames
