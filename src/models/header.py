"""The .myv file header as a value object. Byte packing lives in src/format.py.

Three things this model deliberately does not hold:

- **The magic bytes.** A format constant, not per-file state; they belong beside the struct
  that writes them.
- **A raw pix_fmt id.** It holds the ``PixelFormat`` itself, so it can answer ``frame_shape``
  without anything downstream having to look the id up.
- **Anything about entropy coding beyond one flag.** How payloads are compressed is the
  entropy stage's business; the header only records whether they were.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.models.pixel_format import RGB24, PixelFormat


class Header(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps_num: int = Field(gt=0)
    fps_den: int = Field(gt=0)
    #: Back-patched when the writer closes; 0 while a file is still being written.
    frame_count: int = Field(default=0, ge=0)
    pix_fmt: PixelFormat = RGB24
    #: Whether frame payloads are entropy-coded. Stored as a header flag bit.
    entropy: bool = True
    #: Which entropy codec was used. Stored in the header's reserved byte, so
    #: adding it needed no format version bump: a reader that ignores reserved
    #: assumes "zstd", which is id 0 and therefore what every older file is.
    entropy_codec: str = "zstd"

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Shape of the first plane. Only the whole frame for a single-plane format."""
        return self.pix_fmt.frame_shape(self.width, self.height)

    @property
    def plane_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return self.pix_fmt.plane_shapes(self.width, self.height)

    @property
    def frame_nbytes(self) -> int:
        return self.pix_fmt.frame_nbytes(self.width, self.height)
