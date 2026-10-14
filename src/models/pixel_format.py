"""How raw frame bytes are laid out.

Everything else in the codec talks about frames through one of these, so the RGB assumption
lives in exactly one place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PixelFormat(BaseModel):
    """PixelFormat: id, name, ffmpeg name, channel count.

    ``id`` is what gets stored in the .myv header, so these values are part of the file
    format and must never be reassigned.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    ffmpeg_name: str
    channels: int

    def frame_shape(self, width: int, height: int) -> tuple[int, int, int]:
        """Numpy shape of one frame: rows, columns, channels."""
        return (height, width, self.channels)

    def frame_nbytes(self, width: int, height: int) -> int:
        return width * height * self.channels


RGB24 = PixelFormat(id=0, name="rgb24", ffmpeg_name="rgb24", channels=3)

# Adding YUV later means adding an instance here and reading `channels` instead of
# grepping for the number 3.
_BY_ID = {fmt.id: fmt for fmt in (RGB24,)}


def by_id(fmt_id: int) -> PixelFormat:
    """Map a stored header byte back to a format. Raises rather than returning None."""
    try:
        return _BY_ID[fmt_id]
    except KeyError:
        raise ValueError(
            f"unknown pixel format id {fmt_id}; this file was probably written by a "
            f"newer version of the codec"
        ) from None
