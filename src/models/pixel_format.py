"""How raw frame bytes are laid out.

Everything else in the codec talks about frames through one of these, so the pixel-layout
assumption lives in exactly one place.

**A frame is a tuple of planes.** RGB24 is a one-plane format whose single plane carries
three channels; `yuv420p` is three planes, the last two at half resolution in both axes.
Making the one-plane case a tuple of length one rather than a special case is what keeps
every stage above this module from branching on the format.

The reason subsampled formats exist here at all is measured, not theoretical. Asking ffmpeg
for `rgb24` from a `yuv420p` source makes it *interpolate* three quarters of the chroma, and
the codec then spends bytes losslessly preserving the interpolation: 1.42x more than coding
the source's own planes, with intra prediction alone. See docs/next-lever.md.

**Full-range variants get their own ids.** `yuvj420p` differs from `yuv420p` only by a
colour-range tag, but asking ffmpeg for the wrong one of the two makes swscale apply a
range conversion, which rounds — and a codec that rounds is not lossless. So the stored
format names the exact `pix_fmt` to ask for, and never a compatible-looking neighbour.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PlaneSpec(BaseModel):
    """One plane's geometry, as divisors of the frame's dimensions.

    Dimensions divide with rounding *up*, matching ffmpeg: a 5-pixel-wide `yuv420p` frame
    has a 3-sample-wide chroma plane, not 2. Getting this wrong truncates a column.
    """

    model_config = ConfigDict(frozen=True)

    w_div: int = 1
    h_div: int = 1
    channels: int = 1

    def shape(self, width: int, height: int) -> tuple[int, int, int]:
        return (
            -(-height // self.h_div),
            -(-width // self.w_div),
            self.channels,
        )

    def nbytes(self, width: int, height: int) -> int:
        h, w, c = self.shape(width, height)
        return h * w * c


class PixelFormat(BaseModel):
    """A pixel format: id, names, and the planes a frame is made of.

    ``id`` is what gets stored in the .myv header, so these values are part of the file
    format and must never be reassigned.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    ffmpeg_name: str
    planes: tuple[PlaneSpec, ...]

    @property
    def is_planar(self) -> bool:
        """Whether a frame is more than one plane, and so needs the PLANES mode."""
        return len(self.planes) > 1

    @property
    def channels(self) -> int:
        """Channels in the first plane. Only meaningful for single-plane formats."""
        return self.planes[0].channels

    def plane_shapes(self, width: int, height: int) -> tuple[tuple[int, int, int], ...]:
        return tuple(p.shape(width, height) for p in self.planes)

    def frame_shape(self, width: int, height: int) -> tuple[int, int, int]:
        """Shape of the *first* plane.

        Kept because a single-plane format has exactly one shape and most of the codec was
        written when that was the only possibility. Multi-plane code paths use
        :meth:`plane_shapes`.
        """
        return self.planes[0].shape(width, height)

    def frame_nbytes(self, width: int, height: int) -> int:
        """Bytes in one whole frame, across every plane."""
        return sum(p.nbytes(width, height) for p in self.planes)


_FULL = PlaneSpec(w_div=1, h_div=1, channels=1)
_HALF_BOTH = PlaneSpec(w_div=2, h_div=2, channels=1)
_HALF_WIDE = PlaneSpec(w_div=2, h_div=1, channels=1)

RGB24 = PixelFormat(
    id=0, name="rgb24", ffmpeg_name="rgb24",
    planes=(PlaneSpec(channels=3),),
)
YUV420P = PixelFormat(
    id=1, name="yuv420p", ffmpeg_name="yuv420p",
    planes=(_FULL, _HALF_BOTH, _HALF_BOTH),
)
YUV422P = PixelFormat(
    id=2, name="yuv422p", ffmpeg_name="yuv422p",
    planes=(_FULL, _HALF_WIDE, _HALF_WIDE),
)
YUV444P = PixelFormat(
    id=3, name="yuv444p", ffmpeg_name="yuv444p",
    planes=(_FULL, _FULL, _FULL),
)
# The J variants are the same geometry with a full-range colour tag. Separate ids because
# the tag changes what ffmpeg must be asked for; see the module docstring.
YUVJ420P = PixelFormat(
    id=4, name="yuvj420p", ffmpeg_name="yuvj420p",
    planes=(_FULL, _HALF_BOTH, _HALF_BOTH),
)
YUVJ422P = PixelFormat(
    id=5, name="yuvj422p", ffmpeg_name="yuvj422p",
    planes=(_FULL, _HALF_WIDE, _HALF_WIDE),
)
YUVJ444P = PixelFormat(
    id=6, name="yuvj444p", ffmpeg_name="yuvj444p",
    planes=(_FULL, _FULL, _FULL),
)
GRAY = PixelFormat(
    id=7, name="gray", ffmpeg_name="gray",
    planes=(_FULL,),
)

ALL = (RGB24, YUV420P, YUV422P, YUV444P, YUVJ420P, YUVJ422P, YUVJ444P, GRAY)

_BY_ID = {fmt.id: fmt for fmt in ALL}
_BY_NAME = {fmt.name: fmt for fmt in ALL}

#: Names accepted by ``--pix-fmt``, plus the default that follows the source.
NAMES = ("source",) + tuple(fmt.name for fmt in ALL)


def by_id(fmt_id: int) -> PixelFormat:
    """Map a stored header byte back to a format. Raises rather than returning None."""
    try:
        return _BY_ID[fmt_id]
    except KeyError:
        raise ValueError(
            f"unknown pixel format id {fmt_id}; this file was probably written by a "
            f"newer version of the codec"
        ) from None


def by_name(name: str) -> PixelFormat:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown pixel format {name!r}; pick one of {NAMES}"
        ) from None
