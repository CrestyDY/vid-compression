"""Turning one raw frame buffer into planes, and back.

The model in :mod:`src.models.pixel_format` holds the *geometry*; this holds the two numpy
operations that geometry implies, plus the mapping from whatever ffprobe says the source is
to the format this codec should use for it.

Splitting is a view, not a copy: ``np.frombuffer`` over slices of the same buffer. That
matters at 1080p, where a copy per plane per frame would be 3 MB of pointless memcpy.
"""

from __future__ import annotations

import numpy as np

from src.models.pixel_format import (
    GRAY,
    RGB24,
    YUV420P,
    YUV422P,
    YUV444P,
    YUVJ420P,
    YUVJ422P,
    YUVJ444P,
    PixelFormat,
)

#: What to use for each pix_fmt ffprobe might report.
#:
#: Anything absent falls back to RGB24, which is always safe: ffmpeg will convert, and the
#: result is still coded losslessly — just losslessly *after* a conversion we did not choose.
#: The fallback is reported rather than silent, because "your 10-bit source is being coded
#: as 8-bit RGB" is exactly the kind of thing that should not be discovered from a size
#: report.
_SOURCE_MAP = {
    # True RGB sources. We ask for rgb24; a planar-GBR source is a plane reorder, which
    # swscale does without touching a sample value.
    "rgb24": RGB24,
    "bgr24": RGB24,
    "gbrp": RGB24,
    "rgb0": RGB24,
    "0rgb": RGB24,
    "bgr0": RGB24,
    "0bgr": RGB24,
    "pal8": RGB24,
    # Subsampled YUV: coded in the source's own planes, which is the whole point.
    "yuv420p": YUV420P,
    "yuv422p": YUV422P,
    "yuv444p": YUV444P,
    "yuvj420p": YUVJ420P,
    "yuvj422p": YUVJ422P,
    "yuvj444p": YUVJ444P,
    # Interleaved 4:2:2 holds exactly the samples of yuv422p, reordered.
    "yuyv422": YUV422P,
    "uyvy422": YUV422P,
    # Semi-planar 4:2:0: same samples as yuv420p with the chroma interleaved.
    "nv12": YUV420P,
    "nv21": YUV420P,
    "gray": GRAY,
}


def for_source(pix_fmt: str | None) -> tuple[PixelFormat, str | None]:
    """Pick the format to code a source in, plus a warning if we had to fall back.

    Returns ``(format, note)``. ``note`` is ``None`` when the source's samples are being
    coded as they are, and a sentence to show the user when they are not.
    """
    if not pix_fmt:
        return RGB24, (
            "could not determine the source pixel format; coding as rgb24. If the source "
            "is subsampled YUV this will be larger than it needs to be."
        )
    fmt = _SOURCE_MAP.get(pix_fmt)
    if fmt is not None:
        return fmt, None
    return RGB24, (
        f"source pixel format {pix_fmt!r} is not one this codec stores natively; coding "
        f"as rgb24. The result is still lossless with respect to the rgb24 frames ffmpeg "
        f"produces, but those are a conversion of the source, not the source itself."
    )


def split(buf: bytes, fmt: PixelFormat, width: int, height: int) -> tuple[np.ndarray, ...]:
    """One raw frame buffer -> one array per plane, as views into ``buf``."""
    planes: list[np.ndarray] = []
    offset = 0
    for spec in fmt.planes:
        shape = spec.shape(width, height)
        count = shape[0] * shape[1] * shape[2]
        planes.append(
            np.frombuffer(buf, dtype=np.uint8, count=count, offset=offset).reshape(shape)
        )
        offset += count
    return tuple(planes)


def join(planes: tuple[np.ndarray, ...] | list[np.ndarray]) -> bytes:
    """Planes -> one raw frame buffer, in the order ffmpeg expects them."""
    if len(planes) == 1:
        return np.ascontiguousarray(planes[0]).tobytes()
    return b"".join(np.ascontiguousarray(p).tobytes() for p in planes)
