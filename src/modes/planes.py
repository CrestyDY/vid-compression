"""PLANES — one frame, one independently chosen mode per plane.

Every mode before this one codes a single array. That was enough while a frame *was* a
single array, which it is for RGB24. A subsampled format breaks the assumption: `yuv420p`
is three planes at two different resolutions, and they do not behave alike. Luma carries
the detail and the motion; chroma is a quarter the size, much smoother, and frequently
unchanged between frames where luma is not.

So the right thing is not to code the planes together but to let each pick its own mode,
exactly as TILED lets each tile pick its own — one level further out::

    count   : u8        number of planes
    then per plane, in the format's own order:
        mode    : u8
        length  : uvarint
        payload : length bytes

``count`` duplicates what the header's pixel format already implies. It is stored anyway so
the payload stays independently decodable, which is a property every other mode's payload
has and worth one byte a frame to keep.

**This is not a mode in** :data:`src.modes.MODES`. Those code one plane and share one
signature; this wraps a tuple of them and has its own. Single-plane formats never use it:
an RGB24 file stores the chosen mode in the frame record exactly as it did before, so
nothing about those files changed.
"""

from __future__ import annotations

import numpy as np

from src.utils.bitstream import BitReader, BitWriter, TruncatedStream

#: More planes than any pixel format here has, as a corrupt-payload guard.
MAX_PLANES = 8


def _registry():
    """Look up the mode table lazily, to avoid an import cycle with ``src.modes``."""
    from src.modes import MODES, FrameType

    return MODES, FrameType


def pack(parts: list[tuple[int, bytes]]) -> bytes:
    """Assemble a payload from per-plane ``(frame_type, payload)`` pairs."""
    if not 1 <= len(parts) <= MAX_PLANES:
        raise ValueError(f"a frame has 1..{MAX_PLANES} planes, got {len(parts)}")
    writer = BitWriter()
    writer.u8(len(parts))
    for frame_type, payload in parts:
        writer.u8(int(frame_type))
        writer.uvarint(len(payload))
        writer.blob(payload)
    return writer.getvalue()


def unpack(payload: bytes):
    """Inverse of :func:`pack`. Returns ``[(FrameType, payload), ...]``."""
    _modes, frame_type_enum = _registry()
    reader = BitReader(payload)
    try:
        count = reader.u8()
        if not 1 <= count <= MAX_PLANES:
            raise ValueError(f"PLANES claims {count} planes, out of range 1..{MAX_PLANES}")
        parts = []
        for index in range(count):
            type_id = reader.u8()
            try:
                mode = frame_type_enum(type_id)
            except ValueError:
                raise ValueError(
                    f"PLANES plane {index} has unknown frame type {type_id}"
                ) from None
            length = reader.uvarint()
            parts.append((mode, reader.blob(length)))
    except TruncatedStream as exc:
        raise ValueError(f"PLANES payload is truncated: {exc}") from None
    if reader.remaining():
        raise ValueError(f"PLANES has {reader.remaining()} trailing bytes")
    return parts


def decode(
    payload: bytes,
    prev: tuple[np.ndarray, ...] | None,
    shapes: tuple[tuple[int, int, int], ...],
) -> tuple[np.ndarray, ...]:
    """Reconstruct every plane, each against its own reference plane."""
    modes, frame_type_enum = _registry()
    parts = unpack(payload)
    if len(parts) != len(shapes):
        raise ValueError(
            f"PLANES holds {len(parts)} planes but this pixel format has {len(shapes)}"
        )
    if prev is not None and len(prev) != len(shapes):
        raise ValueError(
            f"reference frame has {len(prev)} planes, expected {len(shapes)}"
        )

    out = []
    for index, ((mode, plane_payload), shape) in enumerate(zip(parts, shapes)):
        if mode is frame_type_enum.PLANES:
            raise ValueError("a PLANES plane may not itself be PLANES; no recursion")
        reference = None if prev is None else prev[index]
        try:
            out.append(modes[mode].decode(plane_payload, reference, shape))
        except KeyError:
            raise ValueError(f"PLANES plane {index} names non-plane mode {mode!r}") from None
    return tuple(out)
