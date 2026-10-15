"""TILED — split the frame into a grid and let each tile pick its own mode.

Every mode before this one commits the *whole frame* to a single representation. That is a
real limitation, and `mixed` was built to expose it: a frame can easily have a static top
half and a moving bottom half, and the frame-level selector has to pick one answer for both,
losing on whichever half disagrees.

TILED divides the frame into a grid and stores an independently chosen mode per tile::

    grid_rows : uvarint
    grid_cols : uvarint
    then per tile, row-major:
        mode    : u8
        length  : uvarint
        payload : length bytes

**On the cost of the selectors.** Four modes need 2 bits each, so the mode bytes are trivial.
The real overhead is the per-tile *lengths*, which is why they are varints rather than the
u32 the rest of the format uses: a tile payload is usually small, and three wasted bytes per
tile per frame is kilobytes per second at sixteen tiles and 25 fps.

**Tiles are independent.** Each mode sees only its own slice of the current and reference
frames, so a MOTION tile can only find matches within its own region. An object crossing a
tile boundary is therefore predicted worse than it would be frame-wide. That cost is real but
bounded: only blocks within `search_range` of a boundary are affected. Sharing the reference
across tiles would fix it at the price of giving every mode an origin offset, which is the
right trade to revisit if boundary loss ever shows up in the numbers.

**No recursion.** A tile may not itself be TILED. Allowing it would turn this into a
quadtree — which is what HEVC does, and the natural next step — but it needs a depth limit
and a cost model that knows when to stop subdividing.

**Where the choosing happens.** This module's ``encode`` ranks tiles by *payload* size, which
is the best it can do without reaching into the entropy stage. The encoder uses
``src.selector`` instead, which ranks each tile after compression — a better criterion that a
mode module has no business knowing about. Both paths produce the same format.
"""

from __future__ import annotations

import numpy as np

from src.models.options import DEFAULT_OPTIONS
from src.utils.bitstream import BitReader, BitWriter, TruncatedStream

MAX_GRID = 64


def _registry():
    """Look up the mode table lazily.

    ``src.modes.__init__`` imports this module, so importing it back at module scope would
    be a cycle. Deferring the lookup to call time keeps the import order irrelevant.
    """
    from src.modes import MODES, TILEABLE, FrameType

    return MODES, TILEABLE, FrameType


def grid_dims(shape: tuple[int, ...], grid: int) -> tuple[int, int]:
    """``grid`` means a ``grid x grid`` arrangement, clamped so no tile is empty."""
    height, width = shape[:2]
    return (max(1, min(grid, height)), max(1, min(grid, width)))


def tiles(shape: tuple[int, ...], rows: int, cols: int):
    """Yield ``(y0, y1, x0, x1)`` for each tile, row-major.

    Boundaries are computed by proportion rather than by a fixed stride, so a 320-wide
    frame in 3 columns gives 106/107/107 rather than 106/106/108 and a ragged remainder.
    """
    height, width = shape[:2]
    for row in range(rows):
        y0 = height * row // rows
        y1 = height * (row + 1) // rows
        for col in range(cols):
            x0 = width * col // cols
            x1 = width * (col + 1) // cols
            yield y0, y1, x0, x1


def pack(rows: int, cols: int, parts: list[tuple[int, bytes]]) -> bytes:
    """Assemble a payload from per-tile ``(frame_type, payload)`` pairs, row-major."""
    if rows * cols != len(parts):
        raise ValueError(f"{rows}x{cols} grid needs {rows * cols} tiles, got {len(parts)}")
    writer = BitWriter()
    writer.uvarint(rows)
    writer.uvarint(cols)
    for frame_type, payload in parts:
        writer.u8(int(frame_type))
        writer.uvarint(len(payload))
        writer.blob(payload)
    return writer.getvalue()


def unpack(payload: bytes):
    """Inverse of :func:`pack`. Returns ``(rows, cols, [(FrameType, payload), ...])``."""
    _modes, _tileable, frame_type_enum = _registry()
    reader = BitReader(payload)
    try:
        rows = reader.uvarint()
        cols = reader.uvarint()
        if not 1 <= rows <= MAX_GRID or not 1 <= cols <= MAX_GRID:
            raise ValueError(f"TILED grid {rows}x{cols} is out of range 1..{MAX_GRID}")
        parts = []
        for tile in range(rows * cols):
            type_id = reader.u8()
            try:
                mode = frame_type_enum(type_id)
            except ValueError:
                raise ValueError(
                    f"TILED tile {tile} has unknown frame type {type_id}"
                ) from None
            length = reader.uvarint()
            parts.append((mode, reader.blob(length)))
    except TruncatedStream as exc:
        raise ValueError(f"TILED payload is truncated: {exc}") from None
    if reader.remaining():
        raise ValueError(f"TILED has {reader.remaining()} trailing bytes")
    return rows, cols, parts


def encode(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> bytes:
    """Choose a mode per tile by smallest payload, and pack the result.

    The standalone path, used when TILED is exercised directly. The encoder goes through
    ``src.selector`` instead, which ranks tiles by post-entropy size.
    """
    if prev is None:
        raise ValueError("TILED needs a reference frame; its tiles are all predicted")
    if cur.shape != prev.shape:
        raise ValueError(f"shape mismatch: current {cur.shape} vs reference {prev.shape}")

    modes, tileable, _ft = _registry()
    grid = (DEFAULT_OPTIONS if opts is None else opts).grid
    rows, cols = grid_dims(cur.shape, grid)

    parts: list[tuple[int, bytes]] = []
    for y0, y1, x0, x1 in tiles(cur.shape, rows, cols):
        tile_cur = np.ascontiguousarray(cur[y0:y1, x0:x1])
        tile_prev = np.ascontiguousarray(prev[y0:y1, x0:x1])
        best: tuple[int, bytes] | None = None
        for frame_type in sorted(tileable):
            payload = modes[frame_type].encode(tile_cur, tile_prev, opts)
            if best is None or len(payload) < len(best[1]):
                best = (frame_type, payload)
        assert best is not None
        parts.append(best)
    return pack(rows, cols, parts)


def decode(payload: bytes, prev: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if prev is None:
        raise ValueError("TILED needs a reference frame to reconstruct against")
    modes, _tileable, frame_type_enum = _registry()
    rows, cols, parts = unpack(payload)

    # A grid finer than the frame would mean empty tiles, so the boundaries the encoder
    # used could not be recovered. grid_dims clamps on the way in, so this only fires on
    # a corrupt payload.
    height, width = shape[:2]
    if rows > height or cols > width:
        raise ValueError(
            f"TILED grid {rows}x{cols} does not fit a {width}x{height} frame"
        )

    out = prev.copy()
    for (y0, y1, x0, x1), (mode, tile_payload) in zip(
        tiles(shape, rows, cols), parts
    ):
        if mode is frame_type_enum.TILED:
            raise ValueError("a TILED tile may not itself be TILED; no recursion")
        tile_shape = (y1 - y0, x1 - x0, shape[2])
        tile_prev = np.ascontiguousarray(prev[y0:y1, x0:x1])
        out[y0:y1, x0:x1] = modes[mode].decode(tile_payload, tile_prev, tile_shape)
    return out


def cost(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> int:
    """Encoded size in bytes. Needs every tile encoded, so it encodes and measures."""
    return len(encode(cur, prev, opts))
