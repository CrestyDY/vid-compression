"""BLOCK_DELTA — store whole changed blocks, dropping per-pixel coordinates.

PIXEL_DELTA spends 4 of every 7 bytes saying *where* a pixel is — measured at 57.1% of the
payload on four of five benchmark clips (docs/m1-results.md, Observation 2). That is fine
when changes are scattered, and terrible when they cluster — which is what changes actually
do. Moving objects, cursors, text, UI repaints: all of them dirty contiguous regions, and
re-stating full coordinates for every pixel in a solid rectangle is pure waste.

This mode divides the frame into a fixed grid and stores, for each block containing any
change, one block index plus every delta in that block::

    block_size : uint8    grid cell side length, so the payload is self-describing
    count      : uint32   number of changed blocks
    then count records:
        index  : uint32   block number, row-major across the grid
        deltas : bh*bw*3 bytes, modular, row-major within the block

The trade is coordinates for density. A changed pixel now costs 3 bytes instead of 7, but
every *un*changed pixel sharing its block costs 3 bytes too. So this wins when changes are
dense within the blocks they touch, and loses when they are sparse and scattered.

Cost for a block is fixed at ``4 + bh*bw*3`` regardless of how much inside it changed, which
makes block size the whole tuning story:

- **Small blocks** track the changed region tightly, wasting few untouched pixels, but pay
  4 bytes of index per block and need many of them.
- **Large blocks** amortise the index but drag in a lot of unchanged pixels.

Edge blocks are clipped to the frame, so dimensions need not divide evenly. Their payload is
correspondingly smaller, which is why records are variable-length and the decoder recomputes
each block's extent rather than trusting a fixed stride.
"""

from __future__ import annotations

import numpy as np

from src.models.options import DEFAULT_OPTIONS

MAX_BLOCKS = 0xFFFFFFFF
_HEADER = 5  # block_size (1) + count (4)
_INDEX = 4


def _block_size(opts) -> int:
    return (DEFAULT_OPTIONS if opts is None else opts).block_size


def _grid(shape: tuple[int, ...], block_size: int) -> tuple[int, int]:
    """How many blocks across and down, including partial edge blocks."""
    height, width = shape[:2]
    return (
        (height + block_size - 1) // block_size,
        (width + block_size - 1) // block_size,
    )


def _extent(index: int, shape: tuple[int, ...], block_size: int, cols: int):
    """The slice a block index refers to, clipped to the frame."""
    height, width = shape[:2]
    by, bx = divmod(index, cols)
    y0, x0 = by * block_size, bx * block_size
    return y0, min(y0 + block_size, height), x0, min(x0 + block_size, width)


def _changed_blocks(cur: np.ndarray, prev: np.ndarray, block_size: int) -> np.ndarray:
    """Indices of every block containing at least one differing pixel.

    Vectorised: build a per-pixel changed mask, pad it up to a whole number of blocks, then
    reduce each block to a single boolean with a reshape. No Python loop over blocks.
    """
    rows, cols = _grid(cur.shape, block_size)
    changed = np.any(cur != prev, axis=2)

    padded = np.zeros((rows * block_size, cols * block_size), dtype=bool)
    padded[: changed.shape[0], : changed.shape[1]] = changed

    per_block = padded.reshape(rows, block_size, cols, block_size).any(axis=(1, 3))
    return np.flatnonzero(per_block.ravel())


def encode(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> bytes:
    if prev is None:
        raise ValueError("BLOCK_DELTA needs a reference frame to diff against")
    if cur.shape != prev.shape:
        raise ValueError(f"shape mismatch: current {cur.shape} vs reference {prev.shape}")

    block_size = _block_size(opts)
    rows, cols = _grid(cur.shape, block_size)
    if rows * cols > MAX_BLOCKS:
        raise ValueError("frame has more blocks than a uint32 index can address")

    indices = _changed_blocks(cur, prev, block_size)
    # One modular delta plane; blocks are sliced out of it rather than recomputed.
    deltas = cur - prev

    out = bytearray()
    out += bytes([block_size])
    out += int(indices.size).to_bytes(4, "little")
    for index in indices:
        y0, y1, x0, x1 = _extent(int(index), cur.shape, block_size, cols)
        out += int(index).to_bytes(_INDEX, "little")
        out += np.ascontiguousarray(deltas[y0:y1, x0:x1]).tobytes()
    return bytes(out)


def decode(payload: bytes, prev: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if len(payload) < _HEADER:
        raise ValueError(f"BLOCK_DELTA payload is {len(payload)} bytes; needs at least 5")
    block_size = payload[0]
    if block_size == 0:
        raise ValueError("BLOCK_DELTA block_size is 0")
    n = int.from_bytes(payload[1:_HEADER], "little")
    if prev is None:
        raise ValueError("BLOCK_DELTA needs a reference frame to reconstruct against")

    rows, cols = _grid(shape, block_size)
    total_blocks = rows * cols
    out = prev.copy()
    offset = _HEADER

    for record in range(n):
        if offset + _INDEX > len(payload):
            raise ValueError(f"BLOCK_DELTA truncated at record {record} of {n}")
        index = int.from_bytes(payload[offset : offset + _INDEX], "little")
        offset += _INDEX
        if index >= total_blocks:
            raise ValueError(
                f"BLOCK_DELTA record {record} names block {index}, but the frame only has "
                f"{total_blocks}"
            )
        y0, y1, x0, x1 = _extent(index, shape, block_size, cols)
        nbytes = (y1 - y0) * (x1 - x0) * shape[2]
        if offset + nbytes > len(payload):
            raise ValueError(f"BLOCK_DELTA truncated inside record {record} of {n}")
        block = np.frombuffer(
            payload, dtype=np.uint8, count=nbytes, offset=offset
        ).reshape(y1 - y0, x1 - x0, shape[2])
        offset += nbytes
        out[y0:y1, x0:x1] += block  # uint8 wraps, undoing the subtraction exactly

    if offset != len(payload):
        raise ValueError(
            f"BLOCK_DELTA has {len(payload) - offset} trailing bytes after {n} records"
        )
    return out


def count(payload: bytes) -> int:
    """How many blocks this payload says changed."""
    if len(payload) < _HEADER:
        raise ValueError(f"BLOCK_DELTA payload is {len(payload)} bytes; needs at least 5")
    return int.from_bytes(payload[1:_HEADER], "little")


def block_size_of(payload: bytes) -> int:
    if not payload:
        raise ValueError("empty BLOCK_DELTA payload")
    return payload[0]


def cost(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> int:
    """Encoded size in bytes, without building the payload. Exact, not an estimate."""
    if prev is None:
        raise ValueError("BLOCK_DELTA needs a reference frame")
    block_size = _block_size(opts)
    _rows, cols = _grid(cur.shape, block_size)
    indices = _changed_blocks(cur, prev, block_size)

    total = _HEADER
    for index in indices:
        y0, y1, x0, x1 = _extent(int(index), cur.shape, block_size, cols)
        total += _INDEX + (y1 - y0) * (x1 - x0) * cur.shape[2]
    return total
