"""MOTION — describe a block as "this region, moved", plus whatever is left over.

`BLOCK_DELTA` says *that* a region changed. It cannot say a region **moved**, so a rigid
object sliding three pixels right costs the same as that region being repainted at random.
Motion compensation is the fix, and it is the single biggest idea in real video codecs::

    current block = reference block, displaced by (dx, dy)
                  + residual

Payload::

    block_size : uint8    self-describing, as with BLOCK_DELTA
    count      : uint32   number of block records
    then count records:
        index    : uint32  block number, row-major across the grid
        dx       : int8    horizontal displacement into the reference frame
        dy       : int8    vertical displacement
        flags    : uint8   bit0 = a residual follows
        residual : bh*bw*3 bytes, modular, only when bit0 is set

A block with no record is unchanged and is copied straight through. A block whose displaced
reference matches exactly costs **7 bytes**, no matter how many pixels it contains — that is
the whole point.

Three things worth understanding:

**Why a residual is still usually needed.** Objects rotate, scale, get lit differently, and
sit on noisy sensors. Perfect block matches are common in synthetic content and rare in real
footage. But the residual after motion compensation is a *smaller and better-behaved signal*
than the residual without it, so it compresses better even when it is not smaller — which is
why MOTION can win on final size while losing on payload size.

**Search is the expensive part.** For each block we try every displacement in a window and
keep the one with the lowest sum-of-absolute-differences. That is quadratic in the search
range, so the search is vectorised across every block at once: for each candidate
displacement we shift the whole reference frame once, then reduce to a per-block SAD with a
single reshape. Candidates are visited nearest-first, so a scene that barely moved settles
early and the loop exits as soon as every block has an exact match.

SAD is a proxy for cost, not cost itself — the displacement with the lowest SAD is not
necessarily the one that encodes smallest. Every real codec makes this approximation,
because evaluating true cost per candidate is unaffordable.

**Out-of-frame displacements are defined, not forbidden.** A block near an edge may point at
a region partly outside the reference. Rather than rejecting those vectors per block, the
predictor zero-fills whatever falls outside, deterministically, on both sides. The search
naturally avoids such vectors because the zeros make the SAD large — but if one is ever
chosen, reconstruction is still exact, because encoder and decoder fill identically.
"""

from __future__ import annotations

import numpy as np

from src.models.options import DEFAULT_OPTIONS

MAX_BLOCKS = 0xFFFFFFFF
_HEADER = 5   # block_size (1) + count (4)
_RECORD = 7   # index (4) + dx (1) + dy (1) + flags (1)
_INDEX = 4
_FLAG_RESIDUAL = 0b1


def _settings(opts) -> tuple[int, int]:
    o = DEFAULT_OPTIONS if opts is None else opts
    return o.block_size, o.search_range


def _grid(shape: tuple[int, ...], block_size: int) -> tuple[int, int]:
    height, width = shape[:2]
    return (
        (height + block_size - 1) // block_size,
        (width + block_size - 1) // block_size,
    )


def _extent(index: int, shape: tuple[int, ...], block_size: int, cols: int):
    height, width = shape[:2]
    by, bx = divmod(index, cols)
    y0, x0 = by * block_size, bx * block_size
    return y0, min(y0 + block_size, height), x0, min(x0 + block_size, width)


def _displaced(ref: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """The whole reference frame shifted so ``out[y, x] == ref[y + dy, x + dx]``.

    Anything outside the reference is zero. Used by the search, which needs one array per
    candidate displacement rather than one per block.
    """
    out = np.zeros_like(ref)
    height, width = ref.shape[:2]
    sy0, sy1 = max(0, dy), min(height, height + dy)
    sx0, sx1 = max(0, dx), min(width, width + dx)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - dy : sy1 - dy, sx0 - dx : sx1 - dx] = ref[sy0:sy1, sx0:sx1]
    return out


def _fetch(
    ref: np.ndarray, y0: int, y1: int, x0: int, x1: int, dy: int, dx: int
) -> np.ndarray:
    """One block's predictor: ``out[i, j] == ref[y0 + i + dy, x0 + j + dx]``, zero outside.

    Must agree exactly with ``_displaced(ref, dy, dx)[y0:y1, x0:x1]`` — a test pins that,
    because a disagreement would corrupt reconstruction only at frame edges.
    """
    out = np.zeros((y1 - y0, x1 - x0, ref.shape[2]), dtype=ref.dtype)
    height, width = ref.shape[:2]
    sy0, sx0 = y0 + dy, x0 + dx
    cy0, cy1 = max(sy0, 0), min(y1 + dy, height)
    cx0, cx1 = max(sx0, 0), min(x1 + dx, width)
    if cy1 > cy0 and cx1 > cx0:
        out[cy0 - sy0 : cy1 - sy0, cx0 - sx0 : cx1 - sx0] = ref[cy0:cy1, cx0:cx1]
    return out


def _block_sums(plane: np.ndarray, rows: int, cols: int, block_size: int) -> np.ndarray:
    """Reduce a per-pixel plane to one value per block, padding the ragged edges with 0."""
    padded = np.zeros((rows * block_size, cols * block_size), dtype=np.int64)
    padded[: plane.shape[0], : plane.shape[1]] = plane
    return padded.reshape(rows, block_size, cols, block_size).sum(axis=(1, 3))


def _candidates(search_range: int) -> list[tuple[int, int]]:
    """Every displacement in the window, nearest first.

    Nearest-first matters: most content barely moves, so the best vector is usually found
    immediately and the loop can stop as soon as every block matches exactly.
    """
    offsets = [
        (dy, dx)
        for dy in range(-search_range, search_range + 1)
        for dx in range(-search_range, search_range + 1)
    ]
    offsets.sort(key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[0]), abs(d[1])))
    return offsets


def _search(
    cur: np.ndarray, prev: np.ndarray, block_size: int, search_range: int
) -> tuple[np.ndarray, np.ndarray]:
    """Best ``(dy, dx)`` per block by lowest SAD. Returns two ``(rows, cols)`` int16 grids."""
    rows, cols = _grid(cur.shape, block_size)
    best_dy = np.zeros((rows, cols), dtype=np.int16)
    best_dx = np.zeros((rows, cols), dtype=np.int16)
    best_sad: np.ndarray | None = None

    for dy, dx in _candidates(search_range):
        ref = _displaced(prev, dy, dx)
        sad_plane = np.abs(cur.astype(np.int32) - ref.astype(np.int32)).sum(
            axis=2, dtype=np.int32
        )
        sad = _block_sums(sad_plane, rows, cols, block_size)

        if best_sad is None:
            best_sad = sad
            continue
        improved = sad < best_sad
        if improved.any():
            best_sad = np.where(improved, sad, best_sad)
            best_dy = np.where(improved, dy, best_dy).astype(np.int16)
            best_dx = np.where(improved, dx, best_dx).astype(np.int16)
        if not best_sad.any():
            break  # every block matches exactly; nothing left to improve

    return best_dy, best_dx


def encode(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> bytes:
    if prev is None:
        raise ValueError("MOTION needs a reference frame to predict from")
    if cur.shape != prev.shape:
        raise ValueError(f"shape mismatch: current {cur.shape} vs reference {prev.shape}")

    block_size, search_range = _settings(opts)
    rows, cols = _grid(cur.shape, block_size)
    if rows * cols > MAX_BLOCKS:
        raise ValueError("frame has more blocks than a uint32 index can address")

    dy_grid, dx_grid = _search(cur, prev, block_size, search_range)
    changed = np.any(cur != prev, axis=2)

    out = bytearray()
    out += bytes([block_size])
    records = bytearray()
    written = 0

    for index in range(rows * cols):
        y0, y1, x0, x1 = _extent(index, cur.shape, block_size, cols)
        if not changed[y0:y1, x0:x1].any():
            continue  # unchanged: no record at all, the decoder copies it through

        by, bx = divmod(index, cols)
        dy, dx = int(dy_grid[by, bx]), int(dx_grid[by, bx])
        predicted = _fetch(prev, y0, y1, x0, x1, dy, dx)
        residual = cur[y0:y1, x0:x1] - predicted  # uint8 wraps: modular, exactly invertible

        has_residual = bool(residual.any())
        records += int(index).to_bytes(_INDEX, "little")
        records += int(dy).to_bytes(1, "little", signed=True)
        records += int(dx).to_bytes(1, "little", signed=True)
        records += bytes([_FLAG_RESIDUAL if has_residual else 0])
        if has_residual:
            records += np.ascontiguousarray(residual).tobytes()
        written += 1

    out += int(written).to_bytes(4, "little")
    out += records
    return bytes(out)


def decode(payload: bytes, prev: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if len(payload) < _HEADER:
        raise ValueError(f"MOTION payload is {len(payload)} bytes; needs at least 5")
    block_size = payload[0]
    if block_size == 0:
        raise ValueError("MOTION block_size is 0")
    n = int.from_bytes(payload[1:_HEADER], "little")
    if prev is None:
        raise ValueError("MOTION needs a reference frame to reconstruct against")

    rows, cols = _grid(shape, block_size)
    total_blocks = rows * cols
    out = prev.copy()
    offset = _HEADER

    for record in range(n):
        if offset + _RECORD > len(payload):
            raise ValueError(f"MOTION truncated at record {record} of {n}")
        index = int.from_bytes(payload[offset : offset + _INDEX], "little")
        dy = int.from_bytes(payload[offset + 4 : offset + 5], "little", signed=True)
        dx = int.from_bytes(payload[offset + 5 : offset + 6], "little", signed=True)
        flags = payload[offset + 6]
        offset += _RECORD

        if index >= total_blocks:
            raise ValueError(
                f"MOTION record {record} names block {index}, but the frame only has "
                f"{total_blocks}"
            )
        y0, y1, x0, x1 = _extent(index, shape, block_size, cols)
        block = _fetch(prev, y0, y1, x0, x1, dy, dx)

        if flags & _FLAG_RESIDUAL:
            nbytes = (y1 - y0) * (x1 - x0) * shape[2]
            if offset + nbytes > len(payload):
                raise ValueError(f"MOTION truncated inside record {record} of {n}")
            residual = np.frombuffer(
                payload, dtype=np.uint8, count=nbytes, offset=offset
            ).reshape(y1 - y0, x1 - x0, shape[2])
            offset += nbytes
            block = block + residual  # uint8 wraps, undoing the subtraction exactly

        out[y0:y1, x0:x1] = block

    if offset != len(payload):
        raise ValueError(
            f"MOTION has {len(payload) - offset} trailing bytes after {n} records"
        )
    return out


def count(payload: bytes) -> int:
    """How many block records this payload holds."""
    if len(payload) < _HEADER:
        raise ValueError(f"MOTION payload is {len(payload)} bytes; needs at least 5")
    return int.from_bytes(payload[1:_HEADER], "little")


def block_size_of(payload: bytes) -> int:
    if not payload:
        raise ValueError("empty MOTION payload")
    return payload[0]


def _walk(payload: bytes):
    """Yield ``(index, dy, dx, flags)`` per record. For tests, `info` and debugging."""
    n = count(payload)
    offset = _HEADER
    for _ in range(n):
        index = int.from_bytes(payload[offset : offset + _INDEX], "little")
        dy = int.from_bytes(payload[offset + 4 : offset + 5], "little", signed=True)
        dx = int.from_bytes(payload[offset + 5 : offset + 6], "little", signed=True)
        flags = payload[offset + 6]
        offset += _RECORD
        yield index, dy, dx, flags
        if flags & _FLAG_RESIDUAL:
            # Residual length needs the grid, which _walk does not know; callers wanting
            # to walk past a residual must use decode. Stop here to stay honest.
            return


def vectors_of(payload: bytes) -> dict[int, tuple[int, int]]:
    """``{block index: (dy, dx)}`` for the records before the first residual."""
    return {index: (dy, dx) for index, dy, dx, _flags in _walk(payload)}


def residual_flags_of(payload: bytes) -> dict[int, bool]:
    """``{block index: carries a residual}`` for the records before the first residual."""
    return {index: bool(flags & _FLAG_RESIDUAL) for index, _dy, _dx, flags in _walk(payload)}


def cost(cur: np.ndarray, prev: np.ndarray | None, opts=None) -> int:
    """Encoded size in bytes.

    Unlike the other modes this is not a closed form: it needs the search to know which
    blocks match exactly and therefore carry no residual. So it encodes and measures.
    That is why MOTION is the mode that makes selection expensive, and why M6's predictor
    exists.
    """
    return len(encode(cur, prev, opts))
