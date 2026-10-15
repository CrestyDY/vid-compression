from __future__ import annotations

import numpy as np
import pytest

from src.models.frame import FrameType
from src.models.options import ModeOptions
from src.modes import TILEABLE, tiled


def opts(grid: int = 4) -> ModeOptions:
    return ModeOptions(grid=grid)


# ------------------------------------------------------------ the grid


def test_grid_dims_is_square_but_clamped_to_the_frame():
    assert tiled.grid_dims((240, 320, 3), 4) == (4, 4)
    # A 2x3 frame cannot hold a 4x4 grid of non-empty tiles.
    assert tiled.grid_dims((2, 3, 3), 4) == (2, 3)


def test_tiles_cover_the_frame_exactly_and_do_not_overlap():
    shape = (24, 32, 3)
    rows, cols = tiled.grid_dims(shape, 3)
    seen = np.zeros(shape[:2], dtype=int)
    for y0, y1, x0, x1 in tiled.tiles(shape, rows, cols):
        assert y1 > y0 and x1 > x0, "a tile must not be empty"
        seen[y0:y1, x0:x1] += 1
    assert (seen == 1).all(), "tiles must partition the frame"


def test_edge_tiles_absorb_the_remainder():
    """A 320-wide frame in 3 columns gives 106/106/108, not a ragged fourth."""
    widths = [x1 - x0 for _y0, _y1, x0, x1 in tiled.tiles((240, 320, 3), 1, 3)]
    assert widths == [106, 107, 107]
    assert sum(widths) == 320


# ------------------------------------------------------------ pack and unpack


def test_pack_round_trips_through_unpack():
    parts = [(FrameType.FULL, b"aaa"), (FrameType.PIXEL_DELTA, b"bb")]
    payload = tiled.pack(1, 2, parts)
    rows, cols, got = tiled.unpack(payload)
    assert (rows, cols) == (1, 2)
    assert got == parts


def test_pack_rejects_a_mismatched_tile_count():
    with pytest.raises(ValueError, match="needs 4 tiles"):
        tiled.pack(2, 2, [(FrameType.FULL, b"x")])


def test_tile_lengths_are_varints():
    """A u32 per tile would waste 3 bytes each; at 16 tiles that is real overhead."""
    payload = tiled.pack(1, 1, [(FrameType.FULL, b"xy")])
    # rows(1) + cols(1) + mode(1) + len(1) + payload(2)
    assert len(payload) == 6


# ------------------------------------------------------------ round trips


def test_round_trips(rng, solid_frame):
    before = solid_frame(height=32, width=32)
    after = before.copy()
    after[0:16, 0:16] = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    payload = tiled.encode(after, before, opts(2))
    assert np.array_equal(tiled.decode(payload, before, after.shape), after)


@pytest.mark.parametrize("grid", [1, 2, 3, 4])
def test_round_trips_at_every_grid(rng, grid):
    before = rng.integers(0, 60, (24, 32, 3), dtype=np.uint8)
    after = before.copy()
    after[4:12, 4:12] = 200
    payload = tiled.encode(after, before, ModeOptions(grid=grid))
    assert np.array_equal(tiled.decode(payload, before, after.shape), after)


def test_tiles_choose_independently(rng, solid_frame):
    """A frame with a changed half and an unchanged half should not pick one answer."""
    before = solid_frame(height=32, width=32)
    after = before.copy()
    after[:, 16:] = rng.integers(0, 256, (32, 16, 3), dtype=np.uint8)

    payload = tiled.encode(after, before, opts(2))
    _rows, _cols, parts = tiled.unpack(payload)
    assert len({mode for mode, _ in parts}) > 1, "expected tiles to differ"
    assert np.array_equal(tiled.decode(payload, before, after.shape), after)


# ------------------------------------------------------------ no recursion


def test_tiled_may_not_contain_tiled():
    """Recursion would make this a quadtree, which needs a depth limit and a stop rule."""
    assert FrameType.TILED not in TILEABLE
    assert FrameType.FULL in TILEABLE


def test_decode_rejects_a_nested_tiled_payload(solid_frame):
    frame = solid_frame(height=16, width=16)
    payload = tiled.pack(1, 1, [(FrameType.TILED, b"")])
    with pytest.raises(ValueError, match="TILED"):
        tiled.decode(payload, frame, frame.shape)


# ------------------------------------------------------------ rejections


def test_encode_needs_a_reference_frame(solid_frame):
    with pytest.raises(ValueError, match="reference frame"):
        tiled.encode(solid_frame(), None, opts())


def test_encode_rejects_a_shape_mismatch(solid_frame):
    with pytest.raises(ValueError, match="shape mismatch"):
        tiled.encode(solid_frame(), solid_frame(height=8), opts())


def test_decode_rejects_a_grid_that_does_not_match_the_frame(solid_frame):
    frame = solid_frame(height=16, width=16)
    payload = tiled.pack(20, 20, [(FrameType.FULL, b"")] * 400)
    with pytest.raises(ValueError, match="does not fit"):
        tiled.decode(payload, frame, frame.shape)


def test_cost_matches_the_encoded_size(rng, solid_frame):
    before = solid_frame(height=32, width=32)
    after = before.copy()
    after[0:8, 0:8] = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert tiled.cost(after, before, opts(2)) == len(tiled.encode(after, before, opts(2)))
