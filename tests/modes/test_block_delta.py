from __future__ import annotations

import numpy as np
import pytest

from src.models.options import ModeOptions
from src.modes import block_delta


def opts(block_size: int = 16) -> ModeOptions:
    return ModeOptions(block_size=block_size)


# ------------------------------------------------------------ the payload is self-describing


def test_block_size_is_stored_in_the_payload(solid_frame):
    """The decoder reads what was done; it is never configured to match the encoder."""
    frame = solid_frame()
    payload = block_delta.encode(frame, frame, opts(8))
    assert block_delta.block_size_of(payload) == 8


def test_an_out_of_range_block_size_is_rejected_at_construction():
    """Caught by the model, before any frame is touched — not inside the mode."""
    for bad in (0, 256):
        with pytest.raises(ValueError, match="block_size"):
            ModeOptions(block_size=bad)


# ------------------------------------------------------------ round trips


@pytest.mark.parametrize("block_size", [1, 4, 8, 16, 32, 255])
def test_round_trips_at_every_block_size(moving_frames, block_size):
    a, b = moving_frames()[:2]
    payload = block_delta.encode(b, a, opts(block_size))
    assert np.array_equal(block_delta.decode(payload, a, b.shape), b)


def test_round_trips_when_dimensions_do_not_divide_evenly(solid_frame, rng):
    """24x32 in 16px blocks leaves 8-tall edge blocks, so records are variable-length."""
    before = solid_frame(height=24, width=32)
    after = rng.integers(0, 256, before.shape, dtype=np.uint8)
    payload = block_delta.encode(after, before, opts(16))
    assert np.array_equal(block_delta.decode(payload, before, after.shape), after)


def test_deltas_are_modular(solid_frame):
    before = solid_frame(color=(2, 2, 2))
    after = solid_frame(color=(250, 250, 250))
    payload = block_delta.encode(after, before, opts(8))
    assert np.array_equal(block_delta.decode(payload, before, after.shape), after)


# ------------------------------------------------------------ only changed blocks are stored


def test_an_unchanged_frame_stores_no_blocks(solid_frame):
    frame = solid_frame()
    payload = block_delta.encode(frame, frame, opts(8))
    assert block_delta.count(payload) == 0
    assert len(payload) == 5  # block_size + uint32 count
    assert np.array_equal(block_delta.decode(payload, frame, frame.shape), frame)


def test_one_changed_pixel_stores_exactly_one_block(solid_frame):
    before = solid_frame(height=32, width=32)
    after = before.copy()
    after[3, 4] = (99, 1, 2)
    payload = block_delta.encode(after, before, opts(16))
    assert block_delta.count(payload) == 1
    assert np.array_equal(block_delta.decode(payload, before, after.shape), after)


def test_changes_in_two_blocks_store_two_blocks(solid_frame):
    before = solid_frame(height=32, width=32)
    after = before.copy()
    after[0, 0] = (1, 2, 3)
    after[20, 20] = (4, 5, 6)
    payload = block_delta.encode(after, before, opts(16))
    assert block_delta.count(payload) == 2


# ------------------------------------------------------------ cost


def test_cost_is_exact(moving_frames):
    a, b = moving_frames()[:2]
    for block_size in (4, 16, 32):
        assert block_delta.cost(b, a, opts(block_size)) == len(
            block_delta.encode(b, a, opts(block_size))
        )


def test_cost_does_not_depend_on_how_much_changed_inside_a_block(solid_frame):
    """A block costs the same whether one pixel or all of them changed."""
    before = solid_frame(height=16, width=16)
    one = before.copy()
    one[0, 0] = (1, 1, 1)
    all_of_it = before.copy()
    all_of_it[:, :] = (9, 9, 9)
    assert block_delta.cost(one, before, opts(16)) == block_delta.cost(
        all_of_it, before, opts(16)
    )


# ------------------------------------------------------------ rejects broken input


def test_encode_needs_a_reference_frame(solid_frame):
    with pytest.raises(ValueError, match="reference frame"):
        block_delta.encode(solid_frame(), None, opts())


def test_encode_rejects_a_shape_mismatch(solid_frame):
    with pytest.raises(ValueError, match="shape mismatch"):
        block_delta.encode(solid_frame(), solid_frame(height=8), opts())


def test_decode_rejects_a_zero_block_size(solid_frame):
    frame = solid_frame()
    with pytest.raises(ValueError, match="block_size is 0"):
        block_delta.decode(b"\x00" + (0).to_bytes(4, "little"), frame, frame.shape)


def test_decode_rejects_a_block_index_past_the_grid(solid_frame):
    frame = solid_frame(height=16, width=16)
    payload = bytes([16]) + (1).to_bytes(4, "little") + (99).to_bytes(4, "little")
    with pytest.raises(ValueError, match="only has"):
        block_delta.decode(payload, frame, frame.shape)


def test_decode_rejects_a_truncated_record(solid_frame):
    frame = solid_frame(height=16, width=16)
    payload = bytes([16]) + (1).to_bytes(4, "little") + (0).to_bytes(4, "little") + b"\x00"
    with pytest.raises(ValueError, match="truncated"):
        block_delta.decode(payload, frame, frame.shape)


def test_decode_rejects_trailing_bytes(solid_frame):
    frame = solid_frame()
    payload = block_delta.encode(frame, frame, opts(8)) + b"junk"
    with pytest.raises(ValueError, match="trailing"):
        block_delta.decode(payload, frame, frame.shape)
