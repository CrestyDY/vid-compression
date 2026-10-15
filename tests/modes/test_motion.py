from __future__ import annotations

import numpy as np
import pytest

from src.models.options import ModeOptions
from src.modes import motion


def opts(block_size: int = 16, search_range: int = 7) -> ModeOptions:
    return ModeOptions(block_size=block_size, search_range=search_range)


def _translated(rng, shift: int, size: int = 64):
    """A textured frame and the same frame shifted `shift` pixels right."""
    bg = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
    a = bg.copy()
    a[16:48, 8:40] = rng.integers(120, 240, (32, 32, 3), dtype=np.uint8)
    b = bg.copy()
    b[16:48, 8 + shift : 40 + shift] = a[16:48, 8:40]
    return a, b


# ------------------------------------------------------------ the payload is self-describing


def test_block_size_is_stored_in_the_payload(solid_frame):
    frame = solid_frame()
    payload = motion.encode(frame, frame, opts(8))
    assert motion.block_size_of(payload) == 8


def test_an_unchanged_frame_stores_no_records(solid_frame):
    frame = solid_frame()
    payload = motion.encode(frame, frame, opts(8))
    assert motion.count(payload) == 0
    assert len(payload) == 6  # block_size + intra filter + uint32 count
    assert np.array_equal(motion.decode(payload, frame, frame.shape), frame)


# ------------------------------------------------------------ round trips


@pytest.mark.parametrize("block_size", [4, 8, 16, 32])
def test_round_trips_at_every_block_size(rng, block_size):
    a, b = _translated(rng, 3)
    payload = motion.encode(b, a, opts(block_size))
    assert np.array_equal(motion.decode(payload, a, b.shape), b)


@pytest.mark.parametrize("shift", [0, 1, 3, 7])
def test_round_trips_at_every_displacement(rng, shift):
    a, b = _translated(rng, shift)
    payload = motion.encode(b, a, opts(16, 7))
    assert np.array_equal(motion.decode(payload, a, b.shape), b)


def test_round_trips_when_dimensions_do_not_divide_evenly(rng):
    a, b = _translated(rng, 2)
    # 24x34 in 16px blocks: ragged on both axes. ascontiguousarray because the modes
    # may be handed a view, and tobytes() on a non-contiguous view would reorder bytes.
    a = np.ascontiguousarray(a[:24, :34])
    b = np.ascontiguousarray(b[:24, :34])
    payload = motion.encode(b, a, opts(16))
    assert np.array_equal(motion.decode(payload, a, b.shape), b)


def test_round_trips_on_random_noise(rng, solid_frame):
    """No block will match; every record carries a full residual."""
    before = solid_frame(height=32, width=32)
    after = rng.integers(0, 256, before.shape, dtype=np.uint8)
    payload = motion.encode(after, before, opts(16))
    assert np.array_equal(motion.decode(payload, before, after.shape), after)


def test_residuals_are_modular(solid_frame):
    before = solid_frame(height=16, width=16, color=(2, 2, 2))
    after = solid_frame(height=16, width=16, color=(250, 250, 250))
    payload = motion.encode(after, before, opts(16))
    assert np.array_equal(motion.decode(payload, before, after.shape), after)


# ------------------------------------------------------------ it actually finds motion


def test_a_translation_is_described_by_vectors_not_residuals(rng):
    """The whole point: a shifted region costs vectors, not a repaint."""
    a, b = _translated(rng, 3)
    payload = motion.encode(b, a, opts(16, 7))

    vectors = motion.vectors_of(payload)
    assert any(v != (0, 0) for v in vectors.values()), "expected a non-zero displacement"

    # And it must be far cheaper than describing the same change as blocks.
    from src.modes import block_delta

    assert len(payload) < len(block_delta.encode(b, a, opts(16)))


def test_an_exactly_matched_block_costs_seven_bytes(solid_frame, rng):
    """A block whose displaced reference matches exactly carries no residual."""
    size = 32
    bg = np.zeros((size, size, 3), np.uint8)
    a = bg.copy()
    a[0:16, 0:16] = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    b = bg.copy()
    b[0:16, 4:20] = a[0:16, 0:16]  # the same texture, moved 4 right

    payload = motion.encode(b, a, opts(16, 7))
    assert np.array_equal(motion.decode(payload, a, b.shape), b)
    # At least one record should be residual-free.
    assert any(not has for has in motion.residual_flags_of(payload).values())


def test_search_range_bounds_what_can_be_found(rng):
    """A displacement outside the window cannot be found, but must still round-trip."""
    a, b = _translated(rng, 9)
    payload = motion.encode(b, a, opts(16, search_range=2))
    assert np.array_equal(motion.decode(payload, a, b.shape), b)


# ------------------------------------------------------------ the shared predictor


def test_whole_frame_and_per_block_fetch_agree(rng):
    """Encoder searches with the whole-frame shift; both sides reconstruct per block.

    If these ever disagree, reconstruction breaks silently at frame edges.
    """
    ref = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    for dy in (-9, -1, 0, 1, 9):
        for dx in (-9, -1, 0, 1, 9):
            whole = motion._displaced(ref, dy, dx)
            for y0, y1, x0, x1 in ((0, 8, 0, 8), (16, 24, 24, 32), (8, 16, 8, 16)):
                block = motion._fetch(ref, y0, y1, x0, x1, dy, dx)
                assert np.array_equal(whole[y0:y1, x0:x1], block), (dy, dx, y0, x0)


# ------------------------------------------------------------ cost and rejections


def test_cost_matches_the_encoded_size(rng):
    a, b = _translated(rng, 3)
    assert motion.cost(b, a, opts(16)) == len(motion.encode(b, a, opts(16)))


def test_encode_needs_a_reference_frame(solid_frame):
    with pytest.raises(ValueError, match="reference frame"):
        motion.encode(solid_frame(), None, opts())


def test_encode_rejects_a_shape_mismatch(solid_frame):
    with pytest.raises(ValueError, match="shape mismatch"):
        motion.encode(solid_frame(), solid_frame(height=8), opts())


def test_search_range_is_bounded_by_int8(solid_frame):
    with pytest.raises(ValueError, match="search_range"):
        ModeOptions(search_range=200)


def test_decode_rejects_a_zero_block_size(solid_frame):
    frame = solid_frame()
    with pytest.raises(ValueError, match="block_size is 0"):
        motion.decode(b"\x00\x00" + (0).to_bytes(4, "little"), frame, frame.shape)


def test_decode_rejects_a_block_index_past_the_grid(solid_frame):
    frame = solid_frame(height=16, width=16)
    payload = (
        bytes([16, 0]) + (1).to_bytes(4, "little")
        + (99).to_bytes(4, "little") + bytes([0, 0, 0])
    )
    with pytest.raises(ValueError, match="only has"):
        motion.decode(payload, frame, frame.shape)


def test_decode_rejects_a_truncated_record(solid_frame):
    frame = solid_frame(height=16, width=16)
    payload = bytes([16, 0]) + (1).to_bytes(4, "little") + b"\x00\x00"
    with pytest.raises(ValueError, match="truncated"):
        motion.decode(payload, frame, frame.shape)


def test_decode_rejects_trailing_bytes(solid_frame):
    frame = solid_frame()
    payload = motion.encode(frame, frame, opts(8)) + b"junk"
    with pytest.raises(ValueError, match="trailing"):
        motion.decode(payload, frame, frame.shape)
