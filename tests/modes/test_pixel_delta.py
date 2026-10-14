from __future__ import annotations

import numpy as np
import pytest

from src.modes import pixel_delta


def test_record_is_seven_bytes():
    assert pixel_delta.RECORD.itemsize == 7


def test_round_trips(moving_frames):
    a, b = moving_frames()[:2]
    payload = pixel_delta.encode(b, a)
    assert np.array_equal(pixel_delta.decode(payload, a, b.shape), b)


def test_identical_frames_cost_only_the_header(solid_frame):
    frame = solid_frame()
    payload = pixel_delta.encode(frame, frame)
    assert pixel_delta.count(payload) == 0
    assert len(payload) == 4
    assert np.array_equal(pixel_delta.decode(payload, frame, frame.shape), frame)


def test_deltas_are_modular(solid_frame):
    """2 -> 250 is -8 mod 256. One byte must cover it without clamping."""
    before = solid_frame(color=(2, 2, 2))
    after = solid_frame(color=(250, 250, 250))
    payload = pixel_delta.encode(after, before)
    assert np.array_equal(pixel_delta.decode(payload, before, after.shape), after)


def test_counts_only_changed_pixels(solid_frame):
    before = solid_frame()
    after = before.copy()
    after[3, 4] = (99, 99, 99)
    after[5, 6, 1] = 7  # a single changed channel still makes the pixel changed
    payload = pixel_delta.encode(after, before)
    assert pixel_delta.count(payload) == 2
    assert set(pixel_delta.coordinates(payload)) == {(3, 4), (5, 6)}


def test_cost_is_exact(moving_frames):
    a, b = moving_frames()[:2]
    assert pixel_delta.cost(b, a) == len(pixel_delta.encode(b, a))


def test_encode_needs_a_reference_frame(solid_frame):
    with pytest.raises(ValueError, match="reference frame"):
        pixel_delta.encode(solid_frame(), None)


def test_encode_rejects_a_shape_mismatch(solid_frame):
    with pytest.raises(ValueError, match="shape mismatch"):
        pixel_delta.encode(solid_frame(), solid_frame(height=8))


def test_decode_rejects_a_truncated_payload(solid_frame):
    frame = solid_frame()
    payload = bytearray(pixel_delta.encode(frame, frame))
    payload[0] = 5  # claim five records that aren't there
    with pytest.raises(ValueError, match="claims 5 records"):
        pixel_delta.decode(bytes(payload), frame, frame.shape)
