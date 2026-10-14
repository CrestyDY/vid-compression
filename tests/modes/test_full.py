from __future__ import annotations

import numpy as np
import pytest

from src.modes import full


def test_round_trips(moving_frames):
    frame = moving_frames()[0]
    payload = full.encode(frame)
    assert np.array_equal(full.decode(payload, None, frame.shape), frame)


def test_needs_no_reference_frame(moving_frames):
    frame = moving_frames()[0]
    assert full.encode(frame, None) == full.encode(frame, frame)


def test_cost_is_exact(moving_frames):
    frame = moving_frames()[0]
    assert full.cost(frame) == len(full.encode(frame))


def test_decode_rejects_a_wrong_length_payload(moving_frames):
    frame = moving_frames()[0]
    with pytest.raises(ValueError, match="expected"):
        full.decode(b"\x00\x01", None, frame.shape)
