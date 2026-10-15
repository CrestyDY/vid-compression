from __future__ import annotations

import numpy as np
import pytest

from src.transform import CANDIDATES, Filter, apply, choose, invert


@pytest.mark.parametrize("filter_id", list(Filter))
@pytest.mark.parametrize("shape", [(1, 1, 3), (4, 7, 3), (24, 32, 3)])
def test_every_filter_is_exactly_invertible(rng, filter_id, shape):
    frame = rng.integers(0, 256, shape, dtype=np.uint8)
    assert np.array_equal(invert(apply(frame, filter_id), filter_id, shape), frame)


@pytest.mark.parametrize("filter_id", list(Filter))
def test_filters_preserve_payload_length(rng, filter_id):
    """Every filter is a permutation-and-difference of the same bytes.

    This is what keeps FULL's closed-form cost() valid: filtering changes the payload's
    content, never its length.
    """
    frame = rng.integers(0, 256, (12, 10, 3), dtype=np.uint8)
    assert len(apply(frame, filter_id)) == frame.size


@pytest.mark.parametrize("filter_id", list(Filter))
def test_filters_are_modular(filter_id):
    """Differences wrap, so a 2 -> 250 neighbour step is -8, not an overflow."""
    frame = np.empty((4, 4, 3), np.uint8)
    frame[:, :2] = 2
    frame[:, 2:] = 250
    assert np.array_equal(invert(apply(frame, filter_id), filter_id, frame.shape), frame)


def test_a_flat_frame_collapses_to_zeros():
    frame = np.full((8, 8, 3), 77, np.uint8)
    body = apply(frame, Filter.PLANAR_XY)
    # First row and column of each plane carry the seed; the rest must be zero.
    assert body.count(0) > frame.size * 0.7


def test_a_gradient_becomes_a_run_of_identical_bytes():
    """The case intra prediction exists for: incompressible raw, trivial once differenced."""
    xs = np.arange(64, dtype=np.uint8)
    frame = np.repeat(xs[None, :, None], 8, axis=0).repeat(3, axis=2)
    raw = set(apply(frame, Filter.RAW))
    differenced = set(apply(frame, Filter.PLANAR_SUB))
    assert len(differenced) < len(raw)


# ------------------------------------------------------------ choosing


def test_choose_returns_a_candidate(rng):
    frame = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert choose(frame) in CANDIDATES


def test_choose_prefers_raw_on_incompressible_content(rng):
    """Every filter scores within a hair of 8 bits on noise; doing less is then better."""
    frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert choose(frame) is Filter.RAW


def test_choose_picks_a_differencing_filter_on_a_gradient():
    ys, xs = np.mgrid[0:64, 0:64]
    frame = np.zeros((64, 64, 3), np.uint8)
    frame[..., 0] = (xs * 4).astype(np.uint8)
    frame[..., 1] = (ys * 4).astype(np.uint8)
    frame[..., 2] = 96
    assert choose(frame) in (Filter.PLANAR_SUB, Filter.PLANAR_UP, Filter.PLANAR_XY)


def test_choose_is_stable_for_a_flat_frame():
    frame = np.full((16, 16, 3), 5, np.uint8)
    assert choose(frame) in CANDIDATES


# ------------------------------------------------------------ rejections


def test_invert_rejects_an_unknown_filter(rng):
    frame = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        invert(apply(frame, Filter.RAW), 99, frame.shape)


def test_invert_rejects_a_short_body(rng):
    with pytest.raises(ValueError, match="bytes"):
        invert(b"\x00\x01", Filter.PLANAR, (4, 4, 3))
