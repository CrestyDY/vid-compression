from __future__ import annotations

import numpy as np
import pytest

from src.models.frame import FrameType
from src.models.options import DEFAULT_OPTIONS
from src.modes import MODES, PLANEABLE, PREDICTED, planes

#: PLANES is a frame-level wrapper, not a mode: it codes a *tuple* of planes and has its
#: own signature, so it is deliberately absent from the registry. Every other frame type
#: must be in there, or a file could name a mode no decoder can run.
_NOT_A_PLANE_MODE = {FrameType.PLANES}


def test_every_frame_type_but_planes_has_a_mode():
    assert set(MODES) == set(FrameType) - _NOT_A_PLANE_MODE


def test_planes_is_not_in_the_registry_and_has_its_own_interface():
    """If PLANES were ever added to MODES, the selector would offer it as a candidate for
    a single plane and recurse. The absence is load-bearing, so it is asserted."""
    assert FrameType.PLANES not in MODES
    assert FrameType.PLANES not in PLANEABLE
    for name in ("pack", "unpack", "decode"):
        assert callable(getattr(planes, name))


def test_every_mode_exposes_the_same_interface():
    for frame_type, mode in MODES.items():
        for name in ("encode", "decode", "cost"):
            assert callable(getattr(mode, name)), f"{frame_type.name}.{name}"


def test_frame_type_values_are_stable():
    """These are written to disk. Changing one silently breaks every existing file."""
    assert FrameType.FULL == 0
    assert FrameType.PIXEL_DELTA == 1
    assert FrameType.TILED == 4
    assert FrameType.PLANES == 5


def test_only_full_needs_no_reference_frame():
    """Every plane mode but FULL reconstructs from a predecessor, so a frame-0 plane
    must be FULL."""
    assert PREDICTED == frozenset(set(MODES) - {FrameType.FULL})
    assert FrameType.FULL not in PREDICTED


# ------------------------------------------------------------ single-channel planes

#: A plane of a subsampled format has one channel, not three. Every mode has to cope:
#: the selector offers all of them for a chroma plane, and a mode that encodes but cannot
#: decode produces a file nobody can read.
_PLANE_SHAPES = [(12, 16, 1), (12, 16, 3)]


@pytest.mark.parametrize("shape", _PLANE_SHAPES)
@pytest.mark.parametrize("frame_type", sorted(MODES))
def test_every_mode_round_trips_a_plane_of_any_channel_count(shape, frame_type, rng):
    mode = MODES[frame_type]
    prev = rng.integers(0, 256, shape, dtype=np.uint8)
    cur = prev.copy()
    cur[3:7, 2:9] = rng.integers(0, 256, (4, 7, shape[2]), dtype=np.uint8)

    reference = None if frame_type is FrameType.FULL else prev
    payload = mode.encode(cur, reference, DEFAULT_OPTIONS)
    got = mode.decode(payload, reference, shape)

    assert got.shape == shape
    assert np.array_equal(got, cur)


@pytest.mark.parametrize("shape", _PLANE_SHAPES)
@pytest.mark.parametrize("frame_type", sorted(MODES))
def test_every_mode_prices_itself_exactly(shape, frame_type, rng):
    """cost() is what the selector ranks on, so it has to equal the real payload length
    on a one-channel plane too, not just on RGB."""
    mode = MODES[frame_type]
    prev = rng.integers(0, 256, shape, dtype=np.uint8)
    cur = prev.copy()
    cur[3:7, 2:9] = rng.integers(0, 256, (4, 7, shape[2]), dtype=np.uint8)

    reference = None if frame_type is FrameType.FULL else prev
    payload = mode.encode(cur, reference, DEFAULT_OPTIONS)
    assert mode.cost(cur, reference, DEFAULT_OPTIONS) == len(payload)
