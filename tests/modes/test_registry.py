from __future__ import annotations

from src.models.frame import FrameType
from src.modes import MODES, PREDICTED


def test_every_frame_type_has_a_mode():
    assert set(MODES) == set(FrameType)


def test_every_mode_exposes_the_same_interface():
    for frame_type, mode in MODES.items():
        for name in ("encode", "decode", "cost"):
            assert callable(getattr(mode, name)), f"{frame_type.name}.{name}"


def test_frame_type_values_are_stable():
    """These are written to disk. Changing one silently breaks every existing file."""
    assert FrameType.FULL == 0
    assert FrameType.PIXEL_DELTA == 1


def test_only_full_needs_no_reference_frame():
    assert PREDICTED == frozenset({FrameType.PIXEL_DELTA})
    assert FrameType.FULL not in PREDICTED
