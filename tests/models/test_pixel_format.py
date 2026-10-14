from __future__ import annotations

import pytest

from src.models.pixel_format import RGB24, by_id


def test_rgb24_layout():
    assert RGB24.frame_shape(320, 240) == (240, 320, 3)
    assert RGB24.frame_nbytes(320, 240) == 320 * 240 * 3


def test_by_id_round_trips():
    assert by_id(RGB24.id) is RGB24


def test_by_id_rejects_unknown():
    with pytest.raises(ValueError, match="unknown pixel format id 99"):
        by_id(99)


def test_pixel_format_is_frozen():
    with pytest.raises(Exception):
        RGB24.channels = 4
