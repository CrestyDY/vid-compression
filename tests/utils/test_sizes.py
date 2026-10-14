from __future__ import annotations

from src.utils.sizes import human_bytes


def test_bytes_print_whole():
    assert human_bytes(7) == "7 B"
    assert human_bytes(0) == "0 B"


def test_larger_units_get_two_decimals():
    assert human_bytes(1536) == "1.50 KB"
    assert human_bytes(1024 * 1024) == "1.00 MB"


def test_spans_ten_orders_of_magnitude():
    assert human_bytes(2 * 1024**3) == "2.00 GB"
