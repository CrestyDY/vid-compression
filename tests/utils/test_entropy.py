from __future__ import annotations

from src.utils import entropy


def test_round_trips_when_enabled():
    payload = b"a" * 1000
    stored = entropy.compress(payload, enabled=True)
    assert len(stored) < len(payload)
    assert entropy.decompress(stored, enabled=True) == payload


def test_is_identity_when_disabled():
    payload = b"abc"
    assert entropy.compress(payload, enabled=False) is payload
    assert entropy.decompress(payload, enabled=False) is payload


def test_round_trips_empty():
    stored = entropy.compress(b"", enabled=True)
    assert entropy.decompress(stored, enabled=True) == b""
