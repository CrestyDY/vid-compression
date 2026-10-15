from __future__ import annotations

import numpy as np
import pytest

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


# ------------------------------------------------------------ the codec choice


@pytest.mark.parametrize("codec", entropy.CODECS)
def test_every_codec_round_trips(rng, codec):
    payload = rng.integers(0, 256, 3000, dtype=np.uint8).tobytes()
    stored = entropy.compress(payload, enabled=True, codec=codec)
    assert entropy.decompress(stored, enabled=True, codec=codec) == payload


@pytest.mark.parametrize("codec", entropy.CODECS)
def test_every_codec_round_trips_empty(codec):
    stored = entropy.compress(b"", enabled=True, codec=codec)
    assert entropy.decompress(stored, enabled=True, codec=codec) == b""


@pytest.mark.parametrize("codec", entropy.CODECS)
def test_every_codec_round_trips_zeros(codec):
    payload = bytes(5000)
    stored = entropy.compress(payload, enabled=True, codec=codec)
    assert entropy.decompress(stored, enabled=True, codec=codec) == payload


def test_auto_never_loses_to_zstd_by_more_than_its_signalling_byte(rng):
    """The guarantee that makes `auto` safe to choose."""
    cases = [
        bytes(4000),                                      # zstd's best case: a long run
        rng.integers(0, 256, 4000, dtype=np.uint8).tobytes(),
        rng.choice([0, 1, 255], size=4000, p=[0.9, 0.05, 0.05]).astype(np.uint8).tobytes(),
    ]
    for payload in cases:
        z = len(entropy.compress(payload, enabled=True, codec="zstd"))
        a = len(entropy.compress(payload, enabled=True, codec="auto"))
        assert a <= z + 1, (a, z)


def test_auto_picks_the_range_coder_when_it_wins(rng):
    """A dense non-repeating but skewed payload is the range coder's case."""
    payload = rng.choice(
        np.arange(16, dtype=np.uint8), size=20_000,
        p=[0.4] + [0.6 / 15] * 15,
    ).tobytes()
    stored = entropy.compress(payload, enabled=True, codec="auto")
    assert stored[0] == 1, "expected auto to have selected the range coder"
    assert entropy.decompress(stored, enabled=True, codec="auto") == payload


def test_unknown_codec_is_rejected():
    with pytest.raises(ValueError, match="unknown entropy codec"):
        entropy.compress(b"x", enabled=True, codec="telepathy")


def test_codec_ids_are_stable():
    """These go in the container header; renumbering breaks existing files."""
    assert entropy.codec_id("zstd") == 0
    assert entropy.codec_id("range") == 1
    assert entropy.codec_id("auto") == 2
    assert entropy.codec_name(0) == "zstd"


def test_a_corrupt_auto_codec_byte_is_reported():
    from src.utils.errors import CorruptPayload

    with pytest.raises(CorruptPayload, match="unknown codec"):
        entropy.decompress(bytes([9]) + b"junk", enabled=True, codec="auto")
