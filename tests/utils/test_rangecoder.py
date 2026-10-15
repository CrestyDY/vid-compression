from __future__ import annotations

import numpy as np
import pytest

from src.utils.rangecoder import (
    BitModel,
    RangeDecoder,
    RangeEncoder,
    decode_bytes,
    encode_bytes,
)

# ------------------------------------------------------------ the coder itself


@pytest.mark.parametrize("bits", [[], [0], [1], [0, 1] * 50, [1] * 200, [0] * 200])
def test_fixed_probability_round_trips(bits):
    enc = RangeEncoder()
    for bit in bits:
        enc.encode_bit(bit, 32768)
    data = enc.finish()

    dec = RangeDecoder(data)
    assert [dec.decode_bit(32768) for _ in bits] == bits


@pytest.mark.parametrize("p0", [1, 100, 1000, 32768, 60000, 65535])
def test_every_probability_round_trips(rng, p0):
    bits = rng.integers(0, 2, 500).tolist()
    enc = RangeEncoder()
    for bit in bits:
        enc.encode_bit(bit, p0)
    data = enc.finish()

    dec = RangeDecoder(data)
    assert [dec.decode_bit(p0) for _ in bits] == bits


def test_round_trips_with_varying_probabilities(rng):
    """The probability may change per bit, as it does under an adaptive model."""
    bits = rng.integers(0, 2, 2000).tolist()
    probs = rng.integers(1, 65536, 2000).tolist()

    enc = RangeEncoder()
    for bit, p0 in zip(bits, probs):
        enc.encode_bit(bit, p0)
    data = enc.finish()

    dec = RangeDecoder(data)
    assert [dec.decode_bit(p0) for p0 in probs] == bits


def test_a_skewed_stream_costs_far_less_than_a_bit_per_symbol():
    """The whole point of arithmetic coding: fractional bits per symbol."""
    n = 4000
    enc = RangeEncoder()
    for _ in range(n):
        enc.encode_bit(0, 65000)  # P(0) ~= 0.992
    data = enc.finish()
    assert len(data) * 8 < n * 0.2, f"{len(data) * 8} bits for {n} near-certain symbols"


# ------------------------------------------------------------ the adaptive model


def test_the_model_learns_a_skew():
    model = BitModel(1)
    start = model.probs[0]
    for _ in range(200):
        model.update(0, 0)
    assert model.probs[0] > start, "P(0) should rise after seeing only zeros"


def test_the_model_stays_in_range():
    model = BitModel(1)
    for _ in range(10_000):
        model.update(0, 1)
    assert 1 <= model.probs[0] <= 65535
    for _ in range(10_000):
        model.update(0, 0)
    assert 1 <= model.probs[0] <= 65535


# ------------------------------------------------------------ bytes end to end


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00",
        bytes(5000),                      # all zeros: the residual common case
        bytes(range(256)),
        b"\x00\x01\xff\xfe" * 500,        # modular deltas around zero
    ],
)
def test_byte_round_trip(payload):
    coded = encode_bytes(payload)
    assert decode_bytes(coded, len(payload)) == payload


def test_byte_round_trip_on_random_data(rng):
    payload = rng.integers(0, 256, 5000, dtype=np.uint8).tobytes()
    assert decode_bytes(encode_bytes(payload), len(payload)) == payload


def test_byte_round_trip_on_realistic_residuals(rng):
    """Mostly zero, occasionally +-small: what an intra-predicted payload looks like."""
    values = rng.choice(
        np.array([0, 1, 255, 2, 254, 7, 249], dtype=np.uint8),
        size=20_000,
        p=[0.80, 0.06, 0.06, 0.03, 0.03, 0.01, 0.01],
    )
    payload = values.tobytes()
    assert decode_bytes(encode_bytes(payload), len(payload)) == payload


def test_zeros_compress_enormously():
    payload = bytes(20_000)
    assert len(encode_bytes(payload)) < 100


def test_incompressible_data_costs_about_what_it_is(rng):
    """Random bytes must not *expand* much: an entropy coder's floor is the entropy."""
    payload = rng.integers(0, 256, 20_000, dtype=np.uint8).tobytes()
    assert len(encode_bytes(payload)) < len(payload) * 1.02


def test_an_empty_stream_for_a_nonempty_payload_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        decode_bytes(b"", 1000)


def test_truncation_yields_wrong_bytes_rather_than_an_error():
    """A documented limitation, not an oversight.

    An arithmetic coder has no internal redundancy: every bit pattern decodes to *some*
    symbol sequence, so a truncated or corrupted stream cannot be detected from the stream
    alone. zstd carries enough framing to notice; this does not.

    Integrity is therefore the container's job, and it already does most of it - the frame
    record stores payload_len and MyvReader rejects a short read. What remains uncovered is
    corruption *within* a correct-length payload, which decodes to wrong pixels silently.
    That is the same exposure --no-entropy already has, and `verify` is what catches it.
    """
    payload = bytes(500) + bytes([0x7F]) * 500
    coded = encode_bytes(payload)
    got = decode_bytes(coded[: len(coded) // 2], len(payload))
    assert len(got) == len(payload)
    assert got != payload
