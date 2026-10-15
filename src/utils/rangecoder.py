"""A context-modelled binary arithmetic coder.

M8's comparison against FFV1 priced what this is for. FFV1 is intra-only - no motion
compensation at all - and it still beat this codec 1.38x on real footage after intra
prediction landed. The remaining difference is the entropy stage: FFV1 picks a probability
model per symbol from its neighbours and adapts it as it codes, while zstd is a
general-purpose LZ matcher that has no idea it is looking at image residuals.

An LZ matcher needs *exact repeats*. Intra-predicted residuals rarely repeat exactly - they
are mostly zero with an occasional small signed value - but their symbol *distribution* is
wildly non-uniform, which is precisely what an arithmetic coder exploits and an LZ matcher
cannot. A symbol with probability 0.99 should cost 0.014 bits; zstd cannot spend less than
roughly a byte on anything it fails to match.

**The algorithm** is the textbook integer arithmetic coder: a 32-bit ``[low, high]`` interval
narrowed per binary decision, renormalised a bit at a time, with an underflow counter for the
straddling case where the interval brackets the midpoint without resolving a bit. Correctness
here is self-consistency rather than conformance to an external format, so it is verified by
round-tripping thousands of random bit sequences at random probabilities.

**The model** is adaptive and contextual. Each byte is coded as 8 binary decisions down a
fixed tree, so the bits already decoded select the node - meaning the coder learns, for
example, that a byte starting 0000 is overwhelmingly likely to continue that way. The tree is
chosen by a *bucket* derived from the previous byte's signed magnitude, so runs of near-zero
residuals get a different, sharper model than busy regions.

**On speed.** Arithmetic coding is inherently sequential: every symbol's interval depends on
the previous one, so it cannot be vectorised the way the rest of this codec is. In pure
Python that costs roughly 8 binary codings plus a model update per byte, which is orders of
magnitude slower than zstd's C. It is correct and it measures the idea; it is not a
replacement for zstd at 1080p, and ``--entropy zstd`` remains the default.
"""

from __future__ import annotations

import numpy as np

_TOP = (1 << 32) - 1
_HALF = 1 << 31
_QUARTER = 1 << 30
_THREE_QUARTERS = 3 << 30

#: Probabilities are 16-bit, expressed as P(bit == 0). Never 0 or 65536: a symbol the coder
#: believes impossible cannot be encoded at all.
_PROB_ONE = 1 << 16
_PROB_MIN = 1
_PROB_MAX = _PROB_ONE - 1

#: How fast the model adapts. Lower is faster and noisier; 5 is the usual compromise.
_RATE = 5


class RangeEncoder:
    """Writes binary decisions into a byte string."""

    __slots__ = ("_low", "_high", "_pending", "_buf", "_cur", "_nbits")

    def __init__(self) -> None:
        self._low = 0
        self._high = _TOP
        self._pending = 0
        self._buf = bytearray()
        self._cur = 0
        self._nbits = 0

    def _put(self, bit: int) -> None:
        self._cur = (self._cur << 1) | bit
        self._nbits += 1
        if self._nbits == 8:
            self._buf.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def _put_with_pending(self, bit: int) -> None:
        self._put(bit)
        inverse = 1 - bit
        for _ in range(self._pending):
            self._put(inverse)
        self._pending = 0

    def encode_bit(self, bit: int, p0: int) -> None:
        """Encode ``bit`` given ``p0``, the 16-bit scaled probability that it is 0."""
        low, high = self._low, self._high
        span = high - low + 1
        mid = low + ((span * p0) >> 16) - 1
        if mid < low:
            mid = low
        elif mid >= high:
            mid = high - 1

        if bit:
            low = mid + 1
        else:
            high = mid

        while True:
            if high < _HALF:
                self._put_with_pending(0)
            elif low >= _HALF:
                self._put_with_pending(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                # The interval straddles the midpoint: we cannot yet say which half we are
                # in, but we know the next two output bits will differ. Remember that.
                self._pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low = (low << 1) & _TOP
            high = ((high << 1) | 1) & _TOP

        self._low, self._high = low, high

    def finish(self) -> bytes:
        """Flush the interval and pad to a whole byte."""
        self._pending += 1
        if self._low < _QUARTER:
            self._put_with_pending(0)
        else:
            self._put_with_pending(1)
        while self._nbits:
            self._put(0)
        return bytes(self._buf)


class RangeDecoder:
    """Reads back what :class:`RangeEncoder` wrote, given the same probabilities."""

    __slots__ = ("_low", "_high", "_value", "_data", "_pos", "_nbits", "_cur")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self._cur = 0
        self._nbits = 0
        self._low = 0
        self._high = _TOP
        self._value = 0
        for _ in range(32):
            self._value = (self._value << 1) | self._get()

    def _get(self) -> int:
        """Next bit, zero-padded past the end.

        Padding rather than raising is deliberate: the encoder's final flush writes only
        enough bits to disambiguate the interval, so a correct stream is routinely read
        slightly past its last written bit.
        """
        if self._nbits == 0:
            if self._pos < len(self._data):
                self._cur = self._data[self._pos]
                self._pos += 1
            else:
                self._cur = 0
            self._nbits = 8
        self._nbits -= 1
        return (self._cur >> self._nbits) & 1

    def decode_bit(self, p0: int) -> int:
        low, high, value = self._low, self._high, self._value
        span = high - low + 1
        mid = low + ((span * p0) >> 16) - 1
        if mid < low:
            mid = low
        elif mid >= high:
            mid = high - 1

        if value <= mid:
            bit = 0
            high = mid
        else:
            bit = 1
            low = mid + 1

        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                low -= _QUARTER
                high -= _QUARTER
                value -= _QUARTER
            else:
                break
            low = (low << 1) & _TOP
            high = ((high << 1) | 1) & _TOP
            value = ((value << 1) | self._get()) & _TOP

        self._low, self._high, self._value = low, high, value
        return bit


class BitModel:
    """Adaptive P(0) per context, as 16-bit scaled integers."""

    __slots__ = ("probs",)

    def __init__(self, contexts: int) -> None:
        self.probs = np.full(contexts, _PROB_ONE >> 1, dtype=np.int64)

    def update(self, ctx: int, bit: int) -> None:
        p = int(self.probs[ctx])
        if bit:
            p -= p >> _RATE
            if p < _PROB_MIN:
                p = _PROB_MIN
        else:
            p += (_PROB_ONE - p) >> _RATE
            if p > _PROB_MAX:
                p = _PROB_MAX
        self.probs[ctx] = p


#: Signed magnitude of a modular byte: 0 and 255 are both "nearly zero", because residuals
#: wrap. Bucketing on this is what lets quiet regions get a sharper model than busy ones.
_BUCKETS = 8
_BUCKET_OF = np.zeros(256, dtype=np.int64)
for _b in range(256):
    _mag = min(_b, 256 - _b)
    if _mag == 0:
        _BUCKET_OF[_b] = 0
    elif _mag == 1:
        _BUCKET_OF[_b] = 1
    elif _mag == 2:
        _BUCKET_OF[_b] = 2
    elif _mag <= 4:
        _BUCKET_OF[_b] = 3
    elif _mag <= 8:
        _BUCKET_OF[_b] = 4
    elif _mag <= 16:
        _BUCKET_OF[_b] = 5
    elif _mag <= 48:
        _BUCKET_OF[_b] = 6
    else:
        _BUCKET_OF[_b] = 7

#: 256 tree nodes per bucket (node indices 1..255 are used; 0 is wasted for clarity).
_CONTEXTS = _BUCKETS * 256


def encode_bytes(payload: bytes) -> bytes:
    """Entropy-code a payload with an order-1 context-modelled arithmetic coder."""
    enc = RangeEncoder()
    model = BitModel(_CONTEXTS)
    probs = model.probs
    update = model.update
    encode_bit = enc.encode_bit
    bucket_of = _BUCKET_OF
    prev = 0

    for byte in payload:
        base = int(bucket_of[prev]) << 8
        node = 1
        for shift in range(7, -1, -1):
            bit = (byte >> shift) & 1
            ctx = base + node
            encode_bit(bit, int(probs[ctx]))
            update(ctx, bit)
            node = (node << 1) | bit
            if node > 255:
                node = 1  # unreachable: the tree is exactly 8 deep
        prev = byte
    return enc.finish()


def decode_bytes(coded: bytes, count: int) -> bytes:
    """Inverse of :func:`encode_bytes`. ``count`` is the original payload length."""
    if count and not coded:
        raise ValueError(f"range-coded stream is empty but {count} bytes were expected")
    dec = RangeDecoder(coded)
    model = BitModel(_CONTEXTS)
    probs = model.probs
    update = model.update
    decode_bit = dec.decode_bit
    bucket_of = _BUCKET_OF

    out = bytearray(count)
    prev = 0
    for index in range(count):
        base = int(bucket_of[prev]) << 8
        node = 1
        for _ in range(8):
            ctx = base + node
            bit = decode_bit(int(probs[ctx]))
            update(ctx, bit)
            node = (node << 1) | bit
        byte = node & 0xFF
        out[index] = byte
        prev = byte
    return bytes(out)
