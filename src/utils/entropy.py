"""The entropy-coding stage - the last box in the pipeline.

Separate from everything else so reports can show pre-entropy and post-entropy sizes
independently: what the codec's representation bought and what generic compression bought
are different questions, and conflating them hides which one is doing the work.

**Two codecs, because they win on different structure.** Measured on this codec's real
payloads (docs/entropy-results.md):

- **zstd** is an LZ matcher. It excels at *exact repeats*, and a `BLOCK_DELTA` payload is
  90% zeros with runs hundreds of bytes long - one LZ match each. It encoded one such
  payload in 84 bytes against the range coder's 257.
- **range** is a context-modelled arithmetic coder. It excels at *skewed but non-repeating*
  distributions, which is what a dense intra-predicted `FULL` payload is. It beat zstd by
  3% there, and 5% on `noise`.

Neither dominates, so ``auto`` prices both per payload and keeps the smaller, paying one
byte to record which - the same argument the codec makes for modes, one layer down. It
cannot lose to zstd by more than that byte, and it picks up the range coder's wins where
they exist.

Which codec a file used is recorded in the container header, so a decoder is never
configured to match an encoder.

**Self-contained framing.** An arithmetic stream has no terminator, so a range-coded payload
carries its own decoded length as a leading varint. Keeping that inside the payload rather
than in the frame record means the container format did not have to change, and a payload
remains independently decodable - the same property every mode's payload has.
"""

from __future__ import annotations

import zstandard

from src.utils.bitstream import BitReader, BitWriter, TruncatedStream
from src.utils.errors import CorruptPayload
from src.utils.rangecoder import decode_bytes, encode_bytes

#: Fixed, so size comparisons across runs and versions stay meaningful.
LEVEL = 10

#: Stored in the container header. Values are part of the file format.
CODEC_ZSTD = 0
CODEC_RANGE = 1
CODEC_AUTO = 2

CODECS = ("zstd", "range", "auto")

_BY_NAME = {"zstd": CODEC_ZSTD, "range": CODEC_RANGE, "auto": CODEC_AUTO}
_BY_ID = {v: k for k, v in _BY_NAME.items()}

#: In ``auto`` each payload carries one byte naming the codec that won.
_PICK_ZSTD = 0
_PICK_RANGE = 1


def codec_id(name: str) -> int:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown entropy codec {name!r}; pick one of {CODECS}") from None


def codec_name(ident: int) -> str:
    try:
        return _BY_ID[ident]
    except KeyError:
        raise ValueError(
            f"unknown entropy codec id {ident}; this file was probably written by a "
            f"newer version of the codec"
        ) from None


def _zstd(payload: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=LEVEL).compress(payload)


def _unzstd(payload: bytes) -> bytes:
    try:
        return zstandard.ZstdDecompressor().decompress(payload)
    except zstandard.ZstdError as exc:
        raise CorruptPayload(str(exc)) from None


def _range(payload: bytes) -> bytes:
    """Length-prefixed, so the stream is self-terminating."""
    writer = BitWriter()
    writer.uvarint(len(payload))
    writer.blob(encode_bytes(payload))
    return writer.getvalue()


def _unrange(payload: bytes) -> bytes:
    reader = BitReader(payload)
    try:
        count = reader.uvarint()
    except TruncatedStream as exc:
        raise CorruptPayload(f"range-coded payload has no length prefix: {exc}") from None
    body = payload[reader.tell() :]
    if count and not body:
        raise CorruptPayload(f"range-coded payload claims {count} bytes but carries none")
    return decode_bytes(body, count)


def compress(payload: bytes, *, enabled: bool, codec: str = "zstd") -> bytes:
    """``enabled`` is keyword-only so a bare ``compress(data, True)`` can't happen."""
    if not enabled:
        return payload  # identity, not a copy
    if codec == "zstd":
        return _zstd(payload)
    if codec == "range":
        return _range(payload)
    if codec == "auto":
        as_zstd = _zstd(payload)
        as_range = _range(payload)
        if len(as_range) < len(as_zstd):
            return bytes([_PICK_RANGE]) + as_range
        return bytes([_PICK_ZSTD]) + as_zstd
    raise ValueError(f"unknown entropy codec {codec!r}; pick one of {CODECS}")


def decompress(payload: bytes, *, enabled: bool, codec: str = "zstd") -> bytes:
    """Undo :func:`compress`."""
    if not enabled:
        return payload
    if codec == "zstd":
        return _unzstd(payload)
    if codec == "range":
        return _unrange(payload)
    if codec == "auto":
        if not payload:
            raise CorruptPayload("auto-coded payload is empty; expected a codec byte")
        pick, body = payload[0], payload[1:]
        if pick == _PICK_ZSTD:
            return _unzstd(body)
        if pick == _PICK_RANGE:
            return _unrange(body)
        raise CorruptPayload(f"auto-coded payload names unknown codec {pick}")
    raise ValueError(f"unknown entropy codec {codec!r}; pick one of {CODECS}")
