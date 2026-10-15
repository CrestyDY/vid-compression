"""Byte-level read/write primitives.

The container packs its header as one ``struct.Struct``, which is one call rather than nine,
so this exists for the payloads that genuinely need variable-length framing. `TILED` is the
first consumer: its per-tile payload lengths are **varints**, because a fixed u32 wastes
three bytes on the common small case, and at sixteen tiles per frame that is kilobytes per
second of pure overhead.

Every read routes through ``_take``, which bounds-checks and raises with the offset and the
shortfall — a truncated payload should say where it ran out, not raise an opaque slice error.
"""

from __future__ import annotations

import struct

_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")


class TruncatedStream(Exception):
    """A read ran past the end of the buffer."""


class BitWriter:
    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = bytearray()

    def u8(self, value: int) -> None:
        self._buf += _U8.pack(value)

    def u16(self, value: int) -> None:
        self._buf += _U16.pack(value)

    def u32(self, value: int) -> None:
        self._buf += _U32.pack(value)

    def uvarint(self, value: int) -> None:
        """Base-128, little end first, high bit set on every byte but the last."""
        if value < 0:
            raise ValueError(f"uvarint cannot encode a negative value: {value}")
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                self._buf.append(byte | 0x80)
            else:
                self._buf.append(byte)
                return

    def blob(self, data: bytes) -> None:
        self._buf += data

    def tell(self) -> int:
        return len(self._buf)

    def getvalue(self) -> bytes:
        return bytes(self._buf)


class BitReader:
    __slots__ = ("_view", "_pos")

    def __init__(self, data: bytes) -> None:
        self._view = memoryview(data).toreadonly()
        self._pos = 0

    def _take(self, n: int) -> memoryview:
        end = self._pos + n
        if end > len(self._view):
            short = end - len(self._view)
            raise TruncatedStream(
                f"needed {n} bytes at offset {self._pos}, {short} short of the end"
            )
        chunk = self._view[self._pos : end]
        self._pos = end
        return chunk

    def u8(self) -> int:
        return _U8.unpack(self._take(1))[0]

    def u16(self) -> int:
        return _U16.unpack(self._take(2))[0]

    def u32(self) -> int:
        return _U32.unpack(self._take(4))[0]

    def uvarint(self) -> int:
        value = 0
        shift = 0
        while True:
            byte = self.u8()
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7

    def blob(self, n: int) -> bytes:
        return bytes(self._take(n))

    def tell(self) -> int:
        return self._pos

    def remaining(self) -> int:
        return len(self._view) - self._pos
