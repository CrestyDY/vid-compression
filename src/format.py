"""The .myv container: a header, then one record per frame.

Deliberately minimal. There is no frame index yet, because without keyframes every seek
would still have to decode from frame 0 — an index would cost bytes and buy nothing. The
``version`` byte is what lets one be added later without breaking existing files.

::

    Header (28 bytes, little-endian)
      magic        4   b"MYV1"
      version      1
      pix_fmt      1   see src.models.pixel_format
      flags        1   bit0 = payloads are entropy-coded
      reserved     1
      width        4
      height       4
      fps_num      4
      fps_den      4   frame rate as a rational, so 30000/1001 stays exact
      frame_count  4   back-patched when the writer closes

    Frame record, repeated frame_count times
      frame_type   1   see src.models.frame.FrameType
      payload_len  4   length as stored, i.e. after entropy coding
      payload      payload_len bytes

The magic carries a trailing version digit so a future incompatible format can be told apart
at a glance, not only by reading the version byte.
"""

from __future__ import annotations

import struct
from typing import BinaryIO, Iterator

from src.models.frame import Frame, FrameType
from src.models.header import Header
from src.models.pixel_format import by_id
from src.utils.errors import BadMyvFile

MAGIC = b"MYV1"
VERSION = 1

_HEADER = struct.Struct("<4sBBBBIIIII")
HEADER_SIZE = _HEADER.size

_FRAME_HEADER = struct.Struct("<BI")
FRAME_HEADER_SIZE = _FRAME_HEADER.size

_FLAG_ENTROPY = 0b1

#: Byte offset of frame_count within the header, for back-patching.
_FRAME_COUNT_OFFSET = HEADER_SIZE - 4


def pack_header(header: Header) -> bytes:
    return _HEADER.pack(
        MAGIC,
        VERSION,
        header.pix_fmt.id,
        _FLAG_ENTROPY if header.entropy else 0,
        0,  # reserved
        header.width,
        header.height,
        header.fps_num,
        header.fps_den,
        header.frame_count,
    )


def unpack_header(data: bytes) -> Header:
    if len(data) < HEADER_SIZE:
        raise BadMyvFile(f"header is too short: {len(data)} of {HEADER_SIZE} bytes")
    magic, version, fmt_id, flags, _res, w, h, num, den, count = _HEADER.unpack(
        data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise BadMyvFile(f"bad magic {magic!r}; this is not a .myv file")
    if version != VERSION:
        raise BadMyvFile(
            f"file format version {version}, but this codec only understands {VERSION}"
        )
    try:
        pix_fmt = by_id(fmt_id)
    except ValueError as exc:
        raise BadMyvFile(str(exc)) from None
    return Header(
        width=w, height=h, fps_num=num, fps_den=den, frame_count=count,
        pix_fmt=pix_fmt, entropy=bool(flags & _FLAG_ENTROPY),
    )


class MyvWriter:
    """Streams frame records out, then back-patches the frame count on close.

    Frames are written as they arrive rather than buffered, so encoding a long video needs
    memory for two frames, not for the whole file. That does mean the output stream has to
    be seekable.
    """

    def __init__(self, stream: BinaryIO, header: Header) -> None:
        self._stream = stream
        self._start = stream.tell()
        self.header = header
        self.frames_written = 0
        self.bytes_written = 0
        self._closed = False
        self._write(pack_header(header))

    def _write(self, data: bytes) -> None:
        self._stream.write(data)
        self.bytes_written += len(data)

    def frame(self, frame_type: FrameType, payload: bytes) -> None:
        self._write(_FRAME_HEADER.pack(int(frame_type), len(payload)))
        self._write(payload)
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        end = self._stream.tell()
        self._stream.seek(self._start + _FRAME_COUNT_OFFSET)
        self._stream.write(struct.pack("<I", self.frames_written))
        self._stream.seek(end)

    def __enter__(self) -> "MyvWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MyvReader:
    """Reads the header eagerly, then yields frame records lazily."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._start = stream.tell()
        self.header = unpack_header(stream.read(HEADER_SIZE))
        self._frames_start = stream.tell()

    def frames(self) -> Iterator[Frame]:
        self._stream.seek(self._frames_start)
        for index in range(self.header.frame_count):
            head = self._stream.read(FRAME_HEADER_SIZE)
            if len(head) < FRAME_HEADER_SIZE:
                raise BadMyvFile(
                    f"file truncated: header promised {self.header.frame_count} frames, "
                    f"stream ended at frame {index}"
                )
            type_id, length = _FRAME_HEADER.unpack(head)
            try:
                frame_type = FrameType(type_id)
            except ValueError:
                raise BadMyvFile(
                    f"frame {index} has unknown frame type {type_id}"
                ) from None
            payload = self._stream.read(length)
            if len(payload) != length:
                raise BadMyvFile(
                    f"frame {index} truncated: expected {length} bytes, "
                    f"got {len(payload)}"
                )
            yield Frame(type=frame_type, payload=payload)
